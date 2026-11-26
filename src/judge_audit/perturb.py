"""Presentation edits applied to one response of a pair (prototype).

Every edit is a pure, deterministic text function. An edit that would leave
the text unchanged is recorded as not applicable instead of producing a
duplicate. Edited versions inherit the original pair's partition, and each
is marked `unreviewed` until two reviewers check it (plan section 4).

Edits (see DECISIONS.md for why these and what is still open):
  neutral_whitespace  no-op: strip trailing spaces per line, end with one newline
  format_bullets      put "- " before each paragraph, keeping its line breaks
                      (2+ paragraphs, no existing list, heading or code block)
  strip_markdown      remove bold/italic markers, heading marks and list markers
  lengthen_filler     add a content-free opening and closing sentence
  content_removal     positive control: drop the second half of the paragraphs
                      (or sentences, for a single paragraph)
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

FILLER_OPEN = "Sure! Here is my response to your request."
FILLER_CLOSE = "I hope this helps! Let me know if you have any other questions."

_LIST_LINE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def neutral_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.rstrip().split("\n")]
    return "\n".join(lines) + "\n"


def format_bullets(text: str) -> str:
    paras = _paragraphs(text)
    if len(paras) < 2 or _LIST_LINE.search(text) or re.search(r"^\s*#", text, re.MULTILINE) or "```" in text:
        return text
    return "\n".join("- " + p.strip().replace("\n", "\n  ") for p in paras)


def strip_markdown(text: str) -> str:
    if "```" in text:  # code blocks: leave alone rather than risk breaking code
        return text
    out = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    out = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", out)
    out = re.sub(r"^\s{0,3}#{1,6}\s+", "", out, flags=re.MULTILINE)
    out = _LIST_LINE.sub("", out)
    return out


def lengthen_filler(text: str) -> str:
    return f"{FILLER_OPEN}\n\n{text.strip()}\n\n{FILLER_CLOSE}"


def _sentences(paragraph: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s]


def content_removal(text: str) -> str:
    paras = _paragraphs(text)
    if len(paras) >= 2:
        return "\n\n".join(paras[: (len(paras) + 1) // 2])
    sentences = _sentences(paras[0]) if paras else []
    if len(sentences) >= 2:
        return " ".join(sentences[: (len(sentences) + 1) // 2])
    return text


EDITS = {
    "neutral_whitespace": neutral_whitespace,
    "format_bullets": format_bullets,
    "strip_markdown": strip_markdown,
    "lengthen_filler": lengthen_filler,
    "content_removal": content_removal,
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_edits(pairs: pd.DataFrame, edit_ids: list[str]) -> pd.DataFrame:
    """One row per (pair, edit, target response). `applicable` is False when the edit changes nothing."""
    unknown = set(edit_ids) - set(EDITS)
    if unknown:
        raise ValueError(f"Unknown edits: {sorted(unknown)}")
    rows = []
    for pair in pairs.itertuples(index=False):
        for edit_id in edit_ids:
            for target in (1, 2):
                original = getattr(pair, f"response_{target}")
                edited = EDITS[edit_id](original)
                rows.append({
                    "pair_id": pair.pair_id, "edit_id": edit_id, "target_response": target,
                    "applicable": edited != original,
                    "original_sha256": sha(original), "edited_sha256": sha(edited),
                    "original_chars": len(original), "edited_chars": len(edited),
                    "edited_text": edited, "validation_status": "unreviewed",
                })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=("h1h3", "audit"), default="h1h3",
                   help="h1h3: ranks 1..edit-k per group; audit: every eligible audit pair (H4)")
    p.add_argument("--pairs", default="data/processed/helpsteer_pairs.parquet")
    p.add_argument("--ranking", default="data/splits/h1h3_ranking.csv")
    p.add_argument("--manifest", default="data/splits/helpsteer_manifest.csv")
    p.add_argument("--edit-k", type=int, default=100)
    p.add_argument("--edits", nargs="+", default=list(EDITS))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    pairs = pd.read_parquet(a.pairs)
    if a.source == "h1h3":
        ranking = pd.read_csv(a.ranking, dtype={"margin_group": str})
        chosen = set(ranking.loc[ranking["rank"] <= a.edit_k, "pair_id"])
    else:
        manifest = pd.read_csv(a.manifest, keep_default_na=False)
        chosen = set(manifest.loc[manifest.partition == "audit", "pair_id"]) & set(pairs.loc[pairs.audit_eligible, "pair_id"])
    pairs = pairs[pairs.pair_id.isin(chosen)]
    edits = build_edits(pairs, a.edits)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    edits.to_parquet(out, index=False)
    summary = edits.groupby("edit_id").applicable.agg(["sum", "size"])
    print(f"Wrote {len(edits)} edit rows for {pairs.pair_id.nunique()} pairs to {out}")
    print(summary.rename(columns={"sum": "applicable", "size": "rows"}).to_string())


if __name__ == "__main__":
    main()
