"""Show source dataset schema and bounded sample records; no data transformation."""
import argparse
import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--revision", default=None)
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not 0 <= a.limit <= 5:
        p.error("--limit must be between 0 and 5 to avoid large accidental dumps")
    resolved_revision = HfApi().dataset_info(a.dataset, revision=a.revision).sha
    ds = load_dataset(a.dataset, data_dir=a.data_dir, split=a.split, revision=resolved_revision)
    result = {
        "dataset": a.dataset, "data_dir": a.data_dir, "split": a.split,
        "requested_revision": a.revision,
        "resolved_revision": resolved_revision,
        "license": ds.info.license,
        "source_fingerprint": ds._fingerprint,
        "rows": len(ds), "features": repr(ds.features), "columns": ds.column_names,
        "internal_split_counts": dict(Counter(str(value) for value in ds["split"])) if "split" in ds.column_names else None,
        "sample_records": [ds[i] for i in range(min(a.limit, len(ds)))],
        "warning": "Samples may contain dataset text. Do not publish without checking licenses/privacy.",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"Saved schema and {len(result['sample_records'])} sample record(s) to {out}")
    print(f"Columns: {ds.column_names}")


if __name__ == "__main__":
    main()
