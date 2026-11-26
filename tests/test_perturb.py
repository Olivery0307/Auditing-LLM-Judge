import pandas as pd
import pytest

from judge_audit.perturb import (
    FILLER_CLOSE,
    FILLER_OPEN,
    build_edits,
    content_removal,
    format_bullets,
    lengthen_filler,
    neutral_whitespace,
    strip_markdown,
)


def test_neutral_whitespace_only_touches_whitespace():
    assert neutral_whitespace("a  \nb") == "a\nb\n"
    assert neutral_whitespace("a\nb\n").split() == ["a", "b"]


def test_format_bullets_needs_plain_paragraphs():
    assert format_bullets("First para.\n\nSecond para.") == "- First para.\n- Second para."
    assert format_bullets("Only one paragraph.") == "Only one paragraph."
    assert format_bullets("Line one\nline two\n\nNext.") == "- Line one\n  line two\n- Next."
    assert format_bullets("Intro.\n\n- already\n- a list") == "Intro.\n\n- already\n- a list"
    assert format_bullets("Text.\n\n```\ncode\n```") == "Text.\n\n```\ncode\n```"


def test_strip_markdown_keeps_words():
    assert strip_markdown("## Title\n**Bold** and *it*\n- item\n1. step") == "Title\nBold and it\nitem\nstep"
    assert strip_markdown("2 * 3 * 4") == "2 * 3 * 4"
    assert strip_markdown("```\n**x**\n```") == "```\n**x**\n```"


def test_lengthen_and_remove():
    assert lengthen_filler(" Body ") == f"{FILLER_OPEN}\n\nBody\n\n{FILLER_CLOSE}"
    assert content_removal("P1.\n\nP2.\n\nP3.\n\nP4.") == "P1.\n\nP2."
    assert content_removal("P1.\n\nP2.\n\nP3.") == "P1.\n\nP2."
    assert content_removal("One. Two. Three. Four.") == "One. Two."
    assert content_removal("Single sentence") == "Single sentence"


def test_build_edits_marks_not_applicable_and_targets_each_side():
    pairs = pd.DataFrame([{"pair_id": "p", "response_1": "One para", "response_2": "A.\n\nB."}])
    edits = build_edits(pairs, ["format_bullets", "lengthen_filler"])
    assert len(edits) == 4
    row = edits.set_index(["edit_id", "target_response"])
    assert not row.loc[("format_bullets", 1), "applicable"]
    assert row.loc[("format_bullets", 2), "edited_text"] == "- A.\n- B."
    assert (edits.validation_status == "unreviewed").all()
    with pytest.raises(ValueError):
        build_edits(pairs, ["nope"])
