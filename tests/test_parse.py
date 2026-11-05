"""Tests for LLM output parsing across greedy, sampled, and S2 generation.

Traces to SPEC §5.2, §15.
"""

from __future__ import annotations

import pytest

from c1.parse import parse_greedy_output, parse_s2_output


def test_parse_greedy_list_markers() -> None:
    """Verify stripping of various list markers ('-', '*', '•', '1.', '1)')."""
    raw = (
        "- E11.9\n"
        "* I10\n"
        "• J45.909\n"
        "1. K21.9\n"
        "2) M54.5\n"
    )
    pred, spans, failed = parse_greedy_output(raw)
    assert not failed
    assert pred == ["E11.9", "I10", "J45.909", "K21.9", "M54.5"]
    assert len(spans) == 5


def test_parse_greedy_trailing_text() -> None:
    """Verify trailing text after whitespace or punctuation is tolerated."""
    raw = (
        "E11.9 - Type 2 diabetes mellitus without complications\n"
        "I10: Essential (primary) hypertension\n"
        "J45.909 (Unspecified asthma)\n"
    )
    pred, spans, failed = parse_greedy_output(raw)
    assert not failed
    assert pred == ["E11.9", "I10", "J45.909"]


def test_parse_greedy_undotted_normalization() -> None:
    """Verify undotted codes are normalized with a dot after the 3rd character."""
    raw = "E119\nI10\nJ45909\n"
    pred, spans, failed = parse_greedy_output(raw)
    assert not failed
    assert pred == ["E11.9", "I10", "J45.909"]


def test_parse_greedy_preserves_invalid_codes() -> None:
    """Verify well-formed but invalid codes are preserved and NEVER repaired."""
    raw = "Z99.999\nE11.9\n"
    pred, spans, failed = parse_greedy_output(raw)
    assert not failed
    # Z99.999 matches syntax pattern but is not in 2024 ICD-10-CM
    assert pred == ["Z99.999", "E11.9"]


def test_parse_greedy_deduplication() -> None:
    """Verify deduplication preserves first occurrence order."""
    raw = (
        "1. E11.9\n"
        "2. I10\n"
        "3. E11.9\n"
        "4. I10\n"
        "5. J45.909\n"
    )
    pred, spans, failed = parse_greedy_output(raw)
    assert not failed
    assert pred == ["E11.9", "I10", "J45.909"]
    assert len(spans) == 3


def test_parse_greedy_empty_and_garbage() -> None:
    """Verify empty or non-code text results in parse_failed=True."""
    raw = "I do not find any codes in this note.\nThere are no diagnoses.\n"
    pred, spans, failed = parse_greedy_output(raw)
    assert failed is True
    assert pred == []
    assert spans == []


def test_parse_greedy_char_spans() -> None:
    """Verify extracted character spans match exact substrings in raw_output."""
    raw = "Line 1 note intro\n- E11.9 (diabetes)\n* I10: hypertension\n"
    pred, spans, failed = parse_greedy_output(raw)
    assert not failed
    assert len(spans) == 2

    span0 = spans[0]
    assert span0["code"] == "E11.9"
    assert raw[span0["start"] : span0["end"]] == span0["raw"]
    assert span0["raw"] == "E11.9"

    span1 = spans[1]
    assert span1["code"] == "I10"
    assert raw[span1["start"] : span1["end"]] == span1["raw"]
    assert span1["raw"] == "I10"


def test_parse_s2_output_formats() -> None:
    """Verify parsing of S2 scores with comma, colon, whitespace, and %."""
    raw = (
        "E11.9, 95\n"
        "I10: 80%\n"
        "J45.909 70\n"
        "K21.9, 150\n"  # should clip to 100 -> 1.0
        "M54.5, -20\n"  # should clip to 0 -> 0.0
    )
    greedy = ["E11.9", "I10", "J45.909", "K21.9", "M54.5", "Z99.999"]
    scores, failed = parse_s2_output(raw, greedy)

    assert scores["E11.9"] == 0.95
    assert scores["I10"] == 0.80
    assert scores["J45.909"] == 0.70
    assert scores["K21.9"] == 1.0
    assert scores["M54.5"] == 0.0
    # Z99.999 was not in S2 output
    assert scores["Z99.999"] is None
    assert failed == ["Z99.999"]
