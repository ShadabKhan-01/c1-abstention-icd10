"""Tests for confidence signals S1 through S5 and S3 token alignment.

Traces to SPEC §6, §12, §15.
"""

from __future__ import annotations

import pytest

from c1.signals import (
    compute_s1,
    compute_s3_from_greedy,
    compute_s5,
    extract_s2,
    extract_s4,
)


def test_s1_computation_and_k_prefixes() -> None:
    """Verify S1 fraction calculation and k-prefix sensitivity (SPEC §6, §12)."""
    # 10 sample runs
    runs = [
        {"run": 1, "codes": ["E11.9", "I10"]},
        {"run": 2, "codes": ["E11.9"]},
        {"run": 3, "codes": ["E11.9", "I10"]},
        {"run": 4, "codes": ["I10"]},
        {"run": 5, "codes": ["E11.9"]},
        {"run": 6, "codes": ["E11.9", "I10"]},
        {"run": 7, "codes": ["E11.9"]},
        {"run": 8, "codes": ["E11.9"]},
        {"run": 9, "codes": ["E11.9", "I10"]},
        {"run": 10, "codes": ["E11.9"]},
    ]

    # Overall (k=10)
    # E11.9 is in 9 of 10 runs -> 0.90
    assert pytest.approx(compute_s1(runs, "E11.9"), 0.001) == 0.90
    # I10 is in 5 of 10 runs -> 0.50
    assert pytest.approx(compute_s1(runs, "I10"), 0.001) == 0.50
    # J45.909 is in 0 of 10 runs -> 0.00
    assert compute_s1(runs, "J45.909") == 0.0

    # Prefix k=3:
    # E11.9 in runs 1, 2, 3 -> 3/3 = 1.0
    assert pytest.approx(compute_s1(runs, "E11.9", k=3), 0.001) == 1.0
    # I10 in runs 1, 3 -> 2/3 = 0.6667
    assert pytest.approx(compute_s1(runs, "I10", k=3), 0.001) == 2 / 3

    # Prefix k=5:
    # E11.9 in runs 1, 2, 3, 5 -> 4/5 = 0.80
    assert pytest.approx(compute_s1(runs, "E11.9", k=5), 0.001) == 0.80
    # I10 in runs 1, 3, 4 -> 3/5 = 0.60
    assert pytest.approx(compute_s1(runs, "I10", k=5), 0.001) == 0.60


def test_s2_extraction_and_failure_handling() -> None:
    """Verify S2 extraction, score clipping, and failure status (SPEC §5.2)."""
    s2_record = {
        "scores": {
            "E11.9": 0.95,
            "I10": 0.80,
            "K21.9": None,
        },
        "failed_codes": ["K21.9", "Z99.999"],
    }

    # Normal scores
    s2_e119, f_e119 = extract_s2(s2_record, "E11.9")
    assert pytest.approx(s2_e119, 0.001) == 0.95
    assert f_e119 is False

    s2_i10, f_i10 = extract_s2(s2_record, "I10")
    assert pytest.approx(s2_i10, 0.001) == 0.80
    assert f_i10 is False

    # Failed / null score
    s2_k21, f_k21 = extract_s2(s2_record, "K21.9")
    assert s2_k21 == 0.0
    assert f_k21 is True

    # Missing from scores dict entirely
    s2_z99, f_z99 = extract_s2(s2_record, "Z99.999")
    assert s2_z99 == 0.0
    assert f_z99 is True

    # None record
    s2_none, f_none = extract_s2(None, "E11.9")
    assert s2_none == 0.0
    assert f_none is True


def test_s3_from_greedy_with_fake_tokenizer() -> None:
    """Verify S3 mean token probability from greedy records with fake tokenizer."""
    fake_vocab = {
        10: "Codes:\n",
        11: "E1",
        12: "1.9",
        13: "\n",
        14: " I10",
        15: "\n",
    }

    def fake_decode(token_ids: list[int]) -> str:
        return "".join(fake_vocab[tid] for tid in token_ids)

    # Generated text: "Codes:\nE11.9\n I10\n"
    # Token IDs: 10, 11, 12, 13, 14, 15
    # Token probs: 0.99, 0.85, 0.95, 0.99, 0.75, 0.99
    raw_text = fake_decode([10, 11, 12, 13, 14, 15])
    greedy_rec = {
        "raw_output": raw_text,
        "token_ids": [10, 11, 12, 13, 14, 15],
        "token_probs": [0.99, 0.85, 0.95, 0.99, 0.75, 0.99],
        "code_char_spans": [
            {"code": "E11.9", "start": raw_text.index("E11.9"), "end": raw_text.index("E11.9") + 5},
            {"code": "I10", "start": raw_text.index("I10"), "end": raw_text.index("I10") + 3},
        ],
    }

    # E11.9 overlaps tokens 11 (0.85) and 12 (0.95) -> mean = 0.90
    s3_e119 = compute_s3_from_greedy(greedy_rec, "E11.9", decode_fn=fake_decode)
    assert pytest.approx(s3_e119, 0.001) == 0.90

    # I10 overlaps token 14 (0.75) -> mean = 0.75
    s3_i10 = compute_s3_from_greedy(greedy_rec, "I10", decode_fn=fake_decode)
    assert pytest.approx(s3_i10, 0.001) == 0.75

    # Missing code returns 0.0
    assert compute_s3_from_greedy(greedy_rec, "Z99.999", decode_fn=fake_decode) == 0.0


def test_s4_extraction_and_fallback() -> None:
    """Verify S4 extraction from s4.jsonl records."""
    s4_rec = {
        "items": [
            {"code": "E11.9", "s4": 0.88},
            {"code": "I10", "s4": 0.65},
        ]
    }

    assert pytest.approx(extract_s4(s4_rec, "E11.9"), 0.001) == 0.88
    assert pytest.approx(extract_s4(s4_rec, "I10"), 0.001) == 0.65
    # Missing code falls back to 0.5
    assert extract_s4(s4_rec, "Z99.999") == 0.5
    assert extract_s4(None, "E11.9") == 0.5


def test_s5_computation() -> None:
    """Verify S5 arithmetic mean of S1 and S4."""
    assert pytest.approx(compute_s5(0.9, 0.7), 0.001) == 0.80
    assert pytest.approx(compute_s5(1.0, 0.0), 0.001) == 0.50
    assert pytest.approx(compute_s5(0.0, 0.0), 0.001) == 0.00
