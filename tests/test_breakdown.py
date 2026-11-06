"""Unit tests for breakdown by note feature and confident mistakes.

Traces to SPEC §11, §15, §18.
"""

from __future__ import annotations

import pandas as pd
import pytest

from c1.breakdown import (
    STRATA_ORDER,
    compute_breakdown_by_feature,
    extract_confident_mistakes,
)
from c1.data import Distractor, GoldCode, Note


def _create_mock_notes_and_codes_for_breakdown() -> tuple[list[Note], pd.DataFrame]:
    # 3 notes with varied features
    notes = [
        # Note 1: implicit gold code, negated distractor
        Note(
            id="n1",
            text="Patient with long history of diabetes. " * 30,  # ~1200 chars
            gold=[GoldCode(code="E11.9", type="implicit", rare=False)],
            distractors=[Distractor(code="I50.9", kind="negated")],
        ),
        # Note 2: explicit gold code, medication distractor
        Note(
            id="n2",
            text="Hypertension diagnosed yesterday.",
            gold=[GoldCode(code="I10", type="explicit", rare=False)],
            distractors=[Distractor(code="Z79.899", kind="distractor_medication")],
        ),
        # Note 3: rare gold code, family_history distractor
        Note(
            id="n3",
            text="Rare metabolic disorder and family history of cancer.",
            gold=[GoldCode(code="E70.0", type="explicit", rare=True)],
            distractors=[Distractor(code="Z80.0", kind="family_history")],
        ),
    ]

    rows = [
        # n1 predicted codes: E11.9 (correct), I50.9 (unsupported, negated distractor, high score)
        {"id": "n1", "model": "mock", "split": "test", "code": "E11.9", "correct": True, "unsupported": False, "invalid": False,
         "cause_tag": "", "s1": 0.95},
        {"id": "n1", "model": "mock", "split": "test", "code": "I50.9", "correct": False, "unsupported": True, "invalid": False,
         "cause_tag": "negated", "s1": 0.90},
        # n2 predicted codes: I10 (correct), Z79.899 (unsupported, medication distractor)
        {"id": "n2", "model": "mock", "split": "test", "code": "I10", "correct": True, "unsupported": False, "invalid": False,
         "cause_tag": "", "s1": 0.85},
        {"id": "n2", "model": "mock", "split": "test", "code": "Z79.899", "correct": False, "unsupported": True, "invalid": False,
         "cause_tag": "distractor_medication", "s1": 0.80},
        # n3 predicted codes: E70.0 (correct), Z80.0 (unsupported, family_history distractor), Z999 (invalid, other)
        {"id": "n3", "model": "mock", "split": "test", "code": "E70.0", "correct": True, "unsupported": False, "invalid": False,
         "cause_tag": "", "s1": 0.75},
        {"id": "n3", "model": "mock", "split": "test", "code": "Z80.0", "correct": False, "unsupported": True, "invalid": False,
         "cause_tag": "family_history", "s1": 0.70},
        {"id": "n3", "model": "mock", "split": "test", "code": "Z999", "correct": False, "unsupported": False, "invalid": True,
         "cause_tag": "other", "s1": 0.65},
    ]
    df = pd.DataFrame(rows)
    return notes, df


def test_by_feature_strata_and_reconciliation_row() -> None:
    """Test that all strata are computed and 'all notes' reconciles with test split."""
    notes, df = _create_mock_notes_and_codes_for_breakdown()
    best_signals = {"mock": "s1"}
    thresholds = {"mock": {"s1": {"0.8": 0.70}}}

    rows = compute_breakdown_by_feature(
        models=["mock"],
        codes_df=df,
        test_notes=notes,
        best_signals=best_signals,
        thresholds=thresholds,
        primary_target=0.8,
    )

    strata_found = [r["stratum"] for r in rows]
    for expected_s in STRATA_ORDER:
        assert expected_s in strata_found

    all_notes_row = next(r for r in rows if r["stratum"] == "all notes")
    assert all_notes_row["n_codes"] == len(df)
    # Total correct is 3 out of 7 -> precision_full = 3/7 = 0.428571
    assert abs(all_notes_row["precision_full"] - round(3 / 7, 6)) < 1e-4

    # Check implicit stratum: only note n1 (2 codes)
    implicit_row = next(r for r in rows if r["stratum"] == "implicit")
    assert implicit_row["n_codes"] == 2


def test_confident_mistakes_filtering_and_truncation() -> None:
    """Confident mistakes contains only wrong kept codes, sorted by score, with 600-char text truncation."""
    notes, df = _create_mock_notes_and_codes_for_breakdown()
    best_signals = {"mock": "s1"}
    # Tau = 0.70 keeps scores >= 0.70 (I50.9 at 0.90, Z79.899 at 0.80, Z80.0 at 0.70)
    # Z999 has score 0.65 < 0.70 so it is withheld!
    thresholds = {"mock": {"s1": {"0.8": 0.70}}}

    mistakes, counts = extract_confident_mistakes(
        models=["mock"],
        codes_df=df,
        test_notes=notes,
        best_signals=best_signals,
        thresholds=thresholds,
        primary_target=0.8,
    )

    # 3 wrong codes were kept (I50.9, Z79.899, Z80.0)
    assert len(mistakes) == 3
    mistake_codes = {m["code"] for m in mistakes}
    assert "E11.9" not in mistake_codes  # correct code never in mistakes
    assert "Z999" not in mistake_codes   # withheld wrong code not in kept mistakes
    assert "I50.9" in mistake_codes
    assert "Z79.899" in mistake_codes
    assert "Z80.0" in mistake_codes

    # Note 1 has >1000 characters, so top-1 mistake for negated (I50.9) must have note_text of exactly 600 characters
    n1_mistake = next(m for m in mistakes if m["code"] == "I50.9")
    assert len(n1_mistake["note_text"]) == 600
    assert n1_mistake["rank_in_tag"] == 1
    assert n1_mistake["cause_tag"] == "negated"
    assert n1_mistake["error_type"] == "unsupported"

