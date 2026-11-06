"""Unit tests for note-level bootstrap, verdict rule, and recommendation logic.

Traces to SPEC §10, §15, §18.
"""

from __future__ import annotations

import pandas as pd
import pytest

from c1.bootstrap import (
    evaluate_recommendation,
    evaluate_verdict,
    run_model_bootstrap,
)
from c1.data import Distractor, GoldCode, Note


# ---------------------------------------------------------------------------
# Verdict Rule Branch Tests (SPEC §10)
# ---------------------------------------------------------------------------


def test_evaluate_verdict_all_pass() -> None:
    """Branch 1: All three conditions pass -> useful."""
    cond1, cond2, cond3, verdict = evaluate_verdict(
        delta_precision_ci=(0.02, 0.08),  # lower > 0
        delta_ucr_ci=(-0.10, -0.01),      # upper < 0
        observed_delta_icr=-0.005,        # observed <= 0
    )
    assert cond1 is True
    assert cond2 is True
    assert cond3 is True
    assert verdict == "useful"


def test_evaluate_verdict_cond1_fails() -> None:
    """Branch 2: Condition (i) fails (precision CI lower <= 0) -> not shown to help."""
    cond1, cond2, cond3, verdict = evaluate_verdict(
        delta_precision_ci=(-0.01, 0.08),  # lower <= 0
        delta_ucr_ci=(-0.10, -0.01),
        observed_delta_icr=-0.005,
    )
    assert cond1 is False
    assert cond2 is True
    assert cond3 is True
    assert verdict == "not shown to help"


def test_evaluate_verdict_cond2_fails() -> None:
    """Branch 3: Condition (ii) fails (UCR CI upper >= 0) -> not shown to help."""
    cond1, cond2, cond3, verdict = evaluate_verdict(
        delta_precision_ci=(0.02, 0.08),
        delta_ucr_ci=(-0.10, 0.01),        # upper >= 0
        observed_delta_icr=-0.005,
    )
    assert cond1 is True
    assert cond2 is False
    assert cond3 is True
    assert verdict == "not shown to help"


def test_evaluate_verdict_cond3_fails() -> None:
    """Branch 4: Condition (iii) fails (observed delta ICR > 0) -> not shown to help."""
    cond1, cond2, cond3, verdict = evaluate_verdict(
        delta_precision_ci=(0.02, 0.08),
        delta_ucr_ci=(-0.10, -0.01),
        observed_delta_icr=0.001,          # observed > 0
    )
    assert cond1 is True
    assert cond2 is True
    assert cond3 is False
    assert verdict == "not shown to help"


# ---------------------------------------------------------------------------
# Recommendation Rule Branch Tests (SPEC §10, paper §4.4)
# ---------------------------------------------------------------------------


def test_evaluate_recommendation_none_useful() -> None:
    """Branch 1: No useful signal -> recommend 'none'."""
    verdicts = {"s1": "not shown to help", "s2": "not shown to help"}
    auroc_cis = {"s1": (0.6, 0.7), "s2": (0.5, 0.6)}
    auroc_means = {"s1": 0.65, "s2": 0.55}
    extra_calls = {"s1": 10.0, "s2": 1.0}

    rec, rationale = evaluate_recommendation(verdicts, auroc_cis, auroc_means, extra_calls)
    assert rec == "none"
    assert "No signal met" in rationale


def test_evaluate_recommendation_single_useful() -> None:
    """Branch 2: Exactly one signal is useful -> recommend that signal."""
    verdicts = {"s1": "not shown to help", "s2": "useful", "s3": "not shown to help"}
    auroc_cis = {"s1": (0.6, 0.7), "s2": (0.7, 0.8), "s3": (0.5, 0.6)}
    auroc_means = {"s1": 0.65, "s2": 0.75, "s3": 0.55}
    extra_calls = {"s1": 10.0, "s2": 1.0, "s3": 0.0}

    rec, rationale = evaluate_recommendation(verdicts, auroc_cis, auroc_means, extra_calls)
    assert rec == "s2"
    assert "Cheapest useful signal" in rationale


def test_evaluate_recommendation_cheapest_when_no_interval_dominance() -> None:
    """Branch 3: Multiple useful signals, cheapest chosen because higher AUROC intervals overlap."""
    verdicts = {"s2": "useful", "s4": "useful"}
    # S4 has higher mean AUROC, but its CI (0.75, 0.85) overlaps S2's CI (0.70, 0.80) (0.75 <= 0.80)
    auroc_cis = {"s2": (0.70, 0.80), "s4": (0.75, 0.85)}
    auroc_means = {"s2": 0.75, "s4": 0.80}
    extra_calls = {"s2": 1.0, "s4": 4.0}

    rec, rationale = evaluate_recommendation(verdicts, auroc_cis, auroc_means, extra_calls)
    assert rec == "s2"
    assert "Cheapest useful signal" in rationale


def test_evaluate_recommendation_dominating_auroc_switches_choice() -> None:
    """Branch 4: Multiple useful signals, higher AUROC signal is chosen because its CI is strictly above cheapest's CI."""
    verdicts = {"s2": "useful", "s1": "useful"}
    # S1 CI (0.85, 0.92) is strictly above S2 CI (0.70, 0.80) because 0.85 > 0.80
    auroc_cis = {"s2": (0.70, 0.80), "s1": (0.85, 0.92)}
    auroc_means = {"s2": 0.75, "s1": 0.88}
    extra_calls = {"s2": 1.0, "s1": 10.0}

    rec, rationale = evaluate_recommendation(verdicts, auroc_cis, auroc_means, extra_calls)
    assert rec == "s1"
    assert "lies entirely above cheapest useful signal s2" in rationale


# ---------------------------------------------------------------------------
# Note-Level Resampling & Determinism Tests (SPEC §10, §15)
# ---------------------------------------------------------------------------


def _create_mock_notes_and_codes() -> tuple[list[Note], pd.DataFrame]:
    notes = [
        Note(id="n1", text="Note 1 text", gold=[GoldCode(code="E11.9", type="explicit", rare=False)]),
        Note(id="n2", text="Note 2 text", gold=[GoldCode(code="I10", type="implicit", rare=False)]),
        Note(id="n3", text="Note 3 text", gold=[GoldCode(code="J44.9", type="explicit", rare=False)]),
    ]
    rows = [
        # Note n1 has 2 codes (1 correct, 1 wrong)
        {"id": "n1", "code": "E11.9", "split": "test", "correct": True, "unsupported": False, "invalid": False,
         "gold_type": "explicit", "gold_rare": False, "s1": 0.9, "s2": 0.8, "s3": 0.7, "s4": 0.85, "s5": 0.875},
        {"id": "n1", "code": "E10.9", "split": "test", "correct": False, "unsupported": True, "invalid": False,
         "gold_type": "", "gold_rare": "", "s1": 0.2, "s2": 0.3, "s3": 0.2, "s4": 0.25, "s5": 0.225},
        # Note n2 has 1 code (correct)
        {"id": "n2", "code": "I10", "split": "test", "correct": True, "unsupported": False, "invalid": False,
         "gold_type": "implicit", "gold_rare": False, "s1": 0.8, "s2": 0.75, "s3": 0.8, "s4": 0.9, "s5": 0.85},
        # Note n3 has 2 codes (both wrong)
        {"id": "n3", "code": "I50.9", "split": "test", "correct": False, "unsupported": True, "invalid": False,
         "gold_type": "", "gold_rare": "", "s1": 0.1, "s2": 0.2, "s3": 0.15, "s4": 0.1, "s5": 0.1},
        {"id": "n3", "code": "Z99.9", "split": "test", "correct": False, "unsupported": False, "invalid": True,
         "gold_type": "", "gold_rare": "", "s1": 0.05, "s2": 0.1, "s3": 0.1, "s4": 0.05, "s5": 0.05},
    ]
    df = pd.DataFrame(rows)
    return notes, df


def test_bootstrap_determinism_same_seed() -> None:
    """Bootstrap with identical seed yields identical intervals."""
    notes, df = _create_mock_notes_and_codes()
    thresholds = {sig: {"0.8": 0.5} for sig in ("s1", "s2", "s3", "s4", "s5")}

    res1, rec1, _ = run_model_bootstrap(
        model_name="mock_model",
        test_df=df,
        test_notes=notes,
        thresholds=thresholds,
        b_repeats=50,
        seed=42,
    )

    res2, rec2, _ = run_model_bootstrap(
        model_name="mock_model",
        test_df=df,
        test_notes=notes,
        thresholds=thresholds,
        b_repeats=50,
        seed=42,
    )

    assert rec1 == rec2
    for r1, r2 in zip(res1, res2):
        assert r1.signal == r2.signal
        assert r1.auroc_mean == r2.auroc_mean
        assert r1.delta_precision_ci_lower == r2.delta_precision_ci_lower
        assert r1.delta_precision_ci_upper == r2.delta_precision_ci_upper
        assert r1.delta_ucr_ci_lower == r2.delta_ucr_ci_lower
        assert r1.delta_ucr_ci_upper == r2.delta_ucr_ci_upper
        assert r1.verdict == r2.verdict

