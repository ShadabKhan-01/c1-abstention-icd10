"""Tests for random withholding and oracle controls.

Traces to SPEC §9, §15.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd

from c1.controls import run_oracle_control, run_random_control
from c1.metrics import compute_kept_metrics


@pytest.fixture
def test_codes_df() -> pd.DataFrame:
    """Fixture DataFrame representing 100 predicted test codes (60 correct, 30 unsupported, 10 invalid)."""
    return pd.DataFrame({
        "correct": [True] * 60 + [False] * 40,
        "unsupported": [False] * 60 + [True] * 30 + [False] * 10,
        "invalid": [False] * 60 + [False] * 30 + [True] * 10,
    })


def test_random_control_100_percent_equals_full_list(test_codes_df: pd.DataFrame) -> None:
    """Verify that random control at 100% coverage reproduces the full list exactly."""
    full_metrics = compute_kept_metrics(
        kept_correct=60,
        kept_unsupported=30,
        kept_invalid=10,
        total_predicted=100,
        total_gold=80,
        total_notes=20,
    )

    rand_ctrl = run_random_control(
        test_df=test_codes_df,
        realised_coverage=1.0,
        total_gold=80,
        total_notes=20,
        repeats=50,
        seed=7,
    )

    assert pytest.approx(rand_ctrl["coverage"], abs=1e-5) == full_metrics.coverage
    assert pytest.approx(rand_ctrl["precision"], abs=1e-5) == full_metrics.precision
    assert pytest.approx(rand_ctrl["recall"], abs=1e-5) == full_metrics.recall
    assert pytest.approx(rand_ctrl["micro_f1"], abs=1e-5) == full_metrics.micro_f1
    assert pytest.approx(rand_ctrl["ucr"], abs=1e-5) == full_metrics.ucr
    assert pytest.approx(rand_ctrl["icr"], abs=1e-5) == full_metrics.icr
    assert pytest.approx(rand_ctrl["review_load"], abs=1e-5) == full_metrics.review_load


def test_random_control_expectation_matches_full_list_precision() -> None:
    """Verify random-control mean precision approx full-list precision on large fixture."""
    rng = np.random.default_rng(2026)
    n = 2000
    p_true = 0.65
    is_corr = rng.uniform(size=n) < p_true

    df = pd.DataFrame({
        "correct": is_corr,
        "unsupported": ~is_corr,
        "invalid": [False] * n,
    })

    full_prec = float(np.mean(is_corr))

    # At 80% coverage:
    rand_ctrl = run_random_control(
        test_df=df,
        realised_coverage=0.8,
        total_gold=1500,
        total_notes=300,
        repeats=100,
        seed=42,
    )

    # In expectation, random control precision equals full-list precision
    assert pytest.approx(rand_ctrl["precision"], abs=0.015) == full_prec


def test_oracle_control_maximum_precision(test_codes_df: pd.DataFrame) -> None:
    """Verify oracle control keeps correct codes first."""
    # At 50% coverage of 100 codes -> 50 codes kept.
    # Since there are 60 correct codes, all 50 kept codes should be correct!
    oracle = run_oracle_control(
        test_df=test_codes_df,
        realised_coverage=0.5,
        total_gold=80,
        total_notes=20,
    )

    assert oracle["coverage"] == 0.5
    assert oracle["precision"] == 1.0  # 50/50 correct
    assert oracle["recall"] == 50 / 80
    assert oracle["ucr"] == 0.0
    assert oracle["icr"] == 0.0


def test_controls_determinism(test_codes_df: pd.DataFrame) -> None:
    """Verify identical seeds produce identical random control results."""
    r1 = run_random_control(test_codes_df, 0.8, 80, 20, repeats=100, seed=7)
    r2 = run_random_control(test_codes_df, 0.8, 80, 20, repeats=100, seed=7)
    assert r1 == r2

