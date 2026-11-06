"""Tests for pure metric functions: AUROC, ECE, and kept-set metrics.

Traces to SPEC §7, §15.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score

from c1.metrics import compute_auroc, compute_ece, compute_kept_metrics


def test_worked_example_1_auroc() -> None:
    """Verify SPEC §15 Worked Example 1:

    correct-code scores {0.9, 0.8, 0.5}, wrong-code scores {0.7, 0.5, 0.2}.
    Pairs: 3 + 3 + 1.5 = 7.5 of 9 => AUROC = 0.8333...
    """
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.8, 0.5, 0.7, 0.5, 0.2]

    auroc = compute_auroc(y_true, y_score)
    expected = 7.5 / 9.0  # 0.8333333333333334
    assert pytest.approx(auroc, abs=1e-5) == expected

    # Cross-check against sklearn.metrics.roc_auc_score
    sklearn_auc = roc_auc_score(y_true, y_score)
    assert pytest.approx(auroc, abs=1e-7) == sklearn_auc


def test_auroc_brute_force_and_sklearn_cross_check() -> None:
    """Verify compute_auroc matches pairwise brute-force and sklearn on random fixtures."""
    rng = np.random.default_rng(2026)
    for _ in range(10):
        n_samples = rng.integers(20, 100)
        y_true = rng.integers(0, 2, size=n_samples)
        # Add some ties by rounding scores to 1 decimal
        y_score = np.round(rng.uniform(0.0, 1.0, size=n_samples), 1)

        if np.sum(y_true) == 0 or np.sum(y_true) == n_samples:
            continue

        calc_auc = compute_auroc(y_true, y_score)
        sk_auc = roc_auc_score(y_true, y_score)
        assert pytest.approx(calc_auc, abs=1e-7) == sk_auc

        # Brute force pairwise calculation
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        concordant = 0.0
        for p in pos_scores:
            for n in neg_scores:
                if p > n:
                    concordant += 1.0
                elif p == n:
                    concordant += 0.5
        bf_auc = concordant / (len(pos_scores) * len(neg_scores))
        assert pytest.approx(calc_auc, abs=1e-7) == bf_auc


def test_auroc_shuffled_and_oracle_sanity() -> None:
    """Verify SPEC §15 sanity checks:

    - shuffled scores => AUROC approx 0.5
    - oracle ordering => AUROC = 1.0
    - single-class edge case => AUROC = NaN
    """
    rng = np.random.default_rng(42)
    n = 2000
    y_true = np.array([1] * 1000 + [0] * 1000)

    # 1. Shuffled random scores
    random_scores = rng.uniform(0.0, 1.0, size=n)
    shuffled_auc = compute_auroc(y_true, random_scores)
    assert 0.46 <= shuffled_auc <= 0.54

    # 2. Oracle ordering
    oracle_scores = np.array([0.9] * 1000 + [0.1] * 1000)
    oracle_auc = compute_auroc(y_true, oracle_scores)
    assert oracle_auc == 1.0

    # 3. Single-class edge case
    assert math.isnan(compute_auroc([1, 1, 1], [0.5, 0.6, 0.7]))
    assert math.isnan(compute_auroc([0, 0, 0], [0.5, 0.6, 0.7]))


def test_worked_example_2_kept_metrics() -> None:
    """Verify SPEC §15 Worked Example 2:

    2 notes, 10 predicted codes (6 correct, 3 unsupported, 1 invalid),
    8 gold codes in total; kept set = 5 codes (4 correct, 1 unsupported, 0 invalid).
    Expected: coverage 0.5, precision 0.8, recall 0.5, micro-F1 0.6154,
    UCR 0.2, ICR 0.0, review load 2.5.
    """
    m = compute_kept_metrics(
        kept_correct=4,
        kept_unsupported=1,
        kept_invalid=0,
        total_predicted=10,
        total_gold=8,
        total_notes=2,
    )

    assert m.coverage == 0.5
    assert m.precision == 0.8
    assert m.recall == 0.5
    assert pytest.approx(m.micro_f1, abs=1e-4) == 0.6154
    assert m.ucr == 0.2
    assert m.icr == 0.0
    assert m.review_load == 2.5


def test_worked_example_3_ece() -> None:
    """Verify SPEC §15 Worked Example 3:

    scores {0.05, 0.25, 0.85, 0.95} with correctness {0, 1, 1, 1} => ECE = 0.25.
    """
    y_score = [0.05, 0.25, 0.85, 0.95]
    y_true = [0, 1, 1, 1]

    ece = compute_ece(y_true, y_score, n_bins=10)
    assert pytest.approx(ece, abs=1e-5) == 0.25


def test_kept_metrics_identities_and_zero_denominators() -> None:
    """Verify identity and zero denominator safety for kept-set metrics."""
    # When predicted == gold and all kept
    m_perfect = compute_kept_metrics(
        kept_correct=10,
        kept_unsupported=0,
        kept_invalid=0,
        total_predicted=10,
        total_gold=10,
        total_notes=2,
    )
    assert m_perfect.coverage == 1.0
    assert m_perfect.precision == 1.0
    assert m_perfect.recall == 1.0
    assert m_perfect.micro_f1 == 1.0
    assert m_perfect.ucr == 0.0
    assert m_perfect.icr == 0.0
    assert m_perfect.review_load == 0.0

    # Zero kept codes
    m_zero = compute_kept_metrics(
        kept_correct=0,
        kept_unsupported=0,
        kept_invalid=0,
        total_predicted=10,
        total_gold=10,
        total_notes=2,
    )
    assert m_zero.coverage == 0.0
    assert math.isnan(m_zero.precision)
    assert m_zero.recall == 0.0
    assert math.isnan(m_zero.micro_f1)
    assert math.isnan(m_zero.ucr)
    assert math.isnan(m_zero.icr)
    assert m_zero.review_load == 5.0
