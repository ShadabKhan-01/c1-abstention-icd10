"""Unit tests for hypotheses H1 through H5 classification branches.

Traces to SPEC §13, §15, §18.
"""

from __future__ import annotations

import pytest

from c1.hypotheses import (
    classify_h1,
    classify_h2,
    classify_h3,
    classify_h4,
    classify_h5,
)


# ---------------------------------------------------------------------------
# H1 Tests (AUROC(S1) > 0.5)
# ---------------------------------------------------------------------------


def test_classify_h1_confirmed() -> None:
    """H1 confirmed when CI lower bound > 0.5."""
    assert classify_h1(ci_lower=0.55, ci_upper=0.75) == "confirmed"


def test_classify_h1_rejected() -> None:
    """H1 rejected when CI upper bound < 0.5."""
    assert classify_h1(ci_lower=0.30, ci_upper=0.48) == "rejected"


def test_classify_h1_inconclusive() -> None:
    """H1 inconclusive when CI contains 0.5."""
    assert classify_h1(ci_lower=0.45, ci_upper=0.65) == "inconclusive"


# ---------------------------------------------------------------------------
# H2 Tests (S1 > S2: ΔAUROC > 0)
# ---------------------------------------------------------------------------


def test_classify_h2_confirmed() -> None:
    """H2 confirmed when ΔAUROC CI lower bound > 0."""
    assert classify_h2(ci_lower=0.02, ci_upper=0.15) == "confirmed"


def test_classify_h2_rejected() -> None:
    """H2 rejected when ΔAUROC CI upper bound < 0."""
    assert classify_h2(ci_lower=-0.12, ci_upper=-0.01) == "rejected"


def test_classify_h2_inconclusive() -> None:
    """H2 inconclusive when ΔAUROC CI contains 0."""
    assert classify_h2(ci_lower=-0.05, ci_upper=0.08) == "inconclusive"


# ---------------------------------------------------------------------------
# H3 Tests (At least one signal is useful at ~80% coverage)
# ---------------------------------------------------------------------------


def test_classify_h3_confirmed() -> None:
    """H3 confirmed when >= 1 signal is verified useful."""
    verdicts = {"s1": "useful", "s2": "not shown to help"}
    condition_cis = {
        "s1": ((0.01, 0.05), (-0.05, -0.01)),
        "s2": ((-0.02, 0.01), (-0.02, 0.02)),
    }
    assert classify_h3(verdicts, condition_cis) == "confirmed"


def test_classify_h3_rejected() -> None:
    """H3 rejected when for every signal, at least one of (i)/(ii) is on the wrong side."""
    verdicts = {"s1": "not shown to help", "s2": "not shown to help"}
    # s1 has precision CI <= 0 (wrong side for cond 1)
    # s2 has UCR CI >= 0 (wrong side for cond 2)
    condition_cis = {
        "s1": ((-0.05, -0.01), (-0.05, 0.02)),  # dp upper <= 0
        "s2": ((-0.01, 0.04), (0.01, 0.08)),    # du lower >= 0
    }
    assert classify_h3(verdicts, condition_cis) == "rejected"


def test_classify_h3_inconclusive() -> None:
    """H3 inconclusive when no signal is useful, but not all intervals are on the wrong side."""
    verdicts = {"s1": "not shown to help", "s2": "not shown to help"}
    # Neither signal is useful, but intervals straddle 0
    condition_cis = {
        "s1": ((-0.02, 0.05), (-0.05, 0.02)),
        "s2": ((-0.03, 0.04), (-0.04, 0.03)),
    }
    assert classify_h3(verdicts, condition_cis) == "inconclusive"


# ---------------------------------------------------------------------------
# H4 Tests (Relative fall: ICR fall - UCR fall > 0)
# ---------------------------------------------------------------------------


def test_classify_h4_na_when_full_icr_zero() -> None:
    """H4 is n/a when full-list ICR is 0."""
    assert classify_h4(full_icr=0.0, ci_lower=0.1, ci_upper=0.3) == "n/a"


def test_classify_h4_confirmed() -> None:
    """H4 confirmed when CI lower bound > 0."""
    assert classify_h4(full_icr=0.05, ci_lower=0.05, ci_upper=0.25) == "confirmed"


def test_classify_h4_rejected() -> None:
    """H4 rejected when CI upper bound < 0."""
    assert classify_h4(full_icr=0.05, ci_lower=-0.20, ci_upper=-0.02) == "rejected"


def test_classify_h4_inconclusive() -> None:
    """H4 inconclusive when CI contains 0."""
    assert classify_h4(full_icr=0.05, ci_lower=-0.08, ci_upper=0.12) == "inconclusive"


# ---------------------------------------------------------------------------
# H5 Tests (Implicit harder than explicit: AUROC_impl - AUROC_expl < 0)
# ---------------------------------------------------------------------------


def test_classify_h5_confirmed() -> None:
    """H5 confirmed when ΔAUROC CI upper bound < 0 (implicit AUROC is strictly lower)."""
    assert classify_h5(ci_lower=-0.15, ci_upper=-0.02) == "confirmed"


def test_classify_h5_rejected() -> None:
    """H5 rejected when ΔAUROC CI lower bound > 0 (implicit AUROC is strictly higher)."""
    assert classify_h5(ci_lower=0.01, ci_upper=0.12) == "rejected"


def test_classify_h5_inconclusive() -> None:
    """H5 inconclusive when ΔAUROC CI contains 0."""
    assert classify_h5(ci_lower=-0.06, ci_upper=0.05) == "inconclusive"
