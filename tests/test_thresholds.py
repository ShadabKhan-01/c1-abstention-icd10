"""Tests for threshold selection, dev-only guard, and persistence.

Traces to SPEC §8, §15.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import pandas as pd

from c1.thresholds import (
    compute_dev_thresholds,
    load_thresholds,
    pick_tau,
    save_thresholds,
)


def test_worked_example_4_pick_tau() -> None:
    """Verify SPEC §15 Worked Example 4:

    dev scores [1, 1, .8, .8, .6, .4, .2, .2, .1, 0]:
    target 0.9 => tau = 0.1
    target 0.8 => tau = 0.2
    target 0.6 => tau = 0.4
    target 1.0 => keeps all (tau = -inf)

    Ties: scores [1, .5, .5, .5, 0], target 0.6 => tau = 0.5 (realised coverage 0.8).
    """
    scores = [1.0, 1.0, 0.8, 0.8, 0.6, 0.4, 0.2, 0.2, 0.1, 0.0]

    assert pytest.approx(pick_tau(scores, 0.9), abs=1e-5) == 0.1
    assert pytest.approx(pick_tau(scores, 0.8), abs=1e-5) == 0.2
    assert pytest.approx(pick_tau(scores, 0.6), abs=1e-5) == 0.4
    tau_100 = pick_tau(scores, 1.0)
    assert math.isinf(tau_100) and tau_100 < 0

    # Ties case:
    tied_scores = [1.0, 0.5, 0.5, 0.5, 0.0]
    tau_tied = pick_tau(tied_scores, 0.6)
    assert pytest.approx(tau_tied, abs=1e-5) == 0.5
    # Realised coverage with tau=0.5 on tied_scores: 4 of 5 = 0.8 >= 0.6
    realised = sum(s >= tau_tied for s in tied_scores) / len(tied_scores)
    assert realised == 0.8


def test_thresholds_file_guard_dev_only() -> None:
    """Verify file guard: compute_dev_thresholds raises ValueError if non-dev split is provided."""
    # Mixed split DataFrame
    mixed_df = pd.DataFrame({
        "split": ["dev", "dev", "test"],
        "s1": [0.8, 0.7, 0.9],
        "correct": [True, False, True],
    })

    with pytest.raises(ValueError, match="File guard violation"):
        compute_dev_thresholds(mixed_df, signals=["s1"], targets=[0.8])

    # Pure dev DataFrame succeeds
    dev_df = pd.DataFrame({
        "split": ["dev", "dev"],
        "s1": [0.8, 0.7],
        "correct": [True, False],
    })
    thresh = compute_dev_thresholds(dev_df, signals=["s1"], targets=[0.8])
    assert "s1" in thresh
    assert "0.8" in thresh["s1"]


def test_save_and_load_thresholds(tmp_path: Path) -> None:
    """Verify thresholds.json atomic saving and loading with dev_codes_hash."""
    thresh_data = {
        "llama3": {
            "s1": {"1.0": -float("inf"), "0.8": 0.75},
        }
    }
    saved_path = save_thresholds(tmp_path, thresh_data, "dev_hash_123")
    assert saved_path.is_file()

    loaded = load_thresholds(saved_path)
    assert loaded["dev_codes_hash"] == "dev_hash_123"
    assert loaded["thresholds"]["llama3"]["s1"]["0.8"] == 0.75
    tau_loaded = loaded["thresholds"]["llama3"]["s1"]["1.0"]
    assert math.isinf(tau_loaded) and tau_loaded < 0
