"""Threshold selection and application for confidence-based abstention.

Computes coverage thresholds tau strictly on dev codes per SPEC §8.
For target 1.0, tau = -infinity (keeps all codes).
For target in {0.9, 0.8, 0.7, 0.6}, tau is the largest value such that
the share of dev codes with score >= tau is >= target.

Traces to SPEC §8, §15.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from c1.utils import sha256_string, write_json_atomic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Threshold Selection (SPEC §8, §15)
# ---------------------------------------------------------------------------


def pick_tau(scores: Sequence[float], target_coverage: float) -> float:
    """Find the largest score threshold tau achieving >= target_coverage on dev codes.

    Rules per SPEC §8:
    - Target 1.0: returns -infinity (keeps all codes).
    - Target in {0.9, 0.8, 0.7, 0.6}: tau is the largest value such that
      count(score >= tau) / len(scores) >= target_coverage.
    - If scores tie, realised coverage may exceed target; tau is not adjusted.

    Args:
        scores: Sequence of dev confidence scores.
        target_coverage: Target coverage fraction (e.g. 1.0, 0.9, 0.8, 0.7, 0.6).

    Returns:
        Float threshold tau. For target >= 1.0, returns -float('inf').
    """
    if target_coverage >= 1.0:
        return -float("inf")

    score_arr = np.asarray(scores, dtype=np.float64)
    n_total = len(score_arr)
    if n_total == 0:
        return -float("inf")

    # Get unique candidate thresholds in descending order
    unique_scores = np.sort(np.unique(score_arr))[::-1]

    # Find the largest score whose share >= target_coverage
    for tau in unique_scores:
        share = np.sum(score_arr >= tau) / n_total
        if share >= target_coverage:
            return float(tau)

    # Fallback if no score meets target (e.g. empty target)
    return float(unique_scores[-1])


# ---------------------------------------------------------------------------
# Dev Threshold Computation & Persistence (SPEC §8)
# ---------------------------------------------------------------------------


def compute_dev_thresholds(
    dev_df: pd.DataFrame,
    signals: Sequence[str] = ("s1", "s2", "s3", "s4", "s5"),
    targets: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6),
) -> dict[str, dict[str, float]]:
    """Compute thresholds for all signals across targets using dev codes only.

    Hard constraint per SPEC §8:
    This function reads only dev split codes. Test labels must NEVER be accessed.

    Args:
        dev_df: DataFrame containing dev codes only (split == 'dev').
        signals: Sequence of signal names (s1..s5).
        targets: Target coverage fractions.

    Returns:
        Dict mapping signal -> target_str -> tau.
    """
    if "split" in dev_df.columns:
        unique_splits = set(dev_df["split"].unique())
        if unique_splits != {"dev"}:
            raise ValueError(
                f"File guard violation: compute_dev_thresholds received non-dev splits: {unique_splits}"
            )

    thresholds: dict[str, dict[str, float]] = {}

    for sig in signals:
        if sig not in dev_df.columns:
            logger.warning("Signal column %s not found in dev DataFrame; skipping", sig)
            continue

        sig_scores = dev_df[sig].astype(float).values
        thresholds[sig] = {}

        for target in targets:
            tau = pick_tau(sig_scores, target)
            thresholds[sig][str(target)] = tau

    return thresholds


def save_thresholds(
    out_dir: str | Path,
    thresholds_by_model: dict[str, dict[str, dict[str, float]]],
    dev_codes_hash: str,
) -> Path:
    """Save thresholds.json atomically with dev codes hash.

    Args:
        out_dir: Run output directory.
        thresholds_by_model: Dict mapping model -> signal -> target -> tau.
        dev_codes_hash: SHA-256 hash of the dev rows of codes.csv.

    Returns:
        Path to saved thresholds.json.
    """
    out_path = Path(out_dir) / "thresholds.json"
    payload = {
        "dev_codes_hash": dev_codes_hash,
        "thresholds": thresholds_by_model,
    }
    write_json_atomic(out_path, payload)
    return out_path


def load_thresholds(path: str | Path) -> dict[str, Any]:
    """Load and return thresholds.json content."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"thresholds.json not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
