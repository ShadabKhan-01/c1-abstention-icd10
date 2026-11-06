"""Pure metric functions for confidence-based abstention in ICD-10 coding.

Provides:
- AUROC: Rank-based with exact tie-breaking (average ranks).
- ECE: Expected Calibration Error across 10 equal-width bins on [0, 1].
- Kept-set metrics: coverage, precision, recall, micro-F1, Unsupported Code Rate (UCR),
  Invalid Code Rate (ICR), and review load (withheld codes per note).

Pure functions, no I/O dependencies.
Traces to SPEC §7, §15.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Data Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeptMetrics:
    """Evaluation metrics for a set of kept codes."""

    coverage: float
    precision: float
    recall: float
    micro_f1: float
    ucr: float
    icr: float
    review_load: float
    kept_count: int
    total_predicted: int
    total_gold: int
    total_notes: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary."""
        return {
            "coverage": self.coverage,
            "precision": self.precision,
            "recall": self.recall,
            "micro_f1": self.micro_f1,
            "ucr": self.ucr,
            "icr": self.icr,
            "review_load": self.review_load,
            "kept_count": self.kept_count,
            "total_predicted": self.total_predicted,
            "total_gold": self.total_gold,
            "total_notes": self.total_notes,
        }


# ---------------------------------------------------------------------------
# AUROC (SPEC §7, §15)
# ---------------------------------------------------------------------------


def compute_auroc(
    y_true: Sequence[bool | int],
    y_score: Sequence[float],
) -> float:
    """Compute AUROC for separating correct (positive) from wrong (negative) codes.

    Uses rank-sum calculation with average ranks for tied scores:
        U1 = R1 - n1 * (n1 + 1) / 2
        AUROC = U1 / (n1 * n0)

    If one class is empty (n1 == 0 or n0 == 0), returns float('nan').

    Args:
        y_true: Binary ground truth labels (True/1 for correct, False/0 for wrong).
        y_score: Continuous confidence scores.

    Returns:
        AUROC in [0.0, 1.0], or float('nan') if undefined.
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_score ({len(y_score)})"
        )

    y_arr = np.asarray(y_true, dtype=bool)
    score_arr = np.asarray(y_score, dtype=np.float64)

    n_pos = int(np.sum(y_arr))
    n_neg = len(y_arr) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Compute ranks with average tie-breaking (1-based ranks)
    ranks = rankdata(score_arr, method="average")
    rank_sum_pos = float(np.sum(ranks[y_arr]))

    u_pos = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    auroc = u_pos / (n_pos * n_neg)

    return float(auroc)


# ---------------------------------------------------------------------------
# ECE (SPEC §7, §15)
# ---------------------------------------------------------------------------


def compute_ece(
    y_true: Sequence[bool | int],
    y_score: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across 10 equal-width bins on [0, 1].

    Bins are [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]. The last bin is closed at 1.0.
    ECE = sum_{b: n_b > 0} (n_b / N) * |accuracy_b - mean_score_b|

    Applicable to probability signals S2, S3, S4 per SPEC §7.

    Args:
        y_true: Binary correctness labels (True/1 for correct, False/0 for wrong).
        y_score: Continuous confidence scores in [0.0, 1.0].
        n_bins: Number of equal-width bins (default 10).

    Returns:
        ECE value in [0.0, 1.0]. If input is empty, returns float('nan').
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_score ({len(y_score)})"
        )

    n_total = len(y_true)
    if n_total == 0:
        return float("nan")

    y_arr = np.asarray(y_true, dtype=bool)
    score_arr = np.asarray(y_score, dtype=np.float64)

    # Bin edges: [0.0, 0.1, ..., 1.0]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for b in range(n_bins):
        low = bin_edges[b]
        high = bin_edges[b + 1]

        if b == n_bins - 1:
            # Last bin is closed on both sides: [0.9, 1.0]
            mask = (score_arr >= low) & (score_arr <= high)
        else:
            mask = (score_arr >= low) & (score_arr < high)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y_arr[mask]))
            bin_conf = float(np.mean(score_arr[mask]))
            ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Kept-Set Metrics (SPEC §7, §15)
# ---------------------------------------------------------------------------


def compute_kept_metrics(
    kept_correct: int,
    kept_unsupported: int,
    kept_invalid: int,
    total_predicted: int,
    total_gold: int,
    total_notes: int,
) -> KeptMetrics:
    """Compute kept-set metrics from category counts and total reference sizes.

    Definitions per SPEC §7:
    - coverage = kept / predicted
    - precision = correct kept / kept
    - recall = correct kept / total gold codes of notes
    - micro-F1 = 2 * P * R / (P + R)
    - Unsupported Code Rate (UCR) = unsupported kept / kept
    - Invalid Code Rate (ICR) = invalid kept / kept
    - review load = (predicted - kept) / total notes

    Zero denominators return float('nan').

    Args:
        kept_correct: Number of correct codes in the kept set.
        kept_unsupported: Number of unsupported codes in the kept set.
        kept_invalid: Number of invalid codes in the kept set.
        total_predicted: Total number of predicted codes in the evaluation split.
        total_gold: Total number of gold codes across notes in the evaluation split.
        total_notes: Total number of notes in the evaluation split.

    Returns:
        KeptMetrics instance.
    """
    kept_count = kept_correct + kept_unsupported + kept_invalid

    coverage = kept_count / total_predicted if total_predicted > 0 else float("nan")
    precision = kept_correct / kept_count if kept_count > 0 else float("nan")
    recall = kept_correct / total_gold if total_gold > 0 else float("nan")

    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0.0:
        micro_f1 = 0.0 if not (math.isnan(precision) or math.isnan(recall)) else float("nan")
    else:
        micro_f1 = (2.0 * precision * recall) / (precision + recall)

    ucr = kept_unsupported / kept_count if kept_count > 0 else float("nan")
    icr = kept_invalid / kept_count if kept_count > 0 else float("nan")

    withheld_count = total_predicted - kept_count
    review_load = withheld_count / total_notes if total_notes > 0 else float("nan")

    return KeptMetrics(
        coverage=coverage,
        precision=precision,
        recall=recall,
        micro_f1=micro_f1,
        ucr=ucr,
        icr=icr,
        review_load=review_load,
        kept_count=kept_count,
        total_predicted=total_predicted,
        total_gold=total_gold,
        total_notes=total_notes,
    )

