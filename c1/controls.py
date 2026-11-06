"""Random withholding and oracle controls for ICD-10 confidence abstention.

Implements:
- Vectorized random withholding control across 100 seeded repeats.
- Oracle control keeping correct codes first in stable order.
- Both controls are evaluated at each signal's realised test coverage.

Traces to SPEC §9, §15.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from c1.metrics import compute_kept_metrics

logger = logging.getLogger(__name__)


def run_random_control(
    test_df: pd.DataFrame,
    realised_coverage: float,
    total_gold: int,
    total_notes: int,
    repeats: int = 100,
    seed: int = 7,
) -> dict[str, float]:
    """Simulate random withholding control keeping round(c * N) codes uniformly at random.

    Vectorized across 100 seeded repeats with numpy per SPEC §9.

    Args:
        test_df: DataFrame of test codes with boolean columns ('correct', 'unsupported', 'invalid').
        realised_coverage: Realised coverage fraction c of the paired signal.
        total_gold: Total number of gold codes in the test split notes.
        total_notes: Total number of notes in the test split.
        repeats: Number of random draws (default 100).
        seed: Random seed for RNG (default 7).

    Returns:
        Dict with mean coverage, precision, recall, micro_f1, ucr, icr, and review_load.
    """
    n_total = len(test_df)
    if n_total == 0:
        return {
            "coverage": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "micro_f1": float("nan"),
            "ucr": float("nan"),
            "icr": float("nan"),
            "review_load": float("nan"),
        }

    k_keep = int(round(realised_coverage * n_total))
    k_keep = max(0, min(n_total, k_keep))

    if k_keep == 0:
        review_load = (n_total - k_keep) / total_notes if total_notes > 0 else float("nan")
        return {
            "coverage": 0.0,
            "precision": float("nan"),
            "recall": 0.0 if total_gold > 0 else float("nan"),
            "micro_f1": 0.0,
            "ucr": float("nan"),
            "icr": float("nan"),
            "review_load": review_load,
        }

    # Extract boolean vectors
    is_correct = test_df["correct"].values.astype(bool)
    is_unsupported = test_df["unsupported"].values.astype(bool)
    is_invalid = test_df["invalid"].values.astype(bool)

    if k_keep == n_total:
        # Full list keeps everything deterministically
        c_count = int(np.sum(is_correct))
        u_count = int(np.sum(is_unsupported))
        inv_count = int(np.sum(is_invalid))
        km = compute_kept_metrics(
            kept_correct=c_count,
            kept_unsupported=u_count,
            kept_invalid=inv_count,
            total_predicted=n_total,
            total_gold=total_gold,
            total_notes=total_notes,
        )
        return {
            "coverage": km.coverage,
            "precision": km.precision,
            "recall": km.recall,
            "micro_f1": km.micro_f1,
            "ucr": km.ucr,
            "icr": km.icr,
            "review_load": km.review_load,
        }

    # Vectorized random sampling: shape (repeats, k_keep)
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(np.arange(n_total), (repeats, 1)), axis=1)
    sampled_indices = perms[:, :k_keep]

    # Compute category counts across rows: shape (repeats,)
    correct_draws = np.sum(is_correct[sampled_indices], axis=1)
    unsupported_draws = np.sum(is_unsupported[sampled_indices], axis=1)
    invalid_draws = np.sum(is_invalid[sampled_indices], axis=1)

    precisions = correct_draws / k_keep
    recalls = correct_draws / total_gold if total_gold > 0 else np.full(repeats, np.nan)

    # Micro-F1 per draw
    denom = precisions + recalls
    with np.errstate(divide="ignore", invalid="ignore"):
        micro_f1s = np.where(denom > 0, 2.0 * precisions * recalls / denom, 0.0)

    ucrs = unsupported_draws / k_keep
    icrs = invalid_draws / k_keep
    review_load = (n_total - k_keep) / total_notes if total_notes > 0 else float("nan")

    return {
        "coverage": float(k_keep / n_total),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "micro_f1": float(np.mean(micro_f1s)),
        "ucr": float(np.mean(ucrs)),
        "icr": float(np.mean(icrs)),
        "review_load": float(review_load),
    }


def run_oracle_control(
    test_df: pd.DataFrame,
    realised_coverage: float,
    total_gold: int,
    total_notes: int,
) -> dict[str, float]:
    """Compute oracle control keeping correct codes first (in stable order), then wrong codes.

    Keeps round(c * N) codes per SPEC §9.

    Args:
        test_df: DataFrame of test codes.
        realised_coverage: Realised coverage fraction c of the paired signal.
        total_gold: Total number of gold codes in the test split notes.
        total_notes: Total number of notes in the test split.

    Returns:
        Dict with oracle kept metrics.
    """
    n_total = len(test_df)
    if n_total == 0:
        return {
            "coverage": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "micro_f1": float("nan"),
            "ucr": float("nan"),
            "icr": float("nan"),
            "review_load": float("nan"),
        }

    k_keep = int(round(realised_coverage * n_total))
    k_keep = max(0, min(n_total, k_keep))

    if k_keep == 0:
        review_load = (n_total - k_keep) / total_notes if total_notes > 0 else float("nan")
        return {
            "coverage": 0.0,
            "precision": float("nan"),
            "recall": 0.0 if total_gold > 0 else float("nan"),
            "micro_f1": 0.0,
            "ucr": float("nan"),
            "icr": float("nan"),
            "review_load": review_load,
        }

    # Stable sort: correct codes first (True > False)
    sorted_df = test_df.sort_values(by="correct", ascending=False, kind="stable")
    kept_df = sorted_df.iloc[:k_keep]

    c_count = int(kept_df["correct"].sum())
    u_count = int(kept_df["unsupported"].sum())
    inv_count = int(kept_df["invalid"].sum())

    km = compute_kept_metrics(
        kept_correct=c_count,
        kept_unsupported=u_count,
        kept_invalid=inv_count,
        total_predicted=n_total,
        total_gold=total_gold,
        total_notes=total_notes,
    )
    return {
        "coverage": km.coverage,
        "precision": km.precision,
        "recall": km.recall,
        "micro_f1": km.micro_f1,
        "ucr": km.ucr,
        "icr": km.icr,
        "review_load": km.review_load,
    }
