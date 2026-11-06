"""Hypotheses H1 through H5 evaluation per SPEC §13 (Table IX).

Implements:
- H1: S1 separates right from wrong (AUROC(S1) > 0.5).
- H2: S1 > S2 (ΔAUROC = AUROC(S1) - AUROC(S2) > 0).
- H3: At least one signal is useful at ~80% coverage.
- H4: Abstention removes invalid codes proportionally more than unsupported
      (relative fall: ICR fall - UCR fall > 0; n/a if full-list ICR is 0).
- H5: Implicit correct codes are harder than explicit
      (ΔAUROC = AUROC_implicit - AUROC_explicit < 0 with shared negative pool).

Statuses: strictly "confirmed", "rejected", "inconclusive" (or "n/a" for H4 when full ICR is 0).

Traces to SPEC §13, §14 (Table IX), §15, §18.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from c1.bootstrap import BootstrapResult

logger = logging.getLogger(__name__)

HYPOTHESES_COLUMNS = [
    "model",
    "hypothesis",
    "claim",
    "estimate",
    "ci_lower",
    "ci_upper",
    "status",
    "details",
]


def classify_h1(ci_lower: float, ci_upper: float) -> str:
    """Classify H1: AUROC(S1) separates right from wrong (> 0.5)."""
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return "inconclusive"
    if ci_lower > 0.5:
        return "confirmed"
    if ci_upper < 0.5:
        return "rejected"
    return "inconclusive"


def classify_h2(ci_lower: float, ci_upper: float) -> str:
    """Classify H2: S1 > S2 (ΔAUROC > 0)."""
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return "inconclusive"
    if ci_lower > 0.0:
        return "confirmed"
    if ci_upper < 0.0:
        return "rejected"
    return "inconclusive"


def classify_h3(
    verdicts: dict[str, str],
    condition_cis: dict[str, tuple[tuple[float, float], tuple[float, float]]],
) -> str:
    """Classify H3: At least one signal is useful at ~80% coverage per SPEC §13.

    Confirmed if >= 1 signal is useful.
    Rejected if for every signal, at least one of (i)/(ii) has its interval entirely
    on the wrong side (precision interval <= 0 or UCR interval >= 0).
    Else inconclusive.
    """
    if any(v == "useful" for v in verdicts.values()):
        return "confirmed"

    all_wrong_side = True
    for sig, (dp_ci, du_ci) in condition_cis.items():
        dp_wrong = dp_ci[1] <= 0.0  # entire precision CI <= 0
        du_wrong = du_ci[0] >= 0.0  # entire UCR CI >= 0
        if not (dp_wrong or du_wrong):
            all_wrong_side = False
            break

    if all_wrong_side and len(condition_cis) > 0:
        return "rejected"

    return "inconclusive"


def classify_h4(full_icr: float, ci_lower: float, ci_upper: float) -> str:
    """Classify H4: Relative fall in ICR > relative fall in UCR."""
    if full_icr <= 0.0:
        return "n/a"
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return "inconclusive"
    if ci_lower > 0.0:
        return "confirmed"
    if ci_upper < 0.0:
        return "rejected"
    return "inconclusive"


def classify_h5(ci_lower: float, ci_upper: float) -> str:
    """Classify H5: Implicit correct codes harder than explicit (ΔAUROC < 0)."""
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return "inconclusive"
    if ci_upper < 0.0:
        return "confirmed"
    if ci_lower > 0.0:
        return "rejected"
    return "inconclusive"


def evaluate_model_hypotheses(
    model_name: str,
    best_signal: str,
    bootstrap_results: Sequence[BootstrapResult],
    hypotheses_dists: dict[str, Any],
    full_icr: float,
) -> list[dict[str, Any]]:
    """Evaluate H1 through H5 for a model using bootstrap distributions per SPEC §13.

    Returns:
        List of dict rows adhering to HYPOTHESES_COLUMNS.
    """
    b_map = {r.signal: r for r in bootstrap_results}
    verdicts = hypotheses_dists.get("verdicts", {r.signal: r.verdict for r in bootstrap_results})

    rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # H1: S1 separates right from wrong (AUROC(S1) > 0.5)
    # -------------------------------------------------------------------------
    s1_res = b_map.get("s1")
    if s1_res:
        h1_est = s1_res.auroc_mean
        h1_lo = s1_res.auroc_ci_lower
        h1_hi = s1_res.auroc_ci_upper
    else:
        h1_est, h1_lo, h1_hi = float("nan"), float("nan"), float("nan")

    h1_status = classify_h1(h1_lo, h1_hi)
    rows.append({
        "model": model_name,
        "hypothesis": "H1",
        "claim": "S1 separates right from wrong (AUROC > 0.5)",
        "estimate": round(h1_est, 6) if not np.isnan(h1_est) else "",
        "ci_lower": round(h1_lo, 6) if not np.isnan(h1_lo) else "",
        "ci_upper": round(h1_hi, 6) if not np.isnan(h1_hi) else "",
        "status": h1_status,
        "details": f"AUROC(S1) 95% CI [{h1_lo:.3f}, {h1_hi:.3f}] vs 0.5",
    })

    # -------------------------------------------------------------------------
    # H2: S1 > S2 (ΔAUROC = AUROC(S1) - AUROC(S2) > 0)
    # -------------------------------------------------------------------------
    delta_s1_s2 = hypotheses_dists.get("delta_auroc_s1_s2_dist", [])
    if delta_s1_s2 and len(delta_s1_s2) >= 10:
        h2_est = float(np.mean(delta_s1_s2))
        h2_lo = float(np.percentile(delta_s1_s2, 2.5))
        h2_hi = float(np.percentile(delta_s1_s2, 97.5))
    else:
        h2_est, h2_lo, h2_hi = float("nan"), float("nan"), float("nan")

    h2_status = classify_h2(h2_lo, h2_hi)
    rows.append({
        "model": model_name,
        "hypothesis": "H2",
        "claim": "S1 outperforms verbalised score S2 (ΔAUROC > 0)",
        "estimate": round(h2_est, 6) if not np.isnan(h2_est) else "",
        "ci_lower": round(h2_lo, 6) if not np.isnan(h2_lo) else "",
        "ci_upper": round(h2_hi, 6) if not np.isnan(h2_hi) else "",
        "status": h2_status,
        "details": f"ΔAUROC(S1 - S2) 95% CI [{h2_lo:.3f}, {h2_hi:.3f}] vs 0.0",
    })

    # -------------------------------------------------------------------------
    # H3: At least one signal is useful at ~80% coverage
    # -------------------------------------------------------------------------
    condition_cis = {
        r.signal: (
            (r.delta_precision_ci_lower, r.delta_precision_ci_upper),
            (r.delta_ucr_ci_lower, r.delta_ucr_ci_upper),
        )
        for r in bootstrap_results
    }
    h3_status = classify_h3(verdicts, condition_cis)
    useful_count = sum(1 for v in verdicts.values() if v == "useful")
    rows.append({
        "model": model_name,
        "hypothesis": "H3",
        "claim": "At least one signal is useful at ~80% coverage",
        "estimate": useful_count,
        "ci_lower": "",
        "ci_upper": "",
        "status": h3_status,
        "details": f"{useful_count}/{len(verdicts)} signals verified useful",
    })

    # -------------------------------------------------------------------------
    # H4: Abstention removes invalid codes proportionally more than unsupported
    # -------------------------------------------------------------------------
    h4_dist = hypotheses_dists.get("h4_relative_fall_dist", {}).get(best_signal, [])
    if h4_dist and len(h4_dist) >= 10:
        h4_est = float(np.mean(h4_dist))
        h4_lo = float(np.percentile(h4_dist, 2.5))
        h4_hi = float(np.percentile(h4_dist, 97.5))
    else:
        h4_est, h4_lo, h4_hi = float("nan"), float("nan"), float("nan")

    h4_status = classify_h4(full_icr, h4_lo, h4_hi)
    rows.append({
        "model": model_name,
        "hypothesis": "H4",
        "claim": "Relative fall in ICR > relative fall in UCR (best signal)",
        "estimate": round(h4_est, 6) if not np.isnan(h4_est) else "",
        "ci_lower": round(h4_lo, 6) if not np.isnan(h4_lo) else "",
        "ci_upper": round(h4_hi, 6) if not np.isnan(h4_hi) else "",
        "status": h4_status,
        "details": f"ICR fall - UCR fall 95% CI [{h4_lo:.3f}, {h4_hi:.3f}] (full ICR={full_icr:.4f})",
    })

    # -------------------------------------------------------------------------
    # H5: Implicit correct codes are harder than explicit (ΔAUROC < 0)
    # -------------------------------------------------------------------------
    h5_dist = hypotheses_dists.get("h5_delta_auroc_dist", {}).get(best_signal, [])
    if h5_dist and len(h5_dist) >= 10:
        h5_est = float(np.mean(h5_dist))
        h5_lo = float(np.percentile(h5_dist, 2.5))
        h5_hi = float(np.percentile(h5_dist, 97.5))
    else:
        h5_est, h5_lo, h5_hi = float("nan"), float("nan"), float("nan")

    h5_status = classify_h5(h5_lo, h5_hi)
    rows.append({
        "model": model_name,
        "hypothesis": "H5",
        "claim": "Implicit codes harder than explicit (AUROC_implicit - AUROC_explicit < 0)",
        "estimate": round(h5_est, 6) if not np.isnan(h5_est) else "",
        "ci_lower": round(h5_lo, 6) if not np.isnan(h5_lo) else "",
        "ci_upper": round(h5_hi, 6) if not np.isnan(h5_hi) else "",
        "status": h5_status,
        "details": f"ΔAUROC(impl - expl) 95% CI [{h5_lo:.3f}, {h5_hi:.3f}] vs 0.0",
    })

    return rows

