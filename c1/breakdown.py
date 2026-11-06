"""Breakdown by note feature and confident mistake extraction.

Implements:
- Note-level non-exclusive strata breakdown (implicit, explicit, negation, family_history,
  distractor_medication, rare, plus "all notes" reconciliation row) per SPEC §11 (Table VI).
- Evaluates test codes using the best signal (highest dev AUROC) at primary target (0.8).
- Confident mistakes extraction (wrong test codes kept by best signal at primary target)
  with top 5 cases per cause tag truncated to 600 characters.

Traces to SPEC §11, §14 (Table VI), §15, §18.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from c1.data import Note, compute_note_features
from c1.metrics import compute_auroc

logger = logging.getLogger(__name__)

STRATA_ORDER = [
    "implicit",
    "explicit",
    "negation",
    "family_history",
    "distractor_medication",
    "rare",
    "all notes",
]

CONFIDENT_MISTAKES_COLUMNS = [
    "model",
    "id",
    "code",
    "score",
    "error_type",
    "cause_tag",
    "rank_in_tag",
    "note_text",
]


def compute_breakdown_by_feature(
    models: Sequence[str],
    codes_df: pd.DataFrame,
    test_notes: Sequence[Note],
    best_signals: dict[str, str],
    thresholds: dict[str, dict[str, dict[str, float]]],
    primary_target: float = 0.8,
) -> list[dict[str, Any]]:
    """Compute Table VI breakdown by note feature per SPEC §11.

    Args:
        models: List of model names.
        codes_df: codes.csv DataFrame.
        test_notes: Sequence of Note instances in the test split.
        best_signals: Map of model_name -> best_signal name.
        thresholds: Map of model -> signal -> {target_str: tau}.
        primary_target: Target coverage fraction (default 0.8).

    Returns:
        List of dict rows for by_feature.csv.
    """
    target_str = str(primary_target)
    notes_by_id = {n.id: n for n in test_notes}
    note_features = {n.id: compute_note_features(n) for n in test_notes}

    # Filter test split codes
    test_codes_df = codes_df[codes_df["split"] == "test"].copy()

    rows: list[dict[str, Any]] = []

    for m in models:
        m_df = test_codes_df[test_codes_df["model"] == m].copy()
        best_sig = best_signals.get(m, "s1")
        tau = float(thresholds.get(m, {}).get(best_sig, {}).get(target_str, -float("inf")))

        for stratum in STRATA_ORDER:
            if stratum == "all notes":
                stratum_note_ids = set(notes_by_id.keys())
            else:
                stratum_note_ids = {nid for nid, feats in note_features.items() if feats.get(stratum, False)}

            s_df = m_df[m_df["id"].isin(stratum_note_ids)]
            n_codes = len(s_df)

            if n_codes == 0:
                rows.append({
                    "model": m,
                    "stratum": stratum,
                    "best_signal": best_sig,
                    "tau": tau if (tau != -float("inf") and not np.isneginf(tau)) else "-inf",
                    "n_codes": 0,
                    "auroc": "",
                    "precision_full": "",
                    "precision_kept": "",
                    "ucr_kept": "",
                })
                continue

            # AUROC
            auc = compute_auroc(s_df["correct"], s_df[best_sig])

            # Full list precision
            c_full = int(s_df["correct"].sum())
            p_full = c_full / n_codes

            # Kept list
            if tau == -float("inf") or np.isneginf(tau):
                kept_df = s_df
            else:
                kept_df = s_df[s_df[best_sig] >= tau]

            n_kept = len(kept_df)
            c_kept = int(kept_df["correct"].sum())
            u_kept = int(kept_df["unsupported"].sum())

            p_kept = c_kept / n_kept if n_kept > 0 else float("nan")
            u_kept_rate = u_kept / n_kept if n_kept > 0 else float("nan")

            rows.append({
                "model": m,
                "stratum": stratum,
                "best_signal": best_sig,
                "tau": round(tau, 6) if (tau != -float("inf") and not np.isneginf(tau)) else "-inf",
                "n_codes": n_codes,
                "auroc": round(auc, 6) if not np.isnan(auc) else "",
                "precision_full": round(p_full, 6),
                "precision_kept": round(p_kept, 6) if not np.isnan(p_kept) else "",
                "ucr_kept": round(u_kept_rate, 6) if not np.isnan(u_kept_rate) else "",
            })

    return rows


def extract_confident_mistakes(
    models: Sequence[str],
    codes_df: pd.DataFrame,
    test_notes: Sequence[Note],
    best_signals: dict[str, str],
    thresholds: dict[str, dict[str, dict[str, float]]],
    primary_target: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    """Extract wrong test codes kept by best signal at primary target per SPEC §11.

    Returns:
        tuple (mistake_rows, cause_counts) where mistake_rows contains columns of
        CONFIDENT_MISTAKES_COLUMNS with top 5 cases per cause tag having note_text
        truncated to 600 characters, and cause_counts maps (model, cause_tag) -> total_count.
    """
    target_str = str(primary_target)
    notes_by_id = {n.id: n for n in test_notes}

    test_codes_df = codes_df[codes_df["split"] == "test"].copy()

    mistake_rows: list[dict[str, Any]] = []
    cause_counts: dict[tuple[str, str], int] = {}

    for m in models:
        m_df = test_codes_df[test_codes_df["model"] == m].copy()
        best_sig = best_signals.get(m, "s1")
        tau = float(thresholds.get(m, {}).get(best_sig, {}).get(target_str, -float("inf")))

        # Filter wrong codes kept
        is_wrong = ~m_df["correct"].values.astype(bool)
        is_kept = (m_df[best_sig].values >= tau) if (tau != -float("inf") and not np.isneginf(tau)) else np.ones(len(m_df), dtype=bool)

        wrong_kept_df = m_df[is_wrong & is_kept].copy()

        # Group by cause_tag
        tags = sorted(wrong_kept_df["cause_tag"].fillna("other").unique())
        # Ensure priority order: negated, family_history, distractor_medication, other, then any others
        priority = {"negated": 1, "family_history": 2, "distractor_medication": 3, "other": 4}
        tags.sort(key=lambda t: (priority.get(t, 5), t))

        for tag in tags:
            tag_df = wrong_kept_df[wrong_kept_df["cause_tag"].fillna("other") == tag].copy()
            total_tag_count = len(tag_df)
            cause_counts[(m, tag)] = total_tag_count

            # Sort descending by score, tie-break by code
            tag_df.sort_values(by=[best_sig, "code"], ascending=[False, True], inplace=True)

            for rank, (_, row) in enumerate(tag_df.iterrows(), start=1):
                nid = str(row["id"])
                note = notes_by_id.get(nid)
                # Truncate synthetic note text to 600 characters for top 5 cases per tag
                note_text_snippet = note.text[:600] if (rank <= 5 and note is not None) else ""

                err_type = "invalid" if bool(row["invalid"]) else "unsupported"

                mistake_rows.append({
                    "model": m,
                    "id": nid,
                    "code": str(row["code"]),
                    "score": round(float(row[best_sig]), 6),
                    "error_type": err_type,
                    "cause_tag": tag,
                    "rank_in_tag": rank,
                    "note_text": note_text_snippet,
                })

    return mistake_rows, cause_counts
