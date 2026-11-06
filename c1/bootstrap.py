"""Note-level bootstrap resampling, verdict evaluation, and recommendation logic.

Implements:
- Note-level bootstrap (B=1000, seeded) resampling test notes with replacement per SPEC §10.
- Vectorized per-note summary statistics and multiplicity weighting.
- Paired random control recomputed inside each resample across 100 draws.
- Metric differences (Δprecision, ΔUCR, ΔICR = signal - random) and 95% percentile intervals.
- AUROC intervals per signal and AUROC differences needed for hypotheses.
- Strict 3-condition verdict rule ("useful" vs "not shown to help").
- Recommendation rule balancing extra calls against non-overlapping higher AUROC.

Traces to SPEC §10, §13, §14 (Table V), §15, §18.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from c1.data import Note
from c1.metrics import compute_auroc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapResult:
    """Summary of bootstrap resampling results for a single (model, signal)."""

    model: str
    signal: str
    target_coverage: float
    realised_coverage: float
    auroc_mean: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    delta_precision_mean: float
    delta_precision_ci_lower: float
    delta_precision_ci_upper: float
    delta_ucr_mean: float
    delta_ucr_ci_lower: float
    delta_ucr_ci_upper: float
    delta_icr_observed: float
    delta_icr_mean: float
    delta_icr_ci_lower: float
    delta_icr_ci_upper: float
    cond1_precision: bool
    cond2_ucr: bool
    cond3_icr: bool
    verdict: str
    skipped_resamples: int


def evaluate_verdict(
    delta_precision_ci: tuple[float, float],
    delta_ucr_ci: tuple[float, float],
    observed_delta_icr: float,
) -> tuple[bool, bool, bool, str]:
    """Evaluate pre-defined 3-condition verdict rule per SPEC §10.

    Conditions:
    (i)   95% interval of Δprecision lies entirely above 0 (ci_lower > 0).
    (ii)  95% interval of ΔUCR lies entirely below 0 (ci_upper < 0).
    (iii) Observed test ΔICR <= 0 (kept codes' Invalid Code Rate is not higher than random).

    Returns:
        tuple (cond1, cond2, cond3, verdict) where verdict is "useful" if all 3 pass,
        else "not shown to help".
    """
    cond1 = bool(delta_precision_ci[0] > 0.0)
    cond2 = bool(delta_ucr_ci[1] < 0.0)
    cond3 = bool(observed_delta_icr <= 0.0)

    verdict = "useful" if (cond1 and cond2 and cond3) else "not shown to help"
    return cond1, cond2, cond3, verdict


def evaluate_recommendation(
    verdicts: dict[str, str],
    auroc_cis: dict[str, tuple[float, float]],
    auroc_means: dict[str, float],
    extra_calls: dict[str, float],
) -> tuple[str, str]:
    """Apply recommendation rule per SPEC §10 and paper §4.4.

    Among useful signals, recommend the one with fewest extra calls, unless
    another useful signal's AUROC 95% interval lies ENTIRELY ABOVE the cheapest
    one's interval (other_ci_lower > cheap_ci_upper), in which case recommend the
    useful signal with the highest AUROC among those. If none is useful, write "none".

    Args:
        verdicts: Map of signal name to verdict ("useful" or "not shown to help").
        auroc_cis: Map of signal name to (ci_lower, ci_upper).
        auroc_means: Map of signal name to mean AUROC.
        extra_calls: Map of signal name to mean extra calls per note.

    Returns:
        tuple of (recommended_signal, rationale).
    """
    useful_signals = [s for s, v in verdicts.items() if v == "useful"]
    if not useful_signals:
        return "none", "No signal met all three usefulness criteria"

    # Sort useful signals by extra calls ascending, breaking ties deterministically
    sorted_by_cost = sorted(useful_signals, key=lambda s: (extra_calls.get(s, 999.0), s))
    cheapest = sorted_by_cost[0]
    cheap_ci_upper = auroc_cis[cheapest][1]

    # Check for useful signals whose AUROC interval is strictly above cheapest's interval
    dominating_signals = [
        s for s in useful_signals
        if s != cheapest and auroc_cis[s][0] > cheap_ci_upper
    ]

    if dominating_signals:
        # Pick the one with the highest mean AUROC among dominating signals
        best_dominating = max(dominating_signals, key=lambda s: (auroc_means.get(s, 0.0), -extra_calls.get(s, 999.0)))
        rationale = (
            f"AUROC 95% CI [{auroc_cis[best_dominating][0]:.3f}, {auroc_cis[best_dominating][1]:.3f}] "
            f"lies entirely above cheapest useful signal {cheapest} CI [{auroc_cis[cheapest][0]:.3f}, {cheap_ci_upper:.3f}]"
        )
        return best_dominating, rationale

    rationale = f"Cheapest useful signal ({extra_calls.get(cheapest, 0.0):.1f} extra calls/note)"
    return cheapest, rationale


def _simulate_random_control_draws(
    is_correct: np.ndarray,
    is_unsupported: np.ndarray,
    is_invalid: np.ndarray,
    k_keep: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Fast simulation of random withholding on a resampled code pool."""
    n_total = len(is_correct)
    if n_total == 0 or k_keep == 0:
        return float("nan"), float("nan"), float("nan")

    if k_keep >= n_total:
        c_count = int(np.sum(is_correct))
        u_count = int(np.sum(is_unsupported))
        inv_count = int(np.sum(is_invalid))
        return c_count / n_total, u_count / n_total, inv_count / n_total

    # Vectorized random permutations across repeats
    perms = rng.permuted(np.tile(np.arange(n_total), (repeats, 1)), axis=1)
    sampled = perms[:, :k_keep]

    correct_draws = np.sum(is_correct[sampled], axis=1)
    unsupported_draws = np.sum(is_unsupported[sampled], axis=1)
    invalid_draws = np.sum(is_invalid[sampled], axis=1)

    mean_p = float(np.mean(correct_draws / k_keep))
    mean_u = float(np.mean(unsupported_draws / k_keep))
    mean_i = float(np.mean(invalid_draws / k_keep))
    return mean_p, mean_u, mean_i


def run_model_bootstrap(
    model_name: str,
    test_df: pd.DataFrame,
    test_notes: Sequence[Note],
    thresholds: dict[str, dict[str, float]],
    primary_target: float = 0.8,
    b_repeats: int = 1000,
    seed: int = 42,
    ctrl_repeats: int = 100,
    ctrl_seed: int = 7,
    extra_calls: dict[str, float] | None = None,
    signals: Sequence[str] = ("s1", "s2", "s3", "s4", "s5"),
) -> tuple[list[BootstrapResult], tuple[str, str], dict[str, Any]]:
    """Run note-level bootstrap for a single model across signals at primary target.

    Args:
        model_name: Name of the model.
        test_df: Test codes DataFrame for this model.
        test_notes: Sequence of test Note objects.
        thresholds: Map of signal -> {target_str: tau}.
        primary_target: Target coverage for Table V and verdicts (default 0.8).
        b_repeats: Number of bootstrap note resamples (default 1000).
        seed: RNG seed for bootstrap note resampling (default 42).
        ctrl_repeats: Repeats for random control inside each resample (default 100).
        ctrl_seed: Base seed for random control RNG inside resamples.
        extra_calls: Map of signal name to extra calls per note (from SPEC §12).
        signals: Signals to evaluate.

    Returns:
        tuple of (bootstrap_results, (recommended_signal, rec_rationale), extra_distributions_for_hypotheses).
    """
    if extra_calls is None:
        extra_calls = {"s1": 10.0, "s2": 1.0, "s3": 0.0, "s4": 4.0, "s5": 14.0}

    # Sort test notes deterministically
    sorted_notes = sorted(test_notes, key=lambda n: n.id)
    n_notes = len(sorted_notes)
    if n_notes == 0 or len(test_df) == 0:
        logger.warning("Empty test notes or test codes for model %s", model_name)
        return [], ("none", "No test data"), {}

    note_id_to_idx = {n.id: i for i, n in enumerate(sorted_notes)}

    # Map each code in test_df to its note index
    code_note_indices = np.array([note_id_to_idx.get(nid, -1) for nid in test_df["id"]], dtype=int)
    correct_arr = test_df["correct"].values.astype(bool)
    unsupported_arr = test_df["unsupported"].values.astype(bool)
    invalid_arr = test_df["invalid"].values.astype(bool)
    gold_type_arr = test_df["gold_type"].fillna("").values.astype(str)

    score_arrays = {sig: test_df[sig].values.astype(float) for sig in signals}

    # Precompute per-note code summary statistics for each signal at primary target
    target_str = str(primary_target)
    taus: dict[str, float] = {}
    for sig in signals:
        tau_val = thresholds.get(sig, {}).get(target_str, -float("inf"))
        taus[sig] = float(tau_val)

    # Note summary arrays: shape (n_notes,)
    total_codes_per_note = np.zeros(n_notes, dtype=int)
    correct_codes_per_note = np.zeros(n_notes, dtype=int)
    unsupported_codes_per_note = np.zeros(n_notes, dtype=int)
    invalid_codes_per_note = np.zeros(n_notes, dtype=int)

    kept_codes_per_note: dict[str, np.ndarray] = {sig: np.zeros(n_notes, dtype=int) for sig in signals}
    kept_correct_per_note: dict[str, np.ndarray] = {sig: np.zeros(n_notes, dtype=int) for sig in signals}
    kept_unsupported_per_note: dict[str, np.ndarray] = {sig: np.zeros(n_notes, dtype=int) for sig in signals}
    kept_invalid_per_note: dict[str, np.ndarray] = {sig: np.zeros(n_notes, dtype=int) for sig in signals}

    for note_idx in range(n_notes):
        mask = (code_note_indices == note_idx)
        total_codes_per_note[note_idx] = int(np.sum(mask))
        correct_codes_per_note[note_idx] = int(np.sum(correct_arr[mask]))
        unsupported_codes_per_note[note_idx] = int(np.sum(unsupported_arr[mask]))
        invalid_codes_per_note[note_idx] = int(np.sum(invalid_arr[mask]))

        for sig in signals:
            tau = taus[sig]
            sig_scores = score_arrays[sig][mask]
            is_kept = (sig_scores >= tau) if (tau != -float("inf") and not np.isneginf(tau)) else np.ones(len(sig_scores), dtype=bool)

            kept_codes_per_note[sig][note_idx] = int(np.sum(is_kept))
            kept_correct_per_note[sig][note_idx] = int(np.sum(correct_arr[mask][is_kept]))
            kept_unsupported_per_note[sig][note_idx] = int(np.sum(unsupported_arr[mask][is_kept]))
            kept_invalid_per_note[sig][note_idx] = int(np.sum(invalid_arr[mask][is_kept]))

    # Observed test metrics on full test set (SPEC §10 condition (iii) requires observed test ΔICR)
    observed_results: dict[str, dict[str, float]] = {}
    for sig in signals:
        n_pred = len(test_df)
        tau = taus[sig]
        kept_mask = (test_df[sig] >= tau) if (tau != -float("inf") and not np.isneginf(tau)) else np.ones(n_pred, dtype=bool)
        k_kept = int(np.sum(kept_mask))
        c_kept = int(np.sum(correct_arr[kept_mask]))
        u_kept = int(np.sum(unsupported_arr[kept_mask]))
        i_kept = int(np.sum(invalid_arr[kept_mask]))

        realised_c = k_kept / n_pred if n_pred > 0 else 0.0
        p_sig = c_kept / k_kept if k_kept > 0 else float("nan")
        u_sig = u_kept / k_kept if k_kept > 0 else float("nan")
        i_sig = i_kept / k_kept if k_kept > 0 else float("nan")

        ctrl_rng = np.random.default_rng(ctrl_seed)
        p_rand, u_rand, i_rand = _simulate_random_control_draws(
            correct_arr, unsupported_arr, invalid_arr, k_keep=k_kept, repeats=ctrl_repeats, rng=ctrl_rng
        )

        observed_results[sig] = {
            "realised_coverage": realised_c,
            "precision": p_sig,
            "ucr": u_sig,
            "icr": i_sig,
            "delta_precision": p_sig - p_rand,
            "delta_ucr": u_sig - u_rand,
            "delta_icr": i_sig - i_rand,
        }

    # Generate bootstrap note resamples (B, n_notes)
    b_rng = np.random.default_rng(seed)
    ctrl_rng = np.random.default_rng(ctrl_seed + 100)

    # Containers for distributions
    auroc_dist: dict[str, list[float]] = {sig: [] for sig in signals}
    skipped_auroc_count: dict[str, int] = {sig: 0 for sig in signals}

    delta_p_dist: dict[str, list[float]] = {sig: [] for sig in signals}
    delta_u_dist: dict[str, list[float]] = {sig: [] for sig in signals}
    delta_i_dist: dict[str, list[float]] = {sig: [] for sig in signals}

    # Distributions needed for hypotheses in SPEC §13
    delta_auroc_s1_s2_dist: list[float] = []
    h4_relative_fall_dist: dict[str, list[float]] = {sig: [] for sig in signals}
    h5_delta_auroc_dist: dict[str, list[float]] = {sig: [] for sig in signals}

    for b in range(b_repeats):
        resampled_note_indices = b_rng.choice(n_notes, size=n_notes, replace=True)
        multiplicities = np.bincount(resampled_note_indices, minlength=n_notes)

        # Multiplicity for each code in test_df
        code_multiplicities = multiplicities[code_note_indices]
        n_resample_codes = int(np.sum(code_multiplicities))

        if n_resample_codes == 0:
            continue

        res_correct = np.repeat(correct_arr, code_multiplicities)
        res_unsupported = np.repeat(unsupported_arr, code_multiplicities)
        res_invalid = np.repeat(invalid_arr, code_multiplicities)
        res_gold_type = np.repeat(gold_type_arr, code_multiplicities)

        # 1. AUROC per signal on this resample
        has_pos = bool(np.any(res_correct))
        has_neg = bool(np.any(~res_correct))
        can_auroc = has_pos and has_neg

        b_aurocs: dict[str, float] = {}
        for sig in signals:
            if can_auroc:
                res_scores = np.repeat(score_arrays[sig], code_multiplicities)
                val = compute_auroc(res_correct, res_scores)
                if not np.isnan(val):
                    auroc_dist[sig].append(val)
                    b_aurocs[sig] = val
                else:
                    skipped_auroc_count[sig] += 1
            else:
                skipped_auroc_count[sig] += 1

        # H2 delta AUROC (S1 - S2)
        if "s1" in b_aurocs and "s2" in b_aurocs:
            delta_auroc_s1_s2_dist.append(b_aurocs["s1"] - b_aurocs["s2"])

        # 2. Kept metrics & paired random control per signal
        # Fast aggregation via dot product
        n_pred_b = int(np.dot(multiplicities, total_codes_per_note))
        c_full_b = int(np.dot(multiplicities, correct_codes_per_note))
        u_full_b = int(np.dot(multiplicities, unsupported_codes_per_note))
        i_full_b = int(np.dot(multiplicities, invalid_codes_per_note))

        full_icr_b = i_full_b / n_pred_b if n_pred_b > 0 else 0.0
        full_ucr_b = u_full_b / n_pred_b if n_pred_b > 0 else 0.0

        for sig in signals:
            k_kept_b = int(np.dot(multiplicities, kept_codes_per_note[sig]))
            c_kept_b = int(np.dot(multiplicities, kept_correct_per_note[sig]))
            u_kept_b = int(np.dot(multiplicities, kept_unsupported_per_note[sig]))
            i_kept_b = int(np.dot(multiplicities, kept_invalid_per_note[sig]))

            if k_kept_b > 0:
                p_sig_b = c_kept_b / k_kept_b
                u_sig_b = u_kept_b / k_kept_b
                i_sig_b = i_kept_b / k_kept_b

                # Paired random control recomputed inside this resample at coverage k_kept_b / n_pred_b
                p_rand_b, u_rand_b, i_rand_b = _simulate_random_control_draws(
                    res_correct, res_unsupported, res_invalid, k_keep=k_kept_b, repeats=ctrl_repeats, rng=ctrl_rng
                )

                delta_p_dist[sig].append(p_sig_b - p_rand_b)
                delta_u_dist[sig].append(u_sig_b - u_rand_b)
                delta_i_dist[sig].append(i_sig_b - i_rand_b)

                # H4: Relative fall difference (ICR fall - UCR fall)
                icr_fall = (full_icr_b - i_sig_b) / full_icr_b if full_icr_b > 0.0 else 0.0
                ucr_fall = (full_ucr_b - u_sig_b) / full_ucr_b if full_ucr_b > 0.0 else 0.0
                h4_relative_fall_dist[sig].append(icr_fall - ucr_fall)

        # H5 AUROC difference (implicit vs explicit) for all signals
        wrong_mask = ~res_correct
        impl_pos = res_correct & (res_gold_type == "implicit")
        expl_pos = res_correct & (res_gold_type == "explicit")

        if np.any(wrong_mask) and np.any(impl_pos) and np.any(expl_pos):
            for sig in signals:
                res_scores = np.repeat(score_arrays[sig], code_multiplicities)

                impl_mask = wrong_mask | impl_pos
                auc_impl = compute_auroc(impl_pos[impl_mask], res_scores[impl_mask])

                expl_mask = wrong_mask | expl_pos
                auc_expl = compute_auroc(expl_pos[expl_mask], res_scores[expl_mask])

                if not np.isnan(auc_impl) and not np.isnan(auc_expl):
                    h5_delta_auroc_dist[sig].append(auc_impl - auc_expl)

    # Assemble BootstrapResult objects
    bootstrap_results: list[BootstrapResult] = []
    verdicts: dict[str, str] = {}
    auroc_cis: dict[str, tuple[float, float]] = {}
    auroc_means: dict[str, float] = {}

    for sig in signals:
        # AUROC CI
        aucs = auroc_dist[sig]
        auc_m = float(np.mean(aucs)) if aucs else float("nan")
        auc_ci = (
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)),
        ) if len(aucs) >= 10 else (float("nan"), float("nan"))

        auroc_means[sig] = auc_m
        auroc_cis[sig] = auc_ci

        # Delta precision CI
        dps = delta_p_dist[sig]
        dp_m = float(np.mean(dps)) if dps else float("nan")
        dp_ci = (
            float(np.percentile(dps, 2.5)),
            float(np.percentile(dps, 97.5)),
        ) if len(dps) >= 10 else (float("nan"), float("nan"))

        # Delta UCR CI
        dus = delta_u_dist[sig]
        du_m = float(np.mean(dus)) if dus else float("nan")
        du_ci = (
            float(np.percentile(dus, 2.5)),
            float(np.percentile(dus, 97.5)),
        ) if len(dus) >= 10 else (float("nan"), float("nan"))

        # Delta ICR CI
        dis = delta_i_dist[sig]
        di_m = float(np.mean(dis)) if dis else float("nan")
        di_ci = (
            float(np.percentile(dis, 2.5)),
            float(np.percentile(dis, 97.5)),
        ) if len(dis) >= 10 else (float("nan"), float("nan"))

        obs_delta_icr = observed_results[sig]["delta_icr"]
        cond1, cond2, cond3, verdict = evaluate_verdict(dp_ci, du_ci, obs_delta_icr)
        verdicts[sig] = verdict

        res = BootstrapResult(
            model=model_name,
            signal=sig,
            target_coverage=primary_target,
            realised_coverage=observed_results[sig]["realised_coverage"],
            auroc_mean=round(auc_m, 6) if not np.isnan(auc_m) else float("nan"),
            auroc_ci_lower=round(auc_ci[0], 6) if not np.isnan(auc_ci[0]) else float("nan"),
            auroc_ci_upper=round(auc_ci[1], 6) if not np.isnan(auc_ci[1]) else float("nan"),
            delta_precision_mean=round(dp_m, 6) if not np.isnan(dp_m) else float("nan"),
            delta_precision_ci_lower=round(dp_ci[0], 6) if not np.isnan(dp_ci[0]) else float("nan"),
            delta_precision_ci_upper=round(dp_ci[1], 6) if not np.isnan(dp_ci[1]) else float("nan"),
            delta_ucr_mean=round(du_m, 6) if not np.isnan(du_m) else float("nan"),
            delta_ucr_ci_lower=round(du_ci[0], 6) if not np.isnan(du_ci[0]) else float("nan"),
            delta_ucr_ci_upper=round(du_ci[1], 6) if not np.isnan(du_ci[1]) else float("nan"),
            delta_icr_observed=round(obs_delta_icr, 6) if not np.isnan(obs_delta_icr) else float("nan"),
            delta_icr_mean=round(di_m, 6) if not np.isnan(di_m) else float("nan"),
            delta_icr_ci_lower=round(di_ci[0], 6) if not np.isnan(di_ci[0]) else float("nan"),
            delta_icr_ci_upper=round(di_ci[1], 6) if not np.isnan(di_ci[1]) else float("nan"),
            cond1_precision=cond1,
            cond2_ucr=cond2,
            cond3_icr=cond3,
            verdict=verdict,
            skipped_resamples=skipped_auroc_count[sig],
        )
        bootstrap_results.append(res)

    rec_sig, rec_rationale = evaluate_recommendation(
        verdicts=verdicts,
        auroc_cis=auroc_cis,
        auroc_means=auroc_means,
        extra_calls=extra_calls,
    )

    hypotheses_dists = {
        "auroc_dist": auroc_dist,
        "delta_auroc_s1_s2_dist": delta_auroc_s1_s2_dist,
        "h4_relative_fall_dist": h4_relative_fall_dist,
        "h5_delta_auroc_dist": h5_delta_auroc_dist,
        "verdicts": verdicts,
        "observed_results": observed_results,
    }

    return bootstrap_results, (rec_sig, rec_rationale), hypotheses_dists
