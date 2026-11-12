"""Table generation and formatting for Paper C1 per SPEC §14.

Implements all paper tables and in-text numbers:
- Table II: Corpus and prediction statistics
- Table III: AUROC [95% CI] and ECE per model and signal
- Table IV(a)/(b): Risk-coverage performance per model for best signal
- Table IV(c): Headline numbers summary (Full list vs ~80% target)
- Table V: Best signal paired deltas, 3 conditions, verdict, and recommendation
- Table VI: Feature strata breakdown and reconciliation row
- Table VII: Cost accounting summary
- Table VIII: k-sensitivity on dev split
- Table IX: Hypotheses H1 through H5 evaluation summary
- paper_numbers.md: In-text numbers summary

Formatting rules per SPEC §14:
- AUROC/ECE/precision/recall/F1 with 3 decimals (0.xxx)
- Coverage and rates as percentages with 1 decimal (xx.x%)
- Intervals as [lo, hi]
- Deltas as signed values (+0.xxx, -0.xxx)
- Em-dash (—) for uncomputed cells
- Footnotes stating best-signal rule, primary target, B, seeds, split sizes, GPU, vocab release
- Mock data guard: refuses mock data unless allow_mock=True; stamps "MOCK DATA" when permitted.

Traces to SPEC §4, §14, §15, §18.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from c1.data import Note, load_corpus
from c1.utils import Config, load_config, model_display_name, write_csv_atomic

logger = logging.getLogger(__name__)

MOCK_BANNER_MD = "> **MOCK DATA — FOR TESTING AND PIPELINE VALIDATION ONLY**\n\n"
MOCK_BANNER_CSV_PREFIX = "# MOCK DATA — FOR TESTING AND PIPELINE VALIDATION ONLY\n"


def fmt_dec3(v: Any) -> str:
    """Format floating point value with 3 decimals or em-dash if NaN/empty."""
    try:
        val = float(v)
        if np.isnan(val):
            return "—"
        return f"{val:.3f}"
    except (ValueError, TypeError):
        return "—"


def fmt_pct1(v: Any) -> str:
    """Format fraction as percentage with 1 decimal or em-dash if NaN/empty."""
    try:
        val = float(v)
        if np.isnan(val):
            return "—"
        return f"{val * 100:.1f}%"
    except (ValueError, TypeError):
        return "—"


def fmt_ci(lo: Any, hi: Any) -> str:
    """Format interval as [0.xxx, 0.xxx] or em-dash if NaN."""
    try:
        v_lo = float(lo)
        v_hi = float(hi)
        if np.isnan(v_lo) or np.isnan(v_hi):
            return "—"
        return f"[{v_lo:.3f}, {v_hi:.3f}]"
    except (ValueError, TypeError):
        return "—"


def fmt_signed_delta(val: Any, lo: Any = None, hi: Any = None) -> str:
    """Format signed delta +0.xxx or +0.xxx [+0.xxx, +0.xxx]."""
    try:
        v = float(val)
        if np.isnan(v):
            return "—"
        sign = "+" if v > 0 else ("-" if v < 0 else " ")
        res = f"{sign}{abs(v):.3f}"
        if lo is not None and hi is not None:
            v_lo = float(lo)
            v_hi = float(hi)
            if not np.isnan(v_lo) and not np.isnan(v_hi):
                s_lo = "+" if v_lo > 0 else ("-" if v_lo < 0 else " ")
                s_hi = "+" if v_hi > 0 else ("-" if v_hi < 0 else " ")
                res += f" [{s_lo}{abs(v_lo):.3f}, {s_hi}{abs(v_hi):.3f}]"
        return res
    except (ValueError, TypeError):
        return "—"


def make_standard_footnote(
    config: Config | None = None,
    best_signals: dict[str, str] | None = None,
    gpu_name: str | None = None,
    n_dev: int = 90,
    n_test: int = 210,
    is_table_ii: bool = False,
) -> str:
    """Generate standardized footnote required by SPEC §14."""
    gen_seed = config.corpus.generator_seed if config else 2026
    split_seed = config.split.seed if config else 2026
    if is_table_ii:
        return (
            f"Footnote: Split sizes: Dev {n_dev} notes, Test {n_test} notes. "
            f"Generator seed: {gen_seed}, Split seed: {split_seed}. "
            f"Vocabulary release: icd10cm_2024. Pinned CDC release."
        )

    b_val = config.bootstrap.B if config else 1000
    p_target = config.primary_target if config else 0.8
    b_seed = config.bootstrap.seed if config else 42
    r_seed = config.random_control.seed if config else 7
    gpu_str = gpu_name if gpu_name else "CUDA GPU"

    best_str = f"Best signals (dev AUROC): {best_signals}. " if best_signals else "Best signal chosen by dev AUROC. "
    return (
        f"Footnote: Split sizes: Dev {n_dev} notes, Test {n_test} notes. "
        f"{best_str}Primary coverage target: {p_target*100:.0f}%. "
        f"Bootstrap resamples: B={b_val} (seed={b_seed}). "
        f"Generator seed: {gen_seed}, Split seed: {split_seed}, Random control seed: {r_seed}. "
        f"Hardware: {gpu_str}. Vocabulary: icd10cm_2024 (CDC pinned release)."
    )


def _write_table_outputs(
    md_content: str,
    df: pd.DataFrame,
    reports_dir: Path,
    base_name: str,
    allow_mock: bool = False,
) -> tuple[Path, Path]:
    """Write table Markdown and CSV files atomically."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{base_name}.md"
    csv_path = reports_dir / f"{base_name}.csv"

    final_md = (MOCK_BANNER_MD + md_content) if allow_mock else md_content
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(final_md)

    write_csv_atomic(csv_path, df)
    logger.info("Saved %s to %s and %s", base_name, md_path, csv_path)
    return md_path, csv_path


# ---------------------------------------------------------------------------
# Table II: Corpus & Prediction Statistics (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_ii(
    notes: list[Note],
    split_df: pd.DataFrame,
    model_predictions: dict[str, dict[str, list[str]]] | None = None,
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Generate Table II: Corpus and prediction statistics."""
    split_map = dict(zip(split_df["id"], split_df["split"]))
    dev_notes = [n for n in notes if split_map.get(n.id) == "dev"]
    test_notes = [n for n in notes if split_map.get(n.id) == "test"]
    partitions = [("dev", dev_notes), ("test", test_notes), ("total", notes)]

    metrics: dict[str, dict[str, Any]] = {
        "notes": {},
        "mean_gold_codes_per_note": {},
        "distinct_gold_codes": {},
        "share_implicit_pct": {},
    }

    for label, subset in partitions:
        n_count = len(subset)
        metrics["notes"][label] = str(n_count)
        if n_count > 0:
            gold_codes = [g.code for n in subset for g in n.gold]
            total_gold = len(gold_codes)
            implicit_gold = sum(1 for n in subset for g in n.gold if g.type == "implicit")
            metrics["mean_gold_codes_per_note"][label] = f"{total_gold / n_count:.3f}"
            metrics["distinct_gold_codes"][label] = str(len(set(gold_codes)))
            share_imp = (implicit_gold / total_gold * 100) if total_gold else 0.0
            metrics["share_implicit_pct"][label] = f"{share_imp:.1f}%"
        else:
            metrics["mean_gold_codes_per_note"][label] = "—"
            metrics["distinct_gold_codes"][label] = "0"
            metrics["share_implicit_pct"][label] = "—"

    # One row per model: those with predictions, else the paper's two (shown as "—").
    model_keys = sorted(model_predictions) if model_predictions else ["llama3", "biomistral"]
    model_display = {k: f"mean_predicted_codes_per_note ({k})" for k in model_keys}

    for m_key in model_keys:
        disp_name = model_display[m_key]
        metrics[disp_name] = {}
        for label, subset in partitions:
            if model_predictions and m_key in model_predictions:
                preds = model_predictions[m_key]
                pred_counts = [len(preds.get(n.id, [])) for n in subset]
                metrics[disp_name][label] = f"{sum(pred_counts) / len(subset):.3f}" if subset else "—"
            else:
                metrics[disp_name][label] = "—"

    rows = []
    display_order = [
        ("notes", "Notes (dev/test)"),
        ("mean_gold_codes_per_note", "Mean gold codes / note"),
        ("distinct_gold_codes", "Distinct gold codes"),
        ("share_implicit_pct", "Share of gold codes that are implicit"),
    ] + [
        (model_display[k], f"Mean predicted codes / note ({model_display_name(config, k)})")
        for k in model_keys
    ]

    for key, human_label in display_order:
        rows.append({
            "metric": human_label,
            "dev": metrics[key]["dev"],
            "test": metrics[key]["test"],
            "total": metrics[key]["total"],
        })

    df = pd.DataFrame(rows)
    footnote = make_standard_footnote(config, n_dev=len(dev_notes), n_test=len(test_notes), is_table_ii=True)

    md_lines = [
        "# Table II: Clinical Corpus and Prediction Statistics",
        "",
        "| Metric | Dev | Test | Total |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(f"| {r['metric']} | {r['dev']} | {r['test']} | {r['total']} |")
    md_lines.extend(["", f"*{footnote}*", ""])
    md_str = "\n".join(md_lines)

    if out_dir:
        _write_table_outputs(md_str, df, Path(out_dir) / "reports", "table_II", allow_mock=allow_mock)

    return md_str, df


def run_table_ii_from_config(
    config_path: str | Path = "configs/c1.yaml",
    out_override: str | Path | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Run Table II generation from configuration paths."""
    cfg = load_config(config_path)
    data_dir = Path(cfg.paths.data_dir)
    corpus_file = data_dir / "corpus.jsonl"
    out_dir = Path(out_override) if out_override else Path(cfg.paths.out_dir) / cfg.run_id
    split_file = out_dir / "split.csv"
    if not split_file.is_file():
        split_file = data_dir / "split.csv"

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}. Run 'make-corpus' first.")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run 'split' first.")

    vocab = None
    try:
        from c1.codes import Vocabulary
        vocab = Vocabulary.load_default()
    except Exception:
        pass
    notes = load_corpus(corpus_file, vocab=vocab)
    split_df = pd.read_csv(split_file)

    # Collect predictions if greedy.jsonl exists
    greedy_file = out_dir / "greedy.jsonl"
    model_preds: dict[str, dict[str, list[str]]] = {}
    if greedy_file.is_file():
        from c1.utils import read_jsonl_resume
        for r in read_jsonl_resume(greedy_file):
            m = r.get("model")
            nid = r.get("id")
            preds = r.get("pred", [])
            if m and nid:
                model_preds.setdefault(m, {})[nid] = preds

    return generate_table_ii(
        notes=notes,
        split_df=split_df,
        model_predictions=model_preds if model_preds else None,
        out_dir=out_dir,
        config=cfg,
        allow_mock=allow_mock,
    )


# ---------------------------------------------------------------------------
# Table III: AUROC & ECE (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_iii(
    metrics_signals_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Generate Table III: AUROC [95% CI] and ECE (S2–S4 only) per model and signal."""
    # Filter test split
    test_metrics = metrics_signals_df[metrics_signals_df["split"] == "test"].copy()
    models = sorted(test_metrics["model"].unique())
    signals = ["s1", "s2", "s3", "s4", "s5"]

    # Map bootstrap CIs: (model, signal) -> (lo, hi)
    b_ci: dict[tuple[str, str], tuple[float, float]] = {}
    for _, r in bootstrap_df.iterrows():
        m = str(r["model"])
        s = str(r["signal"])
        try:
            lo = float(r["auroc_ci_lower"])
            hi = float(r["auroc_ci_upper"])
            b_ci[(m, s)] = (lo, hi)
        except Exception:
            pass

    table_rows = []
    for m in models:
        for s in signals:
            sub = test_metrics[(test_metrics["model"] == m) & (test_metrics["signal"] == s)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            auc_val = r["auroc"]
            lo, hi = b_ci.get((m, s), (float("nan"), float("nan")))

            auc_str = f"{fmt_dec3(auc_val)} {fmt_ci(lo, hi)}" if not np.isnan(float(auc_val)) else "—"

            ece_val = r["ece"]
            ece_str = fmt_dec3(ece_val) if s in ("s2", "s3", "s4") else "—"

            table_rows.append({
                "model": m,
                "signal": s.upper(),
                "auroc_ci": auc_str,
                "ece": ece_str,
            })

    df = pd.DataFrame(table_rows)
    footnote = make_standard_footnote(config)

    md_lines = [
        "# Table III: Discrimination (AUROC) and Calibration (ECE) Across Signals",
        "",
        "| Model | Signal | AUROC [95% CI] | ECE |",
        "|---|---|---|---|",
    ]
    for r in table_rows:
        md_lines.append(f"| {r['model']} | {r['signal']} | {r['auroc_ci']} | {r['ece']} |")
    md_lines.extend(["", f"*{footnote}*", ""])
    md_str = "\n".join(md_lines)

    if out_dir:
        _write_table_outputs(md_str, df, Path(out_dir) / "reports", "table_III", allow_mock=allow_mock)

    return md_str, df


# ---------------------------------------------------------------------------
# Table IV(a)/(b): Abstention Performance Across Targets (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_iv(
    results_test_df: pd.DataFrame,
    results_controls_df: pd.DataFrame,
    best_signals: dict[str, str],
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> dict[str, tuple[str, pd.DataFrame]]:
    """Generate Table IV(a) (LLaMA-3) and Table IV(b) (BioMistral) for the best signal."""
    models = sorted(results_test_df["model"].unique())
    outputs: dict[str, tuple[str, pd.DataFrame]] = {}

    for m in models:
        best_sig = best_signals.get(m, "s1")
        m_test = results_test_df[(results_test_df["model"] == m) & (results_test_df["signal"] == best_sig)].copy()
        m_ctrl = results_controls_df[(results_controls_df["model"] == m) & (results_controls_df["signal"] == best_sig)].copy()

        rows = []

        # 1. Full list (target 1.0)
        full_sub = m_test[m_test["target_coverage"] == 1.0]
        if len(full_sub) > 0:
            fr = full_sub.iloc[0]
            rows.append({
                "setting": "Full list",
                "realised_coverage": fmt_pct1(fr["realised_coverage"]),
                "precision": fmt_dec3(fr["precision"]),
                "recall": fmt_dec3(fr["recall"]),
                "micro_f1": fmt_dec3(fr["micro_f1"]),
                "ucr": fmt_pct1(fr["ucr"]),
                "icr": fmt_pct1(fr["icr"]),
                "review_load": fmt_dec3(fr["review_load"]),
            })

        # 2. Abstention targets 90, 80, 70, 60%
        for target in (0.9, 0.8, 0.7, 0.6):
            t_sub = m_test[m_test["target_coverage"] == target]
            if len(t_sub) > 0:
                tr = t_sub.iloc[0]
                rows.append({
                    "setting": f"Target {target*100:.0f}%",
                    "realised_coverage": fmt_pct1(tr["realised_coverage"]),
                    "precision": fmt_dec3(tr["precision"]),
                    "recall": fmt_dec3(tr["recall"]),
                    "micro_f1": fmt_dec3(tr["micro_f1"]),
                    "ucr": fmt_pct1(tr["ucr"]),
                    "icr": fmt_pct1(tr["icr"]),
                    "review_load": fmt_dec3(tr["review_load"]),
                })

        # 3. Random and Oracle controls at 80% coverage
        for c_type, label in (("random", "Random control (at ~80%)"), ("oracle", "Oracle control (at ~80%)")):
            c_sub = m_ctrl[(m_ctrl["target_coverage"] == 0.8) & (m_ctrl["control_type"] == c_type)]
            if len(c_sub) > 0:
                cr = c_sub.iloc[0]
                rows.append({
                    "setting": label,
                    "realised_coverage": fmt_pct1(cr["realised_coverage"]),
                    "precision": fmt_dec3(cr["precision"]),
                    "recall": fmt_dec3(cr["recall"]),
                    "micro_f1": fmt_dec3(cr["micro_f1"]),
                    "ucr": fmt_pct1(cr["ucr"]),
                    "icr": fmt_pct1(cr["icr"]),
                    "review_load": fmt_dec3(cr["review_load"]),
                })

        df = pd.DataFrame(rows)
        footnote = make_standard_footnote(config, best_signals={m: best_sig})

        # The paper's two models keep their (a)/(b) file names; any other model gets its key.
        suffix = {"llama3": "a_llama3", "biomistral": "b_biomistral"}.get(m, f"_{m}")
        model_title = f"{model_display_name(config, m)} (Best signal: {best_sig.upper()})"

        md_lines = [
            f"# Table IV: Abstention Performance — {model_title}",
            "",
            "| Setting | Coverage | Precision | Recall | Micro-F1 | UCR | ICR | Review Load |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            md_lines.append(
                f"| {r['setting']} | {r['realised_coverage']} | {r['precision']} | {r['recall']} | "
                f"{r['micro_f1']} | {r['ucr']} | {r['icr']} | {r['review_load']} |"
            )
        md_lines.extend(["", f"*{footnote}*", ""])
        md_str = "\n".join(md_lines)

        base_name = f"table_IV{suffix}"
        if out_dir:
            _write_table_outputs(md_str, df, Path(out_dir) / "reports", base_name, allow_mock=allow_mock)

        outputs[m] = (md_str, df)

    return outputs


# ---------------------------------------------------------------------------
# Table IV(c): Headline Numbers Summary (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_iv_c(
    results_test_df: pd.DataFrame,
    best_signals: dict[str, str],
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Generate Table IV(c): Headline numbers summary (Full list vs ~80% target)."""
    models = sorted(results_test_df["model"].unique())
    rows = []

    for m in models:
        best_sig = best_signals.get(m, "s1")
        m_df = results_test_df[(results_test_df["model"] == m) & (results_test_df["signal"] == best_sig)]

        # Full list
        full = m_df[m_df["target_coverage"] == 1.0]
        if len(full) > 0:
            fr = full.iloc[0]
            rows.append({
                "model": m,
                "setting": "Full list",
                "precision": fmt_dec3(fr["precision"]),
                "recall": fmt_dec3(fr["recall"]),
                "ucr": fmt_pct1(fr["ucr"]),
                "icr": fmt_pct1(fr["icr"]),
                "review_load": fmt_dec3(fr["review_load"]),
            })

        # Kept at 80%
        kept = m_df[m_df["target_coverage"] == 0.8]
        if len(kept) > 0:
            kr = kept.iloc[0]
            rows.append({
                "model": m,
                "setting": f"Kept ({best_sig.upper()} at ~80%)",
                "precision": fmt_dec3(kr["precision"]),
                "recall": fmt_dec3(kr["recall"]),
                "ucr": fmt_pct1(kr["ucr"]),
                "icr": fmt_pct1(kr["icr"]),
                "review_load": fmt_dec3(kr["review_load"]),
            })

    df = pd.DataFrame(rows)
    footnote = make_standard_footnote(config, best_signals=best_signals)

    md_lines = [
        "# Table IV(c): Headline Summary — Old (Full list) vs New (~80%)",
        "",
        "| Model | Setting | Precision | Recall | UCR | ICR | Review Load |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(f"| {r['model']} | {r['setting']} | {r['precision']} | {r['recall']} | {r['ucr']} | {r['icr']} | {r['review_load']} |")
    md_lines.extend(["", f"*{footnote}*", ""])
    md_str = "\n".join(md_lines)

    if out_dir:
        _write_table_outputs(md_str, df, Path(out_dir) / "reports", "table_IVc", allow_mock=allow_mock)

    return md_str, df


# ---------------------------------------------------------------------------
# Table V: Paired Differences, Verdict & Recommendation (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_v(
    bootstrap_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    best_signals: dict[str, str],
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Generate Table V: Best signal paired differences, 3 conditions, verdict, and recommendation."""
    models = sorted(bootstrap_df["model"].unique())
    rec_map = dict(zip(recommendation_df["model"], recommendation_df["recommended_signal"]))
    rec_rat = dict(zip(recommendation_df["model"], recommendation_df["rationale"]))

    rows = []
    for m in models:
        best_sig = best_signals.get(m, "s1")
        sub = bootstrap_df[(bootstrap_df["model"] == m) & (bootstrap_df["signal"] == best_sig)]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]

        dp_str = fmt_signed_delta(r["delta_precision_mean"], r["delta_precision_ci_lower"], r["delta_precision_ci_upper"])
        du_str = fmt_signed_delta(r["delta_ucr_mean"], r["delta_ucr_ci_lower"], r["delta_ucr_ci_upper"])
        di_str = fmt_signed_delta(r["delta_icr_observed"])

        c1_str = "Met" if bool(r["cond1_precision"]) else "Failed"
        c2_str = "Met" if bool(r["cond2_ucr"]) else "Failed"
        c3_str = "Met" if bool(r["cond3_icr"]) else "Failed"

        rows.append({
            "model": m,
            "best_signal": best_sig.upper(),
            "delta_precision_ci": dp_str,
            "delta_ucr_ci": du_str,
            "delta_icr_obs": di_str,
            "cond1": c1_str,
            "cond2": c2_str,
            "cond3": c3_str,
            "verdict": str(r["verdict"]),
            "recommendation": str(rec_map.get(m, "none")).upper(),
            "recommendation_rationale": rec_rat.get(m, ""),
        })

    df = pd.DataFrame(rows)
    footnote = make_standard_footnote(config, best_signals=best_signals)

    md_lines = [
        "# Table V: Statistical Verdict and Signal Recommendation (~80% Coverage)",
        "",
        "| Model | Best Signal | ΔPrecision [95% CI] | ΔUCR [95% CI] | Observed ΔICR | Cond (i) | Cond (ii) | Cond (iii) | Verdict | Recommendation |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['best_signal']} | {r['delta_precision_ci']} | {r['delta_ucr_ci']} | "
            f"{r['delta_icr_obs']} | {r['cond1']} | {r['cond2']} | {r['cond3']} | {r['verdict']} | {r['recommendation']} |"
        )
    md_lines.extend(["", f"*{footnote}*", ""])
    md_str = "\n".join(md_lines)

    if out_dir:
        _write_table_outputs(md_str, df, Path(out_dir) / "reports", "table_V", allow_mock=allow_mock)

    return md_str, df


# ---------------------------------------------------------------------------
# Table VI: Feature Breakdown (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_vi(
    by_feature_df: pd.DataFrame,
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Generate Table VI: Feature breakdown across strata per SPEC §11, §14."""
    rows = []
    for _, r in by_feature_df.iterrows():
        rows.append({
            "model": str(r["model"]),
            "stratum": str(r["stratum"]),
            "best_signal": str(r["best_signal"]).upper(),
            "n_codes": str(r["n_codes"]),
            "auroc": fmt_dec3(r["auroc"]),
            "precision_full": fmt_dec3(r["precision_full"]),
            "precision_kept": fmt_dec3(r["precision_kept"]),
            "ucr_kept": fmt_pct1(r["ucr_kept"]),
        })

    df = pd.DataFrame(rows)
    footnote = make_standard_footnote(config)

    md_lines = [
        "# Table VI: Subgroup Performance by Note Feature",
        "",
        "| Model | Stratum | Best Signal | N Codes | AUROC | Precision (Full) | Precision (Kept) | UCR (Kept) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['stratum']} | {r['best_signal']} | {r['n_codes']} | "
            f"{r['auroc']} | {r['precision_full']} | {r['precision_kept']} | {r['ucr_kept']} |"
        )
    md_lines.extend(["", f"*{footnote}*", ""])
    md_str = "\n".join(md_lines)

    if out_dir:
        _write_table_outputs(md_str, df, Path(out_dir) / "reports", "table_VI", allow_mock=allow_mock)

    return md_str, df


# ---------------------------------------------------------------------------
# Table VII: Cost Summary (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_vii(
    cost_df: pd.DataFrame,
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Generate Table VII: Cost accounting summary per SPEC §12, §14."""
    rows = []
    for _, r in cost_df.iterrows():
        rows.append({
            "model": str(r["model"]),
            "item": str(r["item"]).upper(),
            "extra_logical_calls": f"{float(r['extra_logical_calls']):.1f}",
            "physical_generate_calls": f"{float(r['physical_generate_calls']):.1f}",
            "extra_prompt_tokens": f"{float(r['extra_prompt_tokens']):.1f}",
            "extra_gen_tokens": f"{float(r['extra_gen_tokens']):.1f}",
            "extra_total_tokens": f"{float(r['extra_total_tokens']):.1f}",
            "extra_seconds": f"{float(r['extra_seconds']):.2f}",
            "gpu_name": str(r["gpu_name"]),
        })

    df = pd.DataFrame(rows)
    footnote = make_standard_footnote(config, gpu_name=str(cost_df["gpu_name"].iloc[0]) if len(cost_df) > 0 else None)

    md_lines = [
        "# Table VII: Computational Cost Accounting (Mean Per Note)",
        "",
        "| Model | Item | Extra Logical Calls | Physical Generate Calls | Extra Prompt Tokens | Extra Gen Tokens | Extra Total Tokens | Extra Wall-Clock (s) | GPU Device |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['item']} | {r['extra_logical_calls']} | {r['physical_generate_calls']} | "
            f"{r['extra_prompt_tokens']} | {r['extra_gen_tokens']} | {r['extra_total_tokens']} | {r['extra_seconds']} | {r['gpu_name']} |"
        )
    md_lines.extend(["", f"*{footnote}*", ""])
    md_str = "\n".join(md_lines)

    if out_dir:
        _write_table_outputs(md_str, df, Path(out_dir) / "reports", "table_VII", allow_mock=allow_mock)

    return md_str, df


# ---------------------------------------------------------------------------
# Table VIII: k-Sensitivity (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_viii(
    k_sens_df: pd.DataFrame,
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Generate Table VIII: Sensitivity to sample count k on dev split per SPEC §12, §14."""
    rows = []
    for _, r in k_sens_df.iterrows():
        rows.append({
            "model": str(r["model"]),
            "k": str(r["k"]),
            "dev_auroc": fmt_dec3(r["dev_auroc"]),
            "extra_calls_per_note": str(r["extra_calls_per_note"]),
        })

    df = pd.DataFrame(rows)
    footnote = make_standard_footnote(config)

    md_lines = [
        "# Table VIII: Sensitivity to Sample Count k (Dev Split)",
        "",
        "| Model | k Samples | Dev AUROC | Extra Calls / Note |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(f"| {r['model']} | {r['k']} | {r['dev_auroc']} | {r['extra_calls_per_note']} |")
    md_lines.extend(["", f"*{footnote}*", ""])
    md_str = "\n".join(md_lines)

    if out_dir:
        _write_table_outputs(md_str, df, Path(out_dir) / "reports", "table_VIII", allow_mock=allow_mock)

    return md_str, df


# ---------------------------------------------------------------------------
# Table IX: Hypotheses H1–H5 (SPEC §14)
# ---------------------------------------------------------------------------


def generate_table_ix(
    hypotheses_df: pd.DataFrame,
    out_dir: str | Path | None = None,
    config: Config | None = None,
    allow_mock: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Generate Table IX: Evaluation of hypotheses H1 through H5 per SPEC §13, §14."""
    models = sorted(hypotheses_df["model"].unique())
    hypo_names = ["H1", "H2", "H3", "H4", "H5"]

    rows = []
    for h in hypo_names:
        h_sub = hypotheses_df[hypotheses_df["hypothesis"] == h]
        if len(h_sub) == 0:
            continue
        claim = h_sub.iloc[0]["claim"]

        status_parts = []
        details_parts = []
        for m in models:
            m_row = h_sub[h_sub["model"] == m]
            if len(m_row) > 0:
                st = m_row.iloc[0]["status"]
                st = "n/a" if pd.isna(st) else st
                det = m_row.iloc[0]["details"]
                status_parts.append(f"{m}: {st}")
                details_parts.append(f"{m}: {det}")

        rows.append({
            "hypothesis": h,
            "claim": claim,
            "status": "; ".join(status_parts),
            "details": "; ".join(details_parts),
        })

    df = pd.DataFrame(rows)
    footnote = make_standard_footnote(config)

    md_lines = [
        "# Table IX: Evaluation of Pre-Registered Hypotheses (H1–H5)",
        "",
        "| Hypothesis | Claim | Result | Details |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(f"| {r['hypothesis']} | {r['claim']} | {r['status']} | {r['details']} |")
    md_lines.extend(["", f"*{footnote}*", ""])
    md_str = "\n".join(md_lines)

    if out_dir:
        _write_table_outputs(md_str, df, Path(out_dir) / "reports", "table_IX", allow_mock=allow_mock)

    return md_str, df


# ---------------------------------------------------------------------------
# Paper In-Text Numbers (SPEC §14)
# ---------------------------------------------------------------------------


def generate_paper_numbers(
    out_dir: str | Path,
    config: Config | None = None,
    allow_mock: bool = False,
) -> str:
    """Generate paper_numbers.md compiling every in-text number requested by SPEC §14."""
    out_p = Path(out_dir)

    # 1. Metadata
    run_meta_file = out_p / "run_meta.json"
    meta: dict[str, Any] = {}
    if run_meta_file.is_file():
        try:
            with open(run_meta_file, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    # 2. Codes and counts
    codes_file = out_p / "codes.csv"
    n_codes_by_model: dict[str, int] = {}
    parse_failed_by_model: dict[str, int] = {}
    s2_failed_by_model: dict[str, int] = {}
    truncated_by_model: dict[str, int] = {}

    if codes_file.is_file():
        codes_df = pd.read_csv(codes_file)
        for m in sorted(codes_df["model"].unique()):
            m_codes = codes_df[codes_df["model"] == m]
            n_codes_by_model[m] = len(m_codes)
            s2_failed_by_model[m] = int(m_codes["s2_failed"].sum())

    # 3. Greedy records
    greedy_file = out_p / "greedy.jsonl"
    if greedy_file.is_file():
        from c1.utils import read_jsonl_resume
        for r in read_jsonl_resume(greedy_file):
            m = r.get("model", "unknown")
            if r.get("parse_failed"):
                parse_failed_by_model[m] = parse_failed_by_model.get(m, 0) + 1
            if r.get("note_truncated"):
                truncated_by_model[m] = truncated_by_model.get(m, 0) + 1

    # 4. Confident mistakes by cause tag
    mistakes_by_cause: dict[str, int] = {}
    mistakes_file = out_p / "confident_mistakes.csv"
    if mistakes_file.is_file():
        m_df = pd.read_csv(mistakes_file)
        for tag in sorted(m_df["cause_tag"].fillna("other").unique()):
            mistakes_by_cause[tag] = int((m_df["cause_tag"].fillna("other") == tag).sum())

    # 5. Split counts
    split_file = out_p / "split.csv"
    dev_count, test_count = 0, 0
    if split_file.is_file():
        sp_df = pd.read_csv(split_file)
        dev_count = int((sp_df["split"] == "dev").sum())
        test_count = int((sp_df["split"] == "test").sum())

    md_lines = [
        "# In-Text Numbers for Paper C1",
        "",
        "## 1. Experimental Setup & Reproducibility",
        f"- **Generator seed:** {config.corpus.generator_seed if config else meta.get('config', {}).get('corpus', {}).get('generator_seed', 2026)}",
        f"- **Split seed:** {config.split.seed if config else meta.get('config', {}).get('split', {}).get('seed', 2026)}",
        f"- **Sampling decoding seed:** {config.decoding.sample.seed if config else 2026}",
        f"- **Bootstrap seed:** {config.bootstrap.seed if config else 42} (B={config.bootstrap.B if config else 1000})",
        f"- **Random control seed:** {config.random_control.seed if config else 7} (repeats={config.random_control.repeats if config else 100})",
        f"- **Corpus split sizes:** Dev={dev_count} notes, Test={test_count} notes (Total={dev_count+test_count})",
        f"- **Vocabulary release:** icd10cm_2024 (CDC pinned release; hash={meta.get('vocab_hash', '—')[:12]}...)",
        f"- **GPU Hardware:** {meta.get('gpu_name', 'CPU / MockGPU')}",
        "",
        "## 2. Model Predictions & Failure Rates",
    ]

    for m, n_c in n_codes_by_model.items():
        p_fail = parse_failed_by_model.get(m, 0)
        s2_fail = s2_failed_by_model.get(m, 0)
        s2_rate = (s2_fail / n_c * 100) if n_c > 0 else 0.0
        trunc = truncated_by_model.get(m, 0)
        md_lines.extend([
            f"### Model: {m}",
            f"- **Total predicted codes:** {n_c}",
            f"- **Notes with parse failures:** {p_fail}",
            f"- **S2 unreadable scores:** {s2_fail} ({s2_rate:.1f}%)",
            f"- **Truncated notes (shared word limit):** {trunc}",
            "",
        ])

    md_lines.extend([
        "## 3. Confident Mistakes Breakdown by Cause Tag",
    ])
    if mistakes_by_cause:
        for tag, cnt in mistakes_by_cause.items():
            md_lines.append(f"- **{tag}:** {cnt} codes")
    else:
        md_lines.append("- No confident mistakes found.")

    md_lines.extend([
        "",
        "## 4. Environment & Library Versions",
    ])
    lib_vers = meta.get("library_versions", {})
    if lib_vers:
        for lib, ver in lib_vers.items():
            md_lines.append(f"- **{lib}:** {ver}")
    else:
        md_lines.append("- Recorded in run_meta.json.")
    md_lines.append("")

    content = "\n".join(md_lines)
    reports_dir = out_p / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "paper_numbers.md"
    final_content = (MOCK_BANNER_MD + content) if allow_mock else content

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_content)

    logger.info("Saved paper numbers summary to %s", out_file)
    return final_content


# ---------------------------------------------------------------------------
# Master Runner: All Tables & Numbers (SPEC §14)
# ---------------------------------------------------------------------------


def run_all_tables(
    out_dir: str | Path,
    config_path: str | Path = "configs/c1.yaml",
    allow_mock: bool = False,
    selected_table: str = "all",
) -> dict[str, Path]:
    """Render paper Tables II through IX and IV(c) under {out_dir}/reports/.

    Args:
        out_dir: Pipeline output run directory.
        config_path: Path to configuration YAML.
        allow_mock: Whether to tolerate mock outputs and stamp reports.
        selected_table: Specific table to render ("all", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX").

    Returns:
        Dict mapping table identifier to output Path.
    """
    out_p = Path(out_dir)
    cfg = load_config(config_path)

    # 1. Mock mixing guard
    from c1.sanity import check_mock_mixing_guard
    guard = check_mock_mixing_guard(out_p, allow_mock=allow_mock)
    if not guard["passed"]:
        raise ValueError(guard["details"])

    # 2. Check sanity report
    sanity_file = out_p / "sanity_report.json"
    if not sanity_file.is_file():
        raise RuntimeError("Sanity report not found. Run 'c1 sanity' before generating tables.")
    try:
        with open(sanity_file, encoding="utf-8") as f:
            s_rep = json.load(f)
            if not s_rep.get("passed", False):
                raise RuntimeError(f"Sanity verification failed ({s_rep.get('failures')}). Tables cannot be generated.")
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Failed to read sanity_report.json: {e}")

    # Load required dataframes
    metrics_signals_file = out_p / "metrics_signals.csv"
    bootstrap_file = out_p / "bootstrap.csv"
    results_test_file = out_p / "results_test.csv"
    results_controls_file = out_p / "results_controls.csv"
    by_feature_file = out_p / "by_feature.csv"
    cost_file = out_p / "cost.csv"
    k_sens_file = out_p / "k_sens.csv"
    hypotheses_file = out_p / "hypotheses.csv"
    rec_file = out_p / "recommendation.csv"

    # Best signals
    best_signals: dict[str, str] = {}
    if metrics_signals_file.is_file():
        m_df = pd.read_csv(metrics_signals_file)
        dev_m = m_df[m_df["split"] == "dev"]
        for m in sorted(dev_m["model"].unique()):
            m_rows = dev_m[dev_m["model"] == m]
            best_r = max(m_rows.itertuples(), key=lambda r: float(r.auroc) if not np.isnan(float(r.auroc)) else -1.0)
            best_signals[m] = best_r.signal

    table_paths: dict[str, Path] = {}
    reports_dir = out_p / "reports"

    # Table II
    if selected_table in ("all", "II"):
        try:
            run_table_ii_from_config(config_path=config_path, out_override=out_p, allow_mock=allow_mock)
            table_paths["table_II"] = reports_dir / "table_II.md"
        except FileNotFoundError as e:
            logger.warning("Table II skipped (corpus/split file not found): %s", e)

    # Table III
    if selected_table in ("all", "III") and metrics_signals_file.is_file() and bootstrap_file.is_file():
        generate_table_iii(pd.read_csv(metrics_signals_file), pd.read_csv(bootstrap_file), out_p, cfg, allow_mock=allow_mock)
        table_paths["table_III"] = reports_dir / "table_III.md"

    # Table IV(a)/(b) and IV(c)
    if selected_table in ("all", "IV") and results_test_file.is_file() and results_controls_file.is_file():
        generate_table_iv(pd.read_csv(results_test_file), pd.read_csv(results_controls_file), best_signals, out_p, cfg, allow_mock=allow_mock)
        generate_table_iv_c(pd.read_csv(results_test_file), best_signals, out_p, cfg, allow_mock=allow_mock)
        for iv_file in sorted(reports_dir.glob("table_IV*_*.md")):
            key = {"table_IVa_llama3": "table_IVa", "table_IVb_biomistral": "table_IVb"}.get(iv_file.stem, iv_file.stem)
            table_paths[key] = iv_file
        table_paths["table_IVc"] = reports_dir / "table_IVc.md"

    # Table V
    if selected_table in ("all", "V") and bootstrap_file.is_file() and rec_file.is_file():
        generate_table_v(pd.read_csv(bootstrap_file), pd.read_csv(rec_file), best_signals, out_p, cfg, allow_mock=allow_mock)
        table_paths["table_V"] = reports_dir / "table_V.md"

    # Table VI
    if selected_table in ("all", "VI") and by_feature_file.is_file():
        generate_table_vi(pd.read_csv(by_feature_file), out_p, cfg, allow_mock=allow_mock)
        table_paths["table_VI"] = reports_dir / "table_VI.md"

    # Table VII
    if selected_table in ("all", "VII") and cost_file.is_file():
        generate_table_vii(pd.read_csv(cost_file), out_p, cfg, allow_mock=allow_mock)
        table_paths["table_VII"] = reports_dir / "table_VII.md"

    # Table VIII
    if selected_table in ("all", "VIII") and k_sens_file.is_file():
        generate_table_viii(pd.read_csv(k_sens_file), out_p, cfg, allow_mock=allow_mock)
        table_paths["table_VIII"] = reports_dir / "table_VIII.md"

    # Table IX
    if selected_table in ("all", "IX") and hypotheses_file.is_file():
        generate_table_ix(pd.read_csv(hypotheses_file), out_p, cfg, allow_mock=allow_mock)
        table_paths["table_IX"] = reports_dir / "table_IX.md"

    # paper_numbers.md
    if selected_table in ("all", "paper-numbers"):
        generate_paper_numbers(out_p, cfg, allow_mock=allow_mock)
        table_paths["paper_numbers"] = reports_dir / "paper_numbers.md"

    return table_paths
