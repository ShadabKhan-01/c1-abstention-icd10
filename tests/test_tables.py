"""Unit tests for table formatting and rendering per SPEC §14.

Traces to SPEC §4, §14, §15, §18.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from c1.tables import (
    fmt_ci,
    fmt_dec3,
    fmt_pct1,
    fmt_signed_delta,
    generate_paper_numbers,
    generate_table_iii,
    generate_table_iv,
    generate_table_iv_c,
    generate_table_ix,
    generate_table_v,
    generate_table_vi,
    generate_table_vii,
    generate_table_viii,
    run_all_tables,
)
from c1.utils import Config


# ---------------------------------------------------------------------------
# Formatting Helpers Unit Tests (SPEC §14)
# ---------------------------------------------------------------------------


def test_formatting_helpers() -> None:
    """Verify SPEC §14 formatting rules: 3 decimals, 1 decimal %, signed deltas, em-dash."""
    # 3 decimals
    assert fmt_dec3(0.83333) == "0.833"
    assert fmt_dec3(1.0) == "1.000"
    assert fmt_dec3(float("nan")) == "—"
    assert fmt_dec3("") == "—"

    # Percentages
    assert fmt_pct1(0.8524) == "85.2%"
    assert fmt_pct1(1.0) == "100.0%"
    assert fmt_pct1(0.0) == "0.0%"
    assert fmt_pct1(float("nan")) == "—"

    # Confidence intervals
    assert fmt_ci(0.7251, 0.8912) == "[0.725, 0.891]"
    assert fmt_ci(float("nan"), 0.5) == "—"

    # Signed deltas
    assert fmt_signed_delta(0.0523) == "+0.052"
    assert fmt_signed_delta(-0.0418) == "-0.042"
    assert fmt_signed_delta(0.0523, 0.0121, 0.0954) == "+0.052 [+0.012, +0.095]"


# ---------------------------------------------------------------------------
# Tables Rendering & Refusal Tests (SPEC §14, §15)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline_outputs(tmp_path: Path) -> Path:
    """Create a mock outputs directory with analysis files for table rendering."""
    out_dir = tmp_path / "outputs_mock" / "run_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. metrics_signals.csv
    metrics_signals_df = pd.DataFrame([
        {"model": "llama3", "split": "dev", "signal": "s1", "auroc": 0.90, "ece": ""},
        {"model": "llama3", "split": "test", "signal": "s1", "auroc": 0.88, "ece": ""},
        {"model": "llama3", "split": "test", "signal": "s2", "auroc": 0.75, "ece": 0.08},
        {"model": "llama3", "split": "test", "signal": "s3", "auroc": 0.80, "ece": 0.06},
        {"model": "llama3", "split": "test", "signal": "s4", "auroc": 0.85, "ece": 0.05},
        {"model": "llama3", "split": "test", "signal": "s5", "auroc": 0.87, "ece": ""},
    ])
    metrics_signals_df.to_csv(out_dir / "metrics_signals.csv", index=False)

    # 2. bootstrap.csv
    bootstrap_df = pd.DataFrame([
        {
            "model": "llama3", "signal": "s1", "target_coverage": 0.8, "realised_coverage": 0.82,
            "auroc_mean": 0.88, "auroc_ci_lower": 0.82, "auroc_ci_upper": 0.94,
            "delta_precision_mean": 0.08, "delta_precision_ci_lower": 0.02, "delta_precision_ci_upper": 0.14,
            "delta_ucr_mean": -0.06, "delta_ucr_ci_lower": -0.11, "delta_ucr_ci_upper": -0.01,
            "delta_icr_observed": -0.02, "delta_icr_mean": -0.02, "delta_icr_ci_lower": -0.04, "delta_icr_ci_upper": 0.0,
            "cond1_precision": True, "cond2_ucr": True, "cond3_icr": True,
            "verdict": "useful", "skipped_resamples": 0,
        },
        {
            "model": "llama3", "signal": "s2", "target_coverage": 0.8, "realised_coverage": 0.80,
            "auroc_mean": 0.75, "auroc_ci_lower": 0.68, "auroc_ci_upper": 0.82,
            "delta_precision_mean": 0.02, "delta_precision_ci_lower": -0.02, "delta_precision_ci_upper": 0.06,
            "delta_ucr_mean": -0.01, "delta_ucr_ci_lower": -0.05, "delta_ucr_ci_upper": 0.03,
            "delta_icr_observed": 0.0, "delta_icr_mean": 0.0, "delta_icr_ci_lower": -0.02, "delta_icr_ci_upper": 0.02,
            "cond1_precision": False, "cond2_ucr": False, "cond3_icr": True,
            "verdict": "not shown to help", "skipped_resamples": 0,
        },
    ])
    bootstrap_df.to_csv(out_dir / "bootstrap.csv", index=False)

    # 3. recommendation.csv
    pd.DataFrame([
        {"model": "llama3", "recommended_signal": "s1", "rationale": "Cheapest useful signal"},
    ]).to_csv(out_dir / "recommendation.csv", index=False)

    # 4. results_test.csv
    pd.DataFrame([
        {"model": "llama3", "signal": "s1", "target_coverage": 1.0, "tau": "-inf", "realised_coverage": 1.0, "kept_count": 100, "precision": 0.50, "recall": 0.50, "micro_f1": 0.50, "ucr": 0.40, "icr": 0.10, "review_load": 0.0},
        {"model": "llama3", "signal": "s1", "target_coverage": 0.9, "tau": "0.1", "realised_coverage": 0.90, "kept_count": 90, "precision": 0.55, "recall": 0.49, "micro_f1": 0.52, "ucr": 0.38, "icr": 0.07, "review_load": 0.5},
        {"model": "llama3", "signal": "s1", "target_coverage": 0.8, "tau": "0.3", "realised_coverage": 0.82, "kept_count": 82, "precision": 0.60, "recall": 0.48, "micro_f1": 0.53, "ucr": 0.35, "icr": 0.05, "review_load": 1.0},
        {"model": "llama3", "signal": "s1", "target_coverage": 0.7, "tau": "0.5", "realised_coverage": 0.71, "kept_count": 71, "precision": 0.65, "recall": 0.45, "micro_f1": 0.53, "ucr": 0.31, "icr": 0.04, "review_load": 1.5},
        {"model": "llama3", "signal": "s1", "target_coverage": 0.6, "tau": "0.7", "realised_coverage": 0.62, "kept_count": 62, "precision": 0.70, "recall": 0.42, "micro_f1": 0.52, "ucr": 0.28, "icr": 0.02, "review_load": 2.0},
    ]).to_csv(out_dir / "results_test.csv", index=False)

    # 5. results_controls.csv
    pd.DataFrame([
        {"model": "llama3", "signal": "s1", "target_coverage": 0.8, "realised_coverage": 0.82, "control_type": "random", "precision": 0.52, "recall": 0.42, "micro_f1": 0.46, "ucr": 0.41, "icr": 0.07, "review_load": 1.0},
        {"model": "llama3", "signal": "s1", "target_coverage": 0.8, "realised_coverage": 0.82, "control_type": "oracle", "precision": 1.00, "recall": 0.82, "micro_f1": 0.90, "ucr": 0.00, "icr": 0.00, "review_load": 1.0},
    ]).to_csv(out_dir / "results_controls.csv", index=False)

    # 6. by_feature.csv
    pd.DataFrame([
        {"model": "llama3", "stratum": "implicit", "best_signal": "s1", "tau": "0.3", "n_codes": 25, "auroc": 0.82, "precision_full": 0.45, "precision_kept": 0.58, "ucr_kept": 0.36},
        {"model": "llama3", "stratum": "explicit", "best_signal": "s1", "tau": "0.3", "n_codes": 75, "auroc": 0.90, "precision_full": 0.52, "precision_kept": 0.62, "ucr_kept": 0.34},
        {"model": "llama3", "stratum": "all notes", "best_signal": "s1", "tau": "0.3", "n_codes": 100, "auroc": 0.88, "precision_full": 0.50, "precision_kept": 0.60, "ucr_kept": 0.35},
    ]).to_csv(out_dir / "by_feature.csv", index=False)

    # 7. cost.csv
    pd.DataFrame([
        {"model": "llama3", "item": "greedy", "extra_logical_calls": 0.0, "physical_generate_calls": 1.0, "extra_prompt_tokens": 100.0, "extra_gen_tokens": 10.0, "extra_total_tokens": 110.0, "extra_seconds": 0.5, "gpu_name": "TestGPU"},
        {"model": "llama3", "item": "s1", "extra_logical_calls": 10.0, "physical_generate_calls": 1.0, "extra_prompt_tokens": 1000.0, "extra_gen_tokens": 100.0, "extra_total_tokens": 1100.0, "extra_seconds": 5.0, "gpu_name": "TestGPU"},
    ]).to_csv(out_dir / "cost.csv", index=False)

    # 8. k_sens.csv
    pd.DataFrame([
        {"model": "llama3", "k": 3, "dev_auroc": 0.82, "extra_calls_per_note": 3},
        {"model": "llama3", "k": 5, "dev_auroc": 0.86, "extra_calls_per_note": 5},
        {"model": "llama3", "k": 10, "dev_auroc": 0.90, "extra_calls_per_note": 10},
    ]).to_csv(out_dir / "k_sens.csv", index=False)

    # 9. hypotheses.csv
    pd.DataFrame([
        {"model": "llama3", "hypothesis": "H1", "claim": "S1 separates right from wrong", "estimate": 0.88, "ci_lower": 0.82, "ci_upper": 0.94, "status": "confirmed", "details": "AUROC CI > 0.5"},
        {"model": "llama3", "hypothesis": "H2", "claim": "S1 > S2", "estimate": 0.13, "ci_lower": 0.05, "ci_upper": 0.20, "status": "confirmed", "details": "ΔAUROC CI > 0"},
        {"model": "llama3", "hypothesis": "H3", "claim": "At least one signal is useful", "estimate": 1, "ci_lower": "", "ci_upper": "", "status": "confirmed", "details": "1 signal useful"},
        {"model": "llama3", "hypothesis": "H4", "claim": "Relative fall in ICR > UCR", "estimate": 0.12, "ci_lower": 0.02, "ci_upper": 0.22, "status": "confirmed", "details": "CI > 0"},
        {"model": "llama3", "hypothesis": "H5", "claim": "Implicit harder than explicit", "estimate": -0.08, "ci_lower": -0.15, "ci_upper": -0.01, "status": "confirmed", "details": "CI < 0"},
    ]).to_csv(out_dir / "hypotheses.csv", index=False)

    # 10. codes.csv (for paper numbers)
    pd.DataFrame([
        {"model": "llama3", "id": "n1", "split": "dev", "code": "E11.9", "correct": True, "invalid": False, "unsupported": False, "cause_tag": "", "s2_failed": False},
        {"model": "llama3", "id": "n2", "split": "test", "code": "I10", "correct": False, "invalid": False, "unsupported": True, "cause_tag": "negated", "s2_failed": False},
    ]).to_csv(out_dir / "codes.csv", index=False)

    # 11. confident_mistakes.csv
    pd.DataFrame([
        {"model": "llama3", "id": "n2", "code": "I10", "score": 0.85, "error_type": "unsupported", "cause_tag": "negated", "rank_in_tag": 1, "note_text": "Snippet"},
    ]).to_csv(out_dir / "confident_mistakes.csv", index=False)

    # 12. sanity_report.json
    with open(out_dir / "sanity_report.json", "w", encoding="utf-8") as f:
        json.dump({"passed": True, "failures": [], "allow_mock": True}, f)

    return out_dir


def test_table_generation_with_allow_mock(mock_pipeline_outputs: Path) -> None:
    """Verify all tables render and contain MOCK DATA banner when allow_mock=True."""
    out_dir = mock_pipeline_outputs
    tables = run_all_tables(out_dir, allow_mock=True)

    expected_tables = ["table_III", "table_IVa", "table_IVc", "table_V", "table_VI", "table_VII", "table_VIII", "table_IX", "paper_numbers"]
    for t in expected_tables:
        assert t in tables
        p = tables[t]
        assert p.is_file()
        content = p.read_text(encoding="utf-8")
        assert "MOCK DATA" in content, f"Expected MOCK DATA banner in {p}"

    # Also verify CSV files are written and non-empty
    reports_dir = out_dir / "reports"
    for base in ("table_III", "table_IVa_llama3", "table_IVc", "table_V", "table_VI", "table_VII", "table_VIII", "table_IX"):
        csv_file = reports_dir / f"{base}.csv"
        assert csv_file.is_file()
        assert csv_file.stat().st_size > 0


def test_tables_refuses_mock_in_production(mock_pipeline_outputs: Path) -> None:
    """run_all_tables must refuse mock data in production mode."""
    out_dir = mock_pipeline_outputs
    with pytest.raises(ValueError, match="Refusing production run"):
        run_all_tables(out_dir, allow_mock=False)


def test_tables_refuses_when_sanity_failed(mock_pipeline_outputs: Path) -> None:
    """run_all_tables must refuse if sanity_report.json has passed=False."""
    out_dir = mock_pipeline_outputs
    with open(out_dir / "sanity_report.json", "w", encoding="utf-8") as f:
        json.dump({"passed": False, "failures": ["code_identities"]}, f)

    with pytest.raises(RuntimeError, match="Sanity verification failed"):
        run_all_tables(out_dir, allow_mock=True)


def test_table_iv_c_has_paper_rows(mock_pipeline_outputs: Path) -> None:
    """Table IV(c) carries the paper's five measures, copied from results_test.csv."""
    df = pd.read_csv(mock_pipeline_outputs / "results_test.csv")
    md, out = generate_table_iv_c(df, {"llama3": "s1"})
    for col in ("precision", "recall", "ucr", "icr", "review_load"):
        assert col in out.columns
    assert "Recall" in md and "Review Load" in md
    kept = out[out["setting"].str.startswith("Kept")].iloc[0]
    assert kept["recall"] == "0.480" and kept["review_load"] == "1.000"
