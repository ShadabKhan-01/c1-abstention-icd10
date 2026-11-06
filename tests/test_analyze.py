"""Tests for M4 analysis stage, output schemas, and CLI execution.

Traces to SPEC §4, §7, §8, §9, §14, §15.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest
import pandas as pd

from c1.cli import main
from c1.stages import run_analyze_stage
from c1.utils import Config, write_csv_atomic


@pytest.fixture
def mock_analyze_env(tmp_path: Path) -> tuple[Config, Path]:
    """Set up a test environment with mock data/corpus.jsonl, split.csv, and codes.csv."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "outputs" / "test_run"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. corpus.jsonl (10 notes: 4 dev, 6 test)
    corpus_file = data_dir / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for i in range(10):
            nid = f"n{i:04d}"
            # each note has 2 gold codes
            f.write(f'{{"id": "{nid}", "text": "Note {i}", "gold": [{{"code": "E11.9", "type": "explicit", "rare": false}}, {{"code": "I10", "type": "explicit", "rare": false}}], "distractors": []}}\n')

    # 2. split.csv
    split_file = data_dir / "split.csv"
    with open(split_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "split"])
        for i in range(10):
            writer.writerow([f"n{i:04d}", "dev" if i < 4 else "test"])

    # 3. codes.csv (predictions across dev and test notes)
    codes_rows = []
    # Dev: 4 notes * 2 codes = 8 codes
    for i in range(4):
        nid = f"n{i:04d}"
        codes_rows.append({
            "model": "llama3", "id": nid, "split": "dev", "code": "E11.9", "position": 1,
            "correct": True, "invalid": False, "unsupported": False, "cause_tag": "",
            "gold_type": "explicit", "gold_rare": "False",
            "s1": 0.90, "s2": 0.95, "s3": 0.92, "s4": 0.88, "s5": 0.89, "s2_failed": False
        })
        codes_rows.append({
            "model": "llama3", "id": nid, "split": "dev", "code": "I10", "position": 2,
            "correct": False, "invalid": False, "unsupported": True, "cause_tag": "other",
            "gold_type": "", "gold_rare": "",
            "s1": 0.30, "s2": 0.40, "s3": 0.50, "s4": 0.20, "s5": 0.25, "s2_failed": False
        })

    # Test: 6 notes * 2 codes = 12 codes (6 correct, 6 unsupported)
    for i in range(4, 10):
        nid = f"n{i:04d}"
        codes_rows.append({
            "model": "llama3", "id": nid, "split": "test", "code": "E11.9", "position": 1,
            "correct": True, "invalid": False, "unsupported": False, "cause_tag": "",
            "gold_type": "explicit", "gold_rare": "False",
            "s1": 0.85, "s2": 0.90, "s3": 0.91, "s4": 0.85, "s5": 0.85, "s2_failed": False
        })
        codes_rows.append({
            "model": "llama3", "id": nid, "split": "test", "code": "I10", "position": 2,
            "correct": False, "invalid": False, "unsupported": True, "cause_tag": "other",
            "gold_type": "", "gold_rare": "",
            "s1": 0.25, "s2": 0.35, "s3": 0.45, "s4": 0.15, "s5": 0.20, "s2_failed": False
        })

    codes_file = out_dir / "codes.csv"
    write_csv_atomic(codes_file, codes_rows)

    cfg = Config.model_validate({
        "run_id": "test_run",
        "paths": {"data_dir": str(data_dir), "out_dir": str(tmp_path / "outputs")},
        "corpus": {"n_notes": 10, "generator_seed": 2026},
        "split": {"dev_fraction": 0.40, "seed": 2026},
        "models": {
            "llama3": {"hf_id": "meta-llama/Meta-Llama-3-8B-Instruct", "context": 8192},
        },
        "quantisation": {"type": "nf4", "compute_dtype": "float16", "double_quant": True},
        "decoding": {
            "greedy": {"do_sample": False, "max_new_tokens": 200},
            "sample": {"k": 3, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 200, "mode": "batched", "seed": 2026},
        },
        "s2": {"max_new_tokens": 200},
        "s4": {"batch_size": 1},
        "coverage_targets": [1.0, 0.8, 0.6],
        "primary_target": 0.8,
        "random_control": {"repeats": 20, "seed": 7},
        "bootstrap": {"B": 10, "seed": 42, "ci": 0.95},
        "ece_bins": 10,
        "k_sensitivity": [3],
    })

    return cfg, out_dir


def test_run_analyze_stage_full(mock_analyze_env: tuple[Config, Path]) -> None:
    """Verify that run_analyze_stage produces all 4 expected M4 files and valid schemas."""
    cfg, out_dir = mock_analyze_env

    summary = run_analyze_stage(cfg, out_dir)

    assert (out_dir / "thresholds.json").is_file()
    assert (out_dir / "metrics_signals.csv").is_file()
    assert (out_dir / "results_test.csv").is_file()
    assert (out_dir / "results_controls.csv").is_file()

    # 1. Verify metrics_signals.csv
    df_metrics = pd.read_csv(out_dir / "metrics_signals.csv")
    assert len(df_metrics) == 2 * 5  # 2 splits (dev, test) * 5 signals
    assert set(df_metrics.columns) == {"model", "split", "signal", "auroc", "ece"}
    # On test split, S1 cleanly separates correct (0.85) from wrong (0.25) -> AUROC = 1.0
    test_s1_auc = df_metrics[(df_metrics["split"] == "test") & (df_metrics["signal"] == "s1")]["auroc"].values[0]
    assert test_s1_auc == 1.0

    # 2. Verify results_test.csv and 100% full list row identity (SPEC §15)
    df_results = pd.read_csv(out_dir / "results_test.csv")
    assert len(df_results) == 5 * 3  # 5 signals * 3 targets (1.0, 0.8, 0.6)

    # For target 1.0, tau is -inf, kept_count is all 12 test codes, realised_coverage is 1.0
    row_100 = df_results[(df_results["signal"] == "s1") & (df_results["target_coverage"] == 1.0)].iloc[0]
    assert str(row_100["tau"]).lower() in ("-inf", "-infinity")
    assert row_100["kept_count"] == 12
    assert row_100["realised_coverage"] == 1.0
    # 6 correct out of 12 kept -> precision = 0.5
    assert row_100["precision"] == 0.5
    # 6 correct out of 12 gold (6 test notes * 2 gold codes = 12 gold) -> recall = 0.5
    assert row_100["recall"] == 0.5
    assert row_100["ucr"] == 0.5
    assert row_100["icr"] == 0.0

    # 3. Verify results_controls.csv
    df_ctrl = pd.read_csv(out_dir / "results_controls.csv")
    assert len(df_ctrl) == len(df_results) * 2  # random and oracle for each row
    assert set(df_ctrl["control_type"].unique()) == {"random", "oracle"}


def test_cli_analyze_command(mock_analyze_env: tuple[Config, Path], tmp_path: Path) -> None:
    """Verify that 'c1 analyze' executes via CLI without error."""
    cfg, out_dir = mock_analyze_env
    config_path = tmp_path / "c1_test.yaml"
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(), f)

    ret = main(["analyze", "--config", str(config_path), "--out", str(out_dir)])
    assert ret == 0
