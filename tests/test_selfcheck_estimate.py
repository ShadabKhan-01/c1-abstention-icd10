"""Tests for selfcheck diagnostics and runtime estimation CLI commands.

Traces to SPEC §3, §15, §16.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from c1.backend import MockBackend
from c1.cli import main
from c1.codes import Vocabulary
from c1.data import Note
from c1.stages import run_estimate, run_selfcheck
from c1.utils import Config


@pytest.fixture
def mock_dev_notes() -> list[Note]:
    """Create 10 mock dev notes for selfcheck."""
    notes = []
    for i in range(10):
        note = Note.model_validate({
            "id": f"dev{i:04d}",
            "text": f"Patient dev note {i}. Diagnosis of type 2 diabetes and hypertension.",
            "gold": [
                {"code": "E11.9", "type": "explicit", "rare": False},
                {"code": "I10", "type": "explicit", "rare": False},
            ],
            "distractors": [],
        })
        notes.append(note)
    return notes


def test_run_selfcheck_mock(tmp_path: Path, mock_dev_notes: list[Note]) -> None:
    """Verify run_selfcheck produces a valid selfcheck_report.json."""
    out_dir = tmp_path / "selfcheck_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = MockBackend(model_name="mock_model")
    vocab = Vocabulary.load_default()

    cfg_data = {
        "run_id": "test_selfcheck",
        "paths": {"data_dir": "data", "out_dir": str(tmp_path)},
        "corpus": {"n_notes": 10, "generator_seed": 2026},
        "split": {"dev_fraction": 0.40, "seed": 2026},
        "models": {
            "mock_model": {"hf_id": "mock", "context": 2048},
        },
        "quantisation": {"type": "nf4", "compute_dtype": "float16", "double_quant": True},
        "decoding": {
            "greedy": {"do_sample": False, "max_new_tokens": 200},
            "sample": {"k": 3, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 200, "mode": "batched", "seed": 2026},
        },
        "s2": {"max_new_tokens": 200},
        "s4": {"batch_size": 1},
        "coverage_targets": [1.0, 0.8],
        "primary_target": 0.8,
        "random_control": {"repeats": 10, "seed": 7},
        "bootstrap": {"B": 10, "seed": 42, "ci": 0.95},
        "ece_bins": 10,
        "k_sensitivity": [3],
    }
    config = Config.model_validate(cfg_data)

    report = run_selfcheck(config, "mock_model", backend, vocab, mock_dev_notes, out_dir)

    assert (out_dir / "selfcheck_report.json").is_file()
    assert report["backend"] == "mock"
    assert report["greedy_determinism"]["passed"] is True
    assert len(report["greedy_determinism"]["checks"]) == 3
    assert len(report["s3_alignment"]) == 5
    assert report["s4_yes_no_check"]["tested_pairs"] == 10
    assert report["s4_yes_no_check"]["pass_rate"] == 1.0


def test_run_estimate_mock(tmp_path: Path, mock_dev_notes: list[Note]) -> None:
    """Verify run_estimate projects hours for target corpus."""
    backend = MockBackend(model_name="mock_model")
    vocab = Vocabulary.load_default()

    cfg_data = {
        "run_id": "test_estimate",
        "paths": {"data_dir": "data", "out_dir": str(tmp_path)},
        "corpus": {"n_notes": 300, "generator_seed": 2026},
        "split": {"dev_fraction": 0.30, "seed": 2026},
        "models": {
            "mock_model": {"hf_id": "mock", "context": 2048},
        },
        "quantisation": {"type": "nf4", "compute_dtype": "float16", "double_quant": True},
        "decoding": {
            "greedy": {"do_sample": False, "max_new_tokens": 200},
            "sample": {"k": 3, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 200, "mode": "batched", "seed": 2026},
        },
        "s2": {"max_new_tokens": 200},
        "s4": {"batch_size": 1},
        "coverage_targets": [1.0, 0.8],
        "primary_target": 0.8,
        "random_control": {"repeats": 10, "seed": 7},
        "bootstrap": {"B": 10, "seed": 42, "ci": 0.95},
        "ece_bins": 10,
        "k_sensitivity": [3],
    }
    config = Config.model_validate(cfg_data)

    res = run_estimate(config, "mock_model", backend, vocab, mock_dev_notes, sample_size=5)

    assert res["sample_size"] == 5
    assert res["target_notes"] == 300
    assert res["t_total_sec"] > 0
    assert res["sec_per_note"] > 0
    assert res["projected_hours"] >= 0


def test_cli_mock_runs(tmp_path: Path) -> None:
    """Verify that CLI commands 'run', 'selfcheck', and 'estimate' work with --backend mock."""
    out_dir = tmp_path / "mock_cli_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CLI selfcheck
    ret_sc = main(["selfcheck", "--config", "configs/c1.yaml", "--model", "llama3", "--backend", "mock", "--out", str(out_dir)])
    assert ret_sc == 0
    assert (out_dir / "selfcheck_report.json").is_file()

    # CLI estimate
    ret_est = main(["estimate", "--config", "configs/c1.yaml", "--model", "llama3", "--backend", "mock", "--sample-size", "5"])
    assert ret_est == 0

    # CLI run (with limit to 3 notes)
    ret_run = main(["run", "--config", "configs/c1.yaml", "--model", "llama3", "--backend", "mock", "--stages", "greedy,samples,s2,s4", "--limit", "3", "--out", str(out_dir)])
    assert ret_run == 0
    assert (out_dir / "greedy.jsonl").is_file()
    assert (out_dir / "samples.jsonl").is_file()
    assert (out_dir / "s2.jsonl").is_file()
    assert (out_dir / "s4.jsonl").is_file()
    assert (out_dir / "cost_log.jsonl").is_file()
