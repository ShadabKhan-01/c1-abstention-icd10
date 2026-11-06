"""Unit tests for runtime sanity checks per SPEC §15.

Traces to SPEC §4, §15, §18.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from c1.sanity import (
    check_code_identities,
    check_file_completeness,
    check_mock_mixing_guard,
    check_prompts_hash_consistency,
    check_split_integrity,
    check_stage_parity,
    run_sanity_checks,
)
from c1.utils import Config, write_csv_atomic, write_json_atomic


@pytest.fixture
def mock_run_dir(tmp_path: Path) -> Path:
    """Create a fully populated mock output directory."""
    run_dir = tmp_path / "outputs_mock" / "run_01"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage jsonl files (3 notes, 2 models)
    models = ["llama3", "biomistral"]
    for fn in ("greedy.jsonl", "samples.jsonl", "s2.jsonl", "s4.jsonl"):
        p = run_dir / fn
        with open(p, "w", encoding="utf-8") as f:
            for m in models:
                for i in range(1, 4):
                    nid = f"n{i}"
                    rec = {
                        "id": nid,
                        "model": m,
                        "backend": "mock",
                        "prompts_hash": "a1b2c3d4",
                        "pred": ["E11.9"],
                    }
                    f.write(json.dumps(rec) + "\n")

    # 2. split.csv (1 dev, 2 test)
    split_df = pd.DataFrame([
        {"id": "n1", "split": "dev"},
        {"id": "n2", "split": "test"},
        {"id": "n3", "split": "test"},
    ])
    split_df.to_csv(run_dir / "split.csv", index=False)

    # 3. codes.csv (valid identities)
    codes_rows = []
    for m in models:
        codes_rows.extend([
            {"model": m, "id": "n1", "split": "dev", "code": "E11.9", "correct": True, "invalid": False, "unsupported": False, "cause_tag": "", "s2_failed": False},
            {"model": m, "id": "n2", "split": "test", "code": "E11.9", "correct": True, "invalid": False, "unsupported": False, "cause_tag": "", "s2_failed": False},
            {"model": m, "id": "n3", "split": "test", "code": "I10", "correct": False, "invalid": False, "unsupported": True, "cause_tag": "other", "s2_failed": False},
        ])
    pd.DataFrame(codes_rows).to_csv(run_dir / "codes.csv", index=False)

    # 4. Analysis files
    for fn in (
        "metrics_signals.csv", "thresholds.json", "results_test.csv",
        "results_controls.csv", "bootstrap.csv", "by_feature.csv",
        "confident_mistakes.csv", "cost.csv", "k_sens.csv",
        "hypotheses.csv", "recommendation.csv"
    ):
        p = run_dir / fn
        if fn.endswith(".json"):
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"test": true}\n')
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write("col1,col2\n1,2\n")

    return run_dir


def test_mock_mixing_guard_blocks_mock_in_production(mock_run_dir: Path) -> None:
    """Mixing guard must refuse outputs_mock and mock backend in production mode."""
    res = check_mock_mixing_guard(mock_run_dir, allow_mock=False)
    assert res["passed"] is False
    assert "Refusing production run" in res["details"]


def test_mock_mixing_guard_allows_mock_with_flag(mock_run_dir: Path) -> None:
    """Mixing guard permits mock outputs when allow_mock=True."""
    res = check_mock_mixing_guard(mock_run_dir, allow_mock=True)
    assert res["passed"] is True
    assert "Mock data permitted" in res["details"]


def test_split_integrity_disjoint_and_ratio(mock_run_dir: Path) -> None:
    """Split integrity checks disjointness and dev ratio ±1 note."""
    split_file = mock_run_dir / "split.csv"
    res = check_split_integrity(split_file, expected_dev_fraction=0.33)
    assert res["passed"] is True

    # Corrupt split with overlap
    corrupt_split = pd.DataFrame([
        {"id": "n1", "split": "dev"},
        {"id": "n1", "split": "test"},  # overlapping ID
    ])
    corrupt_file = mock_run_dir / "corrupt_split.csv"
    corrupt_split.to_csv(corrupt_file, index=False)
    res_corrupt = check_split_integrity(corrupt_file)
    assert res_corrupt["passed"] is False
    assert "not disjoint" in res_corrupt["details"]


def test_code_identities_check(mock_run_dir: Path) -> None:
    """Code identities verify mutual exclusivity and sum identities."""
    codes_file = mock_run_dir / "codes.csv"
    res = check_code_identities(codes_file)
    assert res["passed"] is True

    # Corrupt codes where both correct and invalid are True
    corrupt_df = pd.DataFrame([
        {"model": "m1", "id": "n1", "split": "test", "code": "E11.9", "correct": True, "invalid": True, "unsupported": False},
    ])
    corrupt_file = mock_run_dir / "corrupt_codes.csv"
    corrupt_df.to_csv(corrupt_file, index=False)
    res_corrupt = check_code_identities(corrupt_file)
    assert res_corrupt["passed"] is False
    assert "violate mutual exclusivity" in res_corrupt["details"]


def test_stage_parity_check(mock_run_dir: Path) -> None:
    """Stage parity verifies all notes exist across greedy, samples, s2, and s4."""
    res = check_stage_parity(mock_run_dir, models=["llama3", "biomistral"])
    assert res["passed"] is True

    # Corrupt parity by removing a record from s4
    s4_file = mock_run_dir / "s4.jsonl"
    with open(s4_file, "w", encoding="utf-8") as f:
        # only write n1
        f.write('{"id": "n1", "model": "llama3"}\n')
    res_corrupt = check_stage_parity(mock_run_dir, models=["llama3"])
    assert res_corrupt["passed"] is False
    assert "greedy=" in res_corrupt["details"]


def test_prompts_hash_consistency(mock_run_dir: Path) -> None:
    """Prompts hash must be single-valued per model and consistent across models."""
    res = check_prompts_hash_consistency(mock_run_dir / "greedy.jsonl")
    assert res["passed"] is True

    # Corrupt by writing different prompt hashes
    greedy_corrupt = mock_run_dir / "greedy_corrupt.jsonl"
    with open(greedy_corrupt, "w", encoding="utf-8") as f:
        f.write('{"id": "n1", "model": "llama3", "prompts_hash": "hash_A"}\n')
        f.write('{"id": "n2", "model": "llama3", "prompts_hash": "hash_B"}\n')
    res_corrupt = check_prompts_hash_consistency(greedy_corrupt)
    assert res_corrupt["passed"] is False
    assert "mixed prompt hashes" in res_corrupt["details"]


def test_run_sanity_checks_end_to_end(mock_run_dir: Path) -> None:
    """End-to-end run_sanity_checks generates sanity_report.json and passes with allow_mock=True."""
    passed, report = run_sanity_checks(mock_run_dir, allow_mock=True)
    assert passed is True
    assert (mock_run_dir / "sanity_report.json").is_file()
    assert report["passed"] is True
    assert len(report["failures"]) == 0
