"""Unit tests for cost accounting and k-sensitivity.

Traces to SPEC §12, §15, §18.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from c1.cost import compute_cost_table
from c1.signals import compute_k_sensitivity


def test_cost_table_calculation(tmp_path: Path) -> None:
    """Test cost accounting table matches SPEC §12 Table VII rules."""
    cost_log_file = tmp_path / "cost_log.jsonl"
    records = [
        # Note 1
        {"model": "m1", "id": "n1", "stage": "greedy", "logical_calls": 1, "prompt_tokens": 100, "gen_tokens": 10, "seconds": 1.0},
        {"model": "m1", "id": "n1", "stage": "samples", "logical_calls": 10, "prompt_tokens": 1000, "gen_tokens": 100, "seconds": 5.0},
        {"model": "m1", "id": "n1", "stage": "s2", "logical_calls": 1, "prompt_tokens": 120, "gen_tokens": 15, "seconds": 0.5},
        {"model": "m1", "id": "n1", "stage": "s4", "logical_calls": 3, "prompt_tokens": 150, "gen_tokens": 0, "seconds": 0.8},
        {"model": "m1", "id": "n1", "stage": "s3_post", "logical_calls": 0, "prompt_tokens": 0, "gen_tokens": 0, "seconds": 0.05},
        # Note 2
        {"model": "m1", "id": "n2", "stage": "greedy", "logical_calls": 1, "prompt_tokens": 100, "gen_tokens": 10, "seconds": 1.0},
        {"model": "m1", "id": "n2", "stage": "samples", "logical_calls": 10, "prompt_tokens": 1000, "gen_tokens": 100, "seconds": 5.0},
        {"model": "m1", "id": "n2", "stage": "s2", "logical_calls": 1, "prompt_tokens": 120, "gen_tokens": 15, "seconds": 0.5},
        {"model": "m1", "id": "n2", "stage": "s4", "logical_calls": 5, "prompt_tokens": 250, "gen_tokens": 0, "seconds": 1.2},
        {"model": "m1", "id": "n2", "stage": "s3_post", "logical_calls": 0, "prompt_tokens": 0, "gen_tokens": 0, "seconds": 0.05},
    ]
    with open(cost_log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cost_rows = compute_cost_table(
        cost_log_file=cost_log_file,
        models=["m1"],
        k_samples=10,
        gpu_name="TestGPU",
    )

    items = {r["item"]: r for r in cost_rows}

    # Greedy baseline
    assert items["greedy"]["extra_logical_calls"] == 0.0
    assert items["greedy"]["physical_generate_calls"] == 1.0
    assert items["greedy"]["extra_seconds"] == 1.0
    assert items["greedy"]["gpu_name"] == "TestGPU"

    # S1 (sampling)
    assert items["s1"]["extra_logical_calls"] == 10.0
    assert items["s1"]["physical_generate_calls"] == 1.0
    assert items["s1"]["extra_seconds"] == 5.0

    # S2 (verbalized)
    assert items["s2"]["extra_logical_calls"] == 1.0
    assert items["s2"]["physical_generate_calls"] == 1.0
    assert items["s2"]["extra_seconds"] == 0.5

    # S3 (token probability)
    assert items["s3"]["extra_logical_calls"] == 0.0
    assert items["s3"]["physical_generate_calls"] == 0.0
    assert items["s3"]["extra_seconds"] == 0.05

    # S4 (mean logical calls across n1 (3) and n2 (5) is 4.0)
    assert items["s4"]["extra_logical_calls"] == 4.0
    assert items["s4"]["physical_generate_calls"] == 0.0
    assert items["s4"]["extra_seconds"] == 1.0  # mean of 0.8 and 1.2

    # S5 (S1 + S4: 10 + 4 = 14 calls)
    assert items["s5"]["extra_logical_calls"] == 14.0
    assert items["s5"]["physical_generate_calls"] == 1.0
    assert items["s5"]["extra_seconds"] == 6.0


def test_k_sensitivity_dev_split_recomputation() -> None:
    """k-sensitivity evaluates on dev split for k in {3, 5, 10} without new LLM calls."""
    dev_codes = pd.DataFrame([
        {"id": "n1", "model": "m1", "split": "dev", "code": "E11.9", "correct": True},
        {"id": "n1", "model": "m1", "split": "dev", "code": "I10", "correct": False},
        {"id": "n2", "model": "m1", "split": "dev", "code": "J44.9", "correct": True},
    ])

    samples_map = {
        "m1": {
            "n1": [
                {"codes": ["E11.9"]},
                {"codes": ["E11.9", "I10"]},
                {"codes": ["E11.9"]},
                {"codes": ["E11.9"]},
                {"codes": ["E11.9"]},
                {"codes": ["E11.9"]},
                {"codes": ["E11.9"]},
                {"codes": ["E11.9"]},
                {"codes": ["E11.9"]},
                {"codes": ["E11.9"]},
            ],
            "n2": [
                {"codes": ["J44.9"]},
                {"codes": ["J44.9"]},
                {"codes": []},
                {"codes": ["J44.9"]},
                {"codes": ["J44.9"]},
                {"codes": ["J44.9"]},
                {"codes": ["J44.9"]},
                {"codes": ["J44.9"]},
                {"codes": ["J44.9"]},
                {"codes": ["J44.9"]},
            ],
        }
    }

    rows = compute_k_sensitivity(
        models=["m1"],
        codes_df=dev_codes,
        samples_map_by_model=samples_map,
        k_values=(3, 5, 10),
    )

    assert len(rows) == 3
    for r, expected_k in zip(rows, (3, 5, 10)):
        assert r["k"] == expected_k
        assert r["extra_calls_per_note"] == expected_k
        assert r["dev_auroc"] != ""
        assert 0.0 <= float(r["dev_auroc"]) <= 1.0

