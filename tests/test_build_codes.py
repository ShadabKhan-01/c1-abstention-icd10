"""Tests for build-codes pipeline stage and codes.csv schema validation.

Traces to SPEC §4, §5.3, §6, §12, §15.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from c1.backend import MockBackend
from c1.cli import main
from c1.codes import Vocabulary
from c1.data import Distractor, GoldCode, Note
from c1.stages import (
    build_codes_stage,
    run_greedy_stage,
    run_s2_stage,
    run_s4_stage,
    run_samples_stage,
)
from c1.utils import Config


@pytest.fixture
def fixture_corpus() -> list[Note]:
    """Create a 5-note test corpus with gold and distractors."""
    notes = [
        Note(
            id="n0001",
            text="Patient 1 with type 2 diabetes and hypertension.",
            gold=[
                GoldCode(code="E11.9", type="explicit", rare=False),
                GoldCode(code="I10", type="explicit", rare=False),
            ],
            distractors=[
                Distractor(code="I50.9", kind="negated"),
            ],
        ),
        Note(
            id="n0002",
            text="Patient 2 with hypertension.",
            gold=[
                GoldCode(code="I10", type="implicit", rare=True),
            ],
            distractors=[
                Distractor(code="E11.9", kind="family_history"),
            ],
        ),
        Note(
            id="n0003",
            text="Patient 3 with diabetes.",
            gold=[
                GoldCode(code="E11.9", type="explicit", rare=False),
            ],
            distractors=[],
        ),
    ]
    return notes


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    return Config.model_validate({
        "run_id": "test_m3",
        "paths": {"data_dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "outputs")},
        "corpus": {"n_notes": 3, "generator_seed": 2026},
        "split": {"dev_fraction": 0.33, "seed": 2026},
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
        "coverage_targets": [1.0, 0.8],
        "primary_target": 0.8,
        "random_control": {"repeats": 10, "seed": 7},
        "bootstrap": {"B": 10, "seed": 42, "ci": 0.95},
        "ece_bins": 10,
        "k_sensitivity": [3],
    })


def test_build_codes_stage_end_to_end(
    tmp_path: Path,
    fixture_corpus: list[Note],
    mock_config: Config,
) -> None:
    """Verify that build_codes_stage produces codes.csv matching SPEC §4 schema

    and logs s3_post costs.
    """
    out_dir = tmp_path / "outputs" / "test_m3"
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = MockBackend(model_name="llama3")
    vocab = Vocabulary.load_default()

    # 1. Run prerequisite M2 generation stages
    run_greedy_stage(mock_config, "llama3", backend, fixture_corpus, out_dir)
    run_samples_stage(mock_config, "llama3", backend, fixture_corpus, out_dir)
    run_s2_stage(mock_config, "llama3", backend, fixture_corpus, out_dir)
    run_s4_stage(mock_config, "llama3", backend, vocab, fixture_corpus, out_dir)

    split_map = {"n0001": "dev", "n0002": "test", "n0003": "test"}

    # 2. Run build_codes_stage
    codes_path, stats = build_codes_stage(
        config=mock_config,
        model_name="llama3",
        out_dir=out_dir,
        vocab=vocab,
        notes=fixture_corpus,
        split_map=split_map,
        decode_fn=backend.decode_tokens,
    )

    assert codes_path.is_file()
    assert stats["total_predicted_codes"] > 0
    assert stats["correct_count"] + stats["invalid_count"] + stats["unsupported_count"] == stats["total_predicted_codes"]

    # 3. Read and validate codes.csv rows
    with open(codes_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == stats["total_predicted_codes"]

    expected_cols = [
        "model", "id", "split", "code", "position", "correct", "invalid", "unsupported",
        "cause_tag", "gold_type", "gold_rare", "s1", "s2", "s3", "s4", "s5", "s2_failed"
    ]

    for row in reader:
        # Check all columns exist
        assert list(row.keys()) == expected_cols

        assert row["model"] == "llama3"
        assert row["id"] in split_map
        assert row["split"] == split_map[row["id"]]
        assert int(row["position"]) >= 1

        # Check boolean correctness partition
        c = row["correct"] == "True"
        inv = row["invalid"] == "True"
        unsupp = row["unsupported"] == "True"
        assert int(c) + int(inv) + int(unsupp) == 1

        # Check signal bounds in [0.0, 1.0]
        s1 = float(row["s1"])
        s2 = float(row["s2"])
        s3 = float(row["s3"])
        s4 = float(row["s4"])
        s5 = float(row["s5"])

        assert 0.0 <= s1 <= 1.0
        assert 0.0 <= s2 <= 1.0
        assert 0.0 <= s3 <= 1.0
        assert 0.0 <= s4 <= 1.0
        assert 0.0 <= s5 <= 1.0

        # Verify S5 = (S1 + S4) / 2
        assert pytest.approx(s5, 0.0001) == (s1 + s4) / 2.0

        # Check metadata consistency
        if c:
            assert row["cause_tag"] == ""
            assert row["gold_type"] in ("explicit", "implicit")
        else:
            assert row["gold_type"] == ""
            assert row["gold_rare"] == ""
            assert row["cause_tag"] in ("negated", "family_history", "distractor_medication", "other")

    # 4. Check cost_log.jsonl has s3_post entries
    cost_file = out_dir / "cost_log.jsonl"
    assert cost_file.is_file()
    with open(cost_file, "r", encoding="utf-8") as f:
        costs = [eval(line.strip()) for line in f]  # or json.loads
    s3_costs = [c for c in costs if c.get("stage") == "s3_post"]
    assert len(s3_costs) == len(fixture_corpus)
    for c in s3_costs:
        assert c["logical_calls"] == 0
        assert c["prompt_tokens"] == 0
        assert c["gen_tokens"] == 0
        assert c["seconds"] >= 0.0


def test_build_codes_cli_mock(tmp_path: Path, mock_config: Config) -> None:
    """Verify build-codes CLI executes with --backend mock."""
    out_dir = tmp_path / "cli_m3_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # First run greedy, samples, s2, s4
    main(["run", "--config", "configs/c1.yaml", "--model", "llama3", "--backend", "mock", "--stages", "greedy,samples,s2,s4", "--limit", "3", "--out", str(out_dir)])

    # Then run build-codes
    ret = main(["build-codes", "--config", "configs/c1.yaml", "--model", "llama3", "--backend", "mock", "--out", str(out_dir)])
    assert ret == 0
    assert (out_dir / "codes.csv").is_file()

