"""Tests for pipeline stages, JSONL schemas, resumability, and cost logging.

Traces to SPEC §3, §4, §15, §16.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from c1.backend import MockBackend
from c1.codes import Vocabulary
from c1.data import Note, load_corpus
from c1.stages import (
    run_greedy_stage,
    run_s2_stage,
    run_s4_stage,
    run_samples_stage,
)
from c1.utils import Config, load_config, read_jsonl_resume


@pytest.fixture
def mock_corpus() -> list[Note]:
    """Create a 5-note mock fixture corpus."""
    notes = []
    for i in range(5):
        note = Note.model_validate({
            "id": f"n{i:04d}",
            "text": f"Patient note {i}. Patient diagnosed with type 2 diabetes and hypertension.",
            "gold": [
                {"code": "E11.9", "type": "explicit", "rare": False},
                {"code": "I10", "type": "explicit", "rare": False},
            ],
            "distractors": [],
        })
        notes.append(note)
    return notes


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Create a test configuration instance."""
    cfg_data = {
        "run_id": "test_run",
        "paths": {"data_dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "outputs")},
        "corpus": {"n_notes": 5, "generator_seed": 2026},
        "split": {"dev_fraction": 0.40, "seed": 2026},
        "models": {
            "llama3": {"hf_id": "meta-llama/Meta-Llama-3-8B-Instruct", "context": 8192},
            "biomistral": {"hf_id": "BioMistral/BioMistral-7B", "context": 2048},
        },
        "quantisation": {"type": "nf4", "compute_dtype": "float16", "double_quant": True},
        "decoding": {
            "greedy": {"do_sample": False, "max_new_tokens": 200},
            "sample": {"k": 3, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 200, "mode": "batched", "seed": 2026},
        },
        "s2": {"max_new_tokens": 200},
        "s4": {"batch_size": 1},
        "coverage_targets": [1.0, 0.9, 0.8, 0.7, 0.6],
        "primary_target": 0.8,
        "random_control": {"repeats": 10, "seed": 7},
        "bootstrap": {"B": 100, "seed": 42, "ci": 0.95},
        "ece_bins": 10,
        "k_sensitivity": [3],
    }
    return Config.model_validate(cfg_data)


def test_full_mock_pipeline_execution(
    tmp_path: Path,
    mock_corpus: list[Note],
    test_config: Config,
) -> None:
    """Verify that a mock run over a fixture corpus produces all four JSONL files

    and cost_log.jsonl with valid schemas per SPEC §4.
    """
    out_dir = tmp_path / "outputs" / "test_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = MockBackend(model_name="llama3")
    vocab = Vocabulary.load_default()

    # Run all 4 stages
    greedy_file = run_greedy_stage(test_config, "llama3", backend, mock_corpus, out_dir)
    samples_file = run_samples_stage(test_config, "llama3", backend, mock_corpus, out_dir)
    s2_file = run_s2_stage(test_config, "llama3", backend, mock_corpus, out_dir)
    s4_file = run_s4_stage(test_config, "llama3", backend, vocab, mock_corpus, out_dir)
    cost_file = out_dir / "cost_log.jsonl"

    assert greedy_file.is_file()
    assert samples_file.is_file()
    assert s2_file.is_file()
    assert s4_file.is_file()
    assert cost_file.is_file()

    # 1. Verify greedy.jsonl schema (SPEC §4)
    greedy_records = read_jsonl_resume(greedy_file)
    assert len(greedy_records) == 5
    for r in greedy_records:
        assert "id" in r
        assert r["model"] == "llama3"
        assert "raw_output" in r
        assert isinstance(r["pred"], list)
        assert isinstance(r["parse_failed"], bool)
        assert isinstance(r["token_ids"], list)
        assert isinstance(r["token_probs"], list)
        assert isinstance(r["code_char_spans"], list)
        assert isinstance(r["prompt_tokens"], int)
        assert isinstance(r["gen_tokens"], int)
        assert isinstance(r["seconds"], float)
        assert isinstance(r["note_truncated"], bool)
        assert "prompts_hash" in r
        assert "template_mode" in r
        assert r["backend"] == "mock"

    # 2. Verify samples.jsonl schema (SPEC §4)
    samples_records = read_jsonl_resume(samples_file)
    assert len(samples_records) == 5
    for r in samples_records:
        assert "id" in r
        assert r["model"] == "llama3"
        assert r["mode"] == "batched"
        assert r["k"] == 3
        assert len(r["runs"]) == 3
        for run in r["runs"]:
            assert "run" in run
            assert "seed" in run
            assert "raw_output" in run
            assert isinstance(run["codes"], list)
        assert isinstance(r["prompt_tokens"], int)
        assert isinstance(r["gen_tokens"], int)
        assert isinstance(r["seconds"], float)

    # 3. Verify s2.jsonl schema (SPEC §4)
    s2_records = read_jsonl_resume(s2_file)
    assert len(s2_records) == 5
    for r in s2_records:
        assert "id" in r
        assert r["model"] == "llama3"
        assert "raw_output" in r
        assert isinstance(r["scores"], dict)
        assert isinstance(r["failed_codes"], list)
        assert isinstance(r["prompt_tokens"], int)
        assert isinstance(r["gen_tokens"], int)
        assert isinstance(r["seconds"], float)

    # 4. Verify s4.jsonl schema (SPEC §4)
    s4_records = read_jsonl_resume(s4_file)
    assert len(s4_records) == 5
    for r in s4_records:
        assert "id" in r
        assert r["model"] == "llama3"
        assert isinstance(r["items"], list)
        for item in r["items"]:
            assert "code" in item
            assert "p_yes" in item
            assert "p_no" in item
            assert "s4" in item
            assert "top_is_yes_or_no" in item
        assert isinstance(r["prompt_tokens"], int)
        assert isinstance(r["seconds"], float)

    # 5. Verify cost_log.jsonl
    cost_records = read_jsonl_resume(cost_file)
    # 5 notes * 4 stages = 20 entries
    assert len(cost_records) == 20
    stages_logged = {c["stage"] for c in cost_records}
    assert stages_logged == {"greedy", "samples", "s2", "s4"}


def test_killing_and_restarting_resumability(
    tmp_path: Path,
    mock_corpus: list[Note],
    test_config: Config,
) -> None:
    """Verify that killing mid-run, appending a corrupted trailing line, and restarting

    loses nothing and duplicates nothing (SPEC §4, §15).
    """
    out_dir = tmp_path / "outputs" / "test_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = MockBackend(model_name="llama3")

    # Step 1: Run greedy stage on first 2 notes only
    partial_notes = mock_corpus[:2]
    greedy_file = run_greedy_stage(test_config, "llama3", backend, partial_notes, out_dir)

    initial_records = read_jsonl_resume(greedy_file)
    assert len(initial_records) == 2
    assert [r["id"] for r in initial_records] == ["n0000", "n0001"]

    # Step 2: Simulate crash leaving a half-written corrupt line
    with open(greedy_file, "a", encoding="utf-8") as f:
        f.write('{"id": "n0002", "model": "llama3", "raw_output": "E11.9\\nI10')  # no closing quote/brace

    # Step 3: Restart stage on all 5 notes
    run_greedy_stage(test_config, "llama3", backend, mock_corpus, out_dir)

    # Step 4: Verify recovered state
    final_records = read_jsonl_resume(greedy_file)
    assert len(final_records) == 5
    recorded_ids = [r["id"] for r in final_records]
    assert recorded_ids == ["n0000", "n0001", "n0002", "n0003", "n0004"]
    # No duplicate IDs
    assert len(set(recorded_ids)) == 5
