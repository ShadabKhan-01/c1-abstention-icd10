"""Unit tests for c1/utils.py and CLI stubs.

Covers atomic writes, truncated-line tolerance, resume behavior, seeding,
configuration validation, SHA-256 hashing, run_meta generation, and CLI stubs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from c1.cli import build_parser, main as cli_main
from c1.utils import (
    Config,
    append_jsonl,
    get_completed_ids,
    load_config,
    read_jsonl_resume,
    set_seed,
    sha256_file,
    sha256_string,
    write_csv_atomic,
    write_json_atomic,
    write_run_meta,
)


# ---------------------------------------------------------------------------
# Seeding Tests
# ---------------------------------------------------------------------------


def test_set_seed() -> None:
    """Verify that set_seed provides deterministic RNG states across python & numpy."""
    set_seed(12345)
    py_val1 = random.random()
    np_val1 = np.random.rand()

    set_seed(12345)
    py_val2 = random.random()
    np_val2 = np.random.rand()

    assert py_val1 == py_val2
    assert np_val1 == np_val2


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------


def test_load_default_config() -> None:
    """Verify loading the repository default configs/c1.yaml."""
    cfg = load_config("configs/c1.yaml")
    assert cfg.run_id == "c1_run01"
    assert cfg.paths.data_dir == "data"
    assert cfg.paths.out_dir == "outputs"
    assert cfg.corpus.n_notes == 300
    assert cfg.corpus.generator_seed == 2026
    assert cfg.split.dev_fraction == 0.30
    assert "llama3" in cfg.models
    assert "biomistral" in cfg.models
    assert cfg.models["llama3"].hf_id == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert cfg.models["llama3"].context == 8192
    assert cfg.models["biomistral"].context == 2048
    assert cfg.decoding.sample.k == 10
    assert cfg.primary_target == 0.8
    assert cfg.coverage_targets == [1.0, 0.9, 0.8, 0.7, 0.6]
    assert cfg.k_sensitivity == [3, 5, 10]


def test_config_validation_rejection(tmp_path: Path) -> None:
    """Verify that malformed or extra keys in config trigger validation errors."""
    invalid_yaml = tmp_path / "invalid.yaml"

    # Extra key should be forbidden (model_config = extra="forbid")
    invalid_yaml.write_text("unknown_extra_field: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(invalid_yaml)

    # Missing file
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "non_existent.yaml")


# ---------------------------------------------------------------------------
# Hashing Tests
# ---------------------------------------------------------------------------


def test_sha256_string_and_file(tmp_path: Path) -> None:
    """Verify SHA-256 file and string hashing functions match hashlib."""
    text = "Confidence-Based Abstention for LLM ICD-10 Coding"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert sha256_string(text) == expected

    test_file = tmp_path / "test.txt"
    test_file.write_text(text, encoding="utf-8")
    assert sha256_file(test_file) == expected


# ---------------------------------------------------------------------------
# Atomic Write Tests
# ---------------------------------------------------------------------------


def test_write_json_atomic(tmp_path: Path) -> None:
    """Verify write_json_atomic writes valid JSON and atomically updates existing files."""
    json_path = tmp_path / "nested" / "meta.json"
    data1 = {"run_id": "c1_run01", "step": 1}

    write_json_atomic(json_path, data1)
    assert json_path.is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data1

    # Overwrite atomically
    data2 = {"run_id": "c1_run01", "step": 2, "extra": "value"}
    write_json_atomic(json_path, data2)

    with open(json_path, "r", encoding="utf-8") as f:
        loaded2 = json.load(f)
    assert loaded2 == data2


def test_write_csv_atomic(tmp_path: Path) -> None:
    """Verify write_csv_atomic writes tabular data atomically from list of dicts or DataFrame."""
    # From list of dicts
    csv_path = tmp_path / "data.csv"
    records = [
        {"id": "n0001", "split": "dev", "val": 0.5},
        {"id": "n0002", "split": "test", "val": 0.8},
    ]
    write_csv_atomic(csv_path, records)
    assert csv_path.is_file()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert reader[0]["id"] == "n0001"
    assert reader[1]["split"] == "test"

    # From pandas DataFrame
    df = pd.DataFrame([{"model": "llama3", "auroc": 0.85}])
    write_csv_atomic(csv_path, df)

    loaded_df = pd.read_csv(csv_path)
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["model"] == "llama3"
    assert pytest.approx(loaded_df.iloc[0]["auroc"]) == 0.85


# ---------------------------------------------------------------------------
# Resumable JSONL Tests
# ---------------------------------------------------------------------------


def test_append_jsonl(tmp_path: Path) -> None:
    """Verify append_jsonl appends valid lines one by one."""
    jsonl_path = tmp_path / "notes.jsonl"
    r1 = {"id": "n0001", "text": "Note 1"}
    r2 = {"id": "n0002", "text": "Note 2"}

    append_jsonl(jsonl_path, r1)
    append_jsonl(jsonl_path, r2)

    records = read_jsonl_resume(jsonl_path)
    assert len(records) == 2
    assert records[0]["id"] == "n0001"
    assert records[1]["id"] == "n0002"
    assert get_completed_ids(jsonl_path) == {"n0001", "n0002"}


def test_read_jsonl_resume_non_existent(tmp_path: Path) -> None:
    """Verify read_jsonl_resume on non-existent file returns empty list."""
    missing = tmp_path / "does_not_exist.jsonl"
    assert read_jsonl_resume(missing) == []
    assert get_completed_ids(missing) == set()


def test_read_jsonl_resume_truncated_last_line(tmp_path: Path) -> None:
    """Verify truncated final line tolerance and file tail cleanup on resume.

    Simulates a Colab disconnect mid-write on note 3:
    1. Writes notes 1 and 2 cleanly.
    2. Appends incomplete / cut-off bytes for note 3.
    3. Resumes reading: notes 1 and 2 are returned; warning is logged; tail is cleaned.
    4. Writes note 3 cleanly.
    5. Reads again: notes 1, 2, and 3 are returned with zero data corruption.
    """
    jsonl_path = tmp_path / "stages.jsonl"

    # Step 1: Write notes 1 and 2
    append_jsonl(jsonl_path, {"id": "n0001", "output": "E11.9"})
    append_jsonl(jsonl_path, {"id": "n0002", "output": "I10"})

    # Step 2: Simulate aborted/truncated write for note 3
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write('{"id": "n0003", "output": "I50')  # No closing quote, brace, or newline

    # Step 3: Resumed reader should tolerate and clean truncated tail
    recovered = read_jsonl_resume(jsonl_path, clean_corrupted_tail=True)
    assert len(recovered) == 2
    assert [r["id"] for r in recovered] == ["n0001", "n0002"]

    # Step 4: Normal resumed execution appends note 3 cleanly
    append_jsonl(jsonl_path, {"id": "n0003", "output": "I50.9"})

    # Step 5: Verify all 3 notes are cleanly read with no duplicate or corruption
    final_records = read_jsonl_resume(jsonl_path)
    assert len(final_records) == 3
    assert [r["id"] for r in final_records] == ["n0001", "n0002", "n0003"]
    assert final_records[2]["output"] == "I50.9"
    assert get_completed_ids(jsonl_path) == {"n0001", "n0002", "n0003"}


def test_read_jsonl_corrupted_middle_line_raises(tmp_path: Path) -> None:
    """Verify that a corrupted intermediate line raises ValueError."""
    jsonl_path = tmp_path / "corrupted.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write('{"id": "n0001"}\n')
        f.write('THIS IS NOT JSON\n')
        f.write('{"id": "n0003"}\n')

    with pytest.raises(ValueError, match="Corrupted record at line 2"):
        read_jsonl_resume(jsonl_path)


# ---------------------------------------------------------------------------
# Run Meta Writer Tests
# ---------------------------------------------------------------------------


def test_write_run_meta(tmp_path: Path) -> None:
    """Verify write_run_meta creates and updates run_meta.json per SPEC §4."""
    out_dir = tmp_path / "run01"
    cfg = load_config("configs/c1.yaml")

    meta = write_run_meta(out_dir, config=cfg, extra={"vocab_hash": "test_vocab_sha"})

    meta_file = out_dir / "run_meta.json"
    assert meta_file.is_file()

    assert meta["python_version"] == sys.version.split()[0]
    assert "platform" in meta
    assert "library_versions" in meta
    assert meta["seeds"]["corpus_generator_seed"] == 2026
    assert meta["vocab_hash"] == "test_vocab_sha"

    # Verify update retains previous extra values
    meta2 = write_run_meta(out_dir, extra={"corpus_hash": "test_corpus_sha"})
    assert meta2["vocab_hash"] == "test_vocab_sha"
    assert meta2["corpus_hash"] == "test_corpus_sha"


# ---------------------------------------------------------------------------
# CLI Stubs & Help Tests
# ---------------------------------------------------------------------------


def test_cli_subparsers_list_all_commands() -> None:
    """Verify build_parser contains all required subcommands from SPEC §4."""
    parser = build_parser()
    expected_commands = {
        "make-corpus",
        "split",
        "selfcheck",
        "estimate",
        "run",
        "build-codes",
        "analyze",
        "tables",
        "figures",
        "sanity",
        "paper-numbers",
    }
    help_text = parser.format_help()
    for cmd in expected_commands:
        assert cmd in help_text


def test_cli_stubs_raise_not_implemented() -> None:
    """Verify that unimplemented CLI commands raise NotImplementedError with their milestone."""
    stubs = {
        "build-codes": "Milestone M3",
        "analyze": "Milestone M4",
        "figures": "Milestone M6",
        "sanity": "Milestone M6",
        "paper-numbers": "Milestone M6",
    }

    for cmd, milestone in stubs.items():
        with pytest.raises(NotImplementedError) as exc_info:
            cli_main([cmd])
        assert milestone in str(exc_info.value)
