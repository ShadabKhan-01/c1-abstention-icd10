"""Unit tests for c1/data.py and c1/tables.py (Table II).

Tests split determinism, disjointness, 30/70 fraction, corpus validation
hard checks per SPEC §1.1, note feature flags per SPEC §11, and Table II rendering.

Traces to SPEC §1.1, §4, §11, §14, §15.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from c1.codes import Vocabulary
from c1.data import (
    Distractor,
    GoldCode,
    Note,
    compute_corpus_stats,
    compute_note_features,
    load_corpus,
    split_corpus,
)
from c1.tables import generate_table_ii


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vocab() -> Vocabulary:
    """Fixture providing a small vocabulary with standard test codes."""
    codes = [
        {"code": "E11.9", "short_desc": "T2DM", "long_desc": "Type 2 diabetes mellitus"},
        {"code": "I10", "short_desc": "HTN", "long_desc": "Essential hypertension"},
        {"code": "I50.9", "short_desc": "HF", "long_desc": "Heart failure"},
        {"code": "E78.5", "short_desc": "Hyperlipidemia", "long_desc": "Hyperlipidemia"},
        {"code": "J44.9", "short_desc": "COPD", "long_desc": "Chronic obstructive pulmonary disease"},
        {"code": "G40.909", "short_desc": "Epilepsy", "long_desc": "Epilepsy, unspecified"},
    ]
    return Vocabulary(codes)


@pytest.fixture
def valid_notes() -> list[Note]:
    """Fixture providing 10 valid Notes with varied features."""
    notes = []
    for i in range(1, 11):
        nid = f"n{i:04d}"
        gold = [
            GoldCode(code="E11.9", type="explicit", rare=False),
            GoldCode(code="I10", type="implicit" if i % 2 == 0 else "explicit", rare=(i == 1)),
        ]
        distractors = [
            Distractor(
                code="I50.9",
                kind="negated" if i % 3 == 0 else ("family_history" if i % 3 == 1 else "distractor_medication"),
            )
        ]
        text = f"Patient {i} presents for review. Past history of type 2 diabetes mellitus. No heart failure."
        notes.append(Note(id=nid, text=text, gold=gold, distractors=distractors))
    return notes


@pytest.fixture
def valid_corpus_file(tmp_path: Path, valid_notes: list[Note]) -> Path:
    """Save valid notes as JSONL file."""
    path = tmp_path / "valid_corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for n in valid_notes:
            f.write(n.model_dump_json() + "\n")
    return path


# ---------------------------------------------------------------------------
# Split Tests (SPEC §4, §15)
# ---------------------------------------------------------------------------


def test_split_determinism(valid_notes: list[Note]) -> None:
    """Verify that split_corpus produces identical assignments for identical seeds."""
    df1 = split_corpus(valid_notes, dev_fraction=0.30, seed=2026)
    df2 = split_corpus(valid_notes, dev_fraction=0.30, seed=2026)

    pd.testing.assert_frame_equal(df1, df2)

    # Different seed should change permutation
    df3 = split_corpus(valid_notes, dev_fraction=0.30, seed=9999)
    assert not df1.equals(df3)


def test_split_disjointness_and_fraction(valid_notes: list[Note]) -> None:
    """Verify dev and test partitions are strictly disjoint and 30/70 proportioned."""
    df = split_corpus(valid_notes, dev_fraction=0.30, seed=2026)
    assert set(df["split"].unique()) == {"dev", "test"}

    dev_ids = set(df[df["split"] == "dev"]["id"])
    test_ids = set(df[df["split"] == "test"]["id"])
    all_ids = set(n.id for n in valid_notes)

    # Disjointness
    assert dev_ids.isdisjoint(test_ids)
    # Completeness
    assert (dev_ids | test_ids) == all_ids

    # 30% dev / 70% test
    # 10 notes * 0.3 = 3 dev, 7 test
    assert len(dev_ids) == 3
    assert len(test_ids) == 7


# ---------------------------------------------------------------------------
# Corpus Loader & Hard Checks (SPEC §1.1)
# ---------------------------------------------------------------------------


def test_load_valid_corpus(valid_corpus_file: Path, mock_vocab: Vocabulary) -> None:
    """Verify loading and validating a valid corpus file."""
    notes = load_corpus(valid_corpus_file, vocab=mock_vocab)
    assert len(notes) == 10
    assert notes[0].id == "n0001"


def test_hard_check_duplicate_ids(tmp_path: Path, mock_vocab: Vocabulary) -> None:
    """Verify duplicate IDs trigger ValueError (SPEC §1.1)."""
    bad_file = tmp_path / "dup_ids.jsonl"
    notes_data = [
        {"id": "n0001", "text": "Note 1", "gold": [{"code": "E11.9", "type": "explicit", "rare": False}], "distractors": []},
        {"id": "n0001", "text": "Note 2", "gold": [{"code": "I10", "type": "explicit", "rare": False}], "distractors": []},
    ]
    with open(bad_file, "w", encoding="utf-8") as f:
        for item in notes_data:
            f.write(json.dumps(item) + "\n")

    with pytest.raises(ValueError, match="Duplicate note ID"):
        load_corpus(bad_file, vocab=mock_vocab)


def test_hard_check_invalid_gold_code(tmp_path: Path, mock_vocab: Vocabulary) -> None:
    """Verify non-existent gold code in vocabulary triggers ValueError (SPEC §1.1)."""
    bad_file = tmp_path / "bad_code.jsonl"
    notes_data = [
        {"id": "n0001", "text": "Note 1", "gold": [{"code": "Z99.999", "type": "explicit", "rare": False}], "distractors": []},
    ]
    with open(bad_file, "w", encoding="utf-8") as f:
        for item in notes_data:
            f.write(json.dumps(item) + "\n")

    with pytest.raises(ValueError, match="invalid gold code"):
        load_corpus(bad_file, vocab=mock_vocab)


def test_hard_check_distractor_gold_overlap(tmp_path: Path, mock_vocab: Vocabulary) -> None:
    """Verify overlapping gold and distractor codes trigger ValueError (SPEC §1.1)."""
    bad_file = tmp_path / "overlap.jsonl"
    notes_data = [
        {
            "id": "n0001",
            "text": "Note 1",
            "gold": [{"code": "E11.9", "type": "explicit", "rare": False}],
            "distractors": [{"code": "E11.9", "kind": "negated"}],
        },
    ]
    with open(bad_file, "w", encoding="utf-8") as f:
        for item in notes_data:
            f.write(json.dumps(item) + "\n")

    with pytest.raises(ValueError, match="both gold and distractors"):
        load_corpus(bad_file, vocab=mock_vocab)


def test_hard_check_implicit_named_in_text(tmp_path: Path, mock_vocab: Vocabulary) -> None:
    """Verify implicit gold code explicit phrase in text triggers ValueError (SPEC §1.1)."""
    bad_file = tmp_path / "implicit_named.jsonl"
    # Create fake template conditions YAML
    cond_file = tmp_path / "conditions.yaml"
    cond_file.write_text(
        "- code: 'E11.9'\n  explicit_phrases:\n    - 'diabetes mellitus'\n",
        encoding="utf-8",
    )

    notes_data = [
        {
            "id": "n0001",
            "text": "Patient has diabetes mellitus.",
            "gold": [{"code": "E11.9", "type": "implicit", "rare": False}],
            "distractors": [],
        },
    ]
    with open(bad_file, "w", encoding="utf-8") as f:
        for item in notes_data:
            f.write(json.dumps(item) + "\n")

    with pytest.raises(ValueError, match="explicit phrase"):
        load_corpus(bad_file, vocab=mock_vocab, conditions_path=cond_file)


# ---------------------------------------------------------------------------
# Note-Level Feature Flags Tests (SPEC §11)
# ---------------------------------------------------------------------------


def test_note_feature_flags() -> None:
    """Verify computation of note-level feature flags per SPEC §11."""
    note = Note(
        id="n0001",
        text="Test note",
        gold=[
            GoldCode(code="E11.9", type="implicit", rare=True),
            GoldCode(code="I10", type="explicit", rare=False),
        ],
        distractors=[
            Distractor(code="I50.9", kind="negated"),
            Distractor(code="J44.9", kind="distractor_medication"),
        ],
    )
    features = compute_note_features(note)
    assert features["implicit"] is True
    assert features["explicit"] is True
    assert features["negation"] is True
    assert features["distractor_medication"] is True
    assert features["family_history"] is False
    assert features["rare"] is True


# ---------------------------------------------------------------------------
# Corpus Stats and Table II Rendering Tests (SPEC §14)
# ---------------------------------------------------------------------------


def test_compute_corpus_stats(valid_notes: list[Note]) -> None:
    """Verify corpus statistics summary."""
    stats = compute_corpus_stats(valid_notes)
    assert stats["n_notes"] == 10
    assert stats["total_gold_codes"] == 20
    assert stats["mean_gold_codes_per_note"] == 2.0
    assert stats["distinct_gold_codes"] == 2
    assert stats["share_implicit"] == 25.0
    assert stats["share_rare"] == 5.0


def test_generate_table_ii(valid_notes: list[Note], tmp_path: Path) -> None:
    """Verify Table II generates valid Markdown and CSV files with SPEC formatting."""
    split_df = split_corpus(valid_notes, dev_fraction=0.30, seed=2026)
    reports_dir = tmp_path / "out"

    md_str, df = generate_table_ii(
        notes=valid_notes,
        split_df=split_df,
        model_predictions=None,
        out_dir=reports_dir,
    )

    # Check Markdown
    assert "# Table II: Clinical Corpus and Prediction Statistics" in md_str
    assert "| Notes (dev/test) | 3 | 7 | 10 |" in md_str
    assert "Mean predicted codes / note (llama3)" in md_str
    assert "| — | — | — |" in md_str
    assert "*Footnote: Split sizes: Dev 3 notes, Test 7 notes." in md_str

    # Check files written
    md_file = reports_dir / "reports" / "table_II.md"
    csv_file = reports_dir / "reports" / "table_II.csv"
    assert md_file.exists()
    assert csv_file.exists()

    saved_df = pd.read_csv(csv_file)
    assert list(saved_df.columns) == ["metric", "dev", "test", "total"]
    assert len(saved_df) == 6


def test_generate_table_ii_with_predictions(valid_notes: list[Note], tmp_path: Path) -> None:
    """Verify Table II computes mean predicted codes when predictions exist."""
    split_df = split_corpus(valid_notes, dev_fraction=0.30, seed=2026)
    model_predictions = {
        "llama3": {n.id: ["E11.9", "I10"] for n in valid_notes},
        "biomistral": {n.id: ["E11.9"] for n in valid_notes},
    }

    md_str, df = generate_table_ii(
        notes=valid_notes,
        split_df=split_df,
        model_predictions=model_predictions,
        out_dir=tmp_path,
    )

    assert "2.000" in md_str  # LLaMA-3: 2 codes/note
    assert "1.000" in md_str  # BioMistral: 1 code/note


def test_cli_m1_pipeline(tmp_path: Path) -> None:
    """Verify CLI execution of make-corpus, split, and tables end-to-end."""
    from c1.cli import main as cli_main

    cfg_data = {
        "run_id": "test_m1",
        "paths": {"data_dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "out")},
        "corpus": {"n_notes": 10, "generator_seed": 42},
        "split": {"dev_fraction": 0.30, "seed": 42},
        "models": {
            "llama3": {"hf_id": "meta-llama/Meta-Llama-3-8B-Instruct", "context": 8192},
            "biomistral": {"hf_id": "BioMistral/BioMistral-7B", "context": 2048},
        },
    }
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(cfg_data, f)

    # 1. Run make-corpus
    ret = cli_main(["make-corpus", "--config", str(cfg_file)])
    assert ret == 0
    corpus_file = tmp_path / "data" / "corpus.jsonl"
    stats_file = tmp_path / "data" / "corpus_stats.json"
    assert corpus_file.exists()
    assert stats_file.exists()

    # 2. Run split
    ret = cli_main(["split", "--config", str(cfg_file)])
    assert ret == 0
    split_file = tmp_path / "out" / "test_m1" / "split.csv"
    assert split_file.exists()

    # 3. Run tables
    ret = cli_main(["tables", "--config", str(cfg_file), "--table", "II"])
    assert ret == 0
    table_md = tmp_path / "out" / "test_m1" / "reports" / "table_II.md"
    table_csv = tmp_path / "out" / "test_m1" / "reports" / "table_II.csv"
    assert table_md.exists()
    assert table_csv.exists()

