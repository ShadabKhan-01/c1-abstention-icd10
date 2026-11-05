"""Corpus loading, validation, generation wrapper, and splitting.

Enforces strict input validation per SPEC §1.1, computes note-level feature
flags per SPEC §11, performs deterministic seeded dev/test splitting,
and computes corpus statistics.

Traces to SPEC §1.1, §4, §11, §15.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

from c1.codes import Vocabulary, normalize
from c1.utils import (
    Config,
    load_config,
    write_csv_atomic,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

# Allowed kinds and types according to SPEC §1.1
GoldType = Literal["explicit", "implicit"]
DistractorKind = Literal["negated", "family_history", "distractor_medication"]


class GoldCode(BaseModel):
    """Single gold-standard ICD-10-CM code for a clinical note."""

    model_config = ConfigDict(extra="forbid")
    code: str
    type: GoldType
    rare: bool = False

    @field_validator("code")
    @classmethod
    def canonical_code(cls, v: str) -> str:
        return normalize(v)


class Distractor(BaseModel):
    """Single distractor condition or medication Tempting a false positive."""

    model_config = ConfigDict(extra="forbid")
    code: str
    kind: DistractorKind

    @field_validator("code")
    @classmethod
    def canonical_code(cls, v: str) -> str:
        return normalize(v)


class Note(BaseModel):
    """Synthetic clinical note record adhering to SPEC §1.1."""

    model_config = ConfigDict(extra="forbid")
    id: str
    text: str
    gold: list[GoldCode]
    distractors: list[Distractor] = Field(default_factory=list)


def compute_note_features(note: Note) -> dict[str, bool]:
    """Compute note-level non-exclusive feature flags per SPEC §11.

    Flags:
        - implicit: >= 1 implicit gold code
        - explicit: >= 1 explicit gold code
        - negation: >= 1 negated distractor
        - family_history: >= 1 family_history distractor
        - distractor_medication: >= 1 distractor_medication distractor
        - rare: >= 1 gold code with rare=true

    Args:
        note: Note instance.

    Returns:
        Dict mapping feature name to boolean flag.
    """
    return {
        "implicit": any(g.type == "implicit" for g in note.gold),
        "explicit": any(g.type == "explicit" for g in note.gold),
        "negation": any(d.kind == "negated" for d in note.distractors),
        "family_history": any(d.kind == "family_history" for d in note.distractors),
        "distractor_medication": any(
            d.kind == "distractor_medication" for d in note.distractors
        ),
        "rare": any(g.rare for g in note.gold),
    }


def load_corpus(
    path: str | Path,
    vocab: Vocabulary | None = None,
    conditions_path: str | Path | None = "vendor/templates/conditions.yaml",
) -> list[Note]:
    """Load clinical notes from JSONL and enforce all hard checks of SPEC §1.1.

    Hard checks:
    1. IDs unique across all notes.
    2. Every gold code valid in the ICD-10-CM vocabulary.
    3. No distractor code is also gold for the same note.
    4. Implicit gold codes are not named explicitly in note text.

    Args:
        path: Path to JSONL corpus file.
        vocab: Vocabulary instance for code validity checking (defaults to 2024 vocab).
        conditions_path: Optional path to template conditions for explicit phrase check.

    Returns:
        List of validated Note instances.

    Raises:
        FileNotFoundError: If corpus file does not exist.
        ValueError: If any validation rule or hard check is violated.
    """
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    if vocab is None:
        vocab = Vocabulary.load_default()

    # Load template conditions for phrase checking if available
    phrases_by_code: dict[str, list[str]] = {}
    if conditions_path:
        cond_file = Path(conditions_path)
        if cond_file.exists():
            with open(cond_file, "r", encoding="utf-8") as f:
                cond_list = yaml.safe_load(f) or []
                for c in cond_list:
                    norm = normalize(c["code"])
                    phrases_by_code[norm] = [
                        p.lower() for p in c.get("explicit_phrases", [])
                    ]

    notes: list[Note] = []
    seen_ids: set[str] = set()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                raw_item = json.loads(line_str)
                note = Note.model_validate(raw_item)
            except Exception as e:
                raise ValueError(
                    f"Line {line_num} in {corpus_path} violates Note schema: {e}"
                ) from e

            # Hard check 1: IDs unique
            if note.id in seen_ids:
                raise ValueError(
                    f"Duplicate note ID '{note.id}' found at line {line_num} in {corpus_path} (SPEC §1.1)"
                )
            seen_ids.add(note.id)

            gold_codes_set: set[str] = set()
            for g in note.gold:
                # Hard check 2: Every gold code valid in vocabulary
                if not vocab.is_valid(g.code):
                    raise ValueError(
                        f"Note '{note.id}' contains invalid gold code '{g.code}' (SPEC §1.1)"
                    )
                gold_codes_set.add(g.code)

            # Hard check 3: No distractor code is also gold for the same note
            for d in note.distractors:
                if d.code in gold_codes_set:
                    raise ValueError(
                        f"Note '{note.id}' has code '{d.code}' in both gold and distractors (SPEC §1.1)"
                    )

            # Hard check 4: Implicit gold codes not named in text
            text_lower = note.text.lower()
            for g in note.gold:
                if g.type == "implicit":
                    # Check known explicit phrases from templates
                    known_phrases = phrases_by_code.get(g.code, [])
                    for phrase in known_phrases:
                        if phrase in text_lower:
                            raise ValueError(
                                f"Note '{note.id}' contains explicit phrase '{phrase}' for implicit code '{g.code}' (SPEC §1.1)"
                            )

            notes.append(note)

    logger.info(f"Loaded and validated {len(notes)} notes from {corpus_path}.")
    return notes


def split_corpus(
    notes: list[Note],
    dev_fraction: float = 0.30,
    seed: int = 2026,
) -> pd.DataFrame:
    """Split corpus into dev and test partitions by seeded permutation.

    Traces to SPEC §2, §4, §15.

    - Sorts note IDs for determinism.
    - Shuffles note IDs with numpy default_rng(seed).
    - dev size = round(len(notes) * dev_fraction).
    - test size = len(notes) - dev size.
    - Ensures dev and test are strictly disjoint.

    Args:
        notes: List of Note instances.
        dev_fraction: Fraction allocated to dev (default 0.30).
        seed: Random seed (default 2026).

    Returns:
        DataFrame with columns ['id', 'split'] where split in {'dev', 'test'}.
    """
    if not notes:
        return pd.DataFrame(columns=["id", "split"])

    sorted_ids = sorted([n.id for n in notes])
    rng = np.random.default_rng(seed)
    shuffled_ids = rng.permutation(sorted_ids).tolist()

    n_dev = round(len(notes) * dev_fraction)
    dev_ids = set(shuffled_ids[:n_dev])

    rows = []
    for nid in sorted_ids:
        split_label = "dev" if nid in dev_ids else "test"
        rows.append({"id": nid, "split": split_label})

    df = pd.DataFrame(rows)
    return df


def compute_corpus_stats(notes: list[Note]) -> dict[str, Any]:
    """Compute corpus-level statistics.

    Args:
        notes: List of validated Note instances.

    Returns:
        Dict with corpus statistics.
    """
    n_notes = len(notes)
    if n_notes == 0:
        return {
            "n_notes": 0,
            "mean_gold_codes_per_note": 0.0,
            "distinct_gold_codes": 0,
            "share_implicit": 0.0,
            "share_rare": 0.0,
        }

    all_gold = [g.code for n in notes for g in n.gold]
    implicit_count = sum(1 for n in notes for g in n.gold if g.type == "implicit")
    rare_count = sum(1 for n in notes for g in n.gold if g.rare)

    total_gold = len(all_gold)
    return {
        "n_notes": n_notes,
        "mean_gold_codes_per_note": round(total_gold / n_notes, 3),
        "distinct_gold_codes": len(set(all_gold)),
        "share_implicit": round((implicit_count / total_gold) * 100, 1)
        if total_gold
        else 0.0,
        "share_rare": round((rare_count / total_gold) * 100, 1) if total_gold else 0.0,
        "total_gold_codes": total_gold,
    }


def make_corpus_from_config(
    config_path: str | Path = "configs/c1.yaml",
    out_override: str | Path | None = None,
) -> Path:
    """Generate synthetic corpus using vendored generator and write stats.

    Args:
        config_path: Path to configuration YAML.
        out_override: Optional path to override data directory.

    Returns:
        Path to generated data/corpus.jsonl file.
    """
    cfg = load_config(config_path)

    data_dir = Path(out_override) if out_override else Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = data_dir / "corpus.jsonl"
    stats_file = data_dir / "corpus_stats.json"

    # Add vendor to path if needed and import vendored generator
    vendor_dir = str(Path("vendor").resolve())
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)

    from synth_data import generate_corpus  # type: ignore

    logger.info(
        f"Generating {cfg.corpus.n_notes} notes with seed {cfg.corpus.generator_seed}..."
    )
    vocab = Vocabulary.load_default()
    generate_corpus(
        n_notes=cfg.corpus.n_notes,
        seed=cfg.corpus.generator_seed,
        out_file=corpus_file,
        conditions_path="vendor/templates/conditions.yaml",
        vocab=vocab,
    )

    # Validate output with load_corpus
    notes = load_corpus(corpus_file, vocab=vocab)
    stats = compute_corpus_stats(notes)
    write_json_atomic(stats_file, stats)
    logger.info(f"Saved corpus to {corpus_file} and stats to {stats_file}.")

    return corpus_file


def run_split_from_config(
    config_path: str | Path = "configs/c1.yaml",
    out_override: str | Path | None = None,
) -> Path:
    """Partition corpus into dev and test splits and save split.csv.

    Args:
        config_path: Path to configuration YAML.
        out_override: Optional run directory override.

    Returns:
        Path to saved split.csv.
    """
    cfg = load_config(config_path)
    data_dir = Path(cfg.paths.data_dir)
    corpus_file = data_dir / "corpus.jsonl"

    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_file}. Run 'make-corpus' first."
        )

    out_dir = Path(out_override) if out_override else Path(cfg.paths.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = out_dir / "split.csv"

    notes = load_corpus(corpus_file)
    split_df = split_corpus(
        notes=notes,
        dev_fraction=cfg.split.dev_fraction,
        seed=cfg.split.seed,
    )

    write_csv_atomic(split_file, split_df)
    logger.info(f"Saved split to {split_file} ({len(split_df)} rows).")
    return split_file

