"""Unit tests for c1/codes.py.

Tests normalization, well-formed regex, and Vocabulary lookups with a tiny fixture.
Traces to SPEC §1.2, §15.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from c1.codes import Vocabulary, is_well_formed, normalize


# ---------------------------------------------------------------------------
# Normalization Tests
# ---------------------------------------------------------------------------


def test_normalize_basic() -> None:
    """Verify whitespace stripping and uppercase conversion."""
    assert normalize("  e11.9  ") == "E11.9"
    assert normalize("i10") == "I10"
    assert normalize("a00") == "A00"


def test_normalize_dot_insertion() -> None:
    """Verify dot insertion after the 3rd character when missing."""
    assert normalize("e119") == "E11.9"
    assert normalize("E119") == "E11.9"
    assert normalize("z9989") == "Z99.89"
    assert normalize("I10") == "I10"  # 3 chars, no dot needed
    assert normalize("E11.9") == "E11.9"  # Already has dot


# ---------------------------------------------------------------------------
# Well-formed Pattern Tests (SPEC §1.2)
# ---------------------------------------------------------------------------


def test_is_well_formed_valid_codes() -> None:
    """Verify valid ICD-10-CM code formats."""
    assert is_well_formed("E11.9")
    assert is_well_formed("e119")  # Normalizes before check
    assert is_well_formed("I10")
    assert is_well_formed("A00.0")
    assert is_well_formed("Z99.89")
    assert is_well_formed("T88.7XXA")


def test_is_well_formed_invalid_codes() -> None:
    """Verify invalid syntax rejected by regex."""
    assert not is_well_formed("")
    assert not is_well_formed("123")
    assert not is_well_formed("E")
    assert not is_well_formed("E1")
    assert not is_well_formed("U07.1")  # 'U' is not in [A-TV-Z]
    assert not is_well_formed("E11.99999")  # > 4 decimal chars
    assert not is_well_formed("INVALID")


# ---------------------------------------------------------------------------
# Vocabulary Wrapper Tests with Tiny Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_vocab_file(tmp_path: Path) -> Path:
    """Create a minimal 2-code vocabulary file fixture."""
    vocab_path = tmp_path / "tiny_vocab.jsonl"
    entries = [
        {
            "code": "E11.9",
            "is_billable": True,
            "short_desc": "Type 2 diabetes",
            "long_desc": "Type 2 diabetes mellitus without complications",
        },
        {
            "code": "I10",
            "is_billable": True,
            "short_desc": "Essential HTN",
            "long_desc": "Essential (primary) hypertension",
        },
    ]
    with open(vocab_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return vocab_path


def test_vocabulary_queries(tiny_vocab_file: Path) -> None:
    """Test is_valid, description, and length on tiny vocabulary."""
    vocab = Vocabulary.from_file(tiny_vocab_file)
    assert len(vocab) == 2

    # is_valid with various forms
    assert vocab.is_valid("E11.9")
    assert vocab.is_valid("e119")  # Normalization handling
    assert vocab.is_valid("I10")
    assert not vocab.is_valid("J44.9")  # Not in tiny fixture
    assert not vocab.is_valid("INVALID")

    # description
    assert (
        vocab.description("E11.9")
        == "Type 2 diabetes mellitus without complications"
    )
    assert vocab.description("e119") == "Type 2 diabetes mellitus without complications"
    assert vocab.description("I10") == "Essential (primary) hypertension"
    assert vocab.description("UNKNOWN") is None


def test_vocabulary_file_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        Vocabulary.from_file(tmp_path / "nonexistent.jsonl")
