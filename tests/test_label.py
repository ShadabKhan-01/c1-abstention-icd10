"""Tests for code correctness labelling and cause tagging.

Traces to SPEC §5.3, §15.
"""

from __future__ import annotations

import pytest

from c1.codes import Vocabulary
from c1.data import Distractor, GoldCode, Note
from c1.label import CodeLabel, label_code, label_note_codes


@pytest.fixture
def test_vocab() -> Vocabulary:
    return Vocabulary.load_default()


@pytest.fixture
def sample_note() -> Note:
    return Note(
        id="n0001",
        text="Sample note mentioning diabetes, hypertension, and denied chest pain.",
        gold=[
            GoldCode(code="E11.9", type="explicit", rare=False),
            GoldCode(code="I10", type="implicit", rare=True),
        ],
        distractors=[
            Distractor(code="R07.9", kind="negated"),
            Distractor(code="Z86.718", kind="family_history"),
            Distractor(code="I50.9", kind="distractor_medication"),
        ],
    )


def test_labelling_mutual_exclusivity(test_vocab: Vocabulary, sample_note: Note) -> None:
    """Verify that exactly one of correct, invalid, unsupported is True (SPEC §5.3)."""
    # 1. Correct code (valid in vocab, in gold)
    lbl_corr = label_code("E11.9", sample_note, test_vocab)
    assert lbl_corr.correct is True
    assert lbl_corr.invalid is False
    assert lbl_corr.unsupported is False
    assert int(lbl_corr.correct) + int(lbl_corr.invalid) + int(lbl_corr.unsupported) == 1

    # 2. Unsupported code (valid in vocab, NOT in gold)
    lbl_unsupp = label_code("J45.909", sample_note, test_vocab)
    assert lbl_unsupp.correct is False
    assert lbl_unsupp.invalid is False
    assert lbl_unsupp.unsupported is True
    assert int(lbl_unsupp.correct) + int(lbl_unsupp.invalid) + int(lbl_unsupp.unsupported) == 1

    # 3. Invalid code (not in vocab)
    lbl_inv = label_code("Z99.999", sample_note, test_vocab)
    assert lbl_inv.correct is False
    assert lbl_inv.invalid is True
    assert lbl_inv.unsupported is False
    assert int(lbl_inv.correct) + int(lbl_inv.invalid) + int(lbl_inv.unsupported) == 1


def test_gold_metadata_propagation(test_vocab: Vocabulary, sample_note: Note) -> None:
    """Verify gold_type and gold_rare propagation for correct codes."""
    lbl_e119 = label_code("E11.9", sample_note, test_vocab)
    assert lbl_e119.correct is True
    assert lbl_e119.gold_type == "explicit"
    assert lbl_e119.gold_rare == "False"
    assert lbl_e119.cause_tag == ""

    lbl_i10 = label_code("I10", sample_note, test_vocab)
    assert lbl_i10.correct is True
    assert lbl_i10.gold_type == "implicit"
    assert lbl_i10.gold_rare == "True"
    assert lbl_i10.cause_tag == ""


def test_cause_tags_and_priority(test_vocab: Vocabulary, sample_note: Note) -> None:
    """Verify cause tagging for wrong codes and priority ordering per SPEC §5.3."""
    # Negated distractor
    lbl_neg = label_code("R07.9", sample_note, test_vocab)
    assert lbl_neg.unsupported is True
    assert lbl_neg.cause_tag == "negated"

    # Family history distractor
    lbl_fh = label_code("Z86.718", sample_note, test_vocab)
    assert lbl_fh.unsupported is True
    assert lbl_fh.cause_tag == "family_history"

    # Distractor medication
    lbl_med = label_code("I50.9", sample_note, test_vocab)
    assert lbl_med.unsupported is True
    assert lbl_med.cause_tag == "distractor_medication"

    # Other wrong code not in distractors
    lbl_oth = label_code("K21.9", sample_note, test_vocab)
    assert lbl_oth.unsupported is True
    assert lbl_oth.cause_tag == "other"

    # Test priority order: note with code matching multiple distractor kinds
    multi_distractor_note = Note(
        id="n0002",
        text="Note text",
        gold=[],
        distractors=[
            Distractor(code="I50.9", kind="distractor_medication"),
            Distractor(code="I50.9", kind="negated"),
        ],
    )
    # Negated takes precedence over distractor_medication
    lbl_prio = label_code("I50.9", multi_distractor_note, test_vocab)
    assert lbl_prio.cause_tag == "negated"


def test_label_note_codes_batch(test_vocab: Vocabulary, sample_note: Note) -> None:
    """Verify label_note_codes batch labels in order."""
    pred = ["E11.9", "R07.9", "Z99.999"]
    labels = label_note_codes(pred, sample_note, test_vocab)
    assert len(labels) == 3
    assert labels[0].correct and labels[0].code == "E11.9"
    assert labels[1].unsupported and labels[1].cause_tag == "negated"
    assert labels[2].invalid and labels[2].cause_tag == "other"

