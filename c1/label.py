"""Ground truth labelling and error cause tagging for predicted ICD-10-CM codes.

Implements the three mutually exclusive correctness categories (correct, invalid, unsupported)
and cause tagging (negated, family_history, distractor_medication, other) per SPEC §5.3.

Traces to SPEC §4, §5.3, §15.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from c1.codes import Vocabulary
from c1.data import Note


# Priority order for distractor cause tags per SPEC §5.3
DISTRACTOR_PRIORITY: tuple[str, ...] = (
    "negated",
    "family_history",
    "distractor_medication",
)


@dataclass(frozen=True)
class CodeLabel:
    """Label and metadata attributes for a single predicted ICD-10-CM code."""

    code: str
    correct: bool
    invalid: bool
    unsupported: bool
    cause_tag: str
    gold_type: str
    gold_rare: str


def label_code(code: str, note: Note, vocab: Vocabulary) -> CodeLabel:
    """Assign correctness label, cause tag, and gold metadata to a predicted code.

    Rules per SPEC §5.3:
    - invalid = code does not exist in vocabulary.
    - correct = code exists in note's gold set.
    - unsupported = code is valid in vocabulary but not in note's gold set.
    - Exactly one of (correct, invalid, unsupported) is True.
    - cause_tag for wrong codes: 'negated' / 'family_history' / 'distractor_medication'
      if code matches a recorded distractor code of that note (resolved by priority order),
      otherwise 'other'. Empty string for correct codes.
    - gold_type ('explicit'/'implicit') and gold_rare ('True'/'False') populated from
      matching gold entry for correct codes; empty string for wrong codes.

    Args:
        code: Normalized uppercase ICD-10-CM code.
        note: Note object containing ground truth gold codes and distractors.
        vocab: Active ICD-10-CM Vocabulary instance.

    Returns:
        CodeLabel instance.
    """
    is_valid = vocab.is_valid(code)
    matching_gold = next((g for g in note.gold if g.code == code), None)

    if not is_valid:
        invalid = True
        correct = False
        unsupported = False
    elif matching_gold is not None:
        invalid = False
        correct = True
        unsupported = False
    else:
        invalid = False
        correct = False
        unsupported = True

    # Sanity identity: exactly one category is True
    if int(correct) + int(invalid) + int(unsupported) != 1:
        raise ValueError(
            f"Mutually exclusive labelling identity failed for code '{code}': "
            f"correct={correct}, invalid={invalid}, unsupported={unsupported}"
        )

    if correct and matching_gold is not None:
        gold_type = matching_gold.type
        gold_rare = str(matching_gold.rare)
        cause_tag = ""
    else:
        gold_type = ""
        gold_rare = ""
        matching_distractors = [d for d in note.distractors if d.code == code]
        if matching_distractors:
            kinds = {d.kind for d in matching_distractors}
            cause_tag = "other"
            for priority_kind in DISTRACTOR_PRIORITY:
                if priority_kind in kinds:
                    cause_tag = priority_kind
                    break
        else:
            cause_tag = "other"

    return CodeLabel(
        code=code,
        correct=correct,
        invalid=invalid,
        unsupported=unsupported,
        cause_tag=cause_tag,
        gold_type=gold_type,
        gold_rare=gold_rare,
    )


def label_note_codes(
    codes: Sequence[str],
    note: Note,
    vocab: Vocabulary,
) -> list[CodeLabel]:
    """Label an ordered list of predicted codes for a note.

    Args:
        codes: Sequence of predicted ICD-10-CM codes.
        note: Note object.
        vocab: Vocabulary instance.

    Returns:
        List of CodeLabel objects corresponding to each code in order.
    """
    return [label_code(c, note, vocab) for c in codes]

