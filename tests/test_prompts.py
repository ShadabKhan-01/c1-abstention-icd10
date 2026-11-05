"""Tests for prompt formatting, template hashing, and note truncation.

Traces to SPEC §3, §4, §5.1, §15.
"""

from __future__ import annotations

import pytest

from c1.prompts import (
    BASE_PROMPT_TEMPLATE,
    NEUTRAL_DESCRIPTION,
    PROMPTS_HASH,
    S2_PROMPT_TEMPLATE,
    S4_PROMPT_TEMPLATE,
    compute_max_note_words,
    compute_prompts_hash,
    format_base_prompt,
    format_s2_prompt,
    format_s4_prompt,
    truncate_note,
)


def test_prompts_hash_determinism() -> None:
    """Verify prompts_hash is stable and deterministic."""
    h1 = compute_prompts_hash()
    h2 = compute_prompts_hash()
    assert h1 == h2
    assert h1 == PROMPTS_HASH
    assert len(h1) == 64  # SHA-256 length


def test_base_prompt_formatting() -> None:
    """Verify base prompt format matches SPEC §5.1."""
    note = "Patient has acute bronchitis."
    prompt = format_base_prompt(note)
    assert "You are a clinical coder." in prompt
    assert "Give one code per line, code only, and nothing else." in prompt
    assert f"Note:\n{note}" in prompt


def test_s2_prompt_formatting() -> None:
    """Verify S2 prompt format matches SPEC §5.1."""
    note = "Patient presents with diabetes."
    codes = ["E11.9", "I10"]
    prompt = format_s2_prompt(note, codes)
    assert "give a score from 0 to 100" in prompt
    assert "`CODE, SCORE`" in prompt
    assert f"Note:\n{note}" in prompt
    assert "Codes:\nE11.9\nI10" in prompt


def test_s4_prompt_formatting_and_neutral_fallback() -> None:
    """Verify S4 prompt formatting and neutral fallback for missing descriptions."""
    note = "Patient note content."

    # With valid description
    p1 = format_s4_prompt(note, "E11.9", "Type 2 diabetes mellitus")
    assert "ICD-10-CM code E11.9 (Type 2 diabetes mellitus)?" in p1
    assert "Answer with Yes or No only." in p1

    # With missing description (invalid codes)
    p2 = format_s4_prompt(note, "Z99.999", None)
    assert f"ICD-10-CM code Z99.999 ({NEUTRAL_DESCRIPTION})?" in p2
    assert "invalid" not in p2.lower()

    # With empty / whitespace description
    p3 = format_s4_prompt(note, "Z99.999", "   ")
    assert f"ICD-10-CM code Z99.999 ({NEUTRAL_DESCRIPTION})?" in p3


def test_truncate_note() -> None:
    """Verify shared note truncation behavior (SPEC §3)."""
    short_note = "Short text with only six words."
    text_out, truncated = truncate_note(short_note, max_words=10)
    assert text_out == short_note
    assert truncated is False

    long_note = " ".join([f"word{i}" for i in range(50)])
    text_out, truncated = truncate_note(long_note, max_words=20)
    assert len(text_out.split()) == 20
    assert truncated is True
    assert text_out.startswith("word0 word1")
    assert text_out.endswith("word19")


def test_compute_max_note_words() -> None:
    """Verify compute_max_note_words yields reasonable limits for BioMistral 2048."""
    max_w = compute_max_note_words(context_window=2048, max_new_tokens=200, reserve_tokens=250)
    assert 900 <= max_w <= 1300

