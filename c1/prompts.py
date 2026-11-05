"""Prompt templates, formatting, hashing, and note truncation for the C1 project.

Provides standardized prompt templates for base code extraction (greedy and samples),
verbalized confidence scoring (S2), and binary verification (S4), along with
deterministic template hashing (prompts_hash) and shared max_note_words truncation.

Traces to SPEC §3, §4, §5.1.
"""

from __future__ import annotations

from typing import Sequence

from c1.utils import sha256_string


# ---------------------------------------------------------------------------
# Prompt Templates (SPEC §5.1)
# ---------------------------------------------------------------------------

BASE_PROMPT_TEMPLATE: str = (
    "You are a clinical coder. Read the clinical note below and list the ICD-10-CM "
    "diagnosis codes it supports. Give one code per line, code only, and nothing else.\n\n"
    "Note:\n{note}"
)

S2_PROMPT_TEMPLATE: str = (
    "You are a clinical coder. Read the note below. For each ICD-10-CM code in the list, "
    "give a score from 0 to 100 for how well the note supports that code. Reply with one line "
    "per code in the form `CODE, SCORE`, and nothing else.\n\n"
    "Note:\n{note}\n\n"
    "Codes:\n{codes}"
)

S4_PROMPT_TEMPLATE: str = (
    "Does the clinical note below support the ICD-10-CM code {code} ({description})? "
    "Answer with Yes or No only.\n\n"
    "Note:\n{note}"
)

NEUTRAL_DESCRIPTION: str = "description not available"

DEFAULT_MAX_NOTE_WORDS: int = 1000


# ---------------------------------------------------------------------------
# Cryptographic Template Hashing (SPEC §4, §5.1)
# ---------------------------------------------------------------------------


def compute_prompts_hash() -> str:
    """Compute deterministic SHA-256 hash across all three prompt templates.

    The hash incorporates the exact character strings of BASE, S2, and S4 templates
    and the neutral description fallback. Prompts may only be modified prior to
    generating test-split outputs.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    canonical_payload = (
        f"BASE:\n{BASE_PROMPT_TEMPLATE}\n"
        f"S2:\n{S2_PROMPT_TEMPLATE}\n"
        f"S4:\n{S4_PROMPT_TEMPLATE}\n"
        f"NEUTRAL_DESC:\n{NEUTRAL_DESCRIPTION}\n"
    )
    return sha256_string(canonical_payload)


PROMPTS_HASH: str = compute_prompts_hash()


# ---------------------------------------------------------------------------
# Prompt Formatters (SPEC §5.1)
# ---------------------------------------------------------------------------


def format_base_prompt(note: str) -> str:
    """Format base prompt for greedy code generation and S1 sampling.

    Args:
        note: Clinical note text (pre-truncated if applicable).

    Returns:
        Formatted prompt string.
    """
    return BASE_PROMPT_TEMPLATE.format(note=note)


def format_s2_prompt(note: str, codes: Sequence[str]) -> str:
    """Format S2 verbalized scoring prompt.

    Args:
        note: Clinical note text (pre-truncated if applicable).
        codes: Sequence of predicted ICD-10-CM codes to evaluate.

    Returns:
        Formatted prompt string.
    """
    codes_block = "\n".join(codes)
    return S2_PROMPT_TEMPLATE.format(note=note, codes=codes_block)


def format_s4_prompt(note: str, code: str, description: str | None = None) -> str:
    """Format S4 verification prompt for a single predicted code.

    If description is missing, empty, or whitespace, the neutral description
    is substituted without leaking invalidity.

    Args:
        note: Clinical note text (pre-truncated if applicable).
        code: ICD-10-CM code to verify.
        description: Vocabulary description or None if unavailable/invalid.

    Returns:
        Formatted prompt string.
    """
    clean_desc = (description or "").strip()
    if not clean_desc:
        clean_desc = NEUTRAL_DESCRIPTION
    return S4_PROMPT_TEMPLATE.format(note=note, code=code, description=clean_desc)


# ---------------------------------------------------------------------------
# Shared Note Truncation (SPEC §3)
# ---------------------------------------------------------------------------


def compute_max_note_words(
    context_window: int = 2048,
    max_new_tokens: int = 200,
    reserve_tokens: int = 250,
    words_per_token: float = 0.75,
) -> int:
    """Compute shared max_note_words so that the longest prompt fits BioMistral.

    Longest prompt is S2: instruction (~55 tokens) + note + code list (~100 tokens)
    + generation (200 tokens) + chat template overhead.
    Available budget for note in 2048 tokens is ~1500 tokens. At ~0.75 words/token,
    this translates to approximately 1000-1100 words.

    Args:
        context_window: Shortest model context window (BioMistral = 2048).
        max_new_tokens: Maximum generated tokens for S2 (200).
        reserve_tokens: Buffer for instructions, code list, and chat template.
        words_per_token: Conservative word-to-token ratio.

    Returns:
        Maximum allowed word count for note text.
    """
    available_tokens = max(context_window - max_new_tokens - reserve_tokens, 100)
    return int(available_tokens * words_per_token)


def truncate_note(text: str, max_words: int = DEFAULT_MAX_NOTE_WORDS) -> tuple[str, bool]:
    """Truncate note text to max_words identically across models.

    Args:
        text: Raw clinical note text.
        max_words: Maximum allowed word count.

    Returns:
        Tuple of (possibly_truncated_text, note_truncated_flag).
    """
    words = text.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        return truncated, True
    return text, False

