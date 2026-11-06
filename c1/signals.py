"""Confidence signals S1 through S5 for LLM predicted ICD-10-CM codes.

Implements the five confidence signals per SPEC §6, §12:
- S1: Sampling agreement across k sampled lists (with k-prefix support for Table VIII).
- S2: Verbalized confidence score with missing/unreadable score failure handling.
- S3: Mean token probability over character spans overlapping the code in raw greedy output.
- S4: Verification probability P(Yes)/(P(Yes)+P(No)) with 0/0 fallback to 0.5.
- S5: Linear combination mean (S1 + S4) / 2.

Traces to SPEC §4, §6, §12, §15.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from c1.backend import align_tokens_to_chars, compute_s3_score, compute_s4_score

logger = logging.getLogger(__name__)


def compute_s1(
    sample_runs: Sequence[dict[str, Any]],
    code: str,
    k: int | None = None,
) -> float:
    """Compute S1 sampling agreement score for a code.

    Defined as the fraction of sampled lists containing the predicted code.
    Supports evaluating over a prefix k (e.g. 3, 5, 10) for Table VIII k-sensitivity.

    Args:
        sample_runs: Sequence of sample run dicts from samples.jsonl, each with 'codes'.
        code: Normalized ICD-10-CM code.
        k: Optional prefix length of runs to consider. If None, considers all runs.

    Returns:
        S1 score in [0.0, 1.0].
    """
    runs = list(sample_runs[:k]) if k is not None else list(sample_runs)
    n_runs = len(runs)
    if n_runs == 0:
        return 0.0

    hits = sum(1 for r in runs if code in r.get("codes", []))
    return hits / n_runs


def extract_s2(
    s2_record: dict[str, Any] | None,
    code: str,
) -> tuple[float, bool]:
    """Extract S2 verbalized score and failure status for a code.

    Rules per SPEC §5.2:
    - Any greedy code without a readable score gets S2 = 0.0 and s2_failed = True.
    - Otherwise returns the normalized parsed score in [0.0, 1.0] and s2_failed = False.

    Args:
        s2_record: Record from s2.jsonl, containing 'scores' and 'failed_codes'.
        code: Normalized ICD-10-CM code.

    Returns:
        Tuple of (s2_score, s2_failed).
    """
    if s2_record is None:
        return 0.0, True

    failed_codes = s2_record.get("failed_codes", [])
    scores = s2_record.get("scores", {})

    if code in failed_codes or code not in scores or scores[code] is None:
        return 0.0, True

    try:
        val = float(scores[code])
        return max(0.0, min(1.0, val)), False
    except (TypeError, ValueError):
        return 0.0, True


def compute_s3_from_greedy(
    greedy_record: dict[str, Any] | None,
    code: str,
    decode_fn: Callable[[Sequence[int]], str] | None = None,
) -> float:
    """Compute S3 score from greedy output character spans and token probabilities.

    Uses monotonic prefix decoding on token IDs to align character offsets, then
    averages probabilities of tokens overlapping the code's character span.

    Args:
        greedy_record: Record from greedy.jsonl containing 'code_char_spans',
            'token_ids', and 'token_probs'.
        code: Normalized ICD-10-CM code.
        decode_fn: Token decoding function mapping token IDs to string.

    Returns:
        S3 score in [0.0, 1.0]. Returns 0.0 if spans or probabilities are missing.
    """
    if greedy_record is None:
        return 0.0

    spans = greedy_record.get("code_char_spans", [])
    matching_span = next((s for s in spans if s.get("code") == code), None)
    if matching_span is None:
        return 0.0

    token_ids = greedy_record.get("token_ids", [])
    token_probs = greedy_record.get("token_probs", [])

    if not token_ids or not token_probs or len(token_ids) != len(token_probs):
        return 0.0

    c_span = (matching_span["start"], matching_span["end"])

    if decode_fn is not None:
        token_spans = align_tokens_to_chars(token_ids, decode_fn)
    else:
        # Fallback if no decode_fn: estimate equal character spans across text
        raw_text = greedy_record.get("raw_output", "")
        text_len = len(raw_text)
        n_toks = len(token_ids)
        token_spans = []
        for i in range(n_toks):
            s = int(i * text_len / n_toks)
            e = int((i + 1) * text_len / n_toks)
            token_spans.append((s, e))

    return compute_s3_score(c_span, token_spans, token_probs)


def extract_s4(
    s4_record: dict[str, Any] | None,
    code: str,
) -> float:
    """Extract S4 verification probability for a code from s4.jsonl.

    Args:
        s4_record: Record from s4.jsonl containing 'items'.
        code: Normalized ICD-10-CM code.

    Returns:
        S4 score in [0.0, 1.0]. Falls back to 0.5 if code record is missing.
    """
    if s4_record is None:
        return 0.5

    items = s4_record.get("items", [])
    matching_item = next((it for it in items if it.get("code") == code), None)

    if matching_item is None:
        return 0.5

    return float(matching_item.get("s4", 0.5))


def compute_s5(s1: float, s4: float) -> float:
    """Compute S5 composite signal as the arithmetic mean of S1 and S4.

    Args:
        s1: S1 sampling agreement score.
        s4: S4 verification probability score.

    Returns:
        S5 score in [0.0, 1.0].
    """
    return (s1 + s4) / 2.0

