"""Line-based parsing for LLM greedy, sampled, and S2 outputs.

Extracts normalized ICD-10-CM codes, extracts character spans for S3 token alignment,
preserves vocabulary-invalid codes without repair, and parses S2 confidence scores.

Traces to SPEC §4, §5.2, §6.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from c1.codes import is_well_formed, normalize


# Regex matching optional leading list markers: "-", "*", "•", "1.", "1)"
_LIST_MARKER_RE = re.compile(r"^\s*([-*•]|\d+[\.\)])\s*")

# Candidate ICD-10 code anywhere in a line (3-character category or dotted/undotted
# 4-7 character code). It must not touch other letters/digits on either side, so
# "ICD10CM" or "HbA1c" never yield a code.
_CANDIDATE_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"([A-Za-z][0-9][0-9A-Za-z](?:\.?[0-9A-Za-z]{1,4})?)"
    r"(?![A-Za-z0-9])(?!\.[A-Za-z0-9])"
)

# Regex for S2 lines: optional marker, CODE, separator (comma, colon, whitespace), SCORE, optional %
_S2_LINE_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[\.\)])?\s*"
    r"([A-Za-z][0-9][0-9A-Za-z](?:\.?[0-9A-Za-z]{1,4})?)\s*"
    r"[,:\s]\s*"
    r"([-+]?[0-9]+(?:\.[0-9]+)?)\s*%?"
)

_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_OUT_OF_100_RE = re.compile(r"(/|out of)\s*100\b", re.IGNORECASE)


def _iter_line_codes(line: str) -> list[tuple[str, str, int, int]]:
    """Find well-formed codes anywhere in one line (SPEC §5.2).

    Returns (normalised code, raw text, start, end) per match, offsets within ``line``.
    A 3-character code without a dot that directly follows a word (e.g. "Vitamin B12",
    "vertebra T12") is skipped: it is ordinary text, not a code. The same code at the
    start of a line, or after punctuation such as "(" or ":", is kept.
    """
    marker = _LIST_MARKER_RE.match(line)
    body_start = marker.end() if marker else 0
    found: list[tuple[str, str, int, int]] = []
    for m in _CANDIDATE_CODE_RE.finditer(line, body_start):
        raw_code = m.group(1)
        norm_code = normalize(raw_code)
        if not is_well_formed(norm_code):
            continue
        before = line[body_start:m.start(1)].rstrip()
        if len(raw_code) == 3 and before and before[-1].isalpha():
            continue
        found.append((norm_code, raw_code, m.start(1), m.end(1)))
    return found


def parse_greedy_output(
    raw_output: str,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    """Parse greedy or sampled model output into normalized ICD-10-CM codes.

    Rules per SPEC §5.2:
    - Line-based; a leading list marker ('-', '*', '•', '1.', '1)') is ignored.
    - Every well-formed code anywhere in a line is taken (models often write
      "Gout (M10.0)" instead of "M10.0"); see ``_iter_line_codes`` for the one exclusion.
    - Normalizes code (inserts dot after 3rd char, uppercase).
    - Never repairs or replaces invalid codes.
    - De-duplicates while strictly preserving first occurrence order.
    - Records character spans [start, end] in raw_output for S3 token alignment.
    - parse_failed is True if and only if zero codes were parsed.

    Args:
        raw_output: Raw text output from model generation.

    Returns:
        Tuple of:
            - pred: List of normalized unique codes in order of first appearance.
            - code_char_spans: List of dicts with 'code', 'start', 'end', 'raw'.
            - parse_failed: True if no codes could be extracted, False otherwise.
    """
    pred: list[str] = []
    code_char_spans: list[dict[str, Any]] = []
    seen_codes: set[str] = set()

    current_offset = 0
    for line in raw_output.splitlines(keepends=True):
        for norm_code, raw_code, start, end in _iter_line_codes(line.rstrip("\r\n")):
            if norm_code in seen_codes:
                continue
            seen_codes.add(norm_code)
            pred.append(norm_code)
            code_char_spans.append(
                {
                    "code": norm_code,
                    "start": current_offset + start,
                    "end": current_offset + end,
                    "raw": raw_code,
                }
            )
        current_offset += len(line)

    parse_failed = len(pred) == 0
    return pred, code_char_spans, parse_failed


def parse_s2_output(
    raw_output: str,
    greedy_codes: Sequence[str],
) -> tuple[dict[str, float | None], list[str]]:
    """Parse S2 verbalized score output matching against greedy predicted codes.

    Rules per SPEC §5.2:
    - Parses lines formatted as `CODE, SCORE` (tolerates ':', whitespace, trailing '%').
    - Otherwise takes the first greedy code found anywhere in the line and the last
      number after it ("Gout (M10.0): 90").
    - Score is clipped to [0, 100] then divided by 100.
    - Codes in S2 response are matched to greedy codes by normalized string.
    - Any greedy code without a readable score gets None (mapped to S2=0 in codes.csv)
      and is recorded in failed_codes.

    Args:
        raw_output: Raw text output from S2 generation.
        greedy_codes: Sequence of predicted ICD-10-CM codes to match.

    Returns:
        Tuple of:
            - scores: Dict mapping greedy code -> parsed score in [0.0, 1.0] or None.
            - failed_codes: List of greedy codes that had no readable score.
    """
    parsed_scores: dict[str, float] = {}

    for line in raw_output.splitlines():
        line_clean = line.strip()
        match = _S2_LINE_RE.match(line_clean)
        if match:
            raw_code = match.group(1)
            raw_score_str = match.group(2)
            try:
                score_val = float(raw_score_str)
                score_val = max(0.0, min(100.0, score_val)) / 100.0
                norm_code = normalize(raw_code)
                parsed_scores[norm_code] = score_val
            except ValueError:
                continue
            continue

        # Fallback for lines such as "Gout (M10.0): 90": the first code in the line that
        # is in the greedy list, scored by the last number written after it.
        wanted = {normalize(c) for c in greedy_codes}
        for norm_code, _raw, _start, end in _iter_line_codes(line_clean):
            if norm_code not in wanted or norm_code in parsed_scores:
                continue
            tail = _OUT_OF_100_RE.sub("", line_clean[end:])
            numbers = _NUMBER_RE.findall(tail)
            if numbers:
                parsed_scores[norm_code] = max(0.0, min(100.0, float(numbers[-1]))) / 100.0
            break

    scores: dict[str, float | None] = {}
    failed_codes: list[str] = []

    for code in greedy_codes:
        norm_code = normalize(code)
        if norm_code in parsed_scores:
            scores[code] = parsed_scores[norm_code]
        else:
            scores[code] = None
            failed_codes.append(code)

    return scores, failed_codes
