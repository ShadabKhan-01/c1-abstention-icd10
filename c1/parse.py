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

# Regex matching candidate ICD-10 code at the start of line text
# Supports 3-character categories or dotted/undotted 4-7 character codes
_CANDIDATE_CODE_RE = re.compile(
    r"^([A-Za-z][0-9][0-9A-Za-z](?:\.?[0-9A-Za-z]{1,4})?)(?=[\s\-_:,;(\[#|]|$)"
)

# Regex for S2 lines: optional marker, CODE, separator (comma, colon, whitespace), SCORE, optional %
_S2_LINE_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[\.\)])?\s*"
    r"([A-Za-z][0-9][0-9A-Za-z](?:\.?[0-9A-Za-z]{1,4})?)\s*"
    r"[,:\s]\s*"
    r"([-+]?[0-9]+(?:\.[0-9]+)?)\s*%?"
)


def parse_greedy_output(
    raw_output: str,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    """Parse greedy or sampled model output into normalized ICD-10-CM codes.

    Rules per SPEC §5.2:
    - Line-based parsing.
    - Strips leading list markers ('-', '*', '•', '1.', '1)').
    - Takes candidate code at start of line (allows trailing text after punctuation/space).
    - Normalizes code (inserts dot after 3rd char, uppercase).
    - Keeps code if it matches well-formed pattern.
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

    # Iterate lines while tracking absolute character offset in raw_output
    current_offset = 0
    lines = raw_output.splitlines(keepends=True)

    for line in lines:
        line_len = len(line)
        line_strip_end = line.rstrip("\r\n")

        # Strip list markers
        marker_match = _LIST_MARKER_RE.match(line_strip_end)
        if marker_match:
            marker_len = marker_match.end()
            line_body = line_strip_end[marker_len:]
            body_start_offset = current_offset + marker_len
        else:
            line_body = line_strip_end.lstrip()
            leading_ws = len(line_strip_end) - len(line_body)
            body_start_offset = current_offset + leading_ws

        # Match candidate code at start of remaining body
        code_match = _CANDIDATE_CODE_RE.match(line_body)
        if code_match:
            raw_code = code_match.group(1)
            norm_code = normalize(raw_code)

            if is_well_formed(norm_code):
                span_start = body_start_offset + code_match.start(1)
                span_end = body_start_offset + code_match.end(1)

                if norm_code not in seen_codes:
                    seen_codes.add(norm_code)
                    pred.append(norm_code)
                    code_char_spans.append(
                        {
                            "code": norm_code,
                            "start": span_start,
                            "end": span_end,
                            "raw": raw_code,
                        }
                    )

        current_offset += line_len

    parse_failed = len(pred) == 0
    return pred, code_char_spans, parse_failed


def parse_s2_output(
    raw_output: str,
    greedy_codes: Sequence[str],
) -> tuple[dict[str, float | None], list[str]]:
    """Parse S2 verbalized score output matching against greedy predicted codes.

    Rules per SPEC §5.2:
    - Parses lines formatted as `CODE, SCORE` (tolerates ':', whitespace, trailing '%').
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
