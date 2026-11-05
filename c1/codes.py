"""ICD-10-CM code normalization, validation, and vocabulary access.

Provides canonical code normalization (uppercase, 3rd-character dot), regex-based
well-formedness checks, and a Vocabulary wrapper with validation and description lookups.

Traces to SPEC §1.2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

logger = logging.getLogger(__name__)

# CDC ICD-10-CM format: 1 letter (except U), 2 digits/chars, optional dot and 1-4 chars.
# Traces to SPEC §1.2.
WELL_FORMED_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$"
)


def normalize(code: str) -> str:
    """Normalize an ICD-10-CM code to canonical dotted uppercase format.

    - Strips whitespace.
    - Converts to uppercase.
    - Inserts a dot after the 3rd character if length > 3 and no dot is present.
      For example, 'e119' -> 'E11.9', 'I10' -> 'I10', 'E11.9' -> 'E11.9'.

    Args:
        code: Raw code string.

    Returns:
        Canonical uppercase dotted code string.
    """
    clean = code.strip().upper()
    if "." in clean:
        return clean

    if len(clean) > 3:
        return f"{clean[:3]}.{clean[3:]}"

    return clean


def is_well_formed(code: str) -> bool:
    """Check if a code matches the ICD-10-CM syntactic pattern.

    Args:
        code: Normalized or raw code string.

    Returns:
        True if the code matches ^[A-TV-Z][0-9][0-9AB](\\.[0-9A-TV-Z]{1,4})?$, else False.
    """
    norm = normalize(code)
    return bool(WELL_FORMED_PATTERN.match(norm))


class Vocabulary:
    """ICD-10-CM Vocabulary lookup and validity checker.

    Wraps the pinned CDC vocabulary release (icd10cm_2024.jsonl).
    Traces to SPEC §1.2.
    """

    def __init__(self, codes: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> None:
        """Initialize vocabulary from code records.

        Args:
            codes: List or dict of code records containing 'code' and descriptions.
        """
        self._codes: dict[str, dict[str, Any]] = {}
        if isinstance(codes, dict):
            for k, item in codes.items():
                self._codes[normalize(k)] = item
        else:
            for item in codes:
                self._codes[normalize(item["code"])] = item

    @classmethod
    def from_file(cls, path: str | Path) -> Vocabulary:
        """Load vocabulary from a processed JSONL file.

        Args:
            path: Path to the JSONL file.

        Returns:
            Vocabulary instance.

        Raises:
            FileNotFoundError: If the vocabulary file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {p}")

        records: list[dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))

        return cls(records)

    @classmethod
    def load_default(
        cls, path: str | Path = "data/vocab/icd10cm_2024.jsonl"
    ) -> Vocabulary:
        """Load the default pinned CDC ICD-10-CM 2024 vocabulary.

        Args:
            path: Path to default vocabulary JSONL.

        Returns:
            Vocabulary instance.
        """
        return cls.from_file(path)

    def is_valid(self, code: str) -> bool:
        """Check if a code exists in the vocabulary.

        Args:
            code: Code string (will be normalized).

        Returns:
            True if code exists in vocabulary, False otherwise.
        """
        norm = normalize(code)
        return norm in self._codes

    def description(self, code: str) -> str | None:
        """Return the description for a code, or None if code is not valid.

        Prefers long_desc, falls back to short_desc.

        Args:
            code: Code string (will be normalized).

        Returns:
            Description string or None.
        """
        norm = normalize(code)
        item = self._codes.get(norm)
        if not item:
            return None
        return item.get("long_desc") or item.get("short_desc")

    def __len__(self) -> int:
        return len(self._codes)

    def __contains__(self, code: str) -> bool:
        return self.is_valid(code)

