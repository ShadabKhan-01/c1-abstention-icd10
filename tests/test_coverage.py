"""Tests verifying coverage matrix integrity.

Traces to SPEC §15, §18.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import re
import pytest


def test_coverage_matrix_traceability() -> None:
    """Parse COVERAGE.md and verify all implemented functions and modules exist."""
    coverage_path = Path("COVERAGE.md")
    assert coverage_path.is_file(), "COVERAGE.md must exist in root."

    content = coverage_path.read_text(encoding="utf-8")

    # Verify milestone traceability headers exist
    assert "## Milestone M0 Implemented Functions Traceability" in content
    assert "## Milestone M1 Implemented Functions Traceability" in content
    assert "## Milestone M2 Implemented Functions Traceability" in content
    assert "## Milestone M3 Implemented Functions Traceability" in content
    assert "## Milestone M4 Implemented Functions Traceability" in content
    assert "## Milestone M5 Implemented Functions Traceability" in content

    # Extract all function references of the form `c1.<module>.<func>`
    func_pattern = re.compile(r"`c1\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)`")
    matches = func_pattern.findall(content)
    assert len(matches) > 0, "Expected function traceability entries in COVERAGE.md"

    for mod_name, item_name in matches:
        full_mod = f"c1.{mod_name}"
        mod = importlib.import_module(full_mod)
        assert hasattr(mod, item_name), f"Expected {item_name} in {full_mod} per COVERAGE.md"

