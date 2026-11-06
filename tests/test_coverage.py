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
    assert "## Milestone M6 Implemented Functions Traceability" in content
    assert "## Milestone M7 Implemented Functions Traceability" in content

    # Extract all function references of the form `c1.<module>.<func>`
    func_pattern = re.compile(r"`c1\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)`")
    matches = func_pattern.findall(content)
    assert len(matches) > 0, "Expected function traceability entries in COVERAGE.md"

    for mod_name, item_name in matches:
        full_mod = f"c1.{mod_name}"
        mod = importlib.import_module(full_mod)
        assert hasattr(mod, item_name), f"Expected {item_name} in {full_mod} per COVERAGE.md"

    # Assert every paper table and figure has a registered producer (SPEC §14, §15)
    import c1.figures as figures
    import c1.tables as tables

    assert hasattr(tables, "generate_table_ii")
    assert hasattr(tables, "generate_table_iii")
    assert hasattr(tables, "generate_table_iv")
    assert hasattr(tables, "generate_table_iv_c")
    assert hasattr(tables, "generate_table_v")
    assert hasattr(tables, "generate_table_vi")
    assert hasattr(tables, "generate_table_vii")
    assert hasattr(tables, "generate_table_viii")
    assert hasattr(tables, "generate_table_ix")
    assert hasattr(tables, "generate_paper_numbers")
    assert hasattr(figures, "generate_fig1_risk_coverage")
    assert hasattr(figures, "generate_fig2_reliability")


def test_all_coverage_rows_checked() -> None:
    """Verify that every row in COVERAGE.md is checked ([x]) with no pending items."""
    coverage_path = Path("COVERAGE.md")
    content = coverage_path.read_text(encoding="utf-8")
    table_lines = [line for line in content.splitlines() if line.startswith("|") and not line.startswith("| Status") and not line.startswith("|:---:")]

    assert len(table_lines) >= 20, "Expected at least 20 coverage table rows."
    for line in table_lines:
        assert "[x]" in line, f"Found unchecked row in COVERAGE.md: {line}"
        assert "[ ]" not in line, f"Found pending [ ] box in COVERAGE.md: {line}"


def test_notebooks_valid_json_and_cli_only() -> None:
    """Verify notebooks are valid JSON, format v4, have clean outputs, and contain only CLI calls."""
    import json

    nb_paths = [
        Path("notebooks/01_run_model.ipynb"),
        Path("notebooks/02_analysis.ipynb"),
    ]

    for p in nb_paths:
        assert p.is_file(), f"Notebook {p} must exist."
        content = p.read_text(encoding="utf-8")
        nb = json.loads(content)

        assert nb.get("nbformat") == 4, f"{p} must be nbformat version 4."
        assert "cells" in nb, f"{p} must contain cells."

        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                # Ensure outputs are empty per git repository hygiene
                assert cell.get("outputs") == [], f"Code cell in {p} must have empty outputs."
                assert cell.get("execution_count") is None, f"Code cell in {p} must have null execution_count."

                # Verify cell source contains no heavy python computation logic (CLI calls or imports only)
                src = "".join(cell.get("source", []))
                for forbidden in ["def ", "class ", "fit(", "train(", "backward()", "optimizer"]:
                    assert forbidden not in src, f"Forbidden logic '{forbidden}' found in {p} code cell."


def test_scope_audit_complete() -> None:
    """Verify SCOPE_AUDIT.md exists and covers all core modules."""
    audit_path = Path("SCOPE_AUDIT.md")
    assert audit_path.is_file(), "SCOPE_AUDIT.md must exist in root."
    content = audit_path.read_text(encoding="utf-8")

    expected_modules = [
        "c1/backend.py", "c1/bootstrap.py", "c1/breakdown.py", "c1/cli.py",
        "c1/codes.py", "c1/controls.py", "c1/cost.py", "c1/data.py",
        "c1/figures.py", "c1/hypotheses.py", "c1/label.py", "c1/metrics.py",
        "c1/parse.py", "c1/prompts.py", "c1/sanity.py", "c1/signals.py",
        "c1/stages.py", "c1/tables.py", "c1/thresholds.py", "c1/utils.py"
    ]
    for mod in expected_modules:
        assert mod in content, f"Expected module {mod} in SCOPE_AUDIT.md"

    # Verify zero unmapped items claim
    assert "**Unmapped Files:** **0**" in content
    assert "**Unmapped Functions:** **0**" in content
    assert "**Coverage Matrix Completion:** **100%**" in content


