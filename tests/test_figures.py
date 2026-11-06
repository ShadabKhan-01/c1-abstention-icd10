"""Unit tests for figure generation per SPEC §14.

Traces to SPEC §4, §14, §15, §18.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from c1.figures import (
    generate_fig1_risk_coverage,
    generate_fig2_reliability,
    run_all_figures,
)


@pytest.fixture
def mock_figures_data(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Create a mock output directory with codes.csv and sanity_report.json."""
    out_dir = tmp_path / "outputs_mock" / "run_fig"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # 2 models, 100 predictions each
    import numpy as np
    rng = np.random.default_rng(42)

    for m in ("llama3", "biomistral"):
        for i in range(100):
            correct = bool(rng.random() > 0.4)
            rows.append({
                "model": m,
                "split": "test",
                "id": f"n{i}",
                "code": f"E11.{i}",
                "correct": correct,
                "invalid": False,
                "unsupported": not correct,
                "s1": float(rng.uniform(0.0, 1.0)),
                "s2": float(rng.uniform(0.0, 1.0)),
                "s3": float(rng.uniform(0.0, 1.0)),
                "s4": float(rng.uniform(0.0, 1.0)),
                "s5": float(rng.uniform(0.0, 1.0)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "codes.csv", index=False)

    # Write passed sanity_report.json
    sanity_report = {
        "passed": True,
        "failures": [],
        "allow_mock": True,
    }
    with open(out_dir / "sanity_report.json", "w", encoding="utf-8") as f:
        json.dump(sanity_report, f)

    return out_dir, df


def test_fig1_risk_coverage_generation(mock_figures_data: tuple[Path, pd.DataFrame]) -> None:
    """Generate Fig 1 risk-coverage curve and assert valid PNG image."""
    out_dir, df = mock_figures_data
    fig_path = generate_fig1_risk_coverage(df, out_dir, allow_mock=True)

    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000  # Valid non-empty PNG


def test_fig2_reliability_generation(mock_figures_data: tuple[Path, pd.DataFrame]) -> None:
    """Generate Fig 2 reliability diagram and assert valid PNG image."""
    out_dir, df = mock_figures_data
    fig_path = generate_fig2_reliability(df, out_dir, allow_mock=True, n_bins=10)

    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000


def test_run_all_figures_refuses_without_sanity(mock_figures_data: tuple[Path, pd.DataFrame]) -> None:
    """run_all_figures must refuse if sanity_report.json is absent or failed."""
    out_dir, _ = mock_figures_data
    sanity_file = out_dir / "sanity_report.json"
    sanity_file.unlink()

    with pytest.raises(RuntimeError, match="Sanity report not found"):
        run_all_figures(out_dir, allow_mock=True)


def test_run_all_figures_refuses_mock_without_flag(mock_figures_data: tuple[Path, pd.DataFrame]) -> None:
    """run_all_figures must refuse mock data in production mode."""
    out_dir, _ = mock_figures_data
    with pytest.raises(ValueError, match="Refusing production run"):
        run_all_figures(out_dir, allow_mock=False)


def test_run_all_figures_success_with_allow_mock(mock_figures_data: tuple[Path, pd.DataFrame]) -> None:
    """run_all_figures renders both figures when allow_mock=True."""
    out_dir, _ = mock_figures_data
    res = run_all_figures(out_dir, allow_mock=True)
    assert "fig1" in res and res["fig1"].is_file()
    assert "fig2" in res and res["fig2"].is_file()
