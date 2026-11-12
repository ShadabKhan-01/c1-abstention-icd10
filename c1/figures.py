"""Figure generation for Paper C1 per SPEC §14.

Implements:
- Fig 1: Risk–coverage curve (x = coverage, y = 1 - precision) per model,
         sweeping tau over unique test scores per signal with horizontal random line.
         Saved as reports/fig1_risk_coverage.png.
- Fig 2: Reliability diagram for S2, S3, S4 across 10 equal-width bins on [0, 1],
         plotting mean score vs observed accuracy with diagonal reference line.
         Saved as reports/fig2_reliability.png.
- Watermark "MOCK DATA" applied when rendered from mock outputs.

Traces to SPEC §4, §7, §14, §15, §18.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c1.utils import Config, model_display_name

logger = logging.getLogger(__name__)

SIGNAL_DISPLAY_NAMES = {
    "s1": "S1 (sampling)",
    "s2": "S2 (verbalised)",
    "s3": "S3 (token prob)",
    "s4": "S4 (yes/no)",
    "s5": "S5 (composite)",
}

SIGNAL_COLORS = {
    "s1": "#1f77b4",  # blue
    "s2": "#ff7f0e",  # orange
    "s3": "#2ca02c",  # green
    "s4": "#d62728",  # red
    "s5": "#9467bd",  # purple
}


def _apply_mock_watermark(ax: plt.Axes) -> None:
    """Draw a prominent semi-transparent MOCK DATA watermark banner."""
    ax.text(
        0.5,
        0.5,
        "MOCK DATA",
        fontsize=24,
        color="red",
        alpha=0.25,
        ha="center",
        va="center",
        transform=ax.transAxes,
        rotation=30,
        weight="bold",
    )


def generate_fig1_risk_coverage(
    codes_df: pd.DataFrame,
    out_dir: Path,
    allow_mock: bool = False,
    signals: Sequence[str] = ("s1", "s2", "s3", "s4", "s5"),
    config: Config | None = None,
) -> Path:
    """Generate Fig. 1: Risk–coverage curves per model per SPEC §14."""
    test_df = codes_df[codes_df["split"] == "test"].copy()
    models = sorted(test_df["model"].unique())
    if not models:
        models = sorted(codes_df["model"].unique())
        test_df = codes_df.copy()

    n_models = max(len(models), 1)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "fig1_risk_coverage.png"

    for idx, m in enumerate(models):
        ax = axes[0, idx]
        m_df = test_df[test_df["model"] == m]
        n_total = len(m_df)

        if n_total == 0:
            ax.set_title(f"{m} (No test codes)")
            continue

        # Horizontal random line = full-list error rate (1 - precision_full)
        c_full = int(m_df["correct"].sum())
        p_full = c_full / n_total if n_total > 0 else 0.0
        rand_risk = 1.0 - p_full

        ax.axhline(
            y=rand_risk,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Random ({rand_risk:.3f})",
            alpha=0.7,
        )

        for sig in signals:
            if sig not in m_df.columns:
                continue

            scores = m_df[sig].values.astype(float)
            corrects = m_df["correct"].values.astype(bool)

            # Sweep unique scores in descending order
            unique_scores = np.sort(np.unique(scores))[::-1]
            covs: list[float] = [0.0]
            risks: list[float] = [0.0]

            for tau in unique_scores:
                kept = (scores >= tau)
                k_count = int(np.sum(kept))
                if k_count == 0:
                    continue
                cov = k_count / n_total
                c_kept = int(np.sum(corrects[kept]))
                p_kept = c_kept / k_count
                risk = 1.0 - p_kept

                covs.append(cov)
                risks.append(risk)

            # Ensure curve ends at full coverage
            if covs[-1] < 1.0:
                covs.append(1.0)
                risks.append(rand_risk)

            label = SIGNAL_DISPLAY_NAMES.get(sig, sig.upper())
            color = SIGNAL_COLORS.get(sig, "gray")
            ax.plot(covs, risks, label=label, color=color, linewidth=2.0)

        model_title = model_display_name(config, m)
        if allow_mock:
            model_title += " [MOCK]"
            _apply_mock_watermark(ax)

        ax.set_title(model_title, fontsize=12, weight="bold")
        ax.set_xlabel("Coverage", fontsize=11)
        ax.set_ylabel("Risk (1 - Precision)", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Fig 1 to %s", out_path)
    return out_path


def generate_fig2_reliability(
    codes_df: pd.DataFrame,
    out_dir: Path,
    allow_mock: bool = False,
    n_bins: int = 10,
    signals: Sequence[str] = ("s2", "s3", "s4"),
    config: Config | None = None,
) -> Path:
    """Generate Fig. 2: Reliability diagram for probability signals (S2, S3, S4) per SPEC §14."""
    test_df = codes_df[codes_df["split"] == "test"].copy()
    models = sorted(test_df["model"].unique())
    if not models:
        models = sorted(codes_df["model"].unique())
        test_df = codes_df.copy()

    n_models = max(len(models), 1)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "fig2_reliability.png"

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for idx, m in enumerate(models):
        ax = axes[0, idx]
        m_df = test_df[test_df["model"] == m]

        # Diagonal reference line (y = x)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Perfect calibration", alpha=0.7)

        for sig in signals:
            if sig not in m_df.columns:
                continue

            scores = m_df[sig].values.astype(float)
            corrects = m_df["correct"].values.astype(bool)

            mean_confs: list[float] = []
            accuracies: list[float] = []

            for b in range(n_bins):
                lo = bin_edges[b]
                hi = bin_edges[b + 1]
                mask = (scores >= lo) & (scores <= hi if b == n_bins - 1 else scores < hi)

                if np.any(mask):
                    mean_conf = float(np.mean(scores[mask]))
                    acc = float(np.mean(corrects[mask]))
                    mean_confs.append(mean_conf)
                    accuracies.append(acc)

            label = SIGNAL_DISPLAY_NAMES.get(sig, sig.upper())
            color = SIGNAL_COLORS.get(sig, "gray")
            ax.plot(mean_confs, accuracies, marker="o", markersize=5, label=label, color=color, linewidth=2.0)

        model_title = model_display_name(config, m)
        if allow_mock:
            model_title += " [MOCK]"
            _apply_mock_watermark(ax)

        ax.set_title(model_title, fontsize=12, weight="bold")
        ax.set_xlabel("Mean Predicted Confidence", fontsize=11)
        ax.set_ylabel("Observed Accuracy", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Fig 2 to %s", out_path)
    return out_path


def run_all_figures(
    out_dir: str | Path,
    config: Config | None = None,
    allow_mock: bool = False,
) -> dict[str, Path]:
    """Render Figures 1 and 2 under {out_dir}/reports/.

    Args:
        out_dir: Pipeline output directory.
        config: Optional Config instance.
        allow_mock: Whether to tolerate mock outputs and apply watermark.

    Returns:
        Dict mapping figure name to output file Path.
    """
    out_p = Path(out_dir)
    codes_file = out_p / "codes.csv"
    if not codes_file.is_file():
        raise FileNotFoundError(f"codes.csv not found in {out_p}. Run build-codes stage first.")

    # Check mock guard
    from c1.sanity import check_mock_mixing_guard
    guard = check_mock_mixing_guard(out_p, allow_mock=allow_mock)
    if not guard["passed"]:
        raise ValueError(guard["details"])

    # Check sanity report
    sanity_file = out_p / "sanity_report.json"
    if not sanity_file.is_file():
        raise RuntimeError("Sanity report not found. Run 'c1 sanity' before generating figures.")
    try:
        import json
        with open(sanity_file, encoding="utf-8") as f:
            s_rep = json.load(f)
            if not s_rep.get("passed", False):
                raise RuntimeError(f"Sanity verification failed ({s_rep.get('failures')}). Figures cannot be generated.")
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Failed to read sanity_report.json: {e}")

    codes_df = pd.read_csv(codes_file)
    n_bins = config.ece_bins if config else 10

    f1 = generate_fig1_risk_coverage(codes_df, out_p, allow_mock=allow_mock, config=config)
    f2 = generate_fig2_reliability(codes_df, out_p, allow_mock=allow_mock, n_bins=n_bins, config=config)

    return {"fig1": f1, "fig2": f2}
