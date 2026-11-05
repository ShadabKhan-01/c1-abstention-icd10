"""Table generation for Paper C1.

Generates Markdown and CSV reports conforming to SPEC §14 formatting rules.
In Milestone M1, produces Table II (Corpus and Prediction Statistics).

Traces to SPEC §14, §18.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from c1.data import Note, load_corpus
from c1.utils import (
    Config,
    load_config,
    write_csv_atomic,
)

logger = logging.getLogger(__name__)


def generate_table_ii(
    notes: list[Note],
    split_df: pd.DataFrame,
    model_predictions: dict[str, dict[str, list[str]]] | None = None,
    out_dir: str | Path | None = None,
    config: Config | None = None,
) -> tuple[str, pd.DataFrame]:
    """Generate Table II: Corpus and prediction statistics.

    Traces to SPEC §14:
    - notes (dev/test)
    - mean gold codes/note
    - distinct gold codes
    - share of gold codes that are implicit
    - mean predicted codes/note per model (emits '—' if not yet available)

    Args:
        notes: List of Note objects.
        split_df: DataFrame with ['id', 'split'].
        model_predictions: Optional mapping of model_name -> {note_id: [predicted_codes]}.
        out_dir: Directory to save reports/table_II.md and table_II.csv.
        config: Optional Config instance for footnotes.

    Returns:
        Tuple of (markdown_string, DataFrame).
    """
    split_map = dict(zip(split_df["id"], split_df["split"]))

    dev_notes = [n for n in notes if split_map.get(n.id) == "dev"]
    test_notes = [n for n in notes if split_map.get(n.id) == "test"]
    partitions = [
        ("dev", dev_notes),
        ("test", test_notes),
        ("total", notes),
    ]

    metrics: dict[str, dict[str, Any]] = {
        "notes": {},
        "mean_gold_codes_per_note": {},
        "distinct_gold_codes": {},
        "share_implicit_pct": {},
    }

    for label, subset in partitions:
        n_count = len(subset)
        metrics["notes"][label] = str(n_count)
        if n_count > 0:
            gold_codes = [g.code for n in subset for g in n.gold]
            total_gold = len(gold_codes)
            implicit_gold = sum(
                1 for n in subset for g in n.gold if g.type == "implicit"
            )

            metrics["mean_gold_codes_per_note"][label] = (
                f"{total_gold / n_count:.3f}"
            )
            metrics["distinct_gold_codes"][label] = str(len(set(gold_codes)))
            share_imp = (implicit_gold / total_gold * 100) if total_gold else 0.0
            metrics["share_implicit_pct"][label] = f"{share_imp:.1f}%"
        else:
            metrics["mean_gold_codes_per_note"][label] = "—"
            metrics["distinct_gold_codes"][label] = "0"
            metrics["share_implicit_pct"][label] = "—"

    # Model predictions (LLaMA-3 and BioMistral)
    model_keys = ["llama3", "biomistral"]
    model_display = {
        "llama3": "mean_predicted_codes_per_note (LLaMA-3)",
        "biomistral": "mean_predicted_codes_per_note (BioMistral)",
    }

    for m_key in model_keys:
        disp_name = model_display[m_key]
        metrics[disp_name] = {}
        for label, subset in partitions:
            if model_predictions and m_key in model_predictions:
                preds = model_predictions[m_key]
                pred_counts = [len(preds.get(n.id, [])) for n in subset]
                if subset:
                    metrics[disp_name][label] = (
                        f"{sum(pred_counts) / len(subset):.3f}"
                    )
                else:
                    metrics[disp_name][label] = "—"
            else:
                metrics[disp_name][label] = "—"

    # Construct DataFrame
    rows = []
    display_order = [
        ("notes", "Notes (dev/test)"),
        ("mean_gold_codes_per_note", "Mean gold codes / note"),
        ("distinct_gold_codes", "Distinct gold codes"),
        ("share_implicit_pct", "Share of gold codes that are implicit"),
        (
            model_display["llama3"],
            "Mean predicted codes / note (LLaMA-3-8B-Instruct)",
        ),
        (
            model_display["biomistral"],
            "Mean predicted codes / note (BioMistral-7B)",
        ),
    ]

    for key, human_label in display_order:
        row_dict = {
            "metric": human_label,
            "dev": metrics[key]["dev"],
            "test": metrics[key]["test"],
            "total": metrics[key]["total"],
        }
        rows.append(row_dict)

    df = pd.DataFrame(rows)

    # Footnotes
    gen_seed = config.corpus.generator_seed if config else 2026
    split_seed = config.split.seed if config else 2026
    n_dev = len(dev_notes)
    n_test = len(test_notes)
    footnote = (
        f"Footnote: Split sizes: Dev {n_dev} notes, Test {n_test} notes. "
        f"Generator seed: {gen_seed}, Split seed: {split_seed}. "
        f"Vocabulary release: icd10cm_2024. Pinned CDC release."
    )

    # Markdown format
    md_lines = [
        "# Table II: Clinical Corpus and Prediction Statistics",
        "",
        "| Metric | Dev | Test | Total |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['metric']} | {r['dev']} | {r['test']} | {r['total']} |"
        )
    md_lines.append("")
    md_lines.append(f"*{footnote}*")
    md_str = "\n".join(md_lines) + "\n"

    # Save to disk if out_dir specified
    if out_dir:
        reports_dir = Path(out_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / "table_II.md"
        csv_path = reports_dir / "table_II.csv"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_str)
        write_csv_atomic(csv_path, df)
        logger.info(f"Rendered Table II to {md_path} and {csv_path}.")

    return md_str, df


def run_table_ii_from_config(
    config_path: str | Path = "configs/c1.yaml",
    out_override: str | Path | None = None,
) -> tuple[str, pd.DataFrame]:
    """Run Table II generation from configuration paths."""
    cfg = load_config(config_path)
    data_dir = Path(cfg.paths.data_dir)
    corpus_file = data_dir / "corpus.jsonl"
    out_dir = (
        Path(out_override) if out_override else Path(cfg.paths.out_dir) / cfg.run_id
    )
    split_file = out_dir / "split.csv"

    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_file}. Run 'make-corpus' first."
        )
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Run 'split' first."
        )

    notes = load_corpus(corpus_file)
    split_df = pd.read_csv(split_file)

    return generate_table_ii(
        notes=notes,
        split_df=split_df,
        model_predictions=None,
        out_dir=out_dir,
        config=cfg,
    )
