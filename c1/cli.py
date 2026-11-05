"""Command-line interface for the C1 project.

Provides CLI entry points for all pipeline stages.
In Milestone M0, all stage commands are stubs that raise NotImplementedError
with the milestone in which they will be implemented.

Traces to SPEC §4.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all pipeline subcommands."""
    parser = argparse.ArgumentParser(
        prog="c1",
        description="Confidence-Based Abstention for LLM ICD-10 Coding (Paper C1)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Milestone M1 commands
    p_corpus = subparsers.add_parser(
        "make-corpus",
        help="Generate synthetic corpus notes (Milestone M1)",
    )
    p_corpus.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_corpus.add_argument("--out", default=None, help="Output run directory")

    p_split = subparsers.add_parser(
        "split",
        help="Split corpus into dev and test partitions (Milestone M1)",
    )
    p_split.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_split.add_argument("--out", default=None, help="Output run directory")

    # Milestone M2 commands
    p_selfcheck = subparsers.add_parser(
        "selfcheck",
        help="Diagnostic checks: GPU, determinism, S3 alignment, S4 Yes/No (Milestone M2)",
    )
    p_selfcheck.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_selfcheck.add_argument("--out", default=None, help="Output run directory")
    p_selfcheck.add_argument("--model", required=False, help="Model to check (llama3 or biomistral)")

    p_estimate = subparsers.add_parser(
        "estimate",
        help="Estimate runtime and projected hours for full corpus (Milestone M2)",
    )
    p_estimate.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_estimate.add_argument("--out", default=None, help="Output run directory")
    p_estimate.add_argument("--model", required=False, help="Model to estimate")

    p_run = subparsers.add_parser(
        "run",
        help="Run model generation stages: greedy, samples, s2, s4 (Milestone M2)",
    )
    p_run.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_run.add_argument("--out", default=None, help="Output run directory")
    p_run.add_argument("--model", required=False, help="Model name")
    p_run.add_argument("--stages", default="greedy,samples,s2,s4", help="Comma-separated stage list")
    p_run.add_argument("--split", default=None, help="Optional split filter (e.g. dev)")
    p_run.add_argument("--limit", type=int, default=None, help="Limit number of notes")

    # Milestone M3 commands
    p_build_codes = subparsers.add_parser(
        "build-codes",
        help="Compute confidence scores S1-S5 and ground truth labels (Milestone M3)",
    )
    p_build_codes.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_build_codes.add_argument("--out", default=None, help="Output run directory")
    p_build_codes.add_argument("--model", required=False, help="Model name")

    # Milestone M4 / M5 commands
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Compute metrics, thresholds, controls, bootstrap, breakdown, and hypotheses (Milestones M4/M5)",
    )
    p_analyze.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_analyze.add_argument("--out", default=None, help="Output run directory")

    # Milestone M6 commands
    p_tables = subparsers.add_parser(
        "tables",
        help="Generate paper Tables II through IX and IV(c) (Milestone M6)",
    )
    p_tables.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_tables.add_argument("--out", default=None, help="Output run directory")

    p_figures = subparsers.add_parser(
        "figures",
        help="Generate paper Figures 1 and 2 (Milestone M6)",
    )
    p_figures.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_figures.add_argument("--out", default=None, help="Output run directory")

    p_sanity = subparsers.add_parser(
        "sanity",
        help="Run verification and sanity checks on real run outputs (Milestone M6)",
    )
    p_sanity.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_sanity.add_argument("--out", default=None, help="Output run directory")
    p_sanity.add_argument("--stage-counts-only", action="store_true", help="Check stage counts only")

    p_paper_numbers = subparsers.add_parser(
        "paper-numbers",
        help="Generate paper_numbers.md summary of in-text values (Milestone M6)",
    )
    p_paper_numbers.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_paper_numbers.add_argument("--out", default=None, help="Output run directory")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point dispatching to pipeline stage stubs.

    Args:
        argv: Command-line arguments. If None, sys.argv[1:] is used.

    Returns:
        Exit code integer.

    Raises:
        NotImplementedError: For all stage commands until implemented in their respective milestones.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    stubs: dict[str, str] = {
        "make-corpus": "Milestone M1",
        "split": "Milestone M1",
        "selfcheck": "Milestone M2",
        "estimate": "Milestone M2",
        "run": "Milestone M2",
        "build-codes": "Milestone M3",
        "analyze": "Milestone M4",
        "tables": "Milestone M6",
        "figures": "Milestone M6",
        "sanity": "Milestone M6",
        "paper-numbers": "Milestone M6",
    }

    if args.command in stubs:
        raise NotImplementedError(stubs[args.command])

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

