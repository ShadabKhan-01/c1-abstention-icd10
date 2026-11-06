"""Command-line interface for the C1 project.

Provides CLI entry points for all pipeline stages.
Milestone M1: make-corpus, split, tables (Table II)
Milestone M2: run (greedy, samples, s2, s4), selfcheck, estimate
Milestone M3: build-codes
Milestone M4: analyze
Milestone M6: tables, figures, sanity, paper-numbers

Traces to SPEC §3, §4, §15, §16.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

logger = logging.getLogger(__name__)


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
    p_selfcheck.add_argument("--model", required=False, default="llama3", help="Model to check (llama3 or biomistral)")
    p_selfcheck.add_argument("--backend", choices=["hf", "mock"], default="hf", help="Backend type (default: hf)")

    p_estimate = subparsers.add_parser(
        "estimate",
        help="Estimate runtime and projected hours for full corpus (Milestone M2)",
    )
    p_estimate.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_estimate.add_argument("--out", default=None, help="Output run directory")
    p_estimate.add_argument("--model", required=False, default="llama3", help="Model to estimate")
    p_estimate.add_argument("--backend", choices=["hf", "mock"], default="hf", help="Backend type (default: hf)")
    p_estimate.add_argument("--sample-size", type=int, default=20, help="Number of notes to benchmark (default: 20)")

    p_run = subparsers.add_parser(
        "run",
        help="Run model generation stages: greedy, samples, s2, s4 (Milestone M2)",
    )
    p_run.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_run.add_argument("--out", default=None, help="Output run directory")
    p_run.add_argument("--model", required=False, default="llama3", help="Model name (llama3 or biomistral)")
    p_run.add_argument("--backend", choices=["hf", "mock"], default="hf", help="Backend type (default: hf)")
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
    p_build_codes.add_argument("--model", required=False, default="llama3", help="Model name (llama3 or biomistral)")
    p_build_codes.add_argument("--backend", choices=["hf", "mock"], default="hf", help="Backend type (default: hf)")

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
        help="Generate paper Tables II through IX and IV(c) (Milestone M1 for Table II, M6 for all)",
    )
    p_tables.add_argument("--config", default="configs/c1.yaml", help="Path to config")
    p_tables.add_argument("--out", default=None, help="Output run directory")
    p_tables.add_argument("--table", default="all", help="Specific table to render (e.g. II or all)")

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


def _create_backend(config: Any, model_name: str, backend_type: str) -> Any:
    """Instantiate and initialize backend."""
    if backend_type == "mock":
        from c1.backend import MockBackend
        return MockBackend(model_name=model_name)

    from c1.backend import HFBackend
    if model_name not in config.models:
        raise ValueError(f"Model '{model_name}' not configured in config.models ({list(config.models.keys())})")

    model_cfg = config.models[model_name]
    backend = HFBackend(
        model_id=model_cfg.hf_id,
        context_limit=model_cfg.context,
        quant_type=config.quantisation.type,
        compute_dtype=config.quantisation.compute_dtype,
        double_quant=config.quantisation.double_quant,
    )
    backend.load()
    return backend


def _load_notes_and_vocab(config: Any, split_filter: str | None = None, limit: int | None = None) -> tuple[Any, list[Any]]:
    """Load vocabulary and filtered notes."""
    from c1.codes import Vocabulary
    from c1.data import load_corpus, load_split

    data_dir = Path(config.paths.data_dir)
    vocab = Vocabulary.load_default()
    corpus_file = data_dir / "corpus.jsonl"
    notes = load_corpus(corpus_file, vocab)

    if split_filter:
        split_file = data_dir / "split.csv"
        if not split_file.is_file():
            split_file = Path(config.paths.out_dir) / config.run_id / "split.csv"
        if split_file.is_file():
            split_map = load_split(split_file)
            notes = [n for n in notes if split_map.get(n.id) == split_filter]
        else:
            logger.warning("Split file %s not found; skipping split filter", split_file)

    if limit is not None:
        notes = notes[:limit]

    return vocab, notes


def run_pipeline_cli(args: argparse.Namespace) -> int:
    """Handler for 'c1 run'."""
    from c1.stages import (
        get_stage_output_dir,
        run_greedy_stage,
        run_s2_stage,
        run_s4_stage,
        run_samples_stage,
    )
    from c1.utils import load_config, write_run_meta

    config = load_config(args.config)
    model_name = args.model
    backend = _create_backend(config, model_name, args.backend)

    try:
        vocab, notes = _load_notes_and_vocab(config, split_filter=args.split, limit=args.limit)
        out_dir = get_stage_output_dir(config, backend, args.out)

        stages = [s.strip().lower() for s in args.stages.split(",") if s.strip()]

        for stage in stages:
            if stage == "greedy":
                run_greedy_stage(config, model_name, backend, notes, out_dir)
            elif stage in ("samples", "sample", "s1"):
                run_samples_stage(config, model_name, backend, notes, out_dir)
            elif stage == "s2":
                run_s2_stage(config, model_name, backend, notes, out_dir)
            elif stage == "s4":
                run_s4_stage(config, model_name, backend, vocab, notes, out_dir)
            else:
                raise ValueError(f"Unknown stage '{stage}'. Must be one of greedy, samples, s2, s4.")

        write_run_meta(out_dir, config, extra={"stages": stages, "model": model_name, "backend": backend.backend_name})
        print(f"Completed stages {stages} for {model_name} in {out_dir}")
        return 0
    finally:
        if hasattr(backend, "unload"):
            backend.unload()


def run_selfcheck_cli(args: argparse.Namespace) -> int:
    """Handler for 'c1 selfcheck'."""
    import json
    from c1.stages import get_stage_output_dir, run_selfcheck
    from c1.utils import load_config

    config = load_config(args.config)
    model_name = args.model
    backend = _create_backend(config, model_name, args.backend)

    try:
        vocab, notes = _load_notes_and_vocab(config, split_filter="dev")
        out_dir = get_stage_output_dir(config, backend, args.out)
        report = run_selfcheck(config, model_name, backend, vocab, notes, out_dir)

        print("\n=== Selfcheck Summary ===")
        print(f"Model: {model_name} ({backend.backend_name})")
        print(f"Environment: {report['environment']}")
        det_pass = report['greedy_determinism']['passed']
        print(f"Greedy Determinism: {'PASSED' if det_pass else 'FAILED'}")
        s4_rate = report['s4_yes_no_check']['pass_rate'] * 100
        print(f"S4 Top-1 Yes/No Rate: {s4_rate:.1f}% ({report['s4_yes_no_check']['top_is_yes_or_no_count']}/{report['s4_yes_no_check']['tested_pairs']})")
        print(f"Detailed report saved to: {out_dir / 'selfcheck_report.json'}\n")
        return 0
    finally:
        if hasattr(backend, "unload"):
            backend.unload()


def run_estimate_cli(args: argparse.Namespace) -> int:
    """Handler for 'c1 estimate'."""
    from c1.stages import run_estimate
    from c1.utils import load_config

    config = load_config(args.config)
    model_name = args.model
    backend = _create_backend(config, model_name, args.backend)

    try:
        vocab, notes = _load_notes_and_vocab(config, split_filter="dev")
        res = run_estimate(config, model_name, backend, vocab, notes, sample_size=args.sample_size)

        print("\n=== Runtime Estimation (SPEC §16) ===")
        print(f"Model: {model_name} ({backend.backend_name})")
        print(f"Benchmark sample: {res['sample_size']} dev notes")
        print(f"Target corpus size: {res['target_notes']} notes")
        print(f"  Greedy stage:  {res['t_greedy_sec']} s")
        print(f"  Samples stage: {res['t_samples_sec']} s")
        print(f"  S2 stage:      {res['t_s2_sec']} s")
        print(f"  S4 stage:      {res['t_s4_sec']} s")
        print(f"Total time for {res['sample_size']} notes: {res['t_total_sec']} s ({res['sec_per_note']} s/note)")
        print(f"Projected total runtime for {res['target_notes']} notes: {res['projected_hours']:.2f} hours\n")
        return 0
    finally:
        if hasattr(backend, "unload"):
            backend.unload()


def run_build_codes_cli(args: argparse.Namespace) -> int:
    """Handler for 'c1 build-codes'."""
    from c1.codes import Vocabulary
    from c1.data import load_corpus, load_split
    from c1.stages import build_codes_stage, get_stage_output_dir
    from c1.utils import load_config, write_run_meta

    config = load_config(args.config)
    model_name = args.model or "llama3"
    backend = _create_backend(config, model_name, args.backend)

    try:
        data_dir = Path(config.paths.data_dir)
        vocab = Vocabulary.load_default()
        notes = load_corpus(data_dir / "corpus.jsonl", vocab)

        split_file = data_dir / "split.csv"
        if not split_file.is_file():
            split_file = Path(config.paths.out_dir) / config.run_id / "split.csv"
        split_map = load_split(split_file) if split_file.is_file() else {}

        out_dir = get_stage_output_dir(config, backend, args.out)

        decode_fn = backend.decode_tokens if hasattr(backend, "decode_tokens") else None
        codes_file, stats = build_codes_stage(
            config=config,
            model_name=model_name,
            out_dir=out_dir,
            vocab=vocab,
            notes=notes,
            split_map=split_map,
            decode_fn=decode_fn,
        )

        write_run_meta(out_dir, config, extra={"command": "build-codes", "model": model_name, "backend": backend.backend_name})

        print(f"\n=== Build Codes Summary ({model_name}) ===")
        print(f"Notes Processed: {stats['n_notes_processed']}")
        print(f"Total Predicted Codes: {stats['total_predicted_codes']}")
        print(f"  Correct:     {stats['correct_count']} ({stats['correct_count'] / max(stats['total_predicted_codes'], 1) * 100:.1f}%)")
        print(f"  Invalid:     {stats['invalid_count']} ({stats['invalid_count'] / max(stats['total_predicted_codes'], 1) * 100:.1f}%)")
        print(f"  Unsupported: {stats['unsupported_count']} ({stats['unsupported_count'] / max(stats['total_predicted_codes'], 1) * 100:.1f}%)")
        print(f"S2 Failure Rate: {stats['s2_failure_rate'] * 100:.1f}% ({stats['s2_failed_count']}/{stats['total_predicted_codes']})")
        print(f"Mean Scores: S1={stats['mean_s1']:.3f}, S2={stats['mean_s2']:.3f}, S3={stats['mean_s3']:.3f}, S4={stats['mean_s4']:.3f}, S5={stats['mean_s5']:.3f}")
        print(f"Saved: {codes_file}\n")
        return 0
    finally:
        if hasattr(backend, "unload"):
            backend.unload()


def run_analyze_cli(args: argparse.Namespace) -> int:
    """Handler for 'c1 analyze'."""
    from c1.stages import run_analyze_stage
    from c1.utils import load_config

    config = load_config(args.config)
    out_dir = Path(args.out) if args.out else Path(config.paths.out_dir) / config.run_id
    if not out_dir.is_dir():
        mock_dir = Path("outputs_mock") / config.run_id
        if mock_dir.is_dir():
            out_dir = mock_dir

    summary = run_analyze_stage(config, out_dir)

    print("\n=== Analysis Summary (Milestone M4) ===")
    print(f"Models: {', '.join(summary['models'])}")
    print(f"Saved: {summary['thresholds_file']}")
    print(f"Saved: {summary['metrics_signals_file']}")
    print(f"Saved: {summary['results_test_file']} ({summary['total_test_rows']} rows)")
    print(f"Saved: {summary['results_controls_file']} ({summary['total_control_rows']} rows)\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point dispatching to pipeline stage handlers.

    Args:
        argv: Command-line arguments. If None, sys.argv[1:] is used.

    Returns:
        Exit code integer.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "make-corpus":
        from c1.data import make_corpus_from_config
        make_corpus_from_config(args.config, args.out)
        return 0

    if args.command == "split":
        from c1.data import run_split_from_config
        run_split_from_config(args.config, args.out)
        return 0

    if args.command == "tables":
        if args.table in ("II", "all"):
            from c1.tables import run_table_ii_from_config
            md_str, _ = run_table_ii_from_config(args.config, args.out)
            print(md_str)
            return 0
        raise NotImplementedError(f"Table {args.table} is not implemented in Milestone M1.")

    if args.command == "run":
        return run_pipeline_cli(args)

    if args.command == "selfcheck":
        return run_selfcheck_cli(args)

    if args.command == "estimate":
        return run_estimate_cli(args)

    if args.command == "build-codes":
        return run_build_codes_cli(args)

    if args.command == "analyze":
        return run_analyze_cli(args)

    stubs: dict[str, str] = {
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
