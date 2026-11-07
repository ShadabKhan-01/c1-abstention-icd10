"""Runtime sanity check engine per SPEC §15.

Implements:
- Output files completeness check
- Disjoint dev/test split check (dev ∩ test = ∅, 30/70 split)
- Note count parity across stages (every note has greedy/samples/s2/s4)
- codes.csv identities (correct + invalid + unsupported = N_pred, mutual exclusivity, precision + UCR + ICR == 1)
- Single-valued prompts_hash per model
- Mock data mixing guard (refuses outputs_mock and "backend": "mock" in production mode)
- Generates sanity_report.json

Traces to SPEC §4, §15, §18.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from c1.utils import Config, load_config, read_jsonl_resume, write_json_atomic

logger = logging.getLogger(__name__)

REQUIRED_STAGE_FILES = [
    "greedy.jsonl",
    "samples.jsonl",
    "s2.jsonl",
    "s4.jsonl",
    "codes.csv",
]

REQUIRED_ANALYSIS_FILES = [
    "metrics_signals.csv",
    "thresholds.json",
    "results_test.csv",
    "results_controls.csv",
    "bootstrap.csv",
    "by_feature.csv",
    "confident_mistakes.csv",
    "cost.csv",
    "k_sens.csv",
    "hypotheses.csv",
    "recommendation.csv",
]


def check_file_completeness(
    out_dir: Path,
    stage_counts_only: bool = False,
) -> dict[str, Any]:
    """Check required output files exist and are non-empty."""
    # codes.csv is built later (notebook 02), so the post-generation check skips it.
    expected = [f for f in REQUIRED_STAGE_FILES if not (stage_counts_only and f == "codes.csv")]
    if not stage_counts_only:
        expected.extend(REQUIRED_ANALYSIS_FILES)

    missing = []
    empty = []
    for fn in expected:
        p = out_dir / fn
        if not p.is_file():
            missing.append(fn)
        elif p.stat().st_size == 0:
            empty.append(fn)

    passed = len(missing) == 0 and len(empty) == 0
    details = f"Verified {len(expected)} required files."
    if missing:
        details += f" Missing: {missing}."
    if empty:
        details += f" Empty: {empty}."

    return {
        "passed": passed,
        "details": details,
        "missing_files": missing,
        "empty_files": empty,
    }


def check_mock_mixing_guard(
    out_dir: Path,
    allow_mock: bool = False,
) -> dict[str, Any]:
    """Check that mock backend rows and outputs_mock are absent in production mode."""
    if allow_mock:
        return {
            "passed": True,
            "details": "Mock data permitted via allow_mock flag.",
            "is_mock": True,
        }

    # Check directory path name
    is_mock_path = "outputs_mock" in str(out_dir.resolve()).replace("\\", "/")
    if is_mock_path:
        return {
            "passed": False,
            "details": "Refusing production run: output directory is under outputs_mock.",
            "is_mock": True,
        }

    # Inspect greedy.jsonl for mock backend records
    greedy_file = out_dir / "greedy.jsonl"
    if greedy_file.is_file():
        records = read_jsonl_resume(greedy_file)
        mock_records = [r for r in records if r.get("backend") == "mock"]
        if mock_records:
            return {
                "passed": False,
                "details": f"Refusing production run: found {len(mock_records)} records stamped with backend='mock'.",
                "is_mock": True,
            }

    return {
        "passed": True,
        "details": "Mixing guard passed: no mock backend rows or paths detected.",
        "is_mock": False,
    }


def check_split_integrity(
    split_file: Path,
    expected_dev_fraction: float = 0.30,
) -> dict[str, Any]:
    """Check that split is disjoint and matches dev fraction ±1 note."""
    if not split_file.is_file():
        return {
            "passed": False,
            "details": f"Split file {split_file} not found.",
        }

    df = pd.read_csv(split_file)
    if "id" not in df.columns or "split" not in df.columns:
        return {
            "passed": False,
            "details": f"split.csv missing required columns: {list(df.columns)}",
        }

    dev_ids = set(df[df["split"] == "dev"]["id"])
    test_ids = set(df[df["split"] == "test"]["id"])

    overlap = dev_ids.intersection(test_ids)
    if overlap:
        return {
            "passed": False,
            "details": f"Split is not disjoint: {len(overlap)} overlapping notes found.",
        }

    total_notes = len(df)
    expected_dev = round(total_notes * expected_dev_fraction)
    actual_dev = len(dev_ids)

    # Disjoint and 30/70 ±1 note per SPEC §15
    fraction_ok = abs(actual_dev - expected_dev) <= 1
    details = f"Total notes: {total_notes}, dev: {actual_dev} (expected {expected_dev}), test: {len(test_ids)}."
    if not fraction_ok:
        details += f" Dev note count deviates from {expected_dev_fraction*100:.0f}% target by > 1 note."

    return {
        "passed": fraction_ok,
        "details": details,
        "dev_count": actual_dev,
        "test_count": len(test_ids),
        "total_count": total_notes,
    }


def check_stage_parity(
    out_dir: Path,
    models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Check that every note has greedy, samples, s2, and s4 records per model."""
    greedy_file = out_dir / "greedy.jsonl"
    samples_file = out_dir / "samples.jsonl"
    s2_file = out_dir / "s2.jsonl"
    s4_file = out_dir / "s4.jsonl"

    for f in (greedy_file, samples_file, s2_file, s4_file):
        if not f.is_file():
            return {
                "passed": False,
                "details": f"Stage file {f.name} missing for parity check.",
            }

    greedy_recs = read_jsonl_resume(greedy_file)
    samples_recs = read_jsonl_resume(samples_file)
    s2_recs = read_jsonl_resume(s2_file)
    s4_recs = read_jsonl_resume(s4_file)

    detected_models = sorted(list({r.get("model") for r in greedy_recs if "model" in r}))
    target_models = models if models else detected_models

    discrepancies = []
    for m in target_models:
        g_ids = {r["id"] for r in greedy_recs if r.get("model") == m and "id" in r}
        sm_ids = {r["id"] for r in samples_recs if r.get("model") == m and "id" in r}
        s2_ids = {r["id"] for r in s2_recs if r.get("model") == m and "id" in r}
        s4_ids = {r["id"] for r in s4_recs if r.get("model") == m and "id" in r}

        if not (g_ids == sm_ids == s2_ids == s4_ids):
            discrepancies.append(
                f"Model {m}: greedy={len(g_ids)}, samples={len(sm_ids)}, s2={len(s2_ids)}, s4={len(s4_ids)}"
            )

    passed = len(discrepancies) == 0
    details = f"Stage note counts equal across stages for models: {target_models}." if passed else "; ".join(discrepancies)

    return {
        "passed": passed,
        "details": details,
        "models": target_models,
    }


def check_code_identities(codes_file: Path) -> dict[str, Any]:
    """Verify ground truth labelling identities on codes.csv per SPEC §5.3, §7, §15."""
    if not codes_file.is_file():
        return {
            "passed": False,
            "details": f"{codes_file} not found.",
        }

    df = pd.read_csv(codes_file)
    n_rows = len(df)
    if n_rows == 0:
        return {
            "passed": False,
            "details": "codes.csv is empty.",
        }

    # 1. Exactly one of correct, invalid, unsupported is true per row
    is_corr = df["correct"].astype(bool)
    is_inv = df["invalid"].astype(bool)
    is_unsup = df["unsupported"].astype(bool)

    sum_labels = is_corr.astype(int) + is_inv.astype(int) + is_unsup.astype(int)
    non_one = int(np.sum(sum_labels != 1)) if hasattr(np, "sum") else int((sum_labels != 1).sum())

    if non_one > 0:
        return {
            "passed": False,
            "details": f"{non_one} rows violate mutual exclusivity (sum != 1).",
        }

    # 2. Total sums equal number of rows
    total_corr = int(is_corr.sum())
    total_inv = int(is_inv.sum())
    total_unsup = int(is_unsup.sum())

    if total_corr + total_inv + total_unsup != n_rows:
        return {
            "passed": False,
            "details": f"Counts ({total_corr}+{total_inv}+{total_unsup}) do not sum to total codes ({n_rows}).",
        }

    # 3. Precision + UCR + ICR == 1.0 on test set per model
    identity_violations = []
    models = sorted(df["model"].unique())
    for m in models:
        test_m = df[(df["model"] == m) & (df["split"] == "test")]
        n_test = len(test_m)
        if n_test > 0:
            p = test_m["correct"].sum() / n_test
            u = test_m["unsupported"].sum() / n_test
            i = test_m["invalid"].sum() / n_test
            if abs(p + u + i - 1.0) > 1e-5:
                identity_violations.append(f"Model {m}: precision+ucr+icr={p+u+i:.6f} != 1.0")

    passed = len(identity_violations) == 0
    details = f"Verified {n_rows} codes: correct={total_corr}, unsupported={total_unsup}, invalid={total_inv}."
    if identity_violations:
        details += f" Violations: {'; '.join(identity_violations)}"

    return {
        "passed": passed,
        "details": details,
        "total_codes": n_rows,
        "correct_count": total_corr,
        "unsupported_count": total_unsup,
        "invalid_count": total_inv,
    }


def check_prompts_hash_consistency(greedy_file: Path) -> dict[str, Any]:
    """Verify prompts_hash is single-valued per model and consistent across models."""
    if not greedy_file.is_file():
        return {
            "passed": False,
            "details": f"{greedy_file} not found.",
        }

    records = read_jsonl_resume(greedy_file)
    hashes_by_model: dict[str, set[str]] = {}
    for r in records:
        m = r.get("model")
        h = r.get("prompts_hash")
        if m and h:
            hashes_by_model.setdefault(m, set()).add(h)

    inconsistencies = []
    all_hashes = set()
    for m, hashes in hashes_by_model.items():
        if len(hashes) > 1:
            inconsistencies.append(f"Model {m} mixed prompt hashes: {hashes}")
        all_hashes.update(hashes)

    if len(all_hashes) > 1:
        inconsistencies.append(f"Prompt hash mismatch across models: {hashes_by_model}")

    passed = len(inconsistencies) == 0
    single_hash = list(all_hashes)[0] if len(all_hashes) == 1 else "mixed"
    details = f"Consistent prompt hash ({single_hash[:8]}...) across {len(hashes_by_model)} models." if passed else "; ".join(inconsistencies)

    return {
        "passed": passed,
        "details": details,
        "hashes_by_model": {m: list(h) for m, h in hashes_by_model.items()},
    }


def run_sanity_checks(
    out_dir: str | Path,
    config: Config | None = None,
    allow_mock: bool = False,
    stage_counts_only: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Run full runtime sanity checks per SPEC §15 and save sanity_report.json.

    Args:
        out_dir: Pipeline output run directory.
        config: Optional Config instance.
        allow_mock: Whether to tolerate mock backend outputs.
        stage_counts_only: If True, only checks stage note count parity.

    Returns:
        tuple (passed: bool, report_dict: dict).
    """
    out_p = Path(out_dir)
    split_file = out_p / "split.csv"
    if not split_file.is_file() and config:
        split_file = Path(config.paths.data_dir) / "split.csv"

    greedy_file = out_p / "greedy.jsonl"
    codes_file = out_p / "codes.csv"

    checks: dict[str, dict[str, Any]] = {}

    # 1. File Completeness
    checks["files_completeness"] = check_file_completeness(out_p, stage_counts_only=stage_counts_only)

    # 2. Mock Mixing Guard
    checks["mock_mixing_guard"] = check_mock_mixing_guard(out_p, allow_mock=allow_mock)

    # 3. Stage Parity
    models = list(config.models.keys()) if config else None
    checks["stage_parity"] = check_stage_parity(out_p, models=models)

    if not stage_counts_only:
        # 4. Split Integrity
        dev_frac = config.split.dev_fraction if config else 0.30
        checks["split_integrity"] = check_split_integrity(split_file, expected_dev_fraction=dev_frac)

        # 5. Codes Identities
        checks["code_identities"] = check_code_identities(codes_file)

        # 6. Prompts Hash Consistency
        checks["prompts_hash"] = check_prompts_hash_consistency(greedy_file)

    all_passed = all(c["passed"] for c in checks.values())
    failures = [name for name, c in checks.items() if not c["passed"]]

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_p),
        "passed": all_passed,
        "allow_mock": allow_mock,
        "stage_counts_only": stage_counts_only,
        "failures": failures,
        "checks": checks,
    }

    report_path = out_p / "sanity_report.json"
    write_json_atomic(report_path, report)
    logger.info("Saved sanity check report to %s (passed=%s)", report_path, all_passed)

    return all_passed, report
