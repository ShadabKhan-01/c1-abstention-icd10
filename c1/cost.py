"""Cost accounting module per SPEC §12 (Table VII).

Implements:
- Aggregates cost_log.jsonl to compute per-model, per-signal mean per note:
  - extra logical calls (S1=k, S2=1, S3=0, S4=mean(N_pred), S5=S1+S4)
  - physical generate calls
  - extra tokens (prompt, gen, total)
  - extra seconds (wall-clock)
- Reports baseline greedy pass metrics and GPU device name.
- Emits cost.csv adhering to Table VII specifications.

Traces to SPEC §4, §12, §14 (Table VII), §18.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

COST_COLUMNS = [
    "model",
    "item",
    "extra_logical_calls",
    "physical_generate_calls",
    "extra_prompt_tokens",
    "extra_gen_tokens",
    "extra_total_tokens",
    "extra_seconds",
    "gpu_name",
]


def detect_gpu_name(run_meta_file: Path | None = None) -> str:
    """Detect GPU name from run_meta.json or system environment."""
    if run_meta_file and run_meta_file.is_file():
        try:
            with open(run_meta_file, encoding="utf-8") as f:
                meta = json.load(f)
                gpu = meta.get("gpu_name")
                if gpu:
                    return str(gpu)
        except Exception:
            pass

    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass

    return "CPU / MockGPU"


def compute_cost_table(
    cost_log_file: Path,
    models: Sequence[str],
    k_samples: int = 10,
    gpu_name: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate cost_log.jsonl into Table VII rows per model and signal.

    Args:
        cost_log_file: Path to cost_log.jsonl.
        models: List of model names.
        k_samples: Number of sample lists k for S1 (default 10).
        gpu_name: Optional GPU name string.

    Returns:
        List of dict rows formatted for cost.csv.
    """
    if gpu_name is None:
        gpu_name = detect_gpu_name(cost_log_file.parent / "run_meta.json")

    # Load cost log records
    records_by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in models}
    if cost_log_file.is_file():
        with open(cost_log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    m = rec.get("model")
                    if m in records_by_model:
                        records_by_model[m].append(rec)
                except Exception:
                    continue

    cost_rows: list[dict[str, Any]] = []

    for m in models:
        m_recs = records_by_model[m]
        # Identify unique notes for this model
        unique_note_ids = {r["id"] for r in m_recs if "id" in r}
        n_notes = max(len(unique_note_ids), 1)

        # Aggregate totals per stage
        stage_totals: dict[str, dict[str, float]] = {
            "greedy": {"logical_calls": 0.0, "prompt_tokens": 0.0, "gen_tokens": 0.0, "seconds": 0.0},
            "samples": {"logical_calls": 0.0, "prompt_tokens": 0.0, "gen_tokens": 0.0, "seconds": 0.0},
            "s2": {"logical_calls": 0.0, "prompt_tokens": 0.0, "gen_tokens": 0.0, "seconds": 0.0},
            "s4": {"logical_calls": 0.0, "prompt_tokens": 0.0, "gen_tokens": 0.0, "seconds": 0.0},
            "s3_post": {"logical_calls": 0.0, "prompt_tokens": 0.0, "gen_tokens": 0.0, "seconds": 0.0},
        }

        for r in m_recs:
            stg = r.get("stage")
            if stg in stage_totals:
                stage_totals[stg]["logical_calls"] += float(r.get("logical_calls", 0))
                stage_totals[stg]["prompt_tokens"] += float(r.get("prompt_tokens", 0))
                stage_totals[stg]["gen_tokens"] += float(r.get("gen_tokens", 0))
                stage_totals[stg]["seconds"] += float(r.get("seconds", 0.0))

        # Calculate means per note
        means: dict[str, dict[str, float]] = {}
        for stg, totals in stage_totals.items():
            means[stg] = {k: v / n_notes for k, v in totals.items()}

        # 1. Greedy baseline row
        g_prompt = means["greedy"]["prompt_tokens"]
        g_gen = means["greedy"]["gen_tokens"]
        g_sec = means["greedy"]["seconds"]
        cost_rows.append({
            "model": m,
            "item": "greedy",
            "extra_logical_calls": 0.0,
            "physical_generate_calls": 1.0,
            "extra_prompt_tokens": round(g_prompt, 2),
            "extra_gen_tokens": round(g_gen, 2),
            "extra_total_tokens": round(g_prompt + g_gen, 2),
            "extra_seconds": round(g_sec, 4),
            "gpu_name": gpu_name,
        })

        # 2. S1 (sampling): extra logical calls = k, 1 physical batched generate call
        s1_prompt = means["samples"]["prompt_tokens"]
        s1_gen = means["samples"]["gen_tokens"]
        s1_sec = means["samples"]["seconds"]
        s1_logical = float(k_samples)
        cost_rows.append({
            "model": m,
            "item": "s1",
            "extra_logical_calls": round(s1_logical, 2),
            "physical_generate_calls": 1.0,
            "extra_prompt_tokens": round(s1_prompt, 2),
            "extra_gen_tokens": round(s1_gen, 2),
            "extra_total_tokens": round(s1_prompt + s1_gen, 2),
            "extra_seconds": round(s1_sec, 4),
            "gpu_name": gpu_name,
        })

        # 3. S2 (verbalized): 1 extra logical call, 1 physical generate call
        s2_prompt = means["s2"]["prompt_tokens"]
        s2_gen = means["s2"]["gen_tokens"]
        s2_sec = means["s2"]["seconds"]
        cost_rows.append({
            "model": m,
            "item": "s2",
            "extra_logical_calls": 1.0,
            "physical_generate_calls": 1.0,
            "extra_prompt_tokens": round(s2_prompt, 2),
            "extra_gen_tokens": round(s2_gen, 2),
            "extra_total_tokens": round(s2_prompt + s2_gen, 2),
            "extra_seconds": round(s2_sec, 4),
            "gpu_name": gpu_name,
        })

        # 4. S3 (token probability): 0 extra logical calls, 0 physical calls, post-processing seconds
        s3_sec = means["s3_post"]["seconds"]
        cost_rows.append({
            "model": m,
            "item": "s3",
            "extra_logical_calls": 0.0,
            "physical_generate_calls": 0.0,
            "extra_prompt_tokens": 0.0,
            "extra_gen_tokens": 0.0,
            "extra_total_tokens": 0.0,
            "extra_seconds": round(s3_sec, 4),
            "gpu_name": gpu_name,
        })

        # 5. S4 (yes/no verification): extra logical calls = number of predicted codes, 0 physical generate calls
        s4_logical = means["s4"]["logical_calls"]
        s4_prompt = means["s4"]["prompt_tokens"]
        s4_gen = means["s4"]["gen_tokens"]
        s4_sec = means["s4"]["seconds"]
        cost_rows.append({
            "model": m,
            "item": "s4",
            "extra_logical_calls": round(s4_logical, 2),
            "physical_generate_calls": 0.0,
            "extra_prompt_tokens": round(s4_prompt, 2),
            "extra_gen_tokens": round(s4_gen, 2),
            "extra_total_tokens": round(s4_prompt + s4_gen, 2),
            "extra_seconds": round(s4_sec, 4),
            "gpu_name": gpu_name,
        })

        # 6. S5 (S1 + S4): extra calls = S1 + S4
        s5_logical = s1_logical + s4_logical
        s5_prompt = s1_prompt + s4_prompt
        s5_gen = s1_gen + s4_gen
        s5_sec = s1_sec + s4_sec
        cost_rows.append({
            "model": m,
            "item": "s5",
            "extra_logical_calls": round(s5_logical, 2),
            "physical_generate_calls": 1.0,
            "extra_prompt_tokens": round(s5_prompt, 2),
            "extra_gen_tokens": round(s5_gen, 2),
            "extra_total_tokens": round(s5_prompt + s5_gen, 2),
            "extra_seconds": round(s5_sec, 4),
            "gpu_name": gpu_name,
        })

    return cost_rows
