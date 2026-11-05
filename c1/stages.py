"""Resumable generation stages, cost logging, selfcheck, and runtime estimation.

Implements the four generation stages:
1. run greedy   -> greedy.jsonl
2. run samples  -> samples.jsonl
3. run s2       -> s2.jsonl
4. run s4       -> s4.jsonl
along with cost_log.jsonl tracking, selfcheck diagnostics, and estimate benchmarking.

Traces to SPEC §3, §4, §5.1, §5.2, §6, §15, §16.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from c1.backend import (
    BaseBackend,
    align_tokens_to_chars,
    compute_s3_score,
    compute_s4_score,
)
from c1.codes import Vocabulary
from c1.data import Note, load_corpus, load_split
from c1.label import label_code
from c1.parse import parse_greedy_output, parse_s2_output
from c1.prompts import (
    DEFAULT_MAX_NOTE_WORDS,
    PROMPTS_HASH,
    format_base_prompt,
    format_s2_prompt,
    format_s4_prompt,
    truncate_note,
)
from c1.signals import (
    compute_s1,
    compute_s3_from_greedy,
    compute_s5,
    extract_s2,
    extract_s4,
)
from c1.utils import (
    Config,
    append_jsonl,
    read_jsonl_resume,
    write_csv_atomic,
    write_json_atomic,
    write_run_meta,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Stage Output Path Resolution
# ---------------------------------------------------------------------------


def get_stage_output_dir(
    config: Config,
    backend: BaseBackend,
    out_override: str | Path | None = None,
) -> Path:
    """Resolve output directory, routing mock runs to outputs_mock/ per AGENTS.md §2."""
    if out_override is not None:
        out_p = Path(out_override)
    else:
        base_dir = "outputs_mock" if backend.backend_name == "mock" else config.paths.out_dir
        out_p = Path(base_dir) / config.run_id

    out_p.mkdir(parents=True, exist_ok=True)
    return out_p


def get_completed_note_ids(file_path: Path, model_name: str) -> set[str]:
    """Retrieve already completed note IDs for a model from a resumable stage JSONL."""
    if not file_path.is_file():
        return set()
    records = read_jsonl_resume(file_path)
    return {
        str(r["id"])
        for r in records
        if r.get("model") == model_name and "id" in r
    }


def append_cost_record(
    cost_file: Path,
    model_name: str,
    note_id: str,
    stage: str,
    logical_calls: int,
    prompt_tokens: int,
    gen_tokens: int,
    seconds: float,
) -> None:
    """Append a cost log entry for a note in a pipeline stage (SPEC §4)."""
    record = {
        "model": model_name,
        "id": note_id,
        "stage": stage,
        "logical_calls": logical_calls,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "seconds": round(seconds, 4),
    }
    append_jsonl(cost_file, record)


# ---------------------------------------------------------------------------
# Stage 1: Greedy Generation (SPEC §4)
# ---------------------------------------------------------------------------


def run_greedy_stage(
    config: Config,
    model_name: str,
    backend: BaseBackend,
    notes: Sequence[Note],
    out_dir: Path,
) -> Path:
    """Run greedy decoding stage producing greedy.jsonl.

    Schema per SPEC §4:
    id, model, raw_output, pred (ordered, deduped, normalised), parse_failed,
    token_ids, token_probs, code_char_spans, prompt_tokens, gen_tokens, seconds,
    note_truncated, prompts_hash, template_mode, backend
    """
    out_file = out_dir / "greedy.jsonl"
    cost_file = out_dir / "cost_log.jsonl"
    completed = get_completed_note_ids(out_file, model_name)

    logger.info("Running greedy stage for %s (%d notes total, %d completed)", model_name, len(notes), len(completed))

    for note in notes:
        if note.id in completed:
            continue

        truncated_text, note_truncated = truncate_note(
            note.text,
            max_words=DEFAULT_MAX_NOTE_WORDS,
        )
        base_prompt = format_base_prompt(truncated_text)
        _, template_mode = backend.format_chat(base_prompt)

        gen_out = backend.generate_greedy(
            base_prompt,
            max_new_tokens=config.decoding.greedy.max_new_tokens,
        )

        pred, code_char_spans, parse_failed = parse_greedy_output(gen_out.text)

        record = {
            "id": note.id,
            "model": model_name,
            "raw_output": gen_out.text,
            "pred": pred,
            "parse_failed": parse_failed,
            "token_ids": gen_out.new_token_ids,
            "token_probs": [round(p, 6) for p in gen_out.token_probs],
            "code_char_spans": code_char_spans,
            "prompt_tokens": gen_out.prompt_tokens,
            "gen_tokens": gen_out.gen_tokens,
            "seconds": round(gen_out.seconds, 4),
            "note_truncated": note_truncated,
            "prompts_hash": PROMPTS_HASH,
            "template_mode": template_mode,
            "backend": backend.backend_name,
        }
        append_jsonl(out_file, record)

        append_cost_record(
            cost_file=cost_file,
            model_name=model_name,
            note_id=note.id,
            stage="greedy",
            logical_calls=1,
            prompt_tokens=gen_out.prompt_tokens,
            gen_tokens=gen_out.gen_tokens,
            seconds=gen_out.seconds,
        )

    return out_file


# ---------------------------------------------------------------------------
# Stage 2: Samples Generation (S1, SPEC §4)
# ---------------------------------------------------------------------------


def run_samples_stage(
    config: Config,
    model_name: str,
    backend: BaseBackend,
    notes: Sequence[Note],
    out_dir: Path,
) -> Path:
    """Run sampled generation stage producing samples.jsonl.

    Schema per SPEC §4:
    id, model, mode, k, runs:[{run, seed, raw_output, codes}], prompt_tokens, gen_tokens, seconds
    """
    out_file = out_dir / "samples.jsonl"
    cost_file = out_dir / "cost_log.jsonl"
    completed = get_completed_note_ids(out_file, model_name)

    sample_cfg = config.decoding.sample
    k = sample_cfg.k
    mode = sample_cfg.mode

    logger.info("Running samples stage for %s (%d notes total, %d completed)", model_name, len(notes), len(completed))

    for note in notes:
        if note.id in completed:
            continue

        truncated_text, _ = truncate_note(
            note.text,
            max_words=DEFAULT_MAX_NOTE_WORDS,
        )
        base_prompt = format_base_prompt(truncated_text)

        # Deterministic note-specific seed
        note_seed_hash = hashlib.sha256(f"{sample_cfg.seed}_{note.id}".encode()).hexdigest()
        note_seed = int(note_seed_hash[:8], 16) % (2**31 - 1)

        gen_outs = backend.generate_samples(
            prompt=base_prompt,
            k=k,
            temperature=sample_cfg.temperature,
            top_p=sample_cfg.top_p,
            seed=note_seed,
            mode=mode,
            max_new_tokens=sample_cfg.max_new_tokens,
        )

        runs: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_gen_tokens = 0
        total_seconds = 0.0

        for i, out in enumerate(gen_outs):
            codes, _, _ = parse_greedy_output(out.text)
            run_seed = note_seed if mode == "batched" else (note_seed + i + 1)
            runs.append(
                {
                    "run": i + 1,
                    "seed": run_seed,
                    "raw_output": out.text,
                    "codes": codes,
                }
            )
            total_prompt_tokens += out.prompt_tokens
            total_gen_tokens += out.gen_tokens
            total_seconds += out.seconds

        record = {
            "id": note.id,
            "model": model_name,
            "mode": mode,
            "k": k,
            "runs": runs,
            "prompt_tokens": total_prompt_tokens,
            "gen_tokens": total_gen_tokens,
            "seconds": round(total_seconds, 4),
        }
        append_jsonl(out_file, record)

        append_cost_record(
            cost_file=cost_file,
            model_name=model_name,
            note_id=note.id,
            stage="samples",
            logical_calls=k,
            prompt_tokens=total_prompt_tokens,
            gen_tokens=total_gen_tokens,
            seconds=total_seconds,
        )

    return out_file


# ---------------------------------------------------------------------------
# Stage 3: S2 Verbalized Scoring (SPEC §4, §5.2)
# ---------------------------------------------------------------------------


def run_s2_stage(
    config: Config,
    model_name: str,
    backend: BaseBackend,
    notes: Sequence[Note],
    out_dir: Path,
) -> Path:
    """Run S2 verbalized scoring stage producing s2.jsonl.

    Schema per SPEC §4:
    id, model, raw_output, scores:{code:value|null}, failed_codes, prompt_tokens, gen_tokens, seconds
    """
    greedy_file = out_dir / "greedy.jsonl"
    if not greedy_file.is_file():
        raise FileNotFoundError(
            f"greedy.jsonl not found at {greedy_file}. Run greedy stage first."
        )

    greedy_records = read_jsonl_resume(greedy_file)
    greedy_map = {
        r["id"]: r.get("pred", [])
        for r in greedy_records
        if r.get("model") == model_name and "id" in r
    }

    out_file = out_dir / "s2.jsonl"
    cost_file = out_dir / "cost_log.jsonl"
    completed = get_completed_note_ids(out_file, model_name)

    logger.info("Running S2 stage for %s (%d notes total, %d completed)", model_name, len(notes), len(completed))

    for note in notes:
        if note.id in completed:
            continue

        greedy_codes = greedy_map.get(note.id, [])

        if not greedy_codes:
            # If no codes predicted in greedy pass, S2 output is trivially empty
            record = {
                "id": note.id,
                "model": model_name,
                "raw_output": "",
                "scores": {},
                "failed_codes": [],
                "prompt_tokens": 0,
                "gen_tokens": 0,
                "seconds": 0.0,
            }
            append_jsonl(out_file, record)
            append_cost_record(
                cost_file=cost_file,
                model_name=model_name,
                note_id=note.id,
                stage="s2",
                logical_calls=0,
                prompt_tokens=0,
                gen_tokens=0,
                seconds=0.0,
            )
            continue

        truncated_text, _ = truncate_note(
            note.text,
            max_words=DEFAULT_MAX_NOTE_WORDS,
        )
        s2_prompt = format_s2_prompt(truncated_text, greedy_codes)

        gen_out = backend.generate_text(
            prompt=s2_prompt,
            max_new_tokens=config.s2.max_new_tokens,
        )

        scores, failed_codes = parse_s2_output(gen_out.text, greedy_codes)

        record = {
            "id": note.id,
            "model": model_name,
            "raw_output": gen_out.text,
            "scores": scores,
            "failed_codes": failed_codes,
            "prompt_tokens": gen_out.prompt_tokens,
            "gen_tokens": gen_out.gen_tokens,
            "seconds": round(gen_out.seconds, 4),
        }
        append_jsonl(out_file, record)

        append_cost_record(
            cost_file=cost_file,
            model_name=model_name,
            note_id=note.id,
            stage="s2",
            logical_calls=1,
            prompt_tokens=gen_out.prompt_tokens,
            gen_tokens=gen_out.gen_tokens,
            seconds=gen_out.seconds,
        )

    return out_file


# ---------------------------------------------------------------------------
# Stage 4: S4 Yes/No Verification (SPEC §4, §5.1, §6)
# ---------------------------------------------------------------------------


def run_s4_stage(
    config: Config,
    model_name: str,
    backend: BaseBackend,
    vocab: Vocabulary,
    notes: Sequence[Note],
    out_dir: Path,
) -> Path:
    """Run S4 verification stage producing s4.jsonl.

    Schema per SPEC §4:
    id, model, items:[{code, p_yes, p_no, s4, top_is_yes_or_no}], prompt_tokens, seconds
    """
    greedy_file = out_dir / "greedy.jsonl"
    if not greedy_file.is_file():
        raise FileNotFoundError(
            f"greedy.jsonl not found at {greedy_file}. Run greedy stage first."
        )

    greedy_records = read_jsonl_resume(greedy_file)
    greedy_map = {
        r["id"]: r.get("pred", [])
        for r in greedy_records
        if r.get("model") == model_name and "id" in r
    }

    out_file = out_dir / "s4.jsonl"
    cost_file = out_dir / "cost_log.jsonl"
    completed = get_completed_note_ids(out_file, model_name)

    yes_tid, no_tid = backend.get_yes_no_token_ids()

    logger.info("Running S4 stage for %s (%d notes total, %d completed)", model_name, len(notes), len(completed))

    for note in notes:
        if note.id in completed:
            continue

        greedy_codes = greedy_map.get(note.id, [])

        if not greedy_codes:
            record = {
                "id": note.id,
                "model": model_name,
                "items": [],
                "prompt_tokens": 0,
                "seconds": 0.0,
            }
            append_jsonl(out_file, record)
            append_cost_record(
                cost_file=cost_file,
                model_name=model_name,
                note_id=note.id,
                stage="s4",
                logical_calls=0,
                prompt_tokens=0,
                gen_tokens=0,
                seconds=0.0,
            )
            continue

        truncated_text, _ = truncate_note(
            note.text,
            max_words=DEFAULT_MAX_NOTE_WORDS,
        )

        items: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_seconds = 0.0

        for code in greedy_codes:
            desc = vocab.description(code)
            s4_prompt = format_s4_prompt(truncated_text, code, desc)

            prob_dict, p_tokens, secs, argmax_tid = backend.next_token_probs(
                prompt=s4_prompt,
                token_ids=[yes_tid, no_tid],
            )

            p_yes = prob_dict.get(yes_tid, 0.0)
            p_no = prob_dict.get(no_tid, 0.0)
            s4_val = compute_s4_score(p_yes, p_no)
            top_is_yes_no = argmax_tid in (yes_tid, no_tid)

            items.append(
                {
                    "code": code,
                    "p_yes": round(p_yes, 6),
                    "p_no": round(p_no, 6),
                    "s4": round(s4_val, 6),
                    "top_is_yes_or_no": top_is_yes_no,
                }
            )
            total_prompt_tokens += p_tokens
            total_seconds += secs

        record = {
            "id": note.id,
            "model": model_name,
            "items": items,
            "prompt_tokens": total_prompt_tokens,
            "seconds": round(total_seconds, 4),
        }
        append_jsonl(out_file, record)

        append_cost_record(
            cost_file=cost_file,
            model_name=model_name,
            note_id=note.id,
            stage="s4",
            logical_calls=len(greedy_codes),
            prompt_tokens=total_prompt_tokens,
            gen_tokens=0,
            seconds=total_seconds,
        )

    return out_file


# ---------------------------------------------------------------------------
# Diagnostics: Selfcheck (SPEC §3, §15)
# ---------------------------------------------------------------------------


def run_selfcheck(
    config: Config,
    model_name: str,
    backend: BaseBackend,
    vocab: Vocabulary,
    notes: Sequence[Note],
    out_dir: Path,
) -> dict[str, Any]:
    """Run diagnostic checks: GPU/VRAM, determinism, S3 alignment, S4 Yes/No check.

    Writes selfcheck_report.json and returns report dict.
    """
    logger.info("Running selfcheck for %s", model_name)
    report: dict[str, Any] = {
        "model": model_name,
        "backend": backend.backend_name,
        "timestamp": time.time(),
    }

    # 1. Environment & Hardware Info
    import platform
    env_info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch
        env_info["torch"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            env_info["vram_gb"] = round(total_mem, 2)
    except ImportError:
        env_info["torch"] = "not_installed"
        env_info["cuda_available"] = False

    report["environment"] = env_info

    # Filter dev notes
    dev_notes = [n for n in notes]  # callers typically provide dev notes

    # 2. Greedy Determinism (3 dev notes)
    det_checks: list[dict[str, Any]] = []
    det_success = True
    for note in dev_notes[:3]:
        truncated_text, _ = truncate_note(note.text, DEFAULT_MAX_NOTE_WORDS)
        prompt = format_base_prompt(truncated_text)

        out1 = backend.generate_greedy(prompt)
        out2 = backend.generate_greedy(prompt)

        identical = (out1.text == out2.text) and (out1.new_token_ids == out2.new_token_ids)
        if not identical:
            det_success = False

        det_checks.append({
            "note_id": note.id,
            "identical": identical,
            "text_match": out1.text == out2.text,
            "token_ids_match": out1.new_token_ids == out2.new_token_ids,
        })
    report["greedy_determinism"] = {
        "passed": det_success,
        "checks": det_checks,
    }

    # 3. S3 Token Alignment (5 dev notes)
    s3_alignments: list[dict[str, Any]] = []
    for note in dev_notes[:5]:
        truncated_text, _ = truncate_note(note.text, DEFAULT_MAX_NOTE_WORDS)
        prompt = format_base_prompt(truncated_text)
        out = backend.generate_greedy(prompt)
        pred, char_spans, _ = parse_greedy_output(out.text)

        token_spans = align_tokens_to_chars(out.new_token_ids, backend.decode_tokens)
        note_codes_info = []

        for code_span in char_spans:
            c_start = code_span["start"]
            c_end = code_span["end"]
            code_str = code_span["code"]

            matched_tokens = []
            for tid, (t_start, t_end), prob in zip(out.new_token_ids, token_spans, out.token_probs):
                if t_start < t_end and max(c_start, t_start) < min(c_end, t_end):
                    token_piece = backend.decode_tokens([tid])
                    matched_tokens.append({
                        "token_id": tid,
                        "token_piece": token_piece,
                        "prob": round(prob, 4),
                        "token_span": [t_start, t_end],
                    })

            s3_val = compute_s3_score((c_start, c_end), token_spans, out.token_probs)
            note_codes_info.append({
                "code": code_str,
                "code_span": [c_start, c_end],
                "matched_tokens": matched_tokens,
                "s3_score": round(s3_val, 4),
            })

        s3_alignments.append({
            "note_id": note.id,
            "pred_codes": pred,
            "alignments": note_codes_info,
        })
    report["s3_alignment"] = s3_alignments

    # 4. S4 Yes/No Token Check (10 dev note/code pairs)
    yes_tid, no_tid = backend.get_yes_no_token_ids()
    s4_checks: list[dict[str, Any]] = []
    top_yes_no_count = 0
    total_pairs = 0

    for note in dev_notes:
        truncated_text, _ = truncate_note(note.text, DEFAULT_MAX_NOTE_WORDS)
        for gold_item in note.gold:
            code = gold_item.code
            desc = vocab.description(code)
            s4_prompt = format_s4_prompt(truncated_text, code, desc)

            prob_dict, _, _, argmax_tid = backend.next_token_probs(
                s4_prompt,
                [yes_tid, no_tid],
            )
            is_top = argmax_tid in (yes_tid, no_tid)
            if is_top:
                top_yes_no_count += 1
            total_pairs += 1

            s4_checks.append({
                "note_id": note.id,
                "code": code,
                "p_yes": round(prob_dict.get(yes_tid, 0.0), 4),
                "p_no": round(prob_dict.get(no_tid, 0.0), 4),
                "argmax_token_id": argmax_tid,
                "top_is_yes_or_no": is_top,
            })
            if total_pairs >= 10:
                break
        if total_pairs >= 10:
            break

    report["s4_yes_no_check"] = {
        "yes_token_id": yes_tid,
        "no_token_id": no_tid,
        "tested_pairs": total_pairs,
        "top_is_yes_or_no_count": top_yes_no_count,
        "pass_rate": round(top_yes_no_count / max(total_pairs, 1), 3),
        "details": s4_checks,
    }

    report_path = out_dir / "selfcheck_report.json"
    write_json_atomic(report_path, report)
    logger.info("Wrote selfcheck report to %s", report_path)
    return report


# ---------------------------------------------------------------------------
# Time Estimation: Estimate (SPEC §16)
# ---------------------------------------------------------------------------


def run_estimate(
    config: Config,
    model_name: str,
    backend: BaseBackend,
    vocab: Vocabulary,
    dev_notes: Sequence[Note],
    sample_size: int = 20,
) -> dict[str, Any]:
    """Benchmark 20 dev notes through all four stages and project full corpus hours.

    Does not fabricate numbers; measures actual elapsed times on dev notes.
    """
    notes_to_run = list(dev_notes[:sample_size])
    n_sample = len(notes_to_run)
    if n_sample == 0:
        raise ValueError("Cannot run estimate: no notes provided.")

    logger.info("Running estimate benchmark on %d notes for model %s", n_sample, model_name)

    t_greedy = 0.0
    t_samples = 0.0
    t_s2 = 0.0
    t_s4 = 0.0

    yes_tid, no_tid = backend.get_yes_no_token_ids()

    for note in notes_to_run:
        truncated_text, _ = truncate_note(note.text, DEFAULT_MAX_NOTE_WORDS)

        # 1. Greedy
        base_prompt = format_base_prompt(truncated_text)
        out_g = backend.generate_greedy(base_prompt, max_new_tokens=config.decoding.greedy.max_new_tokens)
        t_greedy += out_g.seconds
        pred, _, _ = parse_greedy_output(out_g.text)

        # 2. Samples
        out_s = backend.generate_samples(
            prompt=base_prompt,
            k=config.decoding.sample.k,
            temperature=config.decoding.sample.temperature,
            top_p=config.decoding.sample.top_p,
            seed=config.decoding.sample.seed,
            mode=config.decoding.sample.mode,
            max_new_tokens=config.decoding.sample.max_new_tokens,
        )
        t_samples += sum(s.seconds for s in out_s)

        # 3. S2
        if pred:
            s2_prompt = format_s2_prompt(truncated_text, pred)
            out_s2 = backend.generate_text(s2_prompt, max_new_tokens=config.s2.max_new_tokens)
            t_s2 += out_s2.seconds

        # 4. S4
        if pred:
            for code in pred:
                desc = vocab.description(code)
                s4_prompt = format_s4_prompt(truncated_text, code, desc)
                _, _, secs, _ = backend.next_token_probs(s4_prompt, [yes_tid, no_tid])
                t_s4 += secs

    t_total = t_greedy + t_samples + t_s2 + t_s4
    sec_per_note = t_total / n_sample
    target_notes = config.corpus.n_notes
    projected_seconds = sec_per_note * target_notes
    projected_hours = projected_seconds / 3600.0

    res = {
        "model": model_name,
        "sample_size": n_sample,
        "target_notes": target_notes,
        "t_greedy_sec": round(t_greedy, 2),
        "t_samples_sec": round(t_samples, 2),
        "t_s2_sec": round(t_s2, 2),
        "t_s4_sec": round(t_s4, 2),
        "t_total_sec": round(t_total, 2),
        "sec_per_note": round(sec_per_note, 2),
        "projected_hours": round(projected_hours, 2),
    }
    return res


# ---------------------------------------------------------------------------
# Stage 5: Build Codes (codes.csv, SPEC §4)
# ---------------------------------------------------------------------------

CODES_CSV_COLUMNS: list[str] = [
    "model",
    "id",
    "split",
    "code",
    "position",
    "correct",
    "invalid",
    "unsupported",
    "cause_tag",
    "gold_type",
    "gold_rare",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s2_failed",
]


def build_codes_stage(
    config: Config,
    model_name: str,
    out_dir: Path,
    vocab: Vocabulary,
    notes: Sequence[Note],
    split_map: dict[str, str],
    decode_fn: Callable[[Sequence[int]], str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Join stage outputs, ground truth labels, and confidence signals into codes.csv.

    Schema per SPEC §4:
    model, id, split, code, position, correct, invalid, unsupported, cause_tag,
    gold_type, gold_rare, s1, s2, s3, s4, s5, s2_failed

    Also records s3_post execution time in cost_log.jsonl (SPEC §4, §12).
    """
    greedy_file = out_dir / "greedy.jsonl"
    samples_file = out_dir / "samples.jsonl"
    s2_file = out_dir / "s2.jsonl"
    s4_file = out_dir / "s4.jsonl"
    cost_file = out_dir / "cost_log.jsonl"
    codes_file = out_dir / "codes.csv"

    for req_file in (greedy_file, samples_file, s2_file, s4_file):
        if not req_file.is_file():
            raise FileNotFoundError(
                f"Required stage file {req_file.name} not found in {out_dir}. "
                "Run generation stages first."
            )

    greedy_records = read_jsonl_resume(greedy_file)
    samples_records = read_jsonl_resume(samples_file)
    s2_records = read_jsonl_resume(s2_file)
    s4_records = read_jsonl_resume(s4_file)

    greedy_map = {r["id"]: r for r in greedy_records if r.get("model") == model_name and "id" in r}
    samples_map = {r["id"]: r for r in samples_records if r.get("model") == model_name and "id" in r}
    s2_map = {r["id"]: r for r in s2_records if r.get("model") == model_name and "id" in r}
    s4_map = {r["id"]: r for r in s4_records if r.get("model") == model_name and "id" in r}

    logger.info("Building codes.csv for %s across %d notes", model_name, len(notes))

    # Preserve rows for other models if codes.csv already exists
    existing_other_rows: list[dict[str, Any]] = []
    if codes_file.is_file():
        with open(codes_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model") != model_name:
                    existing_other_rows.append(row)

    new_rows: list[dict[str, Any]] = []
    total_s3_time = 0.0

    # Process notes sorted by ID (SPEC §4)
    sorted_notes = sorted(notes, key=lambda n: n.id)

    for note in sorted_notes:
        if note.id not in greedy_map:
            continue

        greedy_rec = greedy_map[note.id]
        samples_rec = samples_map.get(note.id)
        s2_rec = s2_map.get(note.id)
        s4_rec = s4_map.get(note.id)

        pred_codes = greedy_rec.get("pred", [])
        note_split = split_map.get(note.id, "dev")

        sample_runs = samples_rec.get("runs", []) if samples_rec else []

        start_s3_post = time.perf_counter()

        for position, code in enumerate(pred_codes, start=1):
            lbl = label_code(code, note, vocab)
            s1_val = compute_s1(sample_runs, code)
            s2_val, s2_failed = extract_s2(s2_rec, code)
            s3_val = compute_s3_from_greedy(greedy_rec, code, decode_fn)
            s4_val = extract_s4(s4_rec, code)
            s5_val = compute_s5(s1_val, s4_val)

            new_rows.append(
                {
                    "model": model_name,
                    "id": note.id,
                    "split": note_split,
                    "code": code,
                    "position": position,
                    "correct": lbl.correct,
                    "invalid": lbl.invalid,
                    "unsupported": lbl.unsupported,
                    "cause_tag": lbl.cause_tag,
                    "gold_type": lbl.gold_type,
                    "gold_rare": lbl.gold_rare,
                    "s1": round(s1_val, 6),
                    "s2": round(s2_val, 6),
                    "s3": round(s3_val, 6),
                    "s4": round(s4_val, 6),
                    "s5": round(s5_val, 6),
                    "s2_failed": s2_failed,
                }
            )

        elapsed_s3_post = time.perf_counter() - start_s3_post
        total_s3_time += elapsed_s3_post

        append_cost_record(
            cost_file=cost_file,
            model_name=model_name,
            note_id=note.id,
            stage="s3_post",
            logical_calls=0,
            prompt_tokens=0,
            gen_tokens=0,
            seconds=elapsed_s3_post,
        )

    all_rows = existing_other_rows + new_rows
    # Sort deterministically by (model, id, position)
    all_rows.sort(key=lambda r: (r["model"], r["id"], int(r["position"])))

    write_csv_atomic(codes_file, all_rows, fieldnames=CODES_CSV_COLUMNS)

    n_codes = len(new_rows)
    stats: dict[str, Any] = {
        "model": model_name,
        "n_notes_processed": len([n for n in sorted_notes if n.id in greedy_map]),
        "total_predicted_codes": n_codes,
        "correct_count": sum(1 for r in new_rows if r["correct"]),
        "invalid_count": sum(1 for r in new_rows if r["invalid"]),
        "unsupported_count": sum(1 for r in new_rows if r["unsupported"]),
        "s2_failed_count": sum(1 for r in new_rows if r["s2_failed"]),
        "s2_failure_rate": round(sum(1 for r in new_rows if r["s2_failed"]) / max(n_codes, 1), 4),
        "mean_s1": round(sum(r["s1"] for r in new_rows) / max(n_codes, 1), 4) if n_codes else 0.0,
        "mean_s2": round(sum(r["s2"] for r in new_rows) / max(n_codes, 1), 4) if n_codes else 0.0,
        "mean_s3": round(sum(r["s3"] for r in new_rows) / max(n_codes, 1), 4) if n_codes else 0.0,
        "mean_s4": round(sum(r["s4"] for r in new_rows) / max(n_codes, 1), 4) if n_codes else 0.0,
        "mean_s5": round(sum(r["s5"] for r in new_rows) / max(n_codes, 1), 4) if n_codes else 0.0,
        "total_s3_post_seconds": round(total_s3_time, 4),
    }

    logger.info("Successfully built codes.csv at %s (%d rows)", codes_file, len(all_rows))
    return codes_file, stats


