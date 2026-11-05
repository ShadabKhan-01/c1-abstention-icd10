# AGENTS.md — Project rules (read before EVERY task)

**Project:** a reproducible implementation of the paper *"Confidence-Based Abstention for LLM ICD-10 Coding"* (paper C1). It runs two open-weight models (LLaMA-3-8B-Instruct, BioMistral-7B) on a synthetic clinical-note corpus, computes five confidence signals (S1–S5) for every predicted ICD-10-CM code, and measures whether withholding low-confidence codes improves precision and lowers the Unsupported Code Rate versus random withholding.

**Single source of truth:** `SPEC.md` in the repo root. This file only sets behaviour rules.

---

## 1. Scope rules (the most important section)

1. **Build only what traces to a paper element.** Every module, function and output file must appear in `COVERAGE.md` with the paper table / figure / hypothesis / metric / step it serves. Anything without a row is deleted before a milestone is declared done.
2. **Nothing may be missing.** Every row of the coverage matrix in `SPEC.md §18` must be implemented, tested, and produce its output file. A milestone is not done while a row assigned to it is missing.
3. **Do not invent.** If `SPEC.md` is silent or ambiguous, use the default listed in `SPEC.md §17`. If the question is not listed there, **stop and ask the user**, and write it to `QUESTIONS.md`. Never silently choose.
4. **Forbidden (not part of the paper — do not write, suggest, or "improve" with):**
   - extra confidence signals or variants (semantic entropy, verbalised-probability calibration, temperature scaling, isotonic/Platt rescaling, learned classifiers, ensembles beyond S5)
   - fine-tuning or training of any kind; prompt-optimisation loops; hyper-parameter searches other than the dev-set threshold choice and the k-sensitivity in SPEC §12
   - additional models, API/cloud backends (OpenAI, Gemini, etc.), RAG, knowledge graphs, few-shot arms, hierarchical/ancestor matching (the paper lists it as future work), MIMIC-IV code (out of scope), procedure codes
   - web apps, servers, REST APIs, dashboards, databases, Docker, CI pipelines, packaging for PyPI, plugin systems, abstract-factory frameworks
   - metrics the paper does not define (e.g. macro-F1, P@k, hierarchical F1) — only those in SPEC §7
   - generating or "demo" data that looks like results
5. Prefer small, flat, readable modules. No speculative generality.

## 2. Truthfulness rules

1. **Never fabricate or estimate results.** No placeholder numbers. A table cell that cannot be computed is `—`.
2. The only hard-coded numbers allowed in tests are the **constructed worked examples** in `SPEC.md §15` (they are test fixtures, not results).
3. Mock backend output is stamped `"backend":"mock"`, written only under `outputs_mock/`, and can never be read by `tables`/`figures`.
4. Every number in a generated table must be traceable to a file in `outputs/` (record file hashes in `run_meta.json`).

## 3. Execution environment rules

1. **You write the code; the user runs GPU jobs in Google Colab (Pro).** Do not try to download model weights, install CUDA/bitsandbytes locally, or run the HF models on the local machine. Local work = unit tests on CPU with `MockBackend`.
2. Everything that needs the GPU must be reachable through the CLI (`python -m c1.cli …`) so a thin Colab notebook can call it. **No logic inside notebooks.**
3. Colab sessions disconnect. All long stages must be **resumable** (skip finished notes), write append-only JSONL with flush + fsync after every note, tolerate a truncated last line on restart, and take their output path from config (it will point to Google Drive).
4. Do not pin `torch`/`numpy` in requirements (Colab ships its own). Pin only the libraries listed in SPEC §16.
5. **Import `torch`, `transformers` and `bitsandbytes` lazily, only inside `HFBackend`** (inside its methods or `load()`), so the whole test suite runs on a laptop where those packages are not installed.

## 4. Engineering rules

- Python ≥3.10, type hints, docstrings on public functions, `logging` not `print` (except CLI summaries).
- No magic constants: everything configurable is in `configs/c1.yaml` (defaults in SPEC §2).
- Determinism: all randomness flows from config seeds; record seeds and library versions in `run_meta.json`.
- Atomic writes (temp file + rename) for JSON/CSV; schema validation on read (pydantic or dataclass checks).
- Tests: `pytest`, CPU only, no downloads. A milestone's tests must pass before moving on.
- Git: one commit per milestone with a clear message. Do not push, rewrite history, or change git config unless asked. `.gitignore` must cover `data/` **except `data/vocab/`** (public CDC data, must be committed so Colab can clone it), `outputs/`, `outputs_mock/`, `.env`, caches, notebooks' output cells.

## 5. Secrets

HF token and any GitHub token come from Colab Secrets / environment variables only. Never print, log, commit, or write them to files.

## 6. How to report after each milestone

A short message: files added/changed (each with its COVERAGE.md row), tests run and results, deviations from SPEC (should be none), open questions. No marketing language, no interpretation of results.
