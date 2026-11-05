# PROMPTS_AND_RUNBOOK.md — How to build paper C1 with Antigravity and run it on Colab Pro

Three files go into the **root of your new repo**: `AGENTS.md` (rules), `SPEC.md` (the full specification), and this file (for you, not for the agent).

---

## Part 1 — Before you start (you do this)

**1. Create the new repo** (for example `c1-abstention-icd10`) and open its folder in Antigravity.

**2. Put the rules where the agent will read them.** Save `AGENTS.md` in the repo root. If your Antigravity version has a workspace-rules location, also place the same text there. If you are unsure whether it is read automatically, paste this at the start of every task: *"Read AGENTS.md and SPEC.md first and follow them exactly."*

**3. Use planning mode and review the plan** before letting the agent execute each milestone. Keep command approval on, so it cannot install or download things you did not expect.

**4. Gather what the agent cannot invent** (it will stop and ask if missing):
- `vendor/`: copy the **synthetic-data generator** and the **ICD-10-CM vocabulary/validity module** from your earlier repo (the generator must be able to output `gold` with `type` and `rare`, and `distractors`; see SPEC §1.1).
- The **exact base-prompt wording** of your earlier paper's Appendix B, if it differs from SPEC §5.1.
- The processed vocabulary from the earlier repo: `data/vocab/icd10cm_2024.jsonl` and `data/vocab/meta.json` (same release; keep the file names; commit them, they are public CDC data and Colab needs them).
- A **Hugging Face token**, with the **Meta LLaMA-3 licence accepted** on the model page. Store it in Colab Secrets as `HF_TOKEN`.

**5. Confirm these decisions** (the paper leaves them open; SPEC §17 has defaults, change them now or accept them):

| # | Decision | Default |
|---|---|---|
| 1 | Notes in the corpus | 300 (≈90 dev / 210 test); 200 is faster but the dev set is thin |
| 2 | Sampling for S1 | one batched call with 10 sequences (≈10× faster than 10 separate calls); alternative: 10 separate seeded calls |
| 3 | How the best signal is picked | highest **dev** AUROC |
| 4 | Table VI strata | note-level features (§11); H5 uses code-level contrast (§13) |
| 5 | Verdict condition (iii) | observed Δ Invalid Code Rate ≤ 0 |
| 6 | S4 prompt for invalid codes | neutral "description not available" |

Decide 1 and 2 **before the first real run** and never change them afterwards.

---

## Part 2 — Prompts for the Antigravity agent (give one at a time; wait for tests to pass; commit after each)

> Start every prompt with: *"Read AGENTS.md and SPEC.md. Follow them exactly. Build only what this milestone lists. Update COVERAGE.md and ASSUMPTIONS.md. At the end give the milestone report described in AGENTS.md §6."*

### M0 — Scaffold
Create the repository skeleton: package `c1/`, `tests/`, `configs/c1.yaml` (defaults from SPEC §2), `notebooks/` (empty), `vendor/` (do not touch), `requirements-colab.txt`, `requirements-dev.txt` (SPEC §16), `.gitignore`, `README.md` (stub), `COVERAGE.md` (copy SPEC §18 as a checklist), `ASSUMPTIONS.md` (copy SPEC §17), `QUESTIONS.md` (empty). Implement `c1/utils.py` only: seeding, config loading/validation, atomic JSON/CSV write, JSONL append with flush+fsync, resume-safe JSONL reader that tolerates a truncated last line, SHA-256 hashing, and `run_meta.json` writer (SPEC §4). Implement `c1/cli.py` with the command names from SPEC §4 as stubs that raise `NotImplementedError("Milestone Mx")`. Tests for utils (atomic write, truncated-line tolerance, resume). Nothing else.
**Done when:** `pytest` passes; `python -m c1.cli --help` lists every command.

### M1 — Codes, corpus, split, Table II
Implement `c1/codes.py` (normalise, well-formed regex, `Vocabulary` wrapper around `vendor/`: `is_valid`, `description`) and `c1/data.py` (corpus loader with the hard checks of SPEC §1.1, `make-corpus` wrapper that calls the vendored generator with the config seed, `corpus_stats.json`, deterministic 30/70 split by seeded permutation → `split.csv`, note-level feature flags of SPEC §11). Implement the Table II producer. If the vendored generator lacks `type`, `rare` or `distractors`: **stop and list exactly what is missing** before patching anything. Tests: SPEC §15 items for split determinism/disjointness and corpus validation; vocabulary tests with a tiny fixture.
**Done when:** a small fixture corpus validates, splits 30/70 deterministically, and Table II renders from it.

### M2 — Prompts, parsing, backend, generation stages (no scoring yet)
Implement `c1/prompts.py` (SPEC §5.1, with `prompts_hash`), `c1/parse.py` (SPEC §5.2), `c1/backend.py` (`MockBackend` and `HFBackend` per SPEC §3), and the resumable stage runners `run greedy | samples | s2 | s4` writing the files and schemas of SPEC §4, plus `cost_log.jsonl`, shared `max_note_words` truncation, `selfcheck` and `estimate` commands. Do **not** run `HFBackend` locally; test everything else with `MockBackend`. In `HFBackend`, take extra care with: raw-logit probabilities (SPEC §3.1), LLaMA-3 `<|eot_id|>` stop token, decoding only new tokens, the S4 Yes/No token ids, and batch size 1.
**Done when:** a mock run over a fixture corpus produces all four JSONL files; killing and restarting mid-run loses nothing and duplicates nothing; parser tests pass.
**→ Colab Gate A (you run it; see Part 3):** `selfcheck` and `estimate` and a 20-note smoke run for each model. Paste any error or `selfcheck_report.json` back to the agent before continuing.

### M3 — Labels and signals
Implement `c1/label.py` (SPEC §5.3), `c1/signals.py` (S1–S5, SPEC §6, with S3 span-to-token alignment) and the `build-codes` command producing `codes.csv`. Tests: labelling identity, cause tags, S3 alignment on a fake tokenizer, S4 formula including the 0/0 case, S5, S1 on 3/5/10 prefixes, S2 failure handling.
**Done when:** `build-codes` works on mock outputs and all signal tests pass.
**→ Colab Gate B:** run `build-codes` on the 20-note smoke outputs; check that S3 alignment printouts look right and that S4's top token is Yes/No.

### M4 — Metrics, thresholds, controls
Implement `c1/metrics.py` (AUROC, ECE, kept-set metrics, SPEC §7), `c1/thresholds.py` (SPEC §8; thresholds from dev only, written once), `c1/controls.py` (random and oracle, SPEC §9, vectorised), and the part of `analyze` that writes `metrics_signals.csv`, `thresholds.json`, `results_test.csv`, `results_controls.csv`. Implement the four **worked examples of SPEC §15 exactly** as tests (AUROC 0.8333, kept metrics, ECE 0.25, pick_tau), plus the shuffled-score and oracle sanity tests.
**Done when:** all of SPEC §15's metric tests pass and the 100% row equals the full list.

### M5 — Bootstrap, verdict, breakdown, cost, k-sensitivity, hypotheses
Implement `c1/bootstrap.py` (note-level, B=1000, random control recomputed inside each resample, SPEC §10), the verdict and recommendation functions, `c1/breakdown.py` (SPEC §11 including `confident_mistakes.csv`), `cost.csv` (SPEC §12), `k_sens.csv` (dev split, no new LLM calls), `c1/hypotheses.py` (SPEC §13). One unit test per branch of verdict, recommendation and hypothesis status. Keep the bootstrap vectorised (precompute per-note counts; resample by multiplicity weights).
**Done when:** `analyze` produces every file listed in SPEC §4, and bootstrap determinism tests pass.

### M6 — Tables, figures, sanity
Implement `c1/tables.py` (Tables II–IX and IV(c), SPEC §14 formats), `c1/figures.py` (Fig. 1, Fig. 2), `paper_numbers.md`, and `c1/sanity.py` (SPEC §15 runtime checks). `tables`/`figures` must refuse mock data and must refuse to run if `sanity` failed. Add `tests/test_coverage.py`.
**Done when:** on mock data every table and figure renders in `outputs_mock/reports/` with a `MOCK DATA` stamp.

### M7 — Notebooks, README, scope audit
Create `notebooks/01_run_model.ipynb` and `notebooks/02_analysis.ipynb` exactly following Part 3 of PROMPTS_AND_RUNBOOK.md (cells call only the CLI). Write the `README.md` run guide. Then write `SCOPE_AUDIT.md`: every file and function mapped to a COVERAGE.md row; **delete anything unmapped**; list any COVERAGE.md row not yet implemented (there must be none). Run the whole test suite.
**Done when:** `pytest` is green, `SCOPE_AUDIT.md` has no unmapped items, and no COVERAGE.md row is unchecked.

---

## Part 3 — Colab Pro runbook (you run this)

**General:** Runtime → Change runtime type → GPU (T4 is enough; L4 or A100 is faster if available and your compute units allow). Keep the browser tab open during long cells. Use **one session per model** (LLaMA-3 first, then BioMistral). When a model finishes, disconnect and delete the runtime to stop spending compute units. If a session dies, rerun the same cells: every stage resumes.

### Notebook `01_run_model.ipynb` (GPU) — set `MODEL = "llama3"` or `"biomistral"`

```python
# Cell 1 – GPU check
!nvidia-smi

# Cell 2 – Drive (outputs persist here)
from google.colab import drive
drive.mount('/content/drive')
RUN = "/content/drive/MyDrive/c1_runs/c1_run01"      # same path for both models

# Cell 3 – code + dependencies (private repo: put a GitHub token in Colab Secrets as GH_TOKEN)
from google.colab import userdata
gh = userdata.get("GH_TOKEN")
!git clone https://{gh}@github.com/<you>/<repo>.git /content/c1 || (cd /content/c1 && git pull)
%cd /content/c1
!pip -q install -r requirements-colab.txt

# Cell 4 – secrets and cache
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"   # optional: skips re-downloading the model next session

# Cell 5 – corpus + split (first session only; harmless to rerun)
!python -m c1.cli make-corpus --config configs/c1.yaml --out $RUN
!python -m c1.cli split --config configs/c1.yaml --out $RUN

# Cell 6 – checks (Gate A): GPU/library versions, determinism, S3 alignment, S4 Yes/No token
MODEL = "llama3"      # or "biomistral"
!python -m c1.cli selfcheck --config configs/c1.yaml --out $RUN --model $MODEL
!python -m c1.cli estimate  --config configs/c1.yaml --out $RUN --model $MODEL

# Cell 7 – smoke run (20 dev notes, separate output folder)
!python -m c1.cli run --config configs/c1.yaml --out {RUN}_smoke --model $MODEL --stages greedy,samples,s2,s4 --split dev --limit 20
!python -m c1.cli build-codes --config configs/c1.yaml --out {RUN}_smoke --model $MODEL

# Cell 8 – FULL run (only after the smoke run looks right)
!python -m c1.cli run --config configs/c1.yaml --out $RUN --model $MODEL --stages greedy,samples,s2,s4

# Cell 9 – verify completeness
!python -m c1.cli sanity --config configs/c1.yaml --out $RUN --stage-counts-only
```

**What to check at Gate A (and paste back to the agent if wrong):**
- `selfcheck_report.json`: the same greedy output twice; each printed code shows tokens that visibly spell it (S3 alignment); the top next-token for S4 is Yes or No for at least most pairs.
- `estimate` prints the projected hours for the full corpus. Compare with your remaining compute units before Cell 8. A T4 is the slowest option; if the projection is too long, lower `n_notes` (decision 1) **before** the full run, not after.
- Smoke `codes.csv` has all five score columns, no all-zero S2 column, and a sensible mix of correct / unsupported / invalid labels.

### Notebook `02_analysis.ipynb` (CPU runtime is enough)

```python
from google.colab import drive; drive.mount('/content/drive')
RUN = "/content/drive/MyDrive/c1_runs/c1_run01"
%cd /content/c1
!pip -q install -r requirements-colab.txt
for m in ["llama3", "biomistral"]:
    !python -m c1.cli build-codes --config configs/c1.yaml --out $RUN --model {m}
!python -m c1.cli analyze       --config configs/c1.yaml --out $RUN
!python -m c1.cli sanity        --config configs/c1.yaml --out $RUN
!python -m c1.cli tables        --config configs/c1.yaml --out $RUN
!python -m c1.cli figures       --config configs/c1.yaml --out $RUN
!python -m c1.cli paper-numbers --config configs/c1.yaml --out $RUN
```
All tables, figures, `thresholds.json`, `bootstrap.csv`, `hypotheses.csv`, `recommendation.csv` and `paper_numbers.md` appear in `RUN/reports/` on Drive. `analyze` freezes thresholds from dev before reading any test label; do not delete `thresholds.json` to "try again".

---

## Part 4 — Filling the paper

| Paper item | File to copy from |
|---|---|
| Tables II–IX, IV(c) | `reports/table_*.md` (same row/column order as the paper) |
| Figures 1–2 | `reports/fig1_risk_coverage.png`, `fig2_reliability.png` |
| Step-3 and other in-text numbers (seeds, split sizes, GPU, parse and S2 failure rates, truncation, confident mistakes by cause, model revisions, vocabulary release) | `reports/paper_numbers.md` |
| Intro numbers (earlier paper's Unsupported and Invalid rates) | your **earlier** results, not this repo |

Write the discussion only from the printed numbers, and report every hypothesis as the code labelled it (confirmed, rejected or inconclusive). The paper's rule is that failures are reported, not hidden.

---

## Part 5 — If something goes wrong

| Symptom | Likely cause | What to do |
|---|---|---|
| 401/403 downloading LLaMA-3 | licence not accepted or token missing | accept licence on the model page; check `HF_TOKEN` secret |
| CUDA out of memory | other models still loaded, or a long prompt | restart runtime; one model per session; check `note_truncated` counts |
| `bitsandbytes` import error | Colab image mismatch | rerun Cell 3; if persistent paste the full traceback to the agent |
| Session disconnected mid-run | normal | rerun Cells 1–4 and Cell 8; it resumes |
| S4 top token is not Yes/No | tokenizer prefix-space variant | paste `selfcheck_report.json` to the agent; it may adjust the S4 token handling or prompt wording **on dev notes only, before any test run** |
| Very low AUROC for S3 | alignment bug | check the printed code/token pairs in the selfcheck |
