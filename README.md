# Confidence-Based Abstention for LLM ICD-10 Coding

A reproducible reference implementation of the research paper **"Confidence-Based Abstention for LLM ICD-10 Coding"** (paper C1).

---

## 1. Overview

Clinical large language models (LLMs) deployed for automated ICD-10-CM diagnosis coding can generate non-existent codes (*invalid*) or codes unsupported by patient clinical documentation (*unsupported* / hallucinations). This project evaluates whether computing post-generation confidence signals for each predicted code allows an automated system to withhold low-confidence predictions, improving overall precision and lowering the Unsupported Code Rate (UCR) while controlling the Invalid Code Rate (ICR).

Four open-weight models are evaluated across a synthetic clinical-note corpus (keys from `configs/c1.yaml`):
1. `qwen` — **Qwen2.5-7B-Instruct** (`Qwen/Qwen2.5-7B-Instruct`)
2. `biomistral` — **BioMistral-7B** (`BioMistral/BioMistral-7B`, context limit 2,048 tokens)
3. `med42` — **Llama3-Med42-8B** (`m42-health/Llama3-Med42-8B`)
4. `gemma` — **Gemma-2-9B-it** (`google/gemma-2-9b-it`, gated; computes in bfloat16)

The paper draft names LLaMA-3-8B-Instruct and BioMistral-7B; the model set was changed by the author (see `ASSUMPTIONS.md` A27). `llama3` stays in the config but is not run.

## 2. Table I: Confidence Signal Definitions and Decision Rule

| Signal | Name | Type | Definition / Mathematical Formulation | Cost per Note | Range |
|---|---|---|---|---|---|
| **S1** | Sampling Agreement | Frequency | $S1(c) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(c \in Y^{(k)})$, where $Y^{(k)}$ are sampled outputs generated at temperature $T=0.7$, $p=0.95$ ($K=10$). | $K$ sampled sequences | $[0.0, 1.0]$ |
| **S2** | Verbalised Probability | Calibrated self-assessment | Model self-reports probability $(0\text{--}100\%)$ for each predicted code in greedy output via a single follow-up prompt. Parsed as $\frac{\text{score}}{100.0}$. Unreadable scores default to $0.0$. | 1 follow-up call | $[0.0, 1.0]$ |
| **S3** | Token Probability | Logit probability | Mean token probability across newly generated tokens spelling code $c$: $S3(c) = \frac{1}{\|T_c\|} \sum_{t \in T_c} P(t \mid x, t_{<t})$, aligned via character offsets. | 0 (derived from greedy logits) | $[0.0, 1.0]$ |
| **S4** | Verification Prompt Logit | Targeted binary inquiry | Normalised logit probability of top `Yes` vs `No` token when asked whether the patient has condition $c$: $S4(c) = \frac{P(\text{"Yes"})}{P(\text{"Yes"}) + P(\text{"No"})}$. Evaluated with batch size 1. | $\|C_{\text{greedy}}\|$ calls | $[0.0, 1.0]$ |
| **S5** | Ensemble Mean | Hybrid ensemble | Arithmetic mean of orthogonal signals S1 and S4: $S5(c) = \frac{S1(c) + S4(c)}{2}$. | Cost(S1) + Cost(S4) | $[0.0, 1.0]$ |

### Abstention Decision Rule
For confidence signal $S \in \{S1, S2, S3, S4, S5\}$ and operating threshold $\tau$, predicted code $c$ is:
- **Kept** if $S(c) \ge \tau$
- **Withheld** (abstained) if $S(c) < \tau$

Threshold $\tau$ is chosen strictly on the **dev split** ($30\%$ of notes) to achieve target coverage levels $\in \{100\%, 90\%, 80\%, 70\%, 60\%\}$. The primary target is **$80\%$ coverage**. Chosen thresholds are frozen in `thresholds.json` before evaluating the **test split** ($70\%$ of notes).

---

## 3. Repository Architecture

```
c1-abstention-icd10/
├── AGENTS.md                  # Project constraints and development rules
├── SPEC.md                    # Exact technical specification (single source of truth)
├── PROMPTS_AND_RUNBOOK.md     # Milestone prompts and Colab runbook
├── COVERAGE.md                # Paper coverage checklist
├── ASSUMPTIONS.md             # Pinned resolutions of paper ambiguities (SPEC §17)
├── SCOPE_AUDIT.md             # Complete file and function scope audit
├── configs/
│   └── c1.yaml                # Master configuration file
├── data/
│   └── vocab/                 # CDC 2024 ICD-10-CM pinned vocabulary (committed)
├── notebooks/
│   ├── 01_run_model.ipynb     # GPU notebook for model generation stages
│   └── 02_analysis.ipynb      # CPU notebook for analysis, tables, and figures
├── c1/                        # Core Python library
│   ├── backend.py             # HFBackend (NF4 4-bit) and MockBackend
│   ├── bootstrap.py           # Note-level vectorised bootstrap (B=1000)
│   ├── breakdown.py           # Feature strata breakdown and confident mistakes
│   ├── cli.py                 # Unified CLI interface
│   ├── codes.py               # Code normalisation and CDC vocabulary wrapper
│   ├── controls.py            # Random and oracle withholding controls
│   ├── cost.py                # Computational token/second cost accounting
│   ├── data.py                # Corpus loader, schema validation, dev/test split
│   ├── figures.py             # Matplotlib Figure 1 and Figure 2 generator
│   ├── hypotheses.py          # Pre-registered hypotheses (H1–H5) evaluator
│   ├── label.py               # Ground truth labeling (correct, invalid, unsupported)
│   ├── metrics.py             # AUROC, ECE, precision, recall, F1, UCR, ICR, review load
│   ├── parse.py               # Output parsers for greedy and verbalised responses
│   ├── prompts.py             # Base, verbalised (S2), and verification (S4) prompts
│   ├── sanity.py              # 6-check runtime verification suite
│   ├── signals.py             # S1–S5 confidence calculation and k-sensitivity
│   ├── stages.py              # Resumable stage orchestrators (greedy, samples, s2, s4)
│   ├── tables.py              # Tables II–IX and paper_numbers.md formatter
│   ├── thresholds.py          # Dev-set threshold calibration (pick_tau)
│   └── utils.py               # Atomic I/O, hashing, seed control, resumable JSONL
├── tests/                     # Comprehensive test suite (125+ CPU unit tests)
└── vendor/                    # Vendored synthetic note generator & vocabulary
```

---

## 4. Hardware & Environment Requirements

### Google Colab Pro (GPU Environment)
- **Runtime:** CUDA GPU with $\ge 12\text{ GB}$ VRAM (T4, L4, or A100).
- **Disk Persistence:** Google Drive mounted at `/content/drive`.
- **Secrets (Colab Secrets):**
  - `HF_TOKEN`: Hugging Face User Access Token. Accept the licence on the `google/gemma-2-9b-it` model page (and on any other model page that asks) before running that model.
- **Dependencies:** Specified in `requirements-colab.txt`. Torch and CUDA libraries come pre-installed in Colab.

### Local Development (CPU Environment)
- **Runtime:** Python $\ge 3.10$ on any standard CPU laptop (Windows, macOS, Linux).
- **Dependencies:** Specified in `requirements-dev.txt`.
- **Note on PyTorch / Transformers:** Large packages (`torch`, `transformers`, `bitsandbytes`) are imported **lazily** inside `HFBackend`. The entire local test suite runs on CPU with `MockBackend` without requiring CUDA or heavy GPU libraries.

---

## 5. Google Colab Pro Execution Runbook

GPU execution strictly follows `PROMPTS_AND_RUNBOOK.md` Part 3 using thin notebooks that invoke the CLI.

### Step 1: Run `notebooks/01_run_model.ipynb` once per model (GPU)
1. Open `notebooks/01_run_model.ipynb` in Colab and set the runtime to **GPU** (T4 or higher).
2. In **Cell 2** set `MODEL` to one of `"qwen"`, `"biomistral"`, `"med42"`, `"gemma"`.
3. Run **Cells 1–5**: GPU check, Drive mount (`RUN = "/content/drive/MyDrive/c1_runs/c1_run02"`), clone + install, HF token, corpus + split.
4. Run **Cell 6** (`selfcheck`, `estimate`): greedy determinism must pass, the S4 Yes/No rate should be near 100%, and the projected hours must fit your budget.
5. Run **Cell 7** (smoke run, 20 notes into `{RUN}_smoke`).
6. Run **Cell 8** (full run). It is resumable: if Colab disconnects, re-run Cells 1–4 and 8.
7. Run **Cell 9** (note-count check).
8. **Runtime → Disconnect and delete runtime**, then repeat from step 1 with the next model.

Model weights are cached on Colab's local disk (not Drive): Drive corrupts files of this size.

### Step 2: Run `notebooks/02_analysis.ipynb` (CPU is enough)
One cell: `build-codes --model all` (every model that has been run), `analyze`, `sanity`, `tables`, `figures`, `paper-numbers`.

All paper outputs are written directly to `/content/drive/MyDrive/c1_runs/c1_run02/reports/`.

---

## 6. Local Development & Mock Execution

### Running the Unit Test Suite
The complete test suite runs on CPU in under 30 seconds:
```bash
pytest
```

### Running an End-to-End Mock Pipeline
You can test the entire pipeline locally from end to end using `MockBackend`:
```bash
# 1. Split corpus
python -m c1.cli split --config configs/c1.yaml --out outputs_mock/c1_run01

# 2. Run mock generation for both models
python -m c1.cli run --config configs/c1.yaml --backend mock --model llama3 --out outputs_mock/c1_run01
python -m c1.cli run --config configs/c1.yaml --backend mock --model biomistral --out outputs_mock/c1_run01

# 3. Build confidence signals S1-S5 and ground truth labels
python -m c1.cli build-codes --config configs/c1.yaml --backend mock --model llama3 --out outputs_mock/c1_run01
python -m c1.cli build-codes --config configs/c1.yaml --backend mock --model biomistral --out outputs_mock/c1_run01

# 4. Run metrics, dev thresholds, controls, and bootstrap analysis
python -m c1.cli analyze --config configs/c1.yaml --out outputs_mock/c1_run01

# 5. Execute 6-point runtime sanity check
python -m c1.cli sanity --config configs/c1.yaml --allow-mock --out outputs_mock/c1_run01

# 6. Generate all tables, figures, and paper numbers with MOCK DATA stamp
python -m c1.cli tables --config configs/c1.yaml --allow-mock --out outputs_mock/c1_run01
python -m c1.cli figures --config configs/c1.yaml --allow-mock --out outputs_mock/c1_run01
python -m c1.cli paper-numbers --config configs/c1.yaml --allow-mock --out outputs_mock/c1_run01
```

Generated mock reports appear in `outputs_mock/c1_run01/reports/` with a prominent `MOCK DATA` banner. In production mode (without `--allow-mock`), `tables`, `figures`, and `sanity` strictly refuse mock data.

---

## 7. Output Artifacts Directory Layout

Upon completion of the pipeline, `outputs/<run_id>/` contains:

| File | Description | Traces to |
|---|---|---|
| `greedy.jsonl` | Greedy decoded predictions, token IDs, logits, and seconds | SPEC §4 |
| `samples.jsonl` | $K=10$ stochastic decoding samples per note ($T=0.7, p=0.95$) | SPEC §4 |
| `s2.jsonl` | Raw verbalised probability responses and parse status | SPEC §4 |
| `s4.jsonl` | Yes/No verification logit probabilities per candidate code | SPEC §4 |
| `codes.csv` | Code-level table with ground truth labels and S1–S5 scores | SPEC §4, §5.3, §6 |
| `cost_log.jsonl` | Per-stage prompt/generation token counts and elapsed seconds | SPEC §4, §12 |
| `thresholds.json` | Calibrated operating thresholds $\tau$ per signal from dev split | SPEC §4, §8 |
| `metrics_signals.csv` | AUROC and ECE across signals for dev and test splits | SPEC §4, §7 |
| `results_test.csv` | Kept-set metrics on test split across target coverages | SPEC §4, §7 |
| `results_controls.csv` | Random withholding and empirical oracle controls | SPEC §4, §9 |
| `bootstrap.csv` | Note-level paired bootstrap differences ($\Delta\text{P}, \Delta\text{UCR}, \Delta\text{ICR}$) and 95% CIs | SPEC §4, §10 |
| `by_feature.csv` | Stratified performance by note features (rare, mult, dist, impl) | SPEC §4, §11 |
| `confident_mistakes.csv` | High-confidence erroneous predictions ($S \ge \tau_{80}$) with cause tags | SPEC §4, §11 |
| `cost.csv` | Aggregated cost accounting table (time and tokens per stage) | SPEC §4, §12 |
| `k_sens.csv` | S1 AUROC sensitivity across $k \in \{3, 5, 10\}$ prefixes | SPEC §4, §12 |
| `hypotheses.csv` | Empirical evaluation status for hypotheses H1 through H5 | SPEC §4, §13 |
| `recommendation.csv` | Formal 3-condition verdict and signal recommendation | SPEC §4, §10 |
| `sanity_report.json` | Complete audit log of 6 runtime verification checks | SPEC §4, §15 |
| `run_meta.json` | Seeds, vocabulary SHA-256, and pinned library versions | SPEC §4, §15 |

### Generated Reports (`reports/`)
- `table_II.md` / `table_II.csv`: Corpus and prediction statistics
- `table_III.md` / `table_III.csv`: Discrimination (AUROC) and calibration (ECE) across signals
- `table_IVa_llama3.md` / `table_IVa_llama3.csv`: Risk-coverage progression for LLaMA-3-8B-Instruct
- `table_IVb_biomistral.md` / `table_IVb_biomistral.csv`: Risk-coverage progression for BioMistral-7B
- `table_IV_<model>.md` / `.csv`: the same table for every other model (`qwen`, `med42`, `gemma`)
- `table_IVc.md` / `table_IVc.csv`: Headline summary (Full list vs ~80% target)
- `table_V.md` / `table_V.csv`: Statistical verdict and signal recommendation
- `table_VI.md` / `table_VI.csv`: Feature strata breakdown and reconciliation
- `table_VII.md` / `table_VII.csv`: Computational cost accounting
- `table_VIII.md` / `table_VIII.csv`: $k$-sensitivity analysis on dev split
- `table_IX.md` / `table_IX.csv`: Pre-registered hypotheses (H1–H5) evaluation summary
- `fig1_risk_coverage.png`: Publication Figure 1 (risk-coverage curves)
- `fig2_reliability.png`: Publication Figure 2 (reliability diagrams)
- `paper_numbers.md`: In-text values, seeds, hardware, truncation, and error rates

---

## 8. Reproducibility & Truthfulness Guarantees

In accordance with `AGENTS.md`:
1. **Zero Fabrication:** No placeholder values or simulated results. A table cell that cannot be computed renders as an em-dash (`—`).
2. **Resumable Long Stages:** All generation stages skip previously completed notes on restart. JSONL files write with immediate `flush()` and `os.fsync()`.
3. **Atomic File Writes:** CSV and JSON files write to temporary staging files before atomic rename, preventing corruption from unexpected interruptions.
4. **Mock Data Separation:** Mock outputs carry `"backend": "mock"` and write exclusively to `outputs_mock/`. Production table and figure generators refuse mock data unless explicitly overridden with `--allow-mock`.
5. **Deterministic Randomness:** All stochastic processes (split permutation, sampling decoding seeds, random controls, bootstrap resamples) flow deterministically from configuration seeds.
