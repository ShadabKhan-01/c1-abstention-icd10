# Confidence-Based Abstention for LLM ICD-10 Coding

A reproducible reference implementation of the research paper **"Confidence-Based Abstention for LLM ICD-10 Coding"** (paper C1).

---

## 1. Overview

Clinical large language models (LLMs) deployed for automated ICD-10-CM diagnosis coding can generate non-existent codes (*invalid*) or codes unsupported by patient clinical documentation (*unsupported* / hallucinations). This project evaluates whether computing post-generation confidence signals for each predicted code allows an automated system to withhold low-confidence predictions, improving overall precision and lowering the Unsupported Code Rate (UCR) while controlling the Invalid Code Rate (ICR).

Two open-weight models are evaluated across a synthetic clinical-note corpus:
1. **LLaMA-3-8B-Instruct** (`meta-llama/Meta-Llama-3-8B-Instruct`, context limit 8,192 tokens)
2. **BioMistral-7B** (`BioMistral/BioMistral-7B`, context limit 2,048 tokens)

Five confidence signals (**S1–S5**) are calculated for every predicted code and evaluated against a **random-withholding control** (100 repeats) and an empirical **oracle control**.

---

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
  - `HF_TOKEN`: Hugging Face User Access Token (with accepted Meta LLaMA-3 licence).
  - `GH_TOKEN`: GitHub Personal Access Token (for cloning private repository).
- **Dependencies:** Specified in `requirements-colab.txt`. Torch and CUDA libraries come pre-installed in Colab.

### Local Development (CPU Environment)
- **Runtime:** Python $\ge 3.10$ on any standard CPU laptop (Windows, macOS, Linux).
- **Dependencies:** Specified in `requirements-dev.txt`.
- **Note on PyTorch / Transformers:** Large packages (`torch`, `transformers`, `bitsandbytes`) are imported **lazily** inside `HFBackend`. The entire local test suite runs on CPU with `MockBackend` without requiring CUDA or heavy GPU libraries.

---

## 5. Google Colab Pro Execution Runbook

GPU execution strictly follows `PROMPTS_AND_RUNBOOK.md` Part 3 using thin notebooks that invoke the CLI.

### Step 1: Execute `notebooks/01_run_model.ipynb` for LLaMA-3-8B-Instruct
1. Open Google Colab and set runtime to **GPU** (T4 or higher).
2. Open `notebooks/01_run_model.ipynb`.
3. Set `MODEL = "llama3"`.
4. Execute **Cells 1–5**:
   - Checks GPU via `nvidia-smi`.
   - Mounts Google Drive (`RUN = "/content/drive/MyDrive/c1_runs/c1_run01"`).
   - Clones repository and installs `requirements-colab.txt`.
   - Generates corpus (`c1 make-corpus`) and dev/test split (`c1 split`).
5. Execute **Cell 6 (Gate A Checks)**:
   - Runs `c1 selfcheck` and `c1 estimate`.
   - Inspect output: verify greedy determinism passes, S3 token alignment displays visible tokens spelling codes, S4 top token is predominantly `Yes`/`No`, and projected runtime fits your compute unit budget.
6. Execute **Cell 7 (Smoke Run)**:
   - Runs 20 dev notes through all stages into `{RUN}_smoke`.
   - Verifies all 5 confidence signals are populated without error.
7. Execute **Cell 8 (Full Run)**:
   - Resumable full execution across stages `greedy,samples,s2,s4`. Long stages write append-only JSONL with immediate flush and fsync.
8. Execute **Cell 9**:
   - Verifies note count parity across all generation files (`c1 sanity --stage-counts-only`).

### Step 2: Disconnect Runtime and Repeat for BioMistral-7B
1. In Colab: **Runtime → Disconnect and delete runtime** (frees GPU resources and resets RAM).
2. Change runtime to **GPU** and open `notebooks/01_run_model.ipynb`.
3. Set `MODEL = "biomistral"`.
4. Run Cells 1–4, 6–9 (skipping Cell 5 since corpus and split already exist in Drive).

### Step 3: Execute `notebooks/02_analysis.ipynb` (Analysis & Paper Reports)
1. In Colab: Set runtime to **CPU** (standard free runtime is sufficient).
2. Open `notebooks/02_analysis.ipynb` and run:
   - Computes labels and signals (`c1 build-codes`) for both models.
   - Computes metrics, dev thresholds, controls, bootstrap, and hypotheses (`c1 analyze`).
   - Runs full runtime sanity verification (`c1 sanity`).
   - Generates all markdown and CSV tables (`c1 tables`).
   - Renders publication figures (`c1 figures`).
   - Compiles in-text numbers (`c1 paper-numbers`).

All paper outputs are written directly to `/content/drive/MyDrive/c1_runs/c1_run01/reports/`.

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
