# SPEC.md — Implementation specification for paper C1
**"Confidence-Based Abstention for LLM ICD-10 Coding"**

This document defines *exactly* what to build. Section numbers are referenced by `AGENTS.md`, the milestone prompts and `COVERAGE.md`.

---

## 0. What the system does (one paragraph)

For every synthetic note and each of two models, produce a **greedy code list** (the predicted codes). Give each predicted code five confidence scores in [0,1]: **S1** sampling agreement, **S2** verbalised score, **S3** token probability, **S4** yes/no verification probability, **S5** mean of S1 and S4. Label each predicted code as *correct*, *invalid* (not in ICD-10-CM) or *unsupported* (valid but not in the note's gold set). Choose a threshold per signal on a **dev split** so about 90/80/70/60% of codes are kept, apply it to the **test split**, and compare the kept codes with a **random-withholding control** and an **oracle**. Quantify uncertainty with a note-level bootstrap, apply a pre-defined verdict rule, break results down by note feature, log cost, test sensitivity to k, and evaluate hypotheses H1–H5. Emit every table (II–IX) and figure (1–2) of the paper.

The paper's *research output* is tables, figures and verdicts. The program's *per-note output* is: predicted codes + five scores + keep/withhold decisions.

---

## 1. Inputs (contract — validate on load, fail loudly)

### 1.1 Corpus (`data/corpus.jsonl`)
Produced by the **earlier paper's synthetic generator, regenerated with a new seed** (the generator is *vendored*, see 1.3). One JSON object per note:
```json
{"id": "n0001", "text": "...",
 "gold": [{"code": "E11.9", "type": "explicit|implicit", "rare": false}],
 "distractors": [{"code": "I50.9", "kind": "negated|family_history|distractor_medication"}]}
```
- `gold`: the correct codes for the note, each with its type and a rare (long-tail) flag.
- `distractors`: conditions the note *mentions but must not be coded* (negated, family history) or medications that tempt a wrong code, each with the wrong code a model might emit.
- Hard checks: ids unique; every gold code valid in the vocabulary; no distractor code is also gold for the same note; implicit gold codes not named in the text (as the generator guarantees).
- **If the vendored generator does not already emit `type`, `rare` and `distractors`, STOP and report exactly which fields are missing.** Adding them is a minimal patch to `vendor/` listed in `COVERAGE.md`; do not write a new generator.

### 1.2 Vocabulary (`data/vocab/icd10cm_2024.jsonl` + `data/vocab/meta.json`, copied unchanged from the earlier repo; keep the file names; this folder IS committed to git)
The **same ICD-10-CM vocabulary object as the earlier paper** (pinned CDC release; record its name and SHA-256 in `run_meta.json`). *Valid code* = exists in that vocabulary. Needs `description(code)` (used in the S4 prompt). Canonical code form: uppercase, dot after the 3rd character (`E119`→`E11.9`). Well-formed pattern: `^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$`.

### 1.3 Vendored earlier code (`vendor/`)
The user copies the earlier repo's generator and vocabulary modules into `vendor/`. Use them through thin wrappers (`c1/data.py`, `c1/codes.py`). Do not rewrite them. If `vendor/` is empty, stop and ask.

---

## 2. Configuration defaults (`configs/c1.yaml`)

```yaml
run_id: c1_run01
paths: {data_dir: data, out_dir: outputs}        # out_dir will point to Google Drive on Colab
corpus: {n_notes: 300, generator_seed: 2026}
split: {dev_fraction: 0.30, seed: 2026}
models:
  llama3: {hf_id: meta-llama/Meta-Llama-3-8B-Instruct, context: 8192}
  biomistral: {hf_id: BioMistral/BioMistral-7B, context: 2048}
quantisation: {type: nf4, compute_dtype: float16, double_quant: true}
decoding:
  greedy: {do_sample: false, max_new_tokens: 200}
  sample: {k: 10, temperature: 0.7, top_p: 0.95, max_new_tokens: 200, mode: batched, seed: 2026}
s2: {max_new_tokens: 200}
s4: {batch_size: 1}
coverage_targets: [1.0, 0.9, 0.8, 0.7, 0.6]
primary_target: 0.8
random_control: {repeats: 100, seed: 7}
bootstrap: {B: 1000, seed: 42, ci: 0.95}
ece_bins: 10
k_sensitivity: [3, 5, 10]
```
`n_notes` is configurable; the dev/test split is 30/70 of whatever it is.

---

## 3. Models and backend (`c1/backend.py`)

- `HFBackend`: load with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`; `device_map="auto"`; `model.eval()`. T4-compatible (no bf16 assumption).
- Methods (the only ones needed):
  1. `generate_greedy(prompt) -> GenOut(text, new_token_ids, token_probs, prompt_tokens, gen_tokens, seconds)`; `do_sample=False`; probabilities from the **raw logits** (`output_logits=True` if the installed transformers supports it, else `output_scores=True`), softmax in float32, probability of each *chosen* token.
  2. `generate_samples(prompt, k, temperature, top_p, seed, mode) -> list[GenOut]`; `mode=batched`: one `generate` with `num_return_sequences=k` after `torch.manual_seed(seed_for_note)` where `seed_for_note = f(base_seed, note_id)` (deterministic hash); `mode=per_seed_loop`: k separate calls with seeds 1..k. Record the mode in outputs.
  3. `generate_text(prompt, max_new_tokens)` (greedy, used by S2).
  4. `next_token_probs(prompt, token_ids) -> dict` (single forward pass, no generation, used by S4): returns the softmax probability of each requested first-token id at the answer position.
- Chat formatting: **single user message** (no system role) rendered with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`; if a tokenizer has no chat template use the model's documented `[INST] … [/INST]` form. Record `template_mode`.
- LLaMA-3 stop tokens: `eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]`. Decode **only new tokens**. Set `pad_token_id` if missing.
- Truncation: the note is truncated **identically for both models** by a shared `max_note_words` computed once so that the longest prompt (S2: prompt + note + code list + 200 new tokens) fits BioMistral's 2048-token window. Log the number of truncated notes per model; never truncate the instruction or code list.
- Batch size 1 for greedy/S2/S3 (determinism). Free GPU memory between models.
- `MockBackend`: deterministic fake outputs with plausible token probabilities, for CPU tests only (`"backend":"mock"`).
- `selfcheck` CLI (runs on Colab): prints GPU name/VRAM/library versions; greedy determinism (same output twice on 3 dev notes); **S3 alignment** (prints, for 5 dev notes, each code with the decoded token pieces matched to it); **S4 Yes/No check** (on 10 dev (note, code) pairs, reports whether the top-1 next token is the Yes or No token; if not for a model, report which prompt wording change is needed — dev notes only). Writes `selfcheck_report.json`.

---

## 4. Pipeline stages and output files

Order: `corpus → split → [per model] greedy → samples → s2 → s4 → build-codes → analyze → tables/figures → sanity`. All stage outputs live under `{out_dir}/{run_id}/` and each is **resumable per note** (record exists ⇒ skip; tolerate a truncated final line; append + flush + fsync after every note; process notes sorted by id; run all notes, dev and test).

| Stage / command | Output file | One row per | Key fields |
|---|---|---|---|
| `make-corpus` | `data/corpus.jsonl`, `corpus_stats.json` | note | schema 1.1 |
| `split` | `split.csv` | note | `id, split(dev/test)` |
| `run greedy` | `greedy.jsonl` | (model, note) | `id, model, raw_output, pred (ordered, deduped, normalised), parse_failed, token_ids, token_probs, code_char_spans, prompt_tokens, gen_tokens, seconds, note_truncated, prompts_hash, template_mode, backend` |
| `run samples` | `samples.jsonl` | (model, note) | `id, model, mode, k, runs:[{run, seed, raw_output, codes}], prompt_tokens, gen_tokens, seconds` |
| `run s2` | `s2.jsonl` | (model, note) | `id, model, raw_output, scores:{code:value|null}, failed_codes, prompt_tokens, gen_tokens, seconds` |
| `run s4` | `s4.jsonl` | (model, note) | `id, model, items:[{code, p_yes, p_no, s4, top_is_yes_or_no}], prompt_tokens, seconds` |
| `build-codes` | `codes.csv` | (model, note, predicted code) | `model, id, split, code, position, correct, invalid, unsupported, cause_tag, gold_type, gold_rare, s1, s2, s3, s4, s5, s2_failed` |
| `analyze` | `metrics_signals.csv, thresholds.json, results_test.csv, results_controls.csv, bootstrap.csv, by_feature.csv, confident_mistakes.csv, cost.csv, k_sens.csv, hypotheses.csv, recommendation.csv` | | see §7–§13 |
| `tables`, `figures` | `reports/table_*.md/.csv`, `reports/fig1_risk_coverage.png`, `reports/fig2_reliability.png`, `reports/paper_numbers.md` | | §14 |
| `sanity` | `sanity_report.json` | | §15 |
| every command | `run_meta.json` | | config snapshot, seeds, git commit, GPU name, library versions, model revisions (resolved HF commit SHA), vocab hash, corpus hash, prompts hash, file SHA-256s |

Cost logging: every stage appends to `cost_log.jsonl` per (model, note, stage ∈ {greedy, samples, s2, s4, s3_post}): `logical_calls, prompt_tokens, gen_tokens, seconds`.

---

## 5. Prompts, parsing, labelling

### 5.1 Prompts (`c1/prompts.py`; hash all templates into `prompts_hash`; all three must match across models)
- **Base prompt** (greedy and samples): *"You are a clinical coder. Read the clinical note below and list the ICD-10-CM diagnosis codes it supports. Give one code per line, code only, and nothing else.\n\nNote:\n{note}"*
- **S2 prompt:** *"You are a clinical coder. Read the note below. For each ICD-10-CM code in the list, give a score from 0 to 100 for how well the note supports that code. Reply with one line per code in the form `CODE, SCORE`, and nothing else.\n\nNote:\n{note}\n\nCodes:\n{one code per line}"*
- **S4 prompt (one per predicted code):** *"Does the clinical note below support the ICD-10-CM code {code} ({description})? Answer with Yes or No only.\n\nNote:\n{note}"* For a code with **no description in the vocabulary** (invalid codes) use the neutral text `description not available` — never a phrase that reveals invalidity.
- The base prompt text is the earlier paper's Appendix-B wording; if the repo's vendor code contains the earlier paper's exact string, use that instead and say so in `COVERAGE.md`. Prompts may be changed **only before any test-split result is generated**; `sanity` fails if one model's outputs mix prompt hashes.

### 5.2 Parsing (`c1/parse.py`, shared by greedy, samples, S2)
Line-based: strip a leading list marker (`-`, `*`, `•`, `1.`, `1)`), take the code at the start of the line (case-insensitive, then upper-case; allow trailing text after whitespace/punctuation), normalise (insert dot if missing), keep if it matches the well-formed pattern. De-duplicate preserving first occurrence. **Never repair or replace invalid codes.** `parse_failed = (no code parsed)`. Aggregate parse-failure counts per model.
S2 lines: `CODE, SCORE` (tolerate `:` or whitespace as separator and a trailing `%`); score clipped to [0,100] then /100. Codes in the S2 answer are matched to the greedy list by normalised string; any greedy code with no readable score gets **S2 = 0 and `s2_failed=True`**; report the S2 failure rate per model.

### 5.3 Labelling (`c1/label.py`)
For each predicted code: `invalid` = not valid in the vocabulary; `correct` = in the note's gold set; `unsupported` = valid and not in gold. Exactly one is true. Test: counts of the three sum to the number of predicted codes. `cause_tag` for wrong codes: `negated` / `family_history` / `distractor_medication` if the code equals a recorded distractor code of that note (kind of that distractor; if several, take the first in the order negated, family_history, distractor_medication), else `other`. `gold_type`/`gold_rare` for correct codes from the gold entry.

---

## 6. The five signals (`c1/signals.py`) — computed per predicted code of the greedy list

- **S1:** fraction of the k=10 sampled lists (parsed, normalised) that contain the code. Values in {0, 0.1, …, 1}.
- **S2:** §5.2.
- **S3:** mean probability of the tokens that spell the code in the greedy output. Use the stored character span of the code in the raw output (first occurrence on its line) and the tokens whose character offsets overlap that span. Offsets come from re-decoding token by token (or the tokenizer's offset mapping on the decoded text); document the method and unit-test it with a fake tokenizer containing multi-token codes, a code at the end of text, and a code with a leading space token.
- **S4:** `P(Yes)/(P(Yes)+P(No))` from one forward pass of the S4 prompt, using the **first token id** of `"Yes"` and `"No"` in each model's tokenizer (verify in `selfcheck` which id each tokenizer actually produces for the answer position; handle a leading-space variant). If both probabilities are 0 within float precision, S4 = 0.5 and log it.
- **S5:** `(S1 + S4) / 2`.

---

## 7. Metrics (`c1/metrics.py` — pure functions, no I/O)

Per model, unit of analysis = predicted codes of the greedy lists.

- **AUROC** of a signal for separating correct (positive) from wrong (invalid + unsupported) codes; rank-based with tie handling (average ranks); undefined if one class is empty → `n/a`. Computed on the **test** split (dev AUROC also stored, used to pick the best signal).
- **ECE** (S2, S3, S4 only; S1 and S5 are not probabilities): 10 equal-width bins on [0,1] (last bin closed); ECE = Σ (n_b/N)·|accuracy_b − mean_score_b| where accuracy = fraction correct.
- **Metrics on a set of kept codes** (test split; notes' gold codes define recall): coverage = kept/predicted; precision = correct kept / kept; recall = correct kept / total gold codes of the notes; micro-F1 = 2PR/(P+R); **Unsupported Code Rate** = unsupported kept / kept; **Invalid Code Rate** = invalid kept / kept; **review load** = withheld codes per note on average. Zero denominators → `n/a`.
- **Full list** = all predicted codes kept (coverage 100%).

---

## 8. Thresholds (`c1/thresholds.py`)

For each (model, signal) and each target coverage in {0.9, 0.8, 0.7, 0.6}: using **dev codes only**, τ = the **largest** value such that the share of dev codes with score ≥ τ is ≥ the target. For target 1.0, keep everything (τ = −∞), so the 100% row reproduces the full list exactly. Write `thresholds.json` once (include a hash of the dev `codes.csv`); **never recompute thresholds after any test metric exists** and never read test labels while choosing thresholds. Apply τ to the test codes: a code is kept iff score ≥ τ. Report the **realised** test coverage (it differs from the target). Because scores can tie, realised coverage is expected to deviate; do not adjust τ to fix it.

---

## 9. Controls (`c1/controls.py`)

- **Random withholding:** for a given realised coverage c and N predicted test codes, keep `round(c·N)` codes chosen uniformly at random from the pooled test codes; repeat `repeats`=100 times with a seeded RNG; report the mean of each metric. Vectorise with numpy.
- **Oracle:** keep correct codes first (stable order), then wrong codes, up to `round(c·N)`.
- Controls are computed at **each signal's realised coverage** (so they are paired with it) and written to `results_controls.csv`.

---

## 10. Bootstrap, verdict, recommendation (`c1/bootstrap.py`)

- **Note-level bootstrap**, B = 1000, seeded: resample test **notes** with replacement (a note's codes move together). Thresholds stay fixed. In every resample recompute, for each (model, signal, primary target 0.8): the signal's precision, UCR, ICR, coverage, and the **random control at that resample's coverage** (100 draws, same vectorised routine), then the differences Δprecision, ΔUCR, ΔICR = signal − random. Percentile 95% intervals. AUROC intervals per signal and the AUROC differences needed in §13 come from the same resamples (skip resamples where a class is empty; count them).
- **Verdict (fixed before any data is seen).** A signal is **"useful"** for a model at the primary target iff: (i) the 95% interval of Δprecision lies entirely above 0; (ii) the 95% interval of ΔUCR lies entirely below 0; (iii) the **observed test** Δ ICR ≤ 0 (kept codes' Invalid Code Rate is not higher than random's). Otherwise **"not shown to help"**. Report all three conditions for every signal.
- **Recommendation rule (paper §4.4):** among the signals that are useful for a model, recommend the one with the fewest extra calls (from §12), unless another useful signal's AUROC interval lies **entirely above** the cheapest one's, in which case recommend the useful signal with the highest AUROC among those. Write `recommendation.csv`. If none is useful, write "none".
- **Best signal for Tables IV, V and VI** = the signal with the highest **dev** AUROC per model (decided on dev, stated in the tables' footnote).

---

## 11. Breakdown by note feature (`c1/breakdown.py`) → Table VI

Note-level flags from the corpus: `implicit` (≥1 implicit gold code), `explicit` (≥1 explicit gold code), `negation` (≥1 negated distractor), `family_history`, `distractor_medication`, `rare` (≥1 gold code with `rare=true`). Strata are **non-exclusive**. For each stratum and model, using the best signal at the primary target, on test: n predicted codes in the stratum's notes, AUROC (n/a if one class empty), precision full list, precision kept, UCR kept. `by_feature.csv` also contains an **"all notes"** row for reconciliation.
`confident_mistakes.csv`: wrong test codes **kept** by the best signal at the primary target, with columns `model, id, code, score, invalid|unsupported, cause_tag`, plus counts per cause tag (including `other`) and, per tag, the 5 highest-scoring cases with the note text truncated to 600 characters (synthetic text only).

---

## 12. Cost and k-sensitivity

- **Cost (`cost.csv`, Table VII):** per model and signal, mean per note: extra logical calls (S1 = k sampled lists; S2 = 1; S3 = 0 — read from the greedy pass; S4 = number of predicted codes; S5 = S1 + S4), extra tokens (prompt + generated tokens of those calls), extra seconds (wall-clock; S3 = time of the post-processing). Also report the greedy pass itself and physical `generate` calls separately. Record the GPU name next to the times.
- **k-sensitivity (`k_sens.csv`, Table VIII; dev split):** recompute S1 from the first 3, first 5, and all 10 sampled lists; report per model dev AUROC of S1 and extra calls per note (= k). Make no new LLM calls.

---

## 13. Hypotheses (`c1/hypotheses.py`) → Table IX

Evaluate per model; write the per-model status in the table's Result cell (e.g. "LLaMA-3: confirmed; BioMistral: inconclusive"). Statuses: **confirmed / rejected / inconclusive** (inconclusive when the interval contains the no-effect value).
- **H1** (S1 separates right from wrong): AUROC(S1) CI lower end > 0.5 → confirmed; CI upper end < 0.5 → rejected; else inconclusive.
- **H2** (S1 > S2): ΔAUROC = AUROC(S1) − AUROC(S2), CI entirely > 0 → confirmed; entirely < 0 → rejected; else inconclusive.
- **H3** (at least one signal is useful at ~80% coverage): "useful" per §10 for ≥1 signal → confirmed. Otherwise rejected if, for every signal, at least one of conditions (i)/(ii) has its interval entirely on the wrong side; else inconclusive.
- **H4** (abstention removes invalid codes proportionally more than unsupported): relative fall = (full − kept)/full for ICR and UCR for the best signal at the primary target; bootstrap the difference (ICR fall − UCR fall); entirely > 0 → confirmed; entirely < 0 → rejected; else inconclusive; `n/a` if the full-list ICR is 0.
- **H5** (implicit correct codes are harder than explicit): ΔAUROC = AUROC_implicit − AUROC_explicit for the best signal, where **AUROC_f** = AUROC with positives = correct test codes whose gold type is f and negatives = all wrong test codes (same negative pool for both). Confirmed if CI entirely < 0, rejected if entirely > 0, else inconclusive.

---

## 14. Tables and figures (`c1/tables.py`, `c1/figures.py`)

Format everywhere: AUROC/ECE/precision/recall/F1 with 3 decimals; coverage and rates as percentages with 1 decimal; intervals as `[lo, hi]`; Δ as signed values; `—` when not computed; footnotes state the best-signal rule, the primary target, B, seeds, split sizes, GPU and the vocabulary release. Every table is written as Markdown + CSV in the paper's row/column order.

| Output | Content |
|---|---|
| **Table II** | notes (dev/test), mean gold codes/note, distinct gold codes, share of gold codes that are implicit, mean predicted codes/note **per model** |
| **Table III** | per model × signal: AUROC [CI], ECE (S2–S4 only), `—` elsewhere |
| **Table IV(a)/(b)** | per model, best signal: Full list; target 90/80/70/60% rows (realised coverage, precision, recall, micro-F1, UCR, ICR, review load); **Random** and **Oracle** rows at the 80% row's coverage |
| **Table IV(c)** | copy of the headline numbers of IV: full list vs best signal at ~80% (precision, UCR, ICR) — no new computation |
| **Table V** | per model, best signal at 80%: Δprecision, ΔUCR, ΔICR with 95% CI, the three conditions, the verdict, plus the recommendation line |
| **Table VI** | §11 |
| **Table VII** | §12 cost |
| **Table VIII** | §12 k-sensitivity |
| **Table IX** | §13 |
| **Fig. 1** | risk–coverage: x = coverage, y = 1 − precision; per model one panel; one line per signal (sweep τ over the unique test scores) and a horizontal random line (= full-list error) |
| **Fig. 2** | reliability diagram for S2, S3, S4: mean score vs observed accuracy per bin, diagonal reference, per model |
| `paper_numbers.md` | every in-text number the paper asks for: seeds, split sizes, n codes, parse-failure and S2-failure rates, truncated notes, confident-mistake counts by cause, GPU, library versions, model revisions, vocabulary release |

Tables 0, 0A and I contain only text; do not generate them.

---

## 15. Sanity checks and tests (`c1/sanity.py`, `tests/`) — CPU, mock backend, no downloads

**Constructed worked examples (test fixtures only):**
1. *AUROC:* correct-code scores {0.9, 0.8, 0.5}, wrong-code scores {0.7, 0.5, 0.2}. Pairs: 3 + 3 + 1.5 (one tie counts ½) = 7.5 of 9 ⇒ **AUROC = 0.8333**. Also verify against a brute-force pairwise implementation and `sklearn.metrics.roc_auc_score` (test-only dependency) on random fixtures.
2. *Kept metrics:* 2 notes, 10 predicted codes (6 correct, 3 unsupported, 1 invalid), 8 gold codes in total; kept set = 5 codes (4 correct, 1 unsupported, 0 invalid). Expected: coverage 0.5, precision 0.8, recall 0.5, micro-F1 0.6154, UCR 0.2, ICR 0.0, review load 2.5 withheld codes per note.
3. *ECE (10 bins):* scores {0.05, 0.25, 0.85, 0.95} with correctness {0, 1, 1, 1} ⇒ **ECE = 0.25**.
4. *pick_tau:* dev scores [1, 1, .8, .8, .6, .4, .2, .2, .1, 0]: target 0.9 ⇒ τ = 0.1; 0.8 ⇒ τ = 0.2; 0.6 ⇒ τ = 0.4; target 1.0 keeps all. Ties: scores [1, .5, .5, .5, 0], target 0.6 ⇒ τ = 0.5 (realised coverage 0.8).

**Required checks (each its own test):**
- predicted = gold, all scores 1 ⇒ precision = recall = F1 = 1, UCR = ICR = 0.
- **shuffled scores ⇒ AUROC ≈ 0.5** (tolerance on a large synthetic fixture); **oracle ordering ⇒ AUROC = 1**.
- 100%-coverage row equals the full list exactly; random-control mean precision ≈ full-list precision (large fixture) and equals it in expectation analytically.
- correct + invalid + unsupported = number of predicted codes; ICR and UCR identities.
- split is deterministic, disjoint, 30/70 ±1 note; thresholds are computed from dev only (test labels never read; assert by file-guard test).
- parser: list markers, trailing text, undotted codes, duplicates (first kept), invalid codes preserved, empty output ⇒ `parse_failed`.
- S3 alignment on a fake tokenizer (multi-token code, end-of-text code, leading-space token); S4 formula and the 0/0 case; S5 mean; S1 on the 3/5/10 prefixes.
- bootstrap: same seed ⇒ same intervals; resampling is by note; identical systems ⇒ Δ ≈ 0; constructed better system ⇒ interval above 0; skipped single-class resamples counted.
- verdict, recommendation and hypothesis functions: one test per branch (confirmed / rejected / inconclusive; useful / not shown to help; recommendation switch on AUROC interval).
- resume: kill after k notes with the mock backend, restart, no duplicates, no losses, truncated last line tolerated.
- mixing guard: `tables`/`figures` refuse `outputs_mock/` and any file whose rows contain `"backend":"mock"`.
- schema validation of every JSONL/CSV on read; prompts-hash consistency; notes truncated count reported.
- `tests/test_coverage.py`: parses `COVERAGE.md` and asserts every listed module/function/output file exists, and every generated table in §14 has a producer.

**Runtime sanity (`sanity` command, on real outputs):** the above identities on the real `codes.csv`; test notes ∩ dev notes = ∅; counts per model equal across stages (every note has greedy/samples/s2/s4); mock rows absent; prompts hash single-valued per model. Print and save `sanity_report.json`; any failure blocks `tables`.

---

## 16. Environment (Google Colab Pro)

- **Libraries:** `transformers`, `accelerate`, `bitsandbytes`, `numpy`, `scipy`, `pandas`, `scikit-learn` (AUROC cross-check and tests only), `matplotlib`, `pyyaml`, `tqdm`, `pytest`, `pydantic` (or dataclasses). Torch comes from Colab; do not pin or reinstall it. Provide `requirements-colab.txt` (the above without torch/numpy pins, with version ranges verified by the user's smoke run) and `requirements-dev.txt`. Print all versions into `run_meta.json`.
- **GPU:** needs a CUDA GPU with ≥12 GB (T4 works; L4/A100 are faster). Fail with a clear message if CUDA is absent for real runs.
- **Persistence:** outputs and (optionally) the Hugging Face cache live on Google Drive (`out_dir`, `HF_HOME` from config). The repo is cloned into `/content` each session; the HF token is read from Colab Secrets (`google.colab.userdata`) or the `HF_TOKEN` environment variable. LLaMA-3 is gated: the user must accept the licence on the model page first; give a clear error otherwise.
- **Sessions:** run **one model per session**: load once, run all four generation stages in the order of §4, free memory. Cells must be safely re-runnable; no stage may depend on in-memory state.
- **Time budgeting:** the CLI command `estimate` runs 20 dev notes through all four stages for one model and prints the projected total hours for the configured `n_notes` (the paper's work guide asks for this before the full run). Do not hard-code time claims.
- Notebooks (`notebooks/`) only mount Drive, clone/install, log in, call the CLI, and display outputs.

---

## 17. Ambiguities in the paper and the defaults to use (record each in `ASSUMPTIONS.md`; ask the user only if told otherwise)

| # | Ambiguity | Default |
|---|---|---|
| A1 | Table VI feature definition (code-level vs note-level) | note-level non-exclusive strata (§11); H5 uses the code-level shared-negative AUROC (§13) |
| A2 | How the "best signal" is chosen | highest **dev** AUROC per model |
| A3 | Condition (iii) of the verdict (point vs interval) | observed test Δ ICR ≤ 0 |
| A4 | Sampling protocol | `batched` (one call, `num_return_sequences=10`, seeded per note) unless the user sets `per_seed_loop` (seeds 1–10); chosen before any run, never changed |
| A5 | Corpus size | 300 notes (≈90 dev / 210 test); user may lower to 200 |
| A6 | S4 prompt for codes without a description | neutral text `description not available` |
| A7 | Table VIII "calls per note" | extra logical calls = k |
| A8 | Hypothesis status when the two models disagree | report per model in the same cell |
| A9 | Valid code definition | same vocabulary object and definition as the earlier paper |
| A10 | Random control | pooled over the test set, 100 repeats, recomputed inside every bootstrap resample |
| A11 | Realised vs target coverage | thresholds from dev; report realised test coverage; never re-tune |

---

## 18. Coverage matrix (copy into `COVERAGE.md`, fill in file/function names, tick when implemented + tested)

| Paper element | Spec § | Module | Output |
|---|---|---|---|
| Corpus, split, dev/test | 1, 4 | data.py | corpus.jsonl, split.csv |
| Table II | 14 | data.py, tables.py | table_II |
| Greedy lists + parse + S3 | 3, 5, 6 | backend.py, parse.py, signals.py | greedy.jsonl |
| S1 sampling | 6, 12 | backend.py, signals.py | samples.jsonl |
| S2 verbalised | 5, 6 | prompts.py, signals.py | s2.jsonl |
| S4 yes/no | 6 | signals.py | s4.jsonl |
| S5 | 6 | signals.py | codes.csv |
| Labels correct/invalid/unsupported | 5.3 | label.py | codes.csv |
| Table I (definitions) | — | text only | — |
| AUROC, ECE; Table III; Fig. 2 | 7, 14 | metrics.py, figures.py | metrics_signals.csv |
| Thresholds (dev) | 8 | thresholds.py | thresholds.json |
| Table IV(a)(b)(c); Fig. 1 | 7–9, 14 | metrics.py, controls.py | results_test.csv, results_controls.csv |
| Table V + verdict + recommendation | 10 | bootstrap.py | bootstrap.csv, recommendation.csv |
| Table VI + confident mistakes | 11 | breakdown.py | by_feature.csv, confident_mistakes.csv |
| Table VII cost | 12 | cost.py | cost.csv, cost_log.jsonl |
| Table VIII k-sensitivity | 12 | signals.py | k_sens.csv |
| Table IX hypotheses H1–H5 | 13 | hypotheses.py | hypotheses.csv |
| Selfcheck (S3 alignment, S4 token check, determinism) | 3 | backend.py, cli.py | selfcheck_report.json |
| Time estimate | 16 | cli.py | printed |
| Sanity checks | 15 | sanity.py | sanity_report.json |
| In-text numbers | 14 | tables.py | paper_numbers.md |
| Reproducibility record | 4 | utils.py | run_meta.json |
