# COVERAGE.md — Paper Implementation Coverage Matrix

This matrix tracks the implementation and test status of every paper element and code artifact, as mandated by `SPEC.md §18` and `AGENTS.md §1`.

## Paper Elements and Pipeline Stages

| Status | Paper element | Spec § | Module | Functions / Artifacts | Output | Milestone |
|:---:|---|---|---|---|---|:---:|
| [x] | Corpus, split, dev/test | 1, 4 | `c1/data.py` | `load_corpus`, `split_corpus`, `compute_corpus_stats`, `make_corpus_from_config`, `run_split_from_config`, `load_split` | `corpus.jsonl`, `split.csv`, `corpus_stats.json` | M1 |
| [x] | Table II | 14 | `c1/data.py`, `c1/tables.py` | `generate_table_ii`, `run_table_ii_from_config` | `reports/table_II.md`, `reports/table_II.csv` | M1 |
| [x] | Greedy lists + parse + S3 | 3, 5, 6 | `c1/backend.py`, `c1/parse.py`, `c1/stages.py` | `parse_greedy_output`, `align_tokens_to_chars`, `compute_s3_score`, `run_greedy_stage` | `greedy.jsonl` | M2 |
| [x] | S1 sampling | 6, 12 | `c1/backend.py`, `c1/stages.py` | `generate_samples`, `run_samples_stage` | `samples.jsonl` | M2 |
| [x] | S2 verbalised | 5, 6 | `c1/prompts.py`, `c1/parse.py`, `c1/stages.py` | `format_s2_prompt`, `parse_s2_output`, `run_s2_stage` | `s2.jsonl` | M2 |
| [x] | S4 yes/no | 6 | `c1/prompts.py`, `c1/backend.py`, `c1/stages.py` | `format_s4_prompt`, `compute_s4_score`, `run_s4_stage` | `s4.jsonl` | M2 |
| [x] | S5 | 6 | `c1/signals.py`, `c1/stages.py` | `compute_s5`, `build_codes_stage` | `codes.csv` | M3 |
| [x] | Labels correct/invalid/unsupported | 5.3 | `c1/label.py`, `c1/stages.py` | `label_code`, `label_note_codes`, `build_codes_stage` | `codes.csv` | M3 |
| [x] | Table I (definitions) | 6 | `README.md`, `SPEC.md §6` | Signal S1–S5 definitions & abstention rule | `README.md` | M7 |
| [x] | AUROC, ECE; Table III; Fig. 2 | 7, 14 | `c1/metrics.py`, `c1/stages.py` | `compute_auroc`, `compute_ece`, `run_analyze_stage` | `metrics_signals.csv` | M4 |
| [x] | Thresholds (dev) | 8 | `c1/thresholds.py`, `c1/stages.py` | `pick_tau`, `compute_dev_thresholds`, `save_thresholds` | `thresholds.json` | M4 |
| [x] | Table IV(a)(b)(c); Fig. 1 | 7–9, 14 | `c1/metrics.py`, `c1/controls.py`, `c1/stages.py` | `compute_kept_metrics`, `run_random_control`, `run_oracle_control`, `run_analyze_stage` | `results_test.csv`, `results_controls.csv` | M4 |
| [x] | Table V + verdict + recommendation | 10 | `c1/bootstrap.py`, `c1/stages.py` | `run_model_bootstrap`, `evaluate_verdict`, `evaluate_recommendation`, `run_analyze_stage` | `bootstrap.csv`, `recommendation.csv` | M5 |
| [x] | Table VI + confident mistakes | 11 | `c1/breakdown.py`, `c1/stages.py` | `compute_breakdown_by_feature`, `extract_confident_mistakes`, `run_analyze_stage` | `by_feature.csv`, `confident_mistakes.csv` | M5 |
| [x] | Table VII cost | 12 | `c1/cost.py`, `c1/stages.py` | `compute_cost_table`, `append_cost_record`, `run_analyze_stage` | `cost.csv`, `cost_log.jsonl` | M5 |
| [x] | Table VIII k-sensitivity | 12 | `c1/signals.py`, `c1/stages.py` | `compute_k_sensitivity`, `run_analyze_stage` | `k_sens.csv` | M5 |
| [x] | Table IX hypotheses H1–H5 | 13 | `c1/hypotheses.py`, `c1/stages.py` | `evaluate_model_hypotheses`, `classify_h1`, `classify_h2`, `classify_h3`, `classify_h4`, `classify_h5`, `run_analyze_stage` | `hypotheses.csv` | M5 |
| [x] | Selfcheck (S3 alignment, S4 token check, determinism) | 3 | `c1/backend.py`, `c1/stages.py`, `c1/cli.py` | `run_selfcheck`, `run_selfcheck_cli` | `selfcheck_report.json` | M2 |
| [x] | Time estimate | 16 | `c1/stages.py`, `c1/cli.py` | `run_estimate`, `run_estimate_cli` | printed | M2 |
| [x] | Sanity checks | 15 | `c1/sanity.py`, `c1/cli.py` | `run_sanity_checks`, `run_sanity_cli` | `sanity_report.json` | M6 |
| [x] | In-text numbers & Tables II–IX & Figures 1–2 | 14 | `c1/tables.py`, `c1/figures.py`, `c1/cli.py` | `run_all_tables`, `run_all_figures`, `generate_paper_numbers`, `run_tables_cli`, `run_figures_cli` | `reports/table_*.md/.csv`, `reports/fig*.png`, `reports/paper_numbers.md` | M6 |
| [x] | Reproducibility record & utils | 4 | `c1/utils.py` | `set_seed`, `load_config`, `write_json_atomic`, `write_csv_atomic`, `append_jsonl`, `read_jsonl_resume`, `get_completed_ids`, `sha256_file`, `sha256_string`, `write_run_meta` | `run_meta.json` | M0 |
| [x] | CLI Scaffolding & Generation Runners | 4 | `c1/cli.py` | `main`, `build_parser`, `run_pipeline_cli`, `run_selfcheck_cli`, `run_estimate_cli` | CLI entry point | M0, M2 |
| [x] | Notebooks, Run Guide, Scope Audit | 3, 16, 18 | `notebooks/`, `README.md`, `SCOPE_AUDIT.md` | `01_run_model.ipynb`, `02_analysis.ipynb`, `README.md`, `SCOPE_AUDIT.md` | `notebooks/*.ipynb`, `README.md`, `SCOPE_AUDIT.md` | M7 |

## Milestone M0 Implemented Functions Traceability

- `c1.utils.set_seed`: Determinism control (SPEC §4, §15)
- `c1.utils.load_config`: Pydantic schema validation for `configs/c1.yaml` (SPEC §2, §4)
- `c1.utils.write_json_atomic`: Atomic JSON writes with tmp file + fsync + rename (AGENTS.md §4, SPEC §4)
- `c1.utils.write_csv_atomic`: Atomic CSV writes with tmp file + fsync + rename (AGENTS.md §4, SPEC §4)
- `c1.utils.append_jsonl`: Resumable append-only JSONL with flush + fsync (AGENTS.md §3, §4, SPEC §4)
- `c1.utils.read_jsonl_resume`: Resumable JSONL reader with truncated last-line tolerance and clean tail truncation (AGENTS.md §3, SPEC §4, §15)
- `c1.utils.get_completed_ids`: Resumable ID check helper for skipping completed notes (SPEC §4)
- `c1.utils.sha256_file`: SHA-256 file hashing for data/vocab/config/code audit (SPEC §1.2, §4)
- `c1.utils.sha256_string`: SHA-256 string hashing for prompt/template hashing (SPEC §4, §5.1)
- `c1.utils.write_run_meta`: Reproducibility metadata writer (SPEC §4, §15, §16)
- `c1.cli.main`: CLI entry point with command stubs (SPEC §4)

## Milestone M1 Implemented Functions Traceability

- `c1.codes.normalize`: Canonical code normalization with uppercase and 3rd-character dot insertion (SPEC §1.2)
- `c1.codes.is_well_formed`: Well-formed pattern validation regex `^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$` (SPEC §1.2)
- `c1.codes.Vocabulary`: Vocabulary lookup, description, and validity wrapper over pinned CDC 2024 release (SPEC §1.2)
- `vendor.codes`: Shim module forwarding `normalize` and `is_well_formed` to `c1.codes` (SPEC §1.3)
- `vendor.synth_data.generate_record`: Minimal patch emitting structured `gold` with `type` and `rare`, and `distractors` with `kind` (SPEC §1.1, §1.3)
- `vendor.synth_data.generate_corpus`: Unpartitioned corpus generation wrapper (SPEC §1.1, §4)
- `c1.data.GoldCode`, `Distractor`, `Note`: Pydantic schema validation for corpus records (SPEC §1.1)
- `c1.data.compute_note_features`: Note-level non-exclusive feature flags (implicit, explicit, negation, family_history, distractor_medication, rare) (SPEC §11)
- `c1.data.load_corpus`: Strict corpus loader enforcing the four hard checks of SPEC §1.1 (IDs unique, gold codes valid, no distractor in gold, implicit codes not named) (SPEC §1.1)
- `c1.data.split_corpus`: Deterministic seeded 30/70 dev/test split via seeded permutation (SPEC §2, §4, §15)
- `c1.data.compute_corpus_stats`: Corpus summary statistics (SPEC §4)
- `c1.data.make_corpus_from_config`: Generation orchestrator producing `corpus.jsonl` and `corpus_stats.json` (SPEC §4)
- `c1.data.run_split_from_config`: Split runner producing `split.csv` (SPEC §4)
- `c1.data.load_split`: Split mapping loader (SPEC §4)
- `c1.tables.generate_table_ii`: Table II producer rendering Markdown and CSV reports with SPEC §14 formatting (SPEC §14)
- `c1.tables.run_table_ii_from_config`: CLI handler for Table II generation (SPEC §14)

## Milestone M2 Implemented Functions Traceability

- `c1.prompts.BASE_PROMPT_TEMPLATE`, `S2_PROMPT_TEMPLATE`, `S4_PROMPT_TEMPLATE`, `NEUTRAL_DESCRIPTION`: Standardized prompt templates and neutral fallback (SPEC §5.1)
- `c1.prompts.compute_prompts_hash`: Deterministic SHA-256 template hashing across all templates (`PROMPTS_HASH`) (SPEC §4, §5.1)
- `c1.prompts.format_base_prompt`: Formatter for greedy and S1 sampling prompt (SPEC §5.1)
- `c1.prompts.format_s2_prompt`: Formatter for S2 verbalized scoring prompt (SPEC §5.1)
- `c1.prompts.format_s4_prompt`: Formatter for S4 verification prompt with neutral description fallback (SPEC §5.1)
- `c1.prompts.compute_max_note_words`: BioMistral 2048-context shared word limit calculation (SPEC §3)
- `c1.prompts.truncate_note`: Shared note truncation ensuring identical inputs across models (SPEC §3)
- `c1.parse.parse_greedy_output`: Line-based parser extracting ordered, deduplicated, normalized codes, preserving invalid codes, and recording exact character spans for S3 alignment (SPEC §4, §5.2, §6)
- `c1.parse.parse_s2_output`: Line-based S2 score parser clipping to [0, 100]/100 and tracking failed codes (SPEC §5.2)
- `c1.backend.GenOut`: Container for generation text, token IDs, probabilities, token counts, and elapsed time (SPEC §3)
- `c1.backend.BaseBackend`: Abstract backend interface for generation and verification (SPEC §3)
- `c1.backend.MockBackend`: Deterministic CPU mock backend for testing ('backend': 'mock') (SPEC §3, AGENTS.md §2)
- `c1.backend.HFBackend`: 4-bit NF4 quantized GPU backend with lazy torch/transformers imports, T4 compatibility, float32 raw logits, LLaMA-3 stop tokens, only-new-token decoding, batch size 1 (SPEC §3, AGENTS.md §3)
- `c1.backend.align_tokens_to_chars`: Monotonic prefix decoding for exact token-to-character span mapping (SPEC §6, §15)
- `c1.backend.compute_s3_score`: S3 score calculation via mean probability of overlapping tokens (SPEC §6, §15)
- `c1.backend.compute_s4_score`: S4 normalized probability calculation with 0/0 fallback to 0.5 (SPEC §6, §15)
- `c1.stages.run_greedy_stage`: Resumable runner for greedy generation producing `greedy.jsonl` (SPEC §4)
- `c1.stages.run_samples_stage`: Resumable runner for sampled generation producing `samples.jsonl` (SPEC §4)
- `c1.stages.run_s2_stage`: Resumable runner for S2 verbalized scoring producing `s2.jsonl` (SPEC §4)
- `c1.stages.run_s4_stage`: Resumable runner for S4 verification producing `s4.jsonl` (SPEC §4)
- `c1.stages.append_cost_record`: Cost logging for each stage appending to `cost_log.jsonl` (SPEC §4)
- `c1.stages.run_selfcheck`: Diagnostic runner for GPU, determinism, S3 alignment, and S4 Yes/No check producing `selfcheck_report.json` (SPEC §3, §15)
- `c1.stages.run_estimate`: 20-note benchmark projecting full corpus runtime hours (SPEC §16)
- `c1.cli.run_pipeline_cli`: CLI handler for `c1 run` (SPEC §4)
- `c1.cli.run_selfcheck_cli`: CLI handler for `c1 selfcheck` (SPEC §3, §15)
- `c1.cli.run_estimate_cli`: CLI handler for `c1 estimate` (SPEC §16)

## Milestone M3 Implemented Functions Traceability

- `c1.label.CodeLabel`: Dataclass container for code correctness category, cause tag, and gold metadata (SPEC §4, §5.3)
- `c1.label.label_code`: Evaluates code validity, gold set membership, mutually exclusive classification (correct/invalid/unsupported), distractor cause tag priority ('negated' > 'family_history' > 'distractor_medication' > 'other'), and gold type/rare retrieval (SPEC §5.3)
- `c1.label.label_note_codes`: Batch labelling helper for a sequence of predicted codes (SPEC §5.3)
- `c1.signals.compute_s1`: Sampling agreement across k samples with prefix support for Table VIII k-sensitivity (SPEC §6, §12)
- `c1.signals.extract_s2`: Extracts normalized verbalized scores and handles missing score failure mapping (S2 = 0.0, s2_failed = True) (SPEC §5.2, §6)
- `c1.signals.compute_s3_from_greedy`: Computes S3 mean token probability over character spans via monotonic token alignment (SPEC §6, §15)
- `c1.signals.extract_s4`: Extracts normalized verification probability with 0/0 fallback to 0.5 (SPEC §6)
- `c1.signals.compute_s5`: Composite signal mean (S1 + S4) / 2 (SPEC §6)
- `c1.stages.build_codes_stage`: Stage runner joining stage outputs with corpus and vocabulary to produce codes.csv and logging s3_post costs (SPEC §4, §12)
- `c1.cli.run_build_codes_cli`: CLI handler for c1 build-codes (SPEC §4)

## Milestone M4 Implemented Functions Traceability

- `c1.metrics.compute_auroc`: Rank-based AUROC with average ranks for tied scores and nan handling (SPEC §7, §15)
- `c1.metrics.compute_ece`: Expected Calibration Error across 10 equal-width bins on [0, 1] (SPEC §7, §15)
- `c1.metrics.KeptMetrics`: Dataclass container for kept-set metrics (SPEC §7)
- `c1.metrics.compute_kept_metrics`: Evaluates coverage, precision, recall, micro-F1, UCR, ICR, and review load (SPEC §7, §15)
- `c1.thresholds.pick_tau`: Finds largest tau such that dev codes share >= target_coverage (SPEC §8, §15)
- `c1.thresholds.compute_dev_thresholds`: Guarded threshold selection strictly from dev codes (SPEC §8)
- `c1.thresholds.save_thresholds`: Writes thresholds.json with dev_codes_hash (SPEC §8)
- `c1.thresholds.load_thresholds`: Reads and parses thresholds.json (SPEC §8)
- `c1.controls.run_random_control`: Vectorized random withholding across 100 repeats (SPEC §9, §15)
- `c1.controls.run_oracle_control`: Deterministic oracle control keeping correct codes first (SPEC §9, §15)
- `c1.stages.run_analyze_stage`: Orchestrator writing all 11 analysis files (SPEC §4)
- `c1.cli.run_analyze_cli`: CLI handler for c1 analyze (SPEC §4)

## Milestone M5 Implemented Functions Traceability

- `c1.bootstrap.evaluate_verdict`: Pre-defined 3-condition verdict rule evaluating delta precision CI > 0, delta UCR CI < 0, and observed delta ICR <= 0 (SPEC §10)
- `c1.bootstrap.evaluate_recommendation`: Recommendation rule picking cheapest useful signal unless higher AUROC signal interval strictly dominates (SPEC §10, paper §4.4)
- `c1.bootstrap.run_model_bootstrap`: Note-level bootstrap resampling test notes with replacement (B=1000) with multiplicity weighting, recomputing paired 100-repeat random control (SPEC §10, §15)
- `c1.breakdown.compute_breakdown_by_feature`: Non-exclusive note strata evaluation using best signal at primary target with 'all notes' reconciliation row (SPEC §11, Table VI)
- `c1.breakdown.extract_confident_mistakes`: Wrong test codes kept by best signal at primary target, sorted by score, with top-5 note text truncated to 600 characters per cause tag (SPEC §11)
- `c1.cost.compute_cost_table`: Table VII cost aggregator computing per-note mean extra logical calls, physical calls, tokens, wall-clock seconds, and GPU device name (SPEC §12, Table VII)
- `c1.signals.compute_k_sensitivity`: Table VIII k-sensitivity recomputing S1 dev AUROC and extra calls per note on dev split for k in {3, 5, 10} without new LLM calls (SPEC §12, Table VIII)
- `c1.hypotheses.classify_h1`: Classifier for H1 (AUROC(S1) > 0.5) (SPEC §13)
- `c1.hypotheses.classify_h2`: Classifier for H2 (S1 > S2: ΔAUROC > 0) (SPEC §13)
- `c1.hypotheses.classify_h3`: Classifier for H3 (useful signal at ~80% coverage) (SPEC §13)
- `c1.hypotheses.classify_h4`: Classifier for H4 (ICR relative fall > UCR relative fall; n/a if full ICR is 0) (SPEC §13)
- `c1.hypotheses.classify_h5`: Classifier for H5 (implicit correct codes harder than explicit: ΔAUROC < 0) (SPEC §13)
- `c1.hypotheses.evaluate_model_hypotheses`: Full evaluator for hypotheses H1 through H5 producing hypotheses.csv (SPEC §13, Table IX)

## Milestone M6 Implemented Functions Traceability

- `c1.sanity.run_sanity_checks`: Master runtime verification engine auditing file completeness, stage parity, disjoint split, code identities, prompt hash consistency, and mock mixing guard (SPEC §15)
- `c1.sanity.check_file_completeness`: Validates presence and non-empty status of all 17 required pipeline output files (SPEC §4, §15)
- `c1.sanity.check_mock_mixing_guard`: Production mixing guard blocking outputs_mock and mock backend records unless explicitly allowed (SPEC §15, AGENTS.md §2)
- `c1.sanity.check_split_integrity`: Verifies disjoint split sets (dev ∩ test = ∅) and 30/70 proportion ±1 note (SPEC §15)
- `c1.sanity.check_stage_parity`: Verifies note count equality across greedy, samples, s2, and s4 stages (SPEC §15)
- `c1.sanity.check_code_identities`: Asserts correct + invalid + unsupported == N_pred, mutual exclusivity, and precision + UCR + ICR == 1.0 (SPEC §5.3, §7, §15)
- `c1.sanity.check_prompts_hash_consistency`: Verifies prompts_hash is single-valued per model and consistent across models (SPEC §5.1, §15)
- `c1.figures.generate_fig1_risk_coverage`: Renders Fig 1 risk-coverage curves per model with random baseline (SPEC §14)
- `c1.figures.generate_fig2_reliability`: Renders Fig 2 reliability diagrams for probability signals (S2, S3, S4) with diagonal reference (SPEC §14)
- `c1.figures.run_all_figures`: Master figure runner checking mock guard and sanity status before rendering to PNG (SPEC §14, §15)
- `c1.tables.fmt_dec3`: Formats float metrics to 3 decimal places or em-dash (SPEC §14)
- `c1.tables.fmt_pct1`: Formats coverage and error rates as percentages with 1 decimal place (SPEC §14)
- `c1.tables.fmt_ci`: Formats confidence intervals as [lo, hi] (SPEC §14)
- `c1.tables.fmt_signed_delta`: Formats signed differences with confidence intervals (SPEC §14)
- `c1.tables.generate_table_ii`: Generates Table II corpus and prediction statistics (SPEC §14)
- `c1.tables.generate_table_iii`: Generates Table III discrimination (AUROC [CI]) and calibration (ECE) (SPEC §14)
- `c1.tables.generate_table_iv`: Generates Table IV(a)/(b) risk-coverage progression for best signal (SPEC §14)
- `c1.tables.generate_table_iv_c`: Generates Table IV(c) headline numbers summary (SPEC §14)
- `c1.tables.generate_table_v`: Generates Table V statistical verdict and recommendation (SPEC §14)
- `c1.tables.generate_table_vi`: Generates Table VI feature strata breakdown (SPEC §14)
- `c1.tables.generate_table_vii`: Generates Table VII computational cost accounting (SPEC §14)
- `c1.tables.generate_table_viii`: Generates Table VIII k-sensitivity on dev split (SPEC §14)
- `c1.tables.generate_table_ix`: Generates Table IX pre-registered hypotheses evaluation (SPEC §14)
- `c1.tables.generate_paper_numbers`: Generates paper_numbers.md compiling in-text numbers (SPEC §14)
- `c1.tables.run_all_tables`: Master table runner checking mock guard and sanity status before generating all tables and paper numbers (SPEC §14, §15)
- `c1.cli.run_sanity_cli`: CLI handler for c1 sanity (SPEC §15)
- `c1.cli.run_tables_cli`: CLI handler for c1 tables (SPEC §14)
- `c1.cli.run_figures_cli`: CLI handler for c1 figures (SPEC §14)
- `c1.cli.run_paper_numbers_cli`: CLI handler for c1 paper-numbers (SPEC §14)

## Milestone M7 Implemented Functions Traceability

- `notebooks.01_run_model`: GPU execution notebook following Part 3 of PROMPTS_AND_RUNBOOK.md (SPEC §3, §16)
- `notebooks.02_analysis`: CPU analysis, tables, and figures execution notebook (SPEC §14, §16)
- `README.md`: Comprehensive paper run guide, Table I definitions, and Colab Pro/mock execution instructions (SPEC §0, §6, §16)
- `SCOPE_AUDIT.md`: Exhaustive file and function traceability audit and negative scope verification (SPEC §18, AGENTS.md §1)





