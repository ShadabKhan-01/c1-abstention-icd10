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
| [ ] | S5 | 6 | `signals.py` | — | `codes.csv` | M3 |
| [ ] | Labels correct/invalid/unsupported | 5.3 | `label.py` | — | `codes.csv` | M3 |
| [ ] | Table I (definitions) | — | text only | — | — | — |
| [ ] | AUROC, ECE; Table III; Fig. 2 | 7, 14 | `metrics.py`, `figures.py` | — | `metrics_signals.csv` | M4 |
| [ ] | Thresholds (dev) | 8 | `thresholds.py` | — | `thresholds.json` | M4 |
| [ ] | Table IV(a)(b)(c); Fig. 1 | 7–9, 14 | `metrics.py`, `controls.py` | — | `results_test.csv`, `results_controls.csv` | M4 |
| [ ] | Table V + verdict + recommendation | 10 | `bootstrap.py` | — | `bootstrap.csv`, `recommendation.csv` | M5 |
| [ ] | Table VI + confident mistakes | 11 | `breakdown.py` | — | `by_feature.csv`, `confident_mistakes.csv` | M5 |
| [ ] | Table VII cost | 12 | `cost.py`, `c1/stages.py` | `append_cost_record` | `cost.csv`, `cost_log.jsonl` | M5 |
| [ ] | Table VIII k-sensitivity | 12 | `signals.py` | — | `k_sens.csv` | M5 |
| [ ] | Table IX hypotheses H1–H5 | 13 | `hypotheses.py` | — | `hypotheses.csv` | M5 |
| [x] | Selfcheck (S3 alignment, S4 token check, determinism) | 3 | `c1/backend.py`, `c1/stages.py`, `c1/cli.py` | `run_selfcheck`, `run_selfcheck_cli` | `selfcheck_report.json` | M2 |
| [x] | Time estimate | 16 | `c1/stages.py`, `c1/cli.py` | `run_estimate`, `run_estimate_cli` | printed | M2 |
| [ ] | Sanity checks | 15 | `sanity.py` | — | `sanity_report.json` | M6 |
| [ ] | In-text numbers | 14 | `tables.py` | — | `paper_numbers.md` | M6 |
| [x] | Reproducibility record & utils | 4 | `c1/utils.py` | `set_seed`, `load_config`, `write_json_atomic`, `write_csv_atomic`, `append_jsonl`, `read_jsonl_resume`, `get_completed_ids`, `sha256_file`, `sha256_string`, `write_run_meta` | `run_meta.json` | M0 |
| [x] | CLI Scaffolding & Generation Runners | 4 | `c1/cli.py` | `main`, `build_parser`, `run_pipeline_cli`, `run_selfcheck_cli`, `run_estimate_cli` | CLI entry point | M0, M2 |

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
