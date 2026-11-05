# COVERAGE.md — Paper Implementation Coverage Matrix

This matrix tracks the implementation and test status of every paper element and code artifact, as mandated by `SPEC.md §18` and `AGENTS.md §1`.

## Paper Elements and Pipeline Stages

| Status | Paper element | Spec § | Module | Functions / Artifacts | Output | Milestone |
|:---:|---|---|---|---|---|:---:|
| [ ] | Corpus, split, dev/test | 1, 4 | `data.py` | — | `corpus.jsonl`, `split.csv` | M1 |
| [ ] | Table II | 14 | `data.py`, `tables.py` | — | `table_II` | M1 |
| [ ] | Greedy lists + parse + S3 | 3, 5, 6 | `backend.py`, `parse.py`, `signals.py` | — | `greedy.jsonl` | M2 |
| [ ] | S1 sampling | 6, 12 | `backend.py`, `signals.py` | — | `samples.jsonl` | M2 |
| [ ] | S2 verbalised | 5, 6 | `prompts.py`, `signals.py` | — | `s2.jsonl` | M2 |
| [ ] | S4 yes/no | 6 | `signals.py` | — | `s4.jsonl` | M2 |
| [ ] | S5 | 6 | `signals.py` | — | `codes.csv` | M3 |
| [ ] | Labels correct/invalid/unsupported | 5.3 | `label.py` | — | `codes.csv` | M3 |
| [ ] | Table I (definitions) | — | text only | — | — | — |
| [ ] | AUROC, ECE; Table III; Fig. 2 | 7, 14 | `metrics.py`, `figures.py` | — | `metrics_signals.csv` | M4 |
| [ ] | Thresholds (dev) | 8 | `thresholds.py` | — | `thresholds.json` | M4 |
| [ ] | Table IV(a)(b)(c); Fig. 1 | 7–9, 14 | `metrics.py`, `controls.py` | — | `results_test.csv`, `results_controls.csv` | M4 |
| [ ] | Table V + verdict + recommendation | 10 | `bootstrap.py` | — | `bootstrap.csv`, `recommendation.csv` | M5 |
| [ ] | Table VI + confident mistakes | 11 | `breakdown.py` | — | `by_feature.csv`, `confident_mistakes.csv` | M5 |
| [ ] | Table VII cost | 12 | `cost.py` | — | `cost.csv`, `cost_log.jsonl` | M5 |
| [ ] | Table VIII k-sensitivity | 12 | `signals.py` | — | `k_sens.csv` | M5 |
| [ ] | Table IX hypotheses H1–H5 | 13 | `hypotheses.py` | — | `hypotheses.csv` | M5 |
| [ ] | Selfcheck (S3 alignment, S4 token check, determinism) | 3 | `backend.py`, `cli.py` | — | `selfcheck_report.json` | M2 |
| [ ] | Time estimate | 16 | `cli.py` | — | printed | M2 |
| [ ] | Sanity checks | 15 | `sanity.py` | — | `sanity_report.json` | M6 |
| [ ] | In-text numbers | 14 | `tables.py` | — | `paper_numbers.md` | M6 |
| [x] | Reproducibility record & utils | 4 | `c1/utils.py` | `set_seed`, `load_config`, `write_json_atomic`, `write_csv_atomic`, `append_jsonl`, `read_jsonl_resume`, `get_completed_ids`, `sha256_file`, `sha256_string`, `write_run_meta` | `run_meta.json` | M0 |
| [x] | CLI Scaffolding | 4 | `c1/cli.py` | `main`, stubs for all pipeline commands | CLI entry point | M0 |

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

