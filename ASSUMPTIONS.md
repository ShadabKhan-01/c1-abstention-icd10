# ASSUMPTIONS.md — Ambiguities and Chosen Defaults

This document records ambiguities in the paper and the defaults adopted, as specified in `SPEC.md §17`.

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
| A12 | Rare condition definition | Template conditions with weight <= 0.2 (bottom quartile) or explicit rare: true flag are assigned rare: true in generated corpus |
| A13 | Shared note truncation limit | Default `DEFAULT_MAX_NOTE_WORDS = 1000` words (fits BioMistral 2048-token context window minus template and generation margins; SPEC §3) |
| A14 | S2 unreadable score representation | Stored as `null` in `s2.jsonl` with code in `failed_codes`; mapped to `s2 = 0.0` and `s2_failed = True` in `codes.csv` (SPEC §4, §5.2) |
| A15 | S4 zero probability fallback | If $P(\text{Yes}) + P(\text{No}) \le 10^{-12}$, S4 = 0.5 with warning log (SPEC §6) |
| A16 | `codes.csv` empty-field representation | For correct codes, `cause_tag` is empty string `""`; for wrong codes, `gold_type` and `gold_rare` are empty string `""` (SPEC §4, §5.3) |
| A17 | `position` indexing in `codes.csv` | 1-indexed (1 to N) representing the order of codes in greedy `pred` list (SPEC §4) |
| A18 | ECE for non-probability signals | ECE is undefined for S1 and S5 (not calibrated probabilities); stored as NaN in computations and empty string in CSV (SPEC §7, §14) |
| A19 | Zero denominators in kept metrics | Evaluates to NaN (or 0.0 for F1 when P+R == 0) per SPEC §7 zero-denominator rule |
| A20 | Confident mistakes note text truncation | Synthetic note text is truncated by character count (`note.text[:600]`) for top 5 highest-scoring cases per cause tag (SPEC §11) |
| A21 | H4 zero full-list ICR | When full-list ICR is 0.0, H4 status is `n/a` per SPEC §13 |
| A22 | H5 single-class resample skip | Bootstrap resamples with 0 positives or 0 negatives for implicit or explicit gold codes are skipped for H5 calculation |
| A23 | Bootstrap confidence interval calculation | Empirical percentile 95% intervals computed via `[np.percentile(samples, 2.5), np.percentile(samples, 97.5)]` (SPEC §10) |
| A24 | Tables and figures mock data mixing guard | Mock outputs and `outputs_mock` path rejected in production mode; allowed with `--allow-mock` which stamps `MOCK DATA` on markdown, CSV, and PNG reports (SPEC §14, §15, AGENTS.md §2) |
| A25 | Sanity report prerequisite | `tables` and `figures` require `sanity_report.json` with `passed == true` before generating reports (SPEC §15) |
| A26 | Thin Colab notebooks | Notebooks contain zero data/metric processing logic; cells exclusively mount Drive, clone repo, set secrets, and invoke CLI commands (AGENTS.md §3, SPEC §16) |
| A27 | Model set (author decision, deviates from the paper draft and AGENTS.md §1.4) | Run Qwen2.5-7B-Instruct, BioMistral-7B, Llama3-Med42-8B and Gemma-2-9B-it; LLaMA-3-8B-Instruct stays in the config but is not run. The paper text and tables must be updated to name these models |
| A28 | Code parsing (deviates from the original SPEC §5.2 start-of-line rule) | Every well-formed code anywhere in a line is taken, because models write `Gout (M10.0)`. Exception: a 3-character code without a dot directly after a word (`Vitamin B12`) is text, not a code. Codes are still never repaired. First BioMistral run under the old rule: 187/300 notes parsed no code |
| A29 | S2 line fallback | If a line is not `CODE, SCORE`, the first greedy code in the line is scored by the last number after it (`/100` and `out of 100` ignored) |
| A30 | Gemma-2 compute dtype | `bfloat16` (per-model `compute_dtype` override); Gemma-2 is numerically unstable in float16. Other models use float16 |
| A31 | Run id | Results of the four-model study live in `c1_run02`; `c1_run01` (old parser) is not used for any table |
