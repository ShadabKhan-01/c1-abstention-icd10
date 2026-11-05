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

