# Confidence-Based Abstention for LLM ICD-10 Coding

A reproducible implementation of the paper *"Confidence-Based Abstention for LLM ICD-10 Coding"* (paper C1).

## Overview
This project evaluates whether withholding low-confidence predictions from open-weight clinical large language models (LLaMA-3-8B-Instruct and BioMistral-7B) improves coding precision and reduces the Unsupported Code Rate on ICD-10-CM diagnosis coding tasks. Five confidence signals (S1–S5) are compared against a random withholding control and an empirical oracle.

## Repository Layout
- `c1/`: Python package containing all pipeline modules, metrics, and CLI.
- `configs/`: Pipeline configuration (`configs/c1.yaml`).
- `data/`: Dataset storage (`data/vocab/` contains the committed ICD-10-CM vocabulary).
- `notebooks/`: Google Colab notebooks for GPU execution.
- `tests/`: Unit test suite.
- `vendor/`: Vendored synthetic note generator and vocabulary tools.

## Documentation
- `SPEC.md`: Complete implementation specification.
- `AGENTS.md`: Project rules and constraints.
- `PROMPTS_AND_RUNBOOK.md`: Milestone prompts and Colab execution runbook.
- `COVERAGE.md`: Coverage checklist mapping paper elements to code and outputs.
- `ASSUMPTIONS.md`: Default resolutions of paper ambiguities (SPEC §17).
- `QUESTIONS.md`: Tracked open questions.

