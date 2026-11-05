"""Tests for backend interface, MockBackend, S3 alignment, and S4 probabilities.

Traces to SPEC §3, §6, §15.
"""

from __future__ import annotations

import pytest

from c1.backend import (
    MockBackend,
    align_tokens_to_chars,
    compute_s3_score,
    compute_s4_score,
)


def test_mock_backend_interface() -> None:
    """Verify MockBackend generates deterministic outputs with backend='mock'."""
    backend = MockBackend(model_name="mock_llama")
    assert backend.backend_name == "mock"

    # Chat format
    fmt, mode = backend.format_chat("Hello")
    assert "[INST] Hello [/INST]" in fmt
    assert mode == "inst_tags"

    # Greedy
    g_out = backend.generate_greedy("prompt")
    assert "E11.9" in g_out.text
    assert len(g_out.new_token_ids) == len(g_out.token_probs)
    assert g_out.prompt_tokens > 0

    # Samples
    s_outs = backend.generate_samples("prompt", k=5, temperature=0.7, top_p=0.9, seed=42)
    assert len(s_outs) == 5
    for s in s_outs:
        assert s.gen_tokens > 0

    # S2 Text
    t_out = backend.generate_text("prompt")
    assert ", 90" in t_out.text

    # S4 next token probs
    yes_id, no_id = backend.get_yes_no_token_ids()
    prob_dict, p_toks, secs, argmax_id = backend.next_token_probs("prompt", [yes_id, no_id])
    assert prob_dict[yes_id] == 0.85
    assert prob_dict[no_id] == 0.10
    assert argmax_id == yes_id


def test_s3_alignment_fake_tokenizer_cases() -> None:
    """Verify S3 token alignment per SPEC §15:

    1. Multi-token code: code broken into multiple subword tokens.
    2. Code at end of text: code ends at the exact end of raw output.
    3. Leading-space token: token includes a leading space before code.
    """
    # Fake vocabulary map: token_id -> text piece
    fake_vocab = {
        1: "Intro: ",
        2: "E1",       # Multi-token part 1
        3: "1.9",      # Multi-token part 2
        4: "\n",
        5: " I10",     # Leading space token
        6: "\n",
        7: "Z99",      # Multi-token part 1 at end
        8: ".999",     # Multi-token part 2 at end
    }

    def fake_decode(token_ids: list[int]) -> str:
        return "".join(fake_vocab[tid] for tid in token_ids)

    # Generated sequence:
    # "Intro: E11.9\n I10\nZ99.999"
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    # Probabilities for each token:
    # 1: 0.90, 2: 0.80, 3: 0.90, 4: 0.99, 5: 0.70, 6: 0.99, 7: 0.60, 8: 0.50
    probs = [0.90, 0.80, 0.90, 0.99, 0.70, 0.99, 0.60, 0.50]

    full_text = fake_decode(tokens)
    assert full_text == "Intro: E11.9\n I10\nZ99.999"

    spans = align_tokens_to_chars(tokens, fake_decode)
    assert len(spans) == len(tokens)

    # 1. Multi-token code: "E11.9" starts at index 7, ends at 12
    e119_start = full_text.index("E11.9")
    e119_end = e119_start + len("E11.9")
    s3_e119 = compute_s3_score((e119_start, e119_end), spans, probs)
    # Tokens overlapping are tid 2 (0.80) and tid 3 (0.90) -> mean = 0.85
    assert pytest.approx(s3_e119, 0.001) == 0.85

    # 2. Leading space token: "I10" starts at index 14, ends at 17
    # Note: token 5 is " I10" spanning index 13 to 17
    i10_start = full_text.index("I10")
    i10_end = i10_start + len("I10")
    s3_i10 = compute_s3_score((i10_start, i10_end), spans, probs)
    # Overlaps token 5 (0.70)
    assert pytest.approx(s3_i10, 0.001) == 0.70

    # 3. Code at end of text: "Z99.999" at exact end (indices 18 to 25)
    z99_start = full_text.index("Z99.999")
    z99_end = z99_start + len("Z99.999")
    assert z99_end == len(full_text)
    s3_z99 = compute_s3_score((z99_start, z99_end), spans, probs)
    # Overlaps tokens 7 (0.60) and 8 (0.50) -> mean = 0.55
    assert pytest.approx(s3_z99, 0.001) == 0.55


def test_s4_score_computation() -> None:
    """Verify S4 formula P(Yes)/(P(Yes)+P(No)) and 0/0 fallback to 0.5 (SPEC §6, §15)."""
    # Standard cases
    assert pytest.approx(compute_s4_score(0.8, 0.2), 0.0001) == 0.80
    assert pytest.approx(compute_s4_score(0.3, 0.1), 0.0001) == 0.75
    assert pytest.approx(compute_s4_score(0.0, 0.9), 0.0001) == 0.00

    # 0/0 fallback case returns 0.5
    assert compute_s4_score(0.0, 0.0) == 0.5
    assert compute_s4_score(1e-15, 1e-15) == 0.5

