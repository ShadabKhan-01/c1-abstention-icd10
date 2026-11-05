"""Model backends for inference, token alignment, and probability computation.

Provides:
- BaseBackend: Abstract interface for generation and verification.
- MockBackend: Deterministic CPU mock backend for testing ('backend': 'mock').
- HFBackend: GPU backend for 4-bit quantized open-weight LLMs (T4 compatible,
  lazy torch/transformers imports, float32 raw logits, LLaMA-3 stop tokens).
- S3 and S4 scoring functions and monotonic token-to-character alignment.

Traces to SPEC §3, §4, §6, §15, §16.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import gc
import hashlib
import logging
import time
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures (SPEC §3, §4)
# ---------------------------------------------------------------------------


@dataclass
class GenOut:
    """Standardized output container for generation calls."""

    text: str
    new_token_ids: list[int] = field(default_factory=list)
    token_probs: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    gen_tokens: int = 0
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# S3 and S4 Alignment & Probability Helpers (SPEC §6, §15)
# ---------------------------------------------------------------------------


def align_tokens_to_chars(
    token_ids: Sequence[int],
    decode_fn: Callable[[Sequence[int]], str],
) -> list[tuple[int, int]]:
    """Compute character spans [start, end] for each generated token in decoded text.

    Uses monotonic prefix decoding to guarantee exact mapping across any BPE/SentencePiece
    tokenizer without boundary merge discrepancies.

    Args:
        token_ids: Sequence of newly generated token IDs.
        decode_fn: Callable that decodes a sequence of token IDs to string.

    Returns:
        List of (start, end) character offsets in the fully decoded text.
    """
    spans: list[tuple[int, int]] = []
    prev_str = ""

    for i in range(len(token_ids)):
        curr_str = decode_fn(token_ids[: i + 1])
        start = len(prev_str)
        end = len(curr_str)
        spans.append((start, end))
        prev_str = curr_str

    return spans


def compute_s3_score(
    code_span: tuple[int, int],
    token_spans: Sequence[tuple[int, int]],
    token_probs: Sequence[float],
) -> float:
    """Compute S3 score as the mean probability of tokens spelling a code.

    Tokens whose character spans overlap the code's character span are identified.
    Two spans [s1, e1] and [s2, e2] overlap if max(s1, s2) < min(e1, e2) and s1 < e1.

    Args:
        code_span: (start, end) character span of the code in raw output.
        token_spans: Sequence of (start, end) character spans for each generated token.
        token_probs: Sequence of generation probabilities for each generated token.

    Returns:
        Mean probability of overlapping tokens in [0.0, 1.0].
        If no token overlaps, returns 0.0.
    """
    c_start, c_end = code_span
    overlapping_probs: list[float] = []

    for (t_start, t_end), prob in zip(token_spans, token_probs):
        if t_start < t_end and max(c_start, t_start) < min(c_end, t_end):
            overlapping_probs.append(prob)

    if not overlapping_probs:
        return 0.0
    return sum(overlapping_probs) / len(overlapping_probs)


def compute_s4_score(p_yes: float, p_no: float) -> float:
    """Compute normalized S4 verification probability: P(Yes) / (P(Yes) + P(No)).

    If both probabilities are zero within float precision, returns 0.5 per SPEC §6.

    Args:
        p_yes: Softmax probability of Yes token.
        p_no: Softmax probability of No token.

    Returns:
        S4 score in [0.0, 1.0].
    """
    total = p_yes + p_no
    if total <= 1e-12:
        logger.warning("S4 probabilities sum to 0; falling back to 0.5")
        return 0.5
    return p_yes / total


# ---------------------------------------------------------------------------
# Backend Interface (SPEC §3)
# ---------------------------------------------------------------------------


class BaseBackend(ABC):
    """Abstract interface for LLM backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of backend: 'mock' or 'hf'."""
        pass

    @abstractmethod
    def format_chat(self, prompt: str) -> tuple[str, str]:
        """Apply chat template to prompt.

        Returns:
            Tuple of (formatted_prompt_text, template_mode).
        """
        pass

    @abstractmethod
    def generate_greedy(self, prompt: str, max_new_tokens: int = 200) -> GenOut:
        """Run greedy decoding with raw logit token probabilities."""
        pass

    @abstractmethod
    def generate_samples(
        self,
        prompt: str,
        k: int,
        temperature: float,
        top_p: float,
        seed: int,
        mode: str = "batched",
        max_new_tokens: int = 200,
    ) -> list[GenOut]:
        """Run stochastic sampled decoding (S1)."""
        pass

    @abstractmethod
    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> GenOut:
        """Run greedy text generation for S2 scoring."""
        pass

    @abstractmethod
    def next_token_probs(
        self,
        prompt: str,
        token_ids: Sequence[int],
    ) -> tuple[dict[int, float], int, float, int]:
        """Compute next-token softmax probabilities at answer position via single forward pass.

        Returns:
            Tuple of:
                - Dict mapping token_id -> softmax probability.
                - Prompt token count.
                - Elapsed wall-clock seconds.
                - Argmax token ID (top-1 predicted token).
        """
        pass

    @abstractmethod
    def get_yes_no_token_ids(self) -> tuple[int, int]:
        """Return (yes_token_id, no_token_id) for the model's vocabulary."""
        pass

    @abstractmethod
    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        """Decode a sequence of token IDs to text."""
        pass


# ---------------------------------------------------------------------------
# Mock Backend (SPEC §3, AGENTS.md §2)
# ---------------------------------------------------------------------------


class MockBackend(BaseBackend):
    """Deterministic mock backend for CPU testing without GPU or heavy dependencies.

    All outputs are stamped with backend='mock' and written under outputs_mock/.
    """

    def __init__(self, model_name: str = "mock_model") -> None:
        self.model_name = model_name
        self._yes_id = 100
        self._no_id = 101

    @property
    def backend_name(self) -> str:
        return "mock"

    def format_chat(self, prompt: str) -> tuple[str, str]:
        return f"[INST] {prompt} [/INST]", "inst_tags"

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        # Mock decode mapping token IDs to text fragments
        parts = []
        for tid in token_ids:
            if tid == 100:
                parts.append("Yes")
            elif tid == 101:
                parts.append("No")
            elif tid == 201:
                parts.append("E11.9")
            elif tid == 202:
                parts.append("\n")
            elif tid == 203:
                parts.append("I10")
            elif tid == 204:
                parts.append("Z99.999")
            elif tid == 205:
                parts.append(", 90\n")
            elif tid == 206:
                parts.append(", 80\n")
            else:
                parts.append(f" tok{tid}")
        return "".join(parts)

    def generate_greedy(self, prompt: str, max_new_tokens: int = 200) -> GenOut:
        # Generate predictable mock codes
        text = "E11.9\nI10\n"
        token_ids = [201, 202, 203, 202]
        token_probs = [0.95, 0.99, 0.88, 0.99]
        return GenOut(
            text=text,
            new_token_ids=token_ids,
            token_probs=token_probs,
            prompt_tokens=40,
            gen_tokens=len(token_ids),
            seconds=0.01,
        )

    def generate_samples(
        self,
        prompt: str,
        k: int,
        temperature: float,
        top_p: float,
        seed: int,
        mode: str = "batched",
        max_new_tokens: int = 200,
    ) -> list[GenOut]:
        samples = []
        for i in range(k):
            # Deterministically vary sample outputs
            run_seed = seed + i
            if run_seed % 3 == 0:
                text = "E11.9\n"
                tids = [201, 202]
                probs = [0.90, 0.99]
            elif run_seed % 3 == 1:
                text = "E11.9\nI10\n"
                tids = [201, 202, 203, 202]
                probs = [0.92, 0.99, 0.85, 0.99]
            else:
                text = "I10\n"
                tids = [203, 202]
                probs = [0.89, 0.99]

            samples.append(
                GenOut(
                    text=text,
                    new_token_ids=tids,
                    token_probs=probs,
                    prompt_tokens=40,
                    gen_tokens=len(tids),
                    seconds=0.005,
                )
            )
        return samples

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> GenOut:
        # S2 mock scoring
        text = "E11.9, 90\nI10, 80\n"
        token_ids = [201, 205, 203, 206]
        token_probs = [0.95, 0.95, 0.90, 0.90]
        return GenOut(
            text=text,
            new_token_ids=token_ids,
            token_probs=token_probs,
            prompt_tokens=50,
            gen_tokens=len(token_ids),
            seconds=0.01,
        )

    def next_token_probs(
        self,
        prompt: str,
        token_ids: Sequence[int],
    ) -> tuple[dict[int, float], int, float, int]:
        # Return deterministic Yes/No probabilities
        p_yes = 0.85
        p_no = 0.10
        probs = {}
        for tid in token_ids:
            if tid == self._yes_id:
                probs[tid] = p_yes
            elif tid == self._no_id:
                probs[tid] = p_no
            else:
                probs[tid] = 0.01
        return probs, 35, 0.005, self._yes_id

    def get_yes_no_token_ids(self) -> tuple[int, int]:
        return self._yes_id, self._no_id


# ---------------------------------------------------------------------------
# Hugging Face Backend (SPEC §3, AGENTS.md §3)
# ---------------------------------------------------------------------------


class HFBackend(BaseBackend):
    """GPU-accelerated backend using Hugging Face Transformers and BitsAndBytes.

    Requires CUDA GPU with >=12GB VRAM.
    All heavy packages (torch, transformers, bitsandbytes) are loaded lazily.
    """

    def __init__(
        self,
        model_id: str,
        context_limit: int = 2048,
        quant_type: str = "nf4",
        compute_dtype: str = "float16",
        double_quant: bool = True,
    ) -> None:
        self.model_id = model_id
        self.context_limit = context_limit
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.double_quant = double_quant

        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "cuda"
        self._eos_token_ids: list[int] = []
        self._yes_id: int | None = None
        self._no_id: int | None = None
        self._template_mode: str = "chat_template"

    @property
    def backend_name(self) -> str:
        return "hf"

    def load(self) -> None:
        """Load tokenizer and 4-bit quantized model onto GPU.

        Raises:
            RuntimeError: If CUDA is unavailable or torch is missing.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except (ImportError, ModuleNotFoundError) as err:
            raise RuntimeError(
                "PyTorch and Hugging Face Transformers are required for HFBackend but not installed. "
                "GPU runs should be executed in Google Colab (Pro). For local testing, use --backend mock."
            ) from err

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for HFBackend but torch.cuda.is_available() is False."
            )

        logger.info("Loading tokenizer for %s", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Set pad token if missing
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup EOS tokens including LLaMA-3 <|eot_id|>
        eos_ids = [self.tokenizer.eos_token_id]
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if (
                eot_id is not None
                and eot_id != self.tokenizer.unk_token_id
                and eot_id not in eos_ids
            ):
                eos_ids.append(eot_id)
        except Exception:
            pass
        self._eos_token_ids = eos_ids

        # Determine compute dtype
        dtype = torch.float16 if self.compute_dtype == "float16" else torch.bfloat16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=self.double_quant,
        )

        logger.info(
            "Loading 4-bit model %s (device_map='auto', compute_dtype=%s)",
            self.model_id,
            self.compute_dtype,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()

        # Cache Yes and No token IDs
        self._cache_yes_no_tokens()

    def unload(self) -> None:
        """Unload model and free GPU memory between runs."""
        import torch

        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HFBackend unloaded and GPU cache emptied.")

    def _cache_yes_no_tokens(self) -> None:
        """Find and cache the canonical Yes and No token IDs."""
        # Check standard and leading-space tokens
        for candidate in ["Yes", " Yes"]:
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                self._yes_id = ids[0]
                break
        for candidate in ["No", " No"]:
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                self._no_id = ids[0]
                break

        if self._yes_id is None or self._no_id is None:
            raise ValueError(
                f"Could not resolve Yes/No token IDs for tokenizer of {self.model_id}"
            )

    def get_yes_no_token_ids(self) -> tuple[int, int]:
        if self._yes_id is None or self._no_id is None:
            self._cache_yes_no_tokens()
        return self._yes_id, self._no_id  # type: ignore[return-value]

    def format_chat(self, prompt: str) -> tuple[str, str]:
        """Apply chat template or fallback to [INST] tags."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted, "chat_template"
        else:
            return f"[INST] {prompt} [/INST]", "inst_tags"

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_greedy(self, prompt: str, max_new_tokens: int = 200) -> GenOut:
        """Greedy decoding with raw-logit float32 probabilities."""
        import torch

        formatted_prompt, _ = self.format_chat(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        start_t = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=self._eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.perf_counter() - start_t

        new_token_ids = outputs.sequences[0, prompt_len:].tolist()
        gen_tokens = len(new_token_ids)

        # Extract chosen-token probabilities from raw logits (scores)
        token_probs: list[float] = []
        for t in range(gen_tokens):
            step_logits = outputs.scores[t][0].to(torch.float32)
            step_probs = torch.softmax(step_logits, dim=-1)
            chosen_tid = new_token_ids[t]
            token_probs.append(float(step_probs[chosen_tid].item()))

        decoded_text = self.tokenizer.decode(
            outputs.sequences[0, prompt_len:],
            skip_special_tokens=True,
        )

        return GenOut(
            text=decoded_text,
            new_token_ids=new_token_ids,
            token_probs=token_probs,
            prompt_tokens=prompt_len,
            gen_tokens=gen_tokens,
            seconds=elapsed,
        )

    def generate_samples(
        self,
        prompt: str,
        k: int,
        temperature: float,
        top_p: float,
        seed: int,
        mode: str = "batched",
        max_new_tokens: int = 200,
    ) -> list[GenOut]:
        """Stochastic sampled generation for S1."""
        import torch

        formatted_prompt, _ = self.format_chat(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        results: list[GenOut] = []

        if mode == "batched":
            torch.manual_seed(seed)
            start_t = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=k,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self._eos_token_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            elapsed = time.perf_counter() - start_t
            per_sample_elapsed = elapsed / max(k, 1)

            for i in range(k):
                seq = outputs.sequences[i, prompt_len:]
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                results.append(
                    GenOut(
                        text=text,
                        new_token_ids=seq.tolist(),
                        token_probs=[],
                        prompt_tokens=prompt_len,
                        gen_tokens=len(seq),
                        seconds=per_sample_elapsed,
                    )
                )
        else:  # per_seed_loop
            for i in range(1, k + 1):
                torch.manual_seed(seed + i)
                start_t = time.perf_counter()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=1,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=self._eos_token_ids,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                elapsed = time.perf_counter() - start_t
                seq = outputs.sequences[0, prompt_len:]
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                results.append(
                    GenOut(
                        text=text,
                        new_token_ids=seq.tolist(),
                        token_probs=[],
                        prompt_tokens=prompt_len,
                        gen_tokens=len(seq),
                        seconds=elapsed,
                    )
                )

        return results

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> GenOut:
        """Greedy text generation without scores for S2."""
        import torch

        formatted_prompt, _ = self.format_chat(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        start_t = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=self._eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start_t

        seq = outputs.sequences[0, prompt_len:]
        text = self.tokenizer.decode(seq, skip_special_tokens=True)

        return GenOut(
            text=text,
            new_token_ids=seq.tolist(),
            token_probs=[],
            prompt_tokens=prompt_len,
            gen_tokens=len(seq),
            seconds=elapsed,
        )

    def next_token_probs(
        self,
        prompt: str,
        token_ids: Sequence[int],
    ) -> tuple[dict[int, float], int, float, int]:
        """Single forward pass for next-token probabilities (S4)."""
        import torch

        formatted_prompt, _ = self.format_chat(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        start_t = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        elapsed = time.perf_counter() - start_t

        last_logits = outputs.logits[0, -1, :].to(torch.float32)
        probs = torch.softmax(last_logits, dim=-1)
        argmax_tid = int(torch.argmax(last_logits).item())

        token_prob_dict = {tid: float(probs[tid].item()) for tid in token_ids}
        return token_prob_dict, prompt_len, elapsed, argmax_tid
