"""Judge backends. Each returns log-probabilities of "A" and "B" as the first answer token.

- fake: deterministic stand-in for tests and dry runs, with a tunable position and length bias.
- transformers: works on CPU, Apple MPS or CUDA; supports a LoRA adapter. Tested here only with a
  tiny random model.
- vllm: for GPU throughput. NOT TESTED in this repo (no GPU available when written).

Inputs longer than max_input_tokens are skipped and reported, never truncated.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class Score:
    logprob_a: float
    logprob_b: float
    n_input_tokens: int
    skip_reason: str = ""


class FakeBackend:
    """p(A) = sigmoid(position_bias + length_weight * (len(A) - len(B)) / 1000)."""

    name = "fake"

    def __init__(self, position_bias: float = 0.3, length_weight: float = 1.0):
        self.position_bias, self.length_weight = position_bias, length_weight

    def score(self, batch: list[list[dict]]) -> list[Score]:
        out = []
        for messages in batch:
            text = messages[-1]["content"]
            a = re.search(r"\[Response A\]\n(.*?)\n\[End of Response A\]", text, re.DOTALL).group(1)
            b = re.search(r"\[Response B\]\n(.*?)\n\[End of Response B\]", text, re.DOTALL).group(1)
            z = self.position_bias + self.length_weight * (len(a) - len(b)) / 1000
            out.append(Score(-math.log1p(math.exp(-z)), -math.log1p(math.exp(z)), len(text.split())))
        return out


def _answer_token_ids(tokenizer) -> tuple[int, int]:
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"'A' and 'B' must each be one token; got {ids}")
    return ids[0][0], ids[1][0]


class TransformersBackend:
    name = "transformers"

    def __init__(self, model: str, *, revision: str | None = None, dtype: str = "auto", device: str = "auto",
                 adapter: str | None = None, max_input_tokens: int = 8192, batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if dtype == "auto":
            dtype = "bfloat16" if device == "cuda" else "float32"
        self.device, self.dtype = device, dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=getattr(torch, dtype))
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(device).eval()
        self.id_a, self.id_b = _answer_token_ids(self.tokenizer)
        self.max_input_tokens, self.batch_size = max_input_tokens, batch_size

    def _encode(self, messages: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def score(self, batch: list[list[dict]]) -> list[Score]:
        encoded = [self._encode(m) for m in batch]
        results: list[Score | None] = [None] * len(batch)
        runnable = []
        for i, ids in enumerate(encoded):
            if len(ids) > self.max_input_tokens:
                results[i] = Score(math.nan, math.nan, len(ids), "too_long")
            else:
                runnable.append(i)
        for start in range(0, len(runnable), self.batch_size):
            chunk = runnable[start:start + self.batch_size]
            padded = self.tokenizer.pad({"input_ids": [encoded[i] for i in chunk]}, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**padded).logits[:, -1, :].float()
            logprobs = self.torch.log_softmax(logits, dim=-1)
            for row, i in enumerate(chunk):
                results[i] = Score(float(logprobs[row, self.id_a]), float(logprobs[row, self.id_b]), len(encoded[i]))
        return results


class VLLMBackend:
    """Untested sketch for GPU runs. Constrains the first token to A or B and reads its log-probabilities."""

    name = "vllm"

    def __init__(self, model: str, *, revision: str | None = None, dtype: str = "bfloat16",
                 max_input_tokens: int = 8192, seed: int = 0, **_ignored):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, revision=revision, dtype=dtype, max_model_len=max_input_tokens + 8, seed=seed)
        self.tokenizer = self.llm.get_tokenizer()
        self.id_a, self.id_b = _answer_token_ids(self.tokenizer)
        self.params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, allowed_token_ids=[self.id_a, self.id_b])
        self.max_input_tokens = max_input_tokens

    def score(self, batch: list[list[dict]]) -> list[Score]:
        texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        lengths = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
        keep = [i for i, n in enumerate(lengths) if n <= self.max_input_tokens]
        results = [Score(math.nan, math.nan, n, "too_long") for n in lengths]
        outputs = self.llm.generate([texts[i] for i in keep], self.params, use_tqdm=False)
        for i, output in zip(keep, outputs):
            top = output.outputs[0].logprobs[0]
            la = top[self.id_a].logprob if self.id_a in top else math.nan
            lb = top[self.id_b].logprob if self.id_b in top else math.nan
            results[i] = Score(la, lb, lengths[i], "" if not (math.isnan(la) or math.isnan(lb)) else "missing_logprob")
        return results


def make_backend(name: str, **kwargs):
    if name == "fake":
        return FakeBackend()
    if name == "transformers":
        return TransformersBackend(**kwargs)
    if name == "vllm":
        if kwargs.get("adapter"):
            raise ValueError("LoRA adapters are only wired up for the transformers backend")
        return VLLMBackend(**kwargs)
    raise ValueError(f"Unknown backend {name!r}")
