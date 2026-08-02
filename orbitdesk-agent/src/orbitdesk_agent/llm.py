"""Local text-generation backends.

`RealGenerator` loads a small instruction-tuned Hugging Face model
(Qwen2.5-0.5B-Instruct by default -- see config.py) through
`transformers` and runs fully on-device. `MockGenerator` is a
template-based stand-in used by the automated tests so that graph
*routing* can be verified without depending on the exact wording an LLM
would produce (this is required by the assignment), and so the whole
pipeline is runnable before you've downloaded any model weights.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

from .config import MODEL_CONFIG
from .state import RetrievedChunk

SYSTEM_PROMPT = (
    "You are the OrbitDesk support assistant. Answer ONLY using the evidence "
    "passages provided below. Every factual claim must be traceable to one of "
    "the passages. If the evidence does not fully support an answer, say so. "
    "Do not invent steps, error codes, or permissions. Cite each source by its "
    "source_id in square brackets, e.g. [KB-004]. Keep the answer concise and "
    "actionable."
)


def build_prompt(question: str, evidence: List[RetrievedChunk]) -> str:
    evidence_block = "\n\n".join(
        f"[{c['source_id']}] {c['title']}\n{c['text']}" for c in evidence
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER (cite source_ids in brackets):"
    )


class BaseGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, question: str, evidence: List[RetrievedChunk], revision_note: str | None = None) -> str:
        ...


class RealGenerator(BaseGenerator):
    """Wraps a local Hugging Face causal LM via `transformers`."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = MODEL_CONFIG.generation_model_name
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG.generation_model_name, revision=MODEL_CONFIG.generation_model_revision
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.generation_model_name,
            revision=MODEL_CONFIG.generation_model_revision,
            torch_dtype=torch.float32,
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self.load_time_seconds = time.time() - t0

    def generate(self, question: str, evidence: List[RetrievedChunk], revision_note: str | None = None) -> str:
        prompt = build_prompt(question, evidence)
        if revision_note:
            prompt += (
                f"\n\nNOTE: A previous draft failed verification for this reason: "
                f"'{revision_note}'. Correct that issue in this revised answer."
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self._device)
        input_len = inputs["input_ids"].shape[-1]
        t0 = time.time()
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=MODEL_CONFIG.max_new_tokens,
            temperature=MODEL_CONFIG.generation_temperature,
            do_sample=MODEL_CONFIG.generation_temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        self.last_latency_seconds = time.time() - t0
        new_tokens = output_ids[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class MockGenerator(BaseGenerator):
    """Deterministic, template-based generator for tests / offline demos.

    Supports a special trigger: if the question contains "FORCE_BAD_DRAFT"
    and no revision_note has been supplied yet, it deliberately returns an
    answer with no source citations, so the verification-failure -> retry
    path can be exercised in an automated test without depending on real
    model output.
    """

    name = "mock-template-generator"

    def __init__(self) -> None:
        self.load_time_seconds = 0.0
        self.last_latency_seconds = 0.0

    def generate(self, question: str, evidence: List[RetrievedChunk], revision_note: str | None = None) -> str:
        t0 = time.time()
        if "FORCE_BAD_DRAFT" in question and revision_note is None:
            self.last_latency_seconds = time.time() - t0
            return "You should be fine, just try again later."  # no citations -> fails verification

        if not evidence:
            self.last_latency_seconds = time.time() - t0
            return "I don't have enough supported evidence to answer this safely."

        top = evidence[0]
        citation_ids = ", ".join(f"[{c['source_id']}]" for c in evidence[:2])
        answer = (
            f"Based on {citation_ids}: {top['text'].splitlines()[0][:200]} "
            f"See {top['source_id']} for the full documented steps."
        )
        self.last_latency_seconds = time.time() - t0
        return answer


def get_generator(mock: bool) -> BaseGenerator:
    if mock:
        return MockGenerator()
    return RealGenerator()
