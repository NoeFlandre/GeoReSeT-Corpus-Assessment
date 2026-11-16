"""Transformers adapter. Runs on a GPU node only, so unit coverage excludes it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from georeset_corpus_assessment.pipeline import Generation

if TYPE_CHECKING:
    from collections.abc import Sequence

MICRO_BATCH = 64


class TransformersEngine:  # pragma: no cover
    """Greedy chat generation with Hugging Face transformers, in bfloat16."""

    def __init__(self, model_id: str, revision: str, max_new_tokens: int) -> None:
        """Load the model and tokenizer at a pinned revision."""
        import torch  # noqa: PLC0415  # ty: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415  # ty: ignore[import-not-found]

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to("cuda")
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def generate(self, prompts: Sequence[list[dict[str, str]]]) -> list[Generation]:
        """Answer each chat prompt, in micro-batches, greedily."""
        generations: list[Generation] = []
        for start in range(0, len(prompts), MICRO_BATCH):
            generations.extend(
                self._generate_batch(prompts[start : start + MICRO_BATCH])
            )
        return generations

    def _generate_batch(
        self, prompts: Sequence[list[dict[str, str]]]
    ) -> list[Generation]:
        texts = [
            self._tokenizer.apply_chat_template(
                list(prompt),
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]
        inputs = self._tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        with self._torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        texts_out = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [
            Generation(
                text=text,
                finish_reason=finish_reason(
                    tokens, self._max_new_tokens, self._tokenizer.eos_token_id
                ),
            )
            for text, tokens in zip(texts_out, new_tokens.tolist(), strict=True)
        ]


def finish_reason(tokens: list[int], limit: int, eos_token_id: int | None) -> str:
    """Return "stop" when the model ended, or "length" when it hit the limit."""
    if eos_token_id is not None and eos_token_id in tokens:
        return "stop"
    return "length" if len(tokens) >= limit else "stop"
