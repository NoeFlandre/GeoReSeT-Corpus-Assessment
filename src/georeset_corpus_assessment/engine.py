"""vLLM adapter. Runs only on a GPU node, so it is excluded from unit coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from georeset_corpus_assessment.pipeline import Generation

if TYPE_CHECKING:
    from collections.abc import Sequence


class VllmEngine:  # pragma: no cover
    """Greedy chat generation with vLLM, in float16 for GPUs without bfloat16."""

    def __init__(self, model_id: str, revision: str, max_new_tokens: int) -> None:
        """Load the model at a pinned revision."""
        from vllm import LLM, SamplingParams  # noqa: PLC0415  # ty: ignore[import-not-found]

        self._llm = LLM(
            model=model_id,
            revision=revision,
            trust_remote_code=True,
            dtype="float16",
        )
        self._params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def generate(self, prompts: Sequence[list[dict[str, str]]]) -> list[Generation]:
        """Answer each chat prompt in one batched call."""
        outputs = self._llm.chat(list(prompts), self._params, use_tqdm=False)
        generations = []
        for output in outputs:
            first = output.outputs[0]
            generations.append(
                Generation(text=first.text, finish_reason=first.finish_reason or ""),
            )
        return generations
