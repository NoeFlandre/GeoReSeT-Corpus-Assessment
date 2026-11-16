"""Zero-shot prompt that asks the model for finest-level EUNIS classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from georeset_corpus_assessment.legend import Legend

PROMPT_VERSION = "zero-shot-v1"

INSTRUCTIONS = (
    "You label a sentence that describes a place with EUNIS habitat classes. "
    "Choose every class from the list that the sentence clearly describes at "
    "that place. Use only identifiers from the list. Reason briefly, then end "
    "with one final line of the form `FINAL: <identifiers separated by commas>`, "
    "or `FINAL: none` when no class applies."
)


def candidate_block(legend: Legend) -> str:
    """Render the finest-level classes as one identifier and name per line."""
    lines = []
    for item in legend.finest_classes():
        label = item.name or item.identifier
        lines.append(f"- {item.identifier}: {label}")
    return "\n".join(lines)


def build_messages(sentence: str, legend: Legend) -> list[dict[str, str]]:
    """Return the chat messages for one sentence."""
    user = f"Habitat classes:\n{candidate_block(legend)}\n\nSentence:\n{sentence}"
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": user},
    ]
