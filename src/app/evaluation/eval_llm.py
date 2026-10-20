"""Deterministic LLM stub for reproducible RAG evaluation (citations still from retrieval)."""

from __future__ import annotations


class CitationPreservingEvalLLM:
    """
    Skips open-ended generation while keeping the RAG path identical up to citations.

    The answer string is fixed so experiment runs do not depend on Ollama/Cohere availability.
    """

    label = "eval_stub"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return (
            "[eval] Grounded answer placeholder — see citations for retrieved chunk references."
        )
