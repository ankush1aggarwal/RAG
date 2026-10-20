from app.evaluation.eval_llm import CitationPreservingEvalLLM


def test_citation_preserving_eval_llm_is_deterministic():
    llm = CitationPreservingEvalLLM()
    assert llm.label == "eval_stub"
    first = llm.generate("user prompt", system="system")
    second = llm.generate("different prompt", system=None)
    assert first == second
    assert "citations" in first.lower()
