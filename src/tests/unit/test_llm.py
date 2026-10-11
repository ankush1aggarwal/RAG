from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.generation.llm import CohereLLM, OllamaLLM, create_llm_provider


def _chat_response(content: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"message": {"content": content}}
    return response


def test_ollama_generate_user_only():
    client = MagicMock()
    client.post.return_value = _chat_response("answer")

    llm = OllamaLLM(
        base_url="http://ollama:11434",
        model="mistral:latest",
        client=client,
    )
    result = llm.generate("What is APR?")

    assert result == "answer"
    client.post.assert_called_once_with(
        "http://ollama:11434/api/chat",
        json={
            "model": "mistral:latest",
            "messages": [{"role": "user", "content": "What is APR?"}],
            "stream": False,
        },
    )
    response = client.post.return_value
    response.raise_for_status.assert_called_once()


def test_ollama_generate_includes_system_message():
    client = MagicMock()
    client.post.return_value = _chat_response("ok")

    llm = OllamaLLM(client=client)
    llm.generate("q", system="Use only context.")

    _, kwargs = client.post.call_args
    messages = kwargs["json"]["messages"]
    assert messages == [
        {"role": "system", "content": "Use only context."},
        {"role": "user", "content": "q"},
    ]


def test_ollama_strips_trailing_slash_from_base_url():
    client = MagicMock()
    client.post.return_value = _chat_response("x")

    llm = OllamaLLM(base_url="http://localhost:11434/", client=client)
    llm.generate("hi")

    client.post.assert_called_once()
    assert client.post.call_args[0][0] == "http://localhost:11434/api/chat"


def test_ollama_generate_raises_on_unexpected_response_shape():
    client = MagicMock()
    response = MagicMock()
    response.json.return_value = {"unexpected": True}
    client.post.return_value = response

    llm = OllamaLLM(client=client)
    with pytest.raises(ValueError, match="Unexpected Ollama chat response shape"):
        llm.generate("q")


def test_ollama_generate_raises_when_content_not_string():
    client = MagicMock()
    response = MagicMock()
    response.json.return_value = {"message": {"content": 42}}
    client.post.return_value = response

    llm = OllamaLLM(client=client)
    with pytest.raises(ValueError, match="message content must be a string"):
        llm.generate("q")


def test_ollama_generate_propagates_http_errors():
    client = MagicMock()
    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )
    client.post.return_value = response

    llm = OllamaLLM(client=client)
    with pytest.raises(httpx.HTTPStatusError):
        llm.generate("q")


def test_close_closes_owned_client():
    client_instance = MagicMock(spec=httpx.Client)
    with patch("app.generation.llm.httpx.Client", return_value=client_instance):
        llm = OllamaLLM()
    llm.close()
    client_instance.close.assert_called_once()


def test_close_does_not_close_injected_client():
    client = MagicMock(spec=httpx.Client)
    llm = OllamaLLM(client=client)
    llm.close()
    client.close.assert_not_called()


def test_context_manager_closes_owned_client_on_exit():
    client_instance = MagicMock(spec=httpx.Client)
    with patch("app.generation.llm.httpx.Client", return_value=client_instance):
        with OllamaLLM() as llm:
            assert isinstance(llm, OllamaLLM)
    client_instance.close.assert_called_once()


def test_create_llm_provider_ollama():
    llm = create_llm_provider(
        "ollama",
        base_url="http://localhost:11434",
        model="mistral:latest",
        timeout_seconds=30.0,
    )
    assert isinstance(llm, OllamaLLM)
    llm.close()


def test_create_llm_provider_rejects_unknown_method():
    with pytest.raises(ValueError, match="Invalid LLM method"):
        create_llm_provider("openai")  # type: ignore[arg-type]


def test_cohere_generate_user_only():
    client = MagicMock()
    block = MagicMock()
    block.text = "cohere answer"
    client.chat.return_value = MagicMock(message=MagicMock(content=[block]))

    llm = CohereLLM(api_key="test-key", model="command-r-08-2024", client=client)
    result = llm.generate("What is APR?")

    assert result == "cohere answer"
    client.chat.assert_called_once_with(
        model="command-r-08-2024",
        messages=[{"role": "user", "content": "What is APR?"}],
    )


def test_cohere_generate_includes_system_message():
    client = MagicMock()
    block = MagicMock()
    block.text = "ok"
    client.chat.return_value = MagicMock(message=MagicMock(content=[block]))

    llm = CohereLLM(api_key="test-key", client=client)
    llm.generate("q", system="Use only context.")

    messages = client.chat.call_args.kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "Use only context."},
        {"role": "user", "content": "q"},
    ]


def test_create_llm_provider_cohere_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        create_llm_provider("cohere")


def test_create_llm_provider_cohere():
    llm = create_llm_provider("cohere", cohere_api_key="key", cohere_model="command-r-08-2024")
    assert isinstance(llm, CohereLLM)
