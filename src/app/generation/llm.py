"""LLM providers for RAG generation."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from typing import Any, Literal, Protocol

import httpx

from app.cooperative_cancel import QueryCancelledError, active_cancel_event, cooperative_check

LLMMethod = Literal["ollama", "cohere"]


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, system: str | None = None) -> str: ...


def iter_llm_tokens(
    llm: LLMProvider,
    prompt: str,
    *,
    system: str | None = None,
    cancel: threading.Event | None = None,
) -> Iterator[str]:
    """Stream token deltas when supported; otherwise one chunk from ``generate``."""
    cancel = cancel or active_cancel_event()
    stream = getattr(llm, "stream_generate", None)
    if callable(stream):
        yield from stream(prompt, system=system, cancel=cancel)
        return
    cooperative_check()
    yield llm.generate(prompt, system=system)
    cooperative_check()


class OllamaLLM:
    """Ollama chat API client (POST /api/chat, non-streaming)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral:latest",
        timeout_seconds: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        cooperative_check()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unexpected Ollama chat response shape: {data}") from exc
        if not isinstance(content, str):
            raise ValueError(f"Ollama message content must be a string, got {type(content)}")
        cooperative_check()
        return content

    def stream_generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[str]:
        cancel = cancel or active_cancel_event()
        cooperative_check()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if cancel is not None and cancel.is_set():
                    response.close()
                    raise QueryCancelledError()
                cooperative_check()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = data.get("message") if isinstance(data, dict) else None
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content:
                    yield content
                if data.get("done") is True:
                    break

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaLLM:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class CohereLLM:
    """Cohere chat API client (ClientV2 chat, non-streaming)."""

    def __init__(
        self,
        api_key: str,
        model: str = "command-r-08-2024",
        client: Any | None = None,
    ) -> None:
        if client is None:
            import cohere

            self._client = cohere.ClientV2(api_key)
        else:
            self._client = client
        self._model = model

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(model=self._model, messages=messages)
        try:
            content_blocks = response.message.content
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"Unexpected Cohere chat response shape: {response}") from exc
        if not content_blocks:
            raise ValueError("Cohere chat response contained no content blocks")

        parts: list[str] = []
        for block in content_blocks:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        if not parts:
            raise ValueError(f"Cohere chat response had no text content: {response}")
        return "".join(parts)


def create_llm_provider(
    method: LLMMethod,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "mistral:latest",
    timeout_seconds: float = 120.0,
    cohere_api_key: str | None = None,
    cohere_model: str = "command-r-08-2024",
) -> LLMProvider:
    if method == "ollama":
        return OllamaLLM(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    if method == "cohere":
        key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY is required for cohere LLM provider")
        return CohereLLM(api_key=key, model=cohere_model)
    raise ValueError(f"Invalid LLM method: {method}")
