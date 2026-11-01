"""LLM providers for RAG generation."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from typing import Any, Literal, Protocol

import httpx

from app.cooperative_cancel import QueryCancelledError, active_cancel_event, cooperative_check
from app.generation.usage import (
    set_last_llm_usage,
    usage_from_cohere_chat_response,
    usage_from_ollama_chat_payload,
)

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
        set_last_llm_usage(None)
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
        set_last_llm_usage(usage_from_ollama_chat_payload(data))
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
        set_last_llm_usage(None)
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
            last_payload: dict[str, Any] | None = None
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
                if isinstance(data, dict):
                    last_payload = data
                message = data.get("message") if isinstance(data, dict) else None
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content:
                    yield content
                if data.get("done") is True:
                    break
            if last_payload is not None:
                set_last_llm_usage(usage_from_ollama_chat_payload(last_payload))

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
        *,
        cohere_input_usd_per_1m: float | None = None,
        cohere_output_usd_per_1m: float | None = None,
    ) -> None:
        if client is None:
            import cohere

            self._client = cohere.ClientV2(api_key)
        else:
            self._client = client
        self._model = model
        self._cohere_input_usd_per_1m = cohere_input_usd_per_1m
        self._cohere_output_usd_per_1m = cohere_output_usd_per_1m

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        set_last_llm_usage(None)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(model=self._model, messages=messages)
        set_last_llm_usage(
            usage_from_cohere_chat_response(
                response,
                input_usd_per_1m=_cohere_input_usd_per_1m(self),
                output_usd_per_1m=_cohere_output_usd_per_1m(self),
            )
        )
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
    cohere_input_usd_per_1m: float | None = None,
    cohere_output_usd_per_1m: float | None = None,
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
        return CohereLLM(
            api_key=key,
            model=cohere_model,
            cohere_input_usd_per_1m=cohere_input_usd_per_1m,
            cohere_output_usd_per_1m=cohere_output_usd_per_1m,
        )
    raise ValueError(f"Invalid LLM method: {method}")


def _cohere_input_usd_per_1m(llm: CohereLLM) -> float | None:
    return getattr(llm, "_cohere_input_usd_per_1m", None)


def _cohere_output_usd_per_1m(llm: CohereLLM) -> float | None:
    return getattr(llm, "_cohere_output_usd_per_1m", None)
