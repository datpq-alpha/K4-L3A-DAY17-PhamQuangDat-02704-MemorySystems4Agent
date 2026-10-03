from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 5
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 30.0


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""

    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "open-ai": "openai",
        "open_router": "openrouter",
        "open-router": "openrouter",
        "openai-compatible": "custom",
    }
    provider = aliases.get(value.strip().lower(), value.strip().lower())
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if provider not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported provider {value!r}. Choose one of: {choices}.")
    return provider

    raise NotImplementedError


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    common: dict[str, Any] = {
        "model": config.model_name,
        "temperature": config.temperature,
    }

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=config.api_key, **common)
    if provider == "custom":
        if not config.base_url:
            raise ValueError("CUSTOM_BASE_URL is required for provider 'custom'.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=config.api_key or "not-required", base_url=config.base_url, **common)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(google_api_key=config.api_key, **common)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(api_key=config.api_key, **common)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if config.base_url:
            common["base_url"] = config.base_url
        return ChatOllama(**common)
    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(api_key=config.api_key, **common)
        except ImportError:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
                **common,
            )

    raise NotImplementedError


def _retry_after_seconds(error: Exception) -> float | None:
    """Read a provider Retry-After header without depending on a specific SDK."""

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def is_retryable_error(error: Exception) -> bool:
    """Return True for rate limits and temporary provider/network failures."""

    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    name = type(error).__name__.lower()
    message = str(error).lower()
    retry_markers = (
        "ratelimit",
        "rate limit",
        "resourceexhausted",
        "resource exhausted",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connectionerror",
        "connection error",
        "overloaded",
    )
    return any(marker in name or marker in message for marker in retry_markers)


def invoke_with_retry(
    model: Any,
    messages: Any,
    config: ProviderConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
):
    """Invoke a chat model with bounded exponential backoff and jitter.

    `max_retries` is the number of retries after the first request. Permanent
    authentication/validation failures are raised immediately.
    """

    retries = max(0, config.max_retries)
    base_delay = max(0.0, config.retry_initial_delay)
    max_delay = max(base_delay, config.retry_max_delay)

    for attempt in range(retries + 1):
        try:
            return model.invoke(messages)
        except Exception as error:
            if attempt >= retries or not is_retryable_error(error):
                raise
            exponential = min(max_delay, base_delay * (2**attempt))
            retry_after = _retry_after_seconds(error)
            delay = retry_after if retry_after is not None else jitter(exponential * 0.75, exponential * 1.25)
            sleep(min(max_delay, max(0.0, delay)))
