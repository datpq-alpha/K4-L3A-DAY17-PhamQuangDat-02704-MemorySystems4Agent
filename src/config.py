from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # TODO: read env vars for one of the supported providers.
    # Example knobs:
    # - LLM_PROVIDER / LLM_MODEL
    # - OPENAI_API_KEY
    # - GEMINI_API_KEY
    # - ANTHROPIC_API_KEY
    # - OLLAMA_BASE_URL
    # - OPENROUTER_API_KEY
    # - CUSTOM_BASE_URL / CUSTOM_API_KEY
    # TODO: create `root / "state"`.
    # TODO: choose sensible defaults for compact memory.

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        # Offline mode and tests do not require python-dotenv.
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    def env_int(name: str, default: int, minimum: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return max(minimum, int(raw))
        except ValueError as error:
            raise ValueError(f"{name} must be an integer, got {raw!r}.") from error

    def env_float(name: str, default: float, minimum: float = 0.0) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return max(minimum, float(raw))
        except ValueError as error:
            raise ValueError(f"{name} must be a number, got {raw!r}.") from error

    def provider_config(prefix: str, defaults: tuple[str, str]) -> ProviderConfig:
        provider = os.getenv(f"{prefix}_PROVIDER", defaults[0])
        normalized = provider.strip().lower()
        key_names = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "google": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "anthorpic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "custom": "CUSTOM_API_KEY",
        }
        base_names = {
            "ollama": "OLLAMA_BASE_URL",
            "custom": "CUSTOM_BASE_URL",
            "openrouter": "OPENROUTER_BASE_URL",
        }
        return ProviderConfig(
            provider=provider,
            model_name=os.getenv(f"{prefix}_MODEL", defaults[1]),
            temperature=env_float(f"{prefix}_TEMPERATURE", 0.0),
            api_key=os.getenv(key_names.get(normalized, "")) if normalized in key_names else None,
            base_url=os.getenv(base_names.get(normalized, "")) if normalized in base_names else None,
            max_retries=env_int("LLM_MAX_RETRIES", 5, 0),
            retry_initial_delay=env_float("LLM_RETRY_INITIAL_DELAY", 1.0),
            retry_max_delay=env_float("LLM_RETRY_MAX_DELAY", 30.0),
        )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=env_int("COMPACT_THRESHOLD_TOKENS", 1200, 32),
        compact_keep_messages=env_int("COMPACT_KEEP_MESSAGES", 6, 1),
        model=provider_config("LLM", ("openai", "gpt-4o-mini")),
        judge_model=provider_config("JUDGE", ("openai", "gpt-4o-mini")),
    )

    raise NotImplementedError("Students should implement load_config().")
