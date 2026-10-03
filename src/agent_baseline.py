from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from memory_store import extract_profile_updates
from model_provider import build_chat_model
from model_provider import invoke_with_retry, normalize_provider


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None
        if not self.force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

        raise NotImplementedError

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self.sessions.get(thread_id, SessionState()).token_usage

        raise NotImplementedError

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

        raise NotImplementedError

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(
            "\n".join(item["content"] for item in session.messages)
        )
        session.prompt_tokens_processed += prompt_tokens

        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        if _looks_like_recall_question(message):
            response = _facts_response(facts)
        else:
            response = "Mình đã tiếp nhận thông tin trong phiên hiện tại."

        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "content": response,
            "agent_tokens": response_tokens,
            "token_usage": session.token_usage,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

        raise NotImplementedError

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        provider = normalize_provider(self.config.model.provider)
        needs_key = provider in {"openai", "gemini", "anthropic", "openrouter"}
        if needs_key and not self.config.model.api_key:
            return None
        if provider == "custom" and not self.config.model.base_url:
            return None
        return build_chat_model(self.config.model)

        raise NotImplementedError

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(
            "\n".join(item["content"] for item in session.messages)
        )
        result = invoke_with_retry(self.langchain_agent, session.messages, self.config.model)
        response = _message_text(result)
        response_tokens = estimate_tokens(response)
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "content": response,
            "agent_tokens": response_tokens,
            "token_usage": session.token_usage,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }


def _looks_like_recall_question(message: str) -> bool:
    lowered = message.casefold()
    return "?" in message or any(
        marker in lowered
        for marker in ("nhắc lại", "tóm tắt", "bạn biết", "thử nhớ", "mình tên gì")
    )


def _facts_response(facts: dict[str, str]) -> str:
    if not facts:
        return "Mình chưa có thông tin đó trong thread hiện tại."
    labels = {
        "name": "Tên",
        "location": "Nơi ở hiện tại",
        "profession": "Nghề nghiệp hiện tại",
        "technical_interests": "Mối quan tâm kỹ thuật",
        "favorite_drink": "Đồ uống yêu thích",
        "favorite_food": "Món ăn yêu thích",
        "pet": "Thú cưng",
        "response_style": "Style trả lời",
    }
    return "\n".join(
        f"- {labels.get(key, key)}: {value}" for key, value in facts.items()
    )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)
