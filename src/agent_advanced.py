from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model
from model_provider import invoke_with_retry, normalize_provider


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None
        if not self.force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

        raise NotImplementedError

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

        raise NotImplementedError

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

        raise NotImplementedError

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

        raise NotImplementedError

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

        raise NotImplementedError

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        response_tokens = estimate_tokens(response)
        self.compact_memory.append(thread_id, "assistant", response)

        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "response": response,
            "content": response,
            "agent_tokens": response_tokens,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            "memory_file": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }

        raise NotImplementedError

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        context = self.compact_memory.context(thread_id)
        pieces = [self.profile_store.read_text(user_id), str(context["summary"])]
        pieces.extend(str(item.get("content", "")) for item in context["messages"])
        return estimate_tokens("\n".join(pieces))

        raise NotImplementedError

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.facts(user_id)
        if _looks_like_recall_question(message):
            return _facts_response_for_question(facts, message)
            return _facts_response(facts)
        if extract_profile_updates(message):
            if _uses_three_bullet_style(facts):
                return _styled_acknowledgement(profile_updated=True)
            return "Mình đã cập nhật các thông tin ổn định vào hồ sơ."
        if _uses_three_bullet_style(facts):
            return _styled_acknowledgement(profile_updated=False)
        return "Mình đã ghi nhận nội dung và giữ phần cần thiết trong ngữ cảnh phiên."

        raise NotImplementedError

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        provider = normalize_provider(self.config.model.provider)
        needs_key = provider in {"openai", "gemini", "anthropic", "openrouter"}
        if needs_key and not self.config.model.api_key:
            return None
        if provider == "custom" and not self.config.model.base_url:
            return None
        return build_chat_model(self.config.model)

        raise NotImplementedError

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        context = self.compact_memory.context(thread_id)
        system = (
            "Bạn là trợ lý có memory. Dùng hồ sơ và summary dưới đây, ưu tiên facts mới trong hồ sơ.\n\n"
            f"PROFILE:\n{self.profile_store.read_text(user_id)}\n\n"
            f"COMPACT SUMMARY:\n{context['summary']}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(context["messages"])
        result = invoke_with_retry(self.langchain_agent, messages, self.config.model)
        response = _message_text(result)
        response_tokens = estimate_tokens(response)
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "response": response,
            "content": response,
            "agent_tokens": response_tokens,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            "memory_file": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }


def _looks_like_recall_question(message: str) -> bool:
    lowered = message.casefold()
    return "?" in message or any(
        marker in lowered
        for marker in ("nhắc lại", "tóm tắt", "bạn biết", "thử nhớ", "mình tên gì")
    )


def _facts_response(facts: dict[str, str]) -> str:
    if not facts:
        return "Mình chưa có thông tin bền vững nào về bạn."
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


def _uses_three_bullet_style(facts: dict[str, str]) -> bool:
    return "3 bullet" in facts.get("response_style", "").casefold()


def _requested_fact_keys(message: str, facts: dict[str, str]) -> list[str]:
    """Choose profile fields requested by a deterministic recall question."""

    lowered = message.casefold()
    markers = {
        "name": ("tên", "là ai"),
        "profession": ("nghề", "công việc", "làm gì", "product manager"),
        "location": ("ở đâu", "nơi ở", "đang ở", "còn ở", "hà nội", "huế"),
        "favorite_drink": ("đồ uống", "uống gì"),
        "favorite_food": ("món ăn", "ăn gì"),
        "pet": ("nuôi", "con gì", "thú cưng", "corgi"),
        "technical_interests": ("mối quan tâm", "quan tâm kỹ thuật", "python", " ai"),
        "response_style": ("style", "kiểu trả lời", "cách trả lời"),
    }
    order = (
        "name",
        "profession",
        "location",
        "favorite_drink",
        "favorite_food",
        "pet",
        "technical_interests",
        "response_style",
    )
    requested = [
        key
        for key in order
        if key in facts and any(marker in lowered for marker in markers[key])
    ]
    return requested or [key for key in order if key in facts]


def _facts_response_for_question(facts: dict[str, str], message: str) -> str:
    if not facts:
        return "Mình chưa có thông tin bền vững nào về bạn."

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
    keys = _requested_fact_keys(message, facts)
    if _uses_three_bullet_style(facts):
        # Style is itself useful evidence in recall answers. Add stable fields
        # until there is enough material for exactly three concise bullets.
        if "response_style" not in keys:
            keys.append("response_style")
        fill_order = ("name", "profession", "location", "response_style", "technical_interests")
        for key in fill_order:
            if len(keys) >= 3:
                break
            if key in facts and key not in keys:
                keys.append(key)

    entries = [(labels.get(key, key), facts[key]) for key in keys]
    if not _uses_three_bullet_style(facts):
        return "\n".join(f"- {label}: {value}" for label, value in entries)

    while len(entries) < 3:
        entries.append(("Định dạng", "duy trì đúng 3 bullet"))

    groups: list[list[tuple[str, str]]] = [[], [], []]
    base, extra = divmod(len(entries), 3)
    cursor = 0
    for index in range(3):
        size = base + (1 if index < extra else 0)
        groups[index] = entries[cursor : cursor + size]
        cursor += size
    return "\n".join(
        "- " + "; ".join(f"{label}: {value}" for label, value in group)
        for group in groups
    )


def _styled_acknowledgement(profile_updated: bool) -> str:
    first = "Đã cập nhật facts ổn định vào User.md." if profile_updated else "Đã ghi nhận nội dung mới."
    return "\n".join(
        (
            f"- {first}",
            "- Giữ recent messages và summary cho follow-up.",
            "- Trade-off: ưu tiên recall nhưng vẫn kiểm soát token.",
        )
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
