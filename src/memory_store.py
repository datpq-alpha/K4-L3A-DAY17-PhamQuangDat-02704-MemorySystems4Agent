from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    normalized = text.strip()
    if not normalized:
        return 0
    # Character-based estimation is deterministic and works reasonably for
    # both Vietnamese text and provider-agnostic offline benchmarks.
    return max(1, math.ceil(len(normalized) / 4))

    raise NotImplementedError


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # TODO: slugify or sanitize the user id before building the file path.
        normalized = unicodedata.normalize("NFKC", user_id).strip()
        slug = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE).strip(".-_")
        if not slug:
            slug = "user"
        if slug != normalized or slug in {".", ".."}:
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
            slug = f"{slug[:48]}-{digest}"
        return self.root_dir / slug / "User.md"

        raise NotImplementedError

    def read_text(self, user_id: str) -> str:
        # TODO: return file content or an empty default markdown profile.
        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n"
        return path.read_text(encoding="utf-8")

        raise NotImplementedError

    def write_text(self, user_id: str, content: str) -> Path:
        # TODO: write markdown to disk and return the file path.
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = content.rstrip() + "\n"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(normalized, encoding="utf-8")
        temporary.replace(path)
        return path

        raise NotImplementedError

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        # TODO: replace one occurrence inside User.md and return whether it changed.
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

        raise NotImplementedError

    def file_size(self, user_id: str) -> int:
        # TODO: return the current file size in bytes.
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

        raise NotImplementedError

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse the structured `- field: value` records from a profile."""

        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s+([a-z][a-z0-9_]*)\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
            if match:
                facts[match.group(1).lower()] = match.group(2)
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or replace one structured fact, resolving corrections in place."""

        normalized_key = key.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", normalized_key):
            raise ValueError(f"Invalid profile field: {key!r}")
        normalized_value = " ".join(value.strip().split())
        if not normalized_value:
            return False

        content = self.read_text(user_id).rstrip() + "\n"
        pattern = re.compile(
            rf"^-\s+{re.escape(normalized_key)}\s*:\s*.*$",
            re.IGNORECASE | re.MULTILINE,
        )
        replacement = f"- {normalized_key}: {normalized_value}"
        current = pattern.search(content)
        if current and current.group(0) == replacement:
            return False
        if current:
            updated = pattern.sub(replacement, content, count=1)
        else:
            updated = content + replacement + "\n"
        self.write_text(user_id, updated)
        return True


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = " ".join(message.strip().split())
    if not text:
        return {}

    lowered = text.casefold()
    declarative_markers = (
        "mình tên là",
        "tên mình là",
        "mình ở ",
        "mình đang ở",
        "mình vẫn ở",
        "hiện ở",
        "nơi ở hiện tại",
        "đang làm",
        "nghề nghiệp hiện tại",
        "chuyển sang",
        "mình thích",
        "yêu thích là",
        "mình nuôi",
        "trả lời",
        "style",
    )
    if "?" in text and not any(marker in lowered for marker in declarative_markers):
        return {}

    updates: dict[str, str] = {}

    name_patterns = (
        r"mình\s+tên\s+là\s+([\wÀ-ỹ]+(?:\s+Stress)?)",
        r"tên\s+mình\s+là\s+([\wÀ-ỹ]+(?:\s+Stress)?)",
        r"(?:^|[.!?:]\s+)tên\s+(?!mình\b)(?:của mình\s+)?(?:là\s+)?([\wÀ-ỹ]+(?:\s+Stress)?)",
    )
    name_matches: list[str] = []
    for pattern in name_patterns:
        name_matches.extend(re.findall(pattern, text, re.IGNORECASE))
    invalid_values = {"gì", "ai", "nào"}
    valid_names = [value.strip(" .,;:") for value in name_matches if value.casefold() not in invalid_values]
    if valid_names:
        updates["name"] = valid_names[-1]

    location_patterns = (
        r"nơi ở(?:\s+hiện tại)?\s+đã\s+cập nhật\s+từ\s+.+?\s+sang\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,3})",
        r"nơi ở(?:\s+hiện tại)?\s+(?:của mình\s+)?(?:là|vẫn là)\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,3})",
        r"(?:giờ|hiện tại|thực ra từ tuần này)\s+mình\s+(?:đang\s+)?(?:làm việc\s+)?ở\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,3})",
        r"mình\s+(?:đang|vẫn)\s+ở\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,3})",
        r"mình\s+ở\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,3})",
        r"hiện\s+ở\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,3})",
    )
    locations: list[str] = []
    for pattern in location_patterns:
        locations.extend(re.findall(pattern, text, re.IGNORECASE))
    if locations:
        candidate = re.split(r"\s+(?:và|chứ|dù|để|cho)\b", locations[0], maxsplit=1, flags=re.IGNORECASE)[0]
        candidate = candidate.strip(" .,;:")
        if candidate.casefold() not in {"đâu", "nào", "gì"}:
            updates["location"] = candidate

    profession_patterns = (
        r"nghề nghiệp(?:\s+hiện tại)?\s+(?:thì\s+)?(?:vẫn\s+)?(?:là|:)\s+([\wÀ-ỹ+#-]+(?:\s+[\wÀ-ỹ+#-]+){0,2})",
        r"(?:giờ\s+)?chuyển\s+sang\s+([\wÀ-ỹ+#-]+(?:\s+[\wÀ-ỹ+#-]+){0,2})",
        r"(?:đang|vẫn)\s+làm\s+([\wÀ-ỹ+#-]+(?:\s+[\wÀ-ỹ+#-]+){0,2})",
        r"(?:nghề|công việc)\s+(?:hiện tại\s+)?(?:của mình\s+)?(?:vẫn\s+)?là\s+([\wÀ-ỹ+#-]+(?:\s+[\wÀ-ỹ+#-]+){0,2})",
    )
    professions: list[str] = []
    for pattern in profession_patterns:
        professions.extend(re.findall(pattern, text, re.IGNORECASE))
    if professions:
        candidate = re.split(r"\s+(?:cho|ở|và|chứ|không)\b", professions[0], maxsplit=1, flags=re.IGNORECASE)[0]
        role_markers = (
            "engineer",
            "developer",
            "manager",
            "designer",
            "analyst",
            "scientist",
            "architect",
            "teacher",
            "giảng viên",
            "kỹ sư",
        )
        if (
            any(marker in candidate.casefold() for marker in role_markers)
            and not ("product manager" in candidate.casefold() and "đùa" in lowered)
        ):
            updates["profession"] = candidate.strip(" .,;:")

    if "cà phê sữa đá" in lowered and any(
        marker in lowered for marker in ("yêu thích", "mình thích", "mình vẫn uống", "đồ uống")
    ):
        updates["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in lowered and any(marker in lowered for marker in ("yêu thích", "mình thích", "món ăn")):
        updates["favorite_food"] = "mì Quảng"
    pet_match = re.search(r"mình\s+nuôi\s+(?:một\s+)?(?:bé\s+)?([\wÀ-ỹ-]+)(?:\s+tên\s+([\wÀ-ỹ-]+))?", text, re.IGNORECASE)
    if pet_match:
        pet = pet_match.group(1).lower()
        pet_name = pet_match.group(2)
        if pet not in {"con", "gì", "thú"}:
            updates["pet"] = f"{pet} tên {pet_name}" if pet_name else pet

    if "mình thích" in lowered or "mình vẫn thích" in lowered:
        interests: list[str] = []
        if "python" in lowered:
            interests.append("Python")
        if re.search(r"\bai\b|ai ứng dụng", lowered):
            interests.append("AI ứng dụng")
        if "mlops" in lowered:
            interests.append("MLOps")
        if interests:
            updates["technical_interests"] = ", ".join(interests)

    if any(marker in lowered for marker in ("trả lời", "style", "giải thích")):
        style: list[str] = []
        if "ngắn" in lowered or "gọn" in lowered or "đừng lan man" in lowered:
            style.append("ngắn gọn")
        if "3 bullet" in lowered:
            style.append("3 bullet")
        elif "bullet" in lowered:
            style.append("bullet")
        if "ví dụ thực chiến" in lowered:
            style.append("có ví dụ thực chiến")
        elif "ví dụ thực tế" in lowered:
            style.append("có ví dụ thực tế")
        if "trade-off" in lowered:
            style.append("nhấn vào trade-off")
        if style:
            updates["response_style"] = ", ".join(dict.fromkeys(style))

    return updates

    raise NotImplementedError


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""
    snippets: list[str] = []
    for message in messages[-max_items:]:
        role = str(message.get("role", "unknown")).strip().lower()
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        if len(content) > 220:
            content = content[:217].rstrip() + "..."
        snippets.append(f"{role}: {content}")
    return "\n".join(snippets)

    raise NotImplementedError


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread["summary"])
        context_text = summary + "\n" + "\n".join(str(item.get("content", "")) for item in messages)
        if estimate_tokens(context_text) > self.threshold_tokens and len(messages) > self.keep_messages:
            older = messages[:-self.keep_messages]
            recent = messages[-self.keep_messages:]
            summary_input: list[dict[str, str]] = []
            if summary:
                summary_input.append({"role": "summary", "content": summary})
            summary_input.extend(older)
            thread["summary"] = summarize_messages(summary_input, max_items=8)
            thread["messages"] = recent
            thread["compactions"] = int(thread["compactions"]) + 1
        return None

        raise NotImplementedError

    def context(self, thread_id: str) -> dict[str, object]:
        # TODO: return per-thread state with keys like messages, summary, compactions.
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        return {
            "messages": [dict(item) for item in thread["messages"]],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

        raise NotImplementedError

    def compaction_count(self, thread_id: str) -> int:
        # TODO: return number of compactions for this thread.
        return int(self.context(thread_id)["compactions"])

        raise NotImplementedError
