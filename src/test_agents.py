from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import ProviderConfig, invoke_with_retry


def make_config(tmp_path: Path):
    """Student TODO: build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    config = load_config(tmp_path)
    config.compact_threshold_tokens = 100
    config.compact_keep_messages = 2
    return config

    raise NotImplementedError


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text("dungct", "# User Profile\n- location: Đà Nẵng")

    assert path.name == "User.md"
    assert "Đà Nẵng" in store.read_text("dungct")
    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert store.edit_text("dungct", "không tồn tại", "x") is False
    assert store.facts("dungct")["location"] == "Huế"
    assert store.file_size("dungct") > 0
    return None

    raise NotImplementedError


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    manager = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    for index in range(8):
        manager.append("thread", "user", f"Turn {index}: " + "nội dung dài " * 12)

    context = manager.context("thread")
    assert manager.compaction_count("thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= 2
    return None

    raise NotImplementedError


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    fact = "Mình tên là DũngCT và hiện tại mình đang ở Huế."
    question = "Sang thread mới, nhắc lại giúp mình tên và nơi ở hiện tại?"

    baseline.reply("dungct", "baseline-old", fact)
    advanced.reply("dungct", "advanced-old", fact)
    baseline_answer = baseline.reply("dungct", "baseline-new", question)["response"]
    advanced_answer = advanced.reply("dungct", "advanced-new", question)["response"]

    assert "DũngCT" not in baseline_answer
    assert "Huế" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "Huế" in advanced_answer
    return None

    raise NotImplementedError


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for index in range(14):
        message = f"Đây là turn {index}. " + ("Thông tin nhiễu cho stress test. " * 25)
        baseline.reply("stress", "long-baseline", message)
        advanced.reply("stress", "long-advanced", message)

    assert advanced.compaction_count("long-advanced") > 0
    assert advanced.prompt_token_usage("long-advanced") < baseline.prompt_token_usage("long-baseline")
    return None

    raise NotImplementedError


def test_structured_extraction_handles_corrections_and_noise() -> None:
    correction = extract_profile_updates(
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer. "
        "Nơi ở đã cập nhật từ Huế sang Đà Nẵng."
    )
    noise = extract_profile_updates(
        "Hà Nội chỉ là nơi mình đi họp, còn product manager chỉ là câu đùa."
    )
    recall_question = extract_profile_updates(
        "Tên mình là gì và mình thích kiểu trả lời như thế nào?"
    )

    assert correction["profession"] == "MLOps engineer"
    assert correction["location"] == "Đà Nẵng"
    assert "profession" not in noise
    assert "location" not in noise
    assert "name" not in recall_question


def test_llm_retry_uses_exponential_backoff() -> None:
    class RateLimitError(Exception):
        status_code = 429

    class FlakyModel:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls < 3:
                raise RateLimitError("free-tier rate limit")
            return "ok"

    delays: list[float] = []
    model = FlakyModel()
    config = ProviderConfig(
        provider="openai",
        model_name="test",
        temperature=0,
        max_retries=3,
        retry_initial_delay=1,
        retry_max_delay=10,
    )
    result = invoke_with_retry(
        model,
        [],
        config,
        sleep=delays.append,
        jitter=lambda low, high: (low + high) / 2,
    )

    assert result == "ok"
    assert model.calls == 3
    assert delays == [1.0, 2.0]
