from __future__ import annotations

from dataclasses import dataclass
import json
import tempfile
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Benchmark file must contain a JSON list: {path}")
    required = {"id", "user_id", "turns", "recall_questions"}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not required.issubset(item):
            missing = required.difference(item if isinstance(item, dict) else {})
            raise ValueError(f"Invalid conversation at index {index}; missing {sorted(missing)}")
    return data

    raise NotImplementedError


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0
    normalized = answer.casefold()
    matches = sum(1 for fact in expected if fact.casefold() in normalized)
    ratio = matches / len(expected)
    if ratio == 0:
        return 0.0
    if ratio == 1:
        return 1.0
    return 0.5

    raise NotImplementedError


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""

    if not answer.strip():
        return 0.0
    normalized = answer.casefold()
    coverage = 1.0 if not expected else sum(
        1 for fact in expected if fact.casefold() in normalized
    ) / len(expected)
    presentation = 1.0 if len(answer) <= 700 and ("- " in answer or len(answer) <= 220) else 0.5
    return round(0.85 * coverage + 0.15 * presentation, 3)

    raise NotImplementedError


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    thread_ids: set[str] = set()
    user_ids = {str(item["user_id"]) for item in conversations}
    before_sizes = {
        user_id: agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0
        for user_id in user_ids
    }

    for conversation in conversations:
        thread_id = f"benchmark:{conversation['id']}:conversation"
        thread_ids.add(thread_id)
        for turn in conversation["turns"]:
            agent.reply(str(conversation["user_id"]), thread_id, str(turn))

    recall_scores: list[float] = []
    quality_scores: list[float] = []
    for conversation in conversations:
        for index, recall in enumerate(conversation["recall_questions"]):
            thread_id = f"benchmark:{conversation['id']}:recall:{index}"
            thread_ids.add(thread_id)
            result = agent.reply(
                str(conversation["user_id"]),
                thread_id,
                str(recall["question"]),
            )
            answer = str(result["response"])
            expected = [str(value) for value in recall["expected_contains"]]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    after_sizes = {
        user_id: agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0
        for user_id in user_ids
    }
    memory_growth = sum(max(0, after_sizes[user_id] - before_sizes[user_id]) for user_id in user_ids)
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids),
        recall_score=round(average_recall, 3),
        response_quality=round(average_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )

    raise NotImplementedError


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]

    def line(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([line(headers), separator, *(line(row) for row in values)])

    raise NotImplementedError


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    # TODO:
    # - load both datasets from root/data
    # - initialize baseline and advanced agents
    # - run benchmarks
    # - print comparison tables
    suites = (
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    )
    with tempfile.TemporaryDirectory(
        prefix="day17-benchmark-",
        dir=config.state_dir,
        ignore_cleanup_errors=True,
    ) as temporary:
        temporary_root = Path(temporary)
        for suite_index, (title, path) in enumerate(suites):
            conversations = load_conversations(path)
            baseline_config = dataclass_replace(config, state_dir=temporary_root / f"{suite_index}-baseline")
            advanced_config = dataclass_replace(config, state_dir=temporary_root / f"{suite_index}-advanced")
            rows = [
                run_agent_benchmark(
                    "Baseline",
                    BaselineAgent(baseline_config, force_offline=True),
                    conversations,
                    baseline_config,
                ),
                run_agent_benchmark(
                    "Advanced",
                    AdvancedAgent(advanced_config, force_offline=True),
                    conversations,
                    advanced_config,
                ),
            ]
            print(f"\n## {title}\n")
            print(format_rows(rows))
    return None

    raise NotImplementedError


if __name__ == "__main__":
    main()
