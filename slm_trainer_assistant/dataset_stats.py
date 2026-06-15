"""Small dataset summary helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from slm_trainer_assistant.dataset_validator import validate_jsonl_file
from slm_trainer_assistant.schemas import EvalExample, TrainingExample


@dataclass(frozen=True)
class DatasetStats:
    total: int
    example_types: Counter[str]
    categories: Counter[str]
    difficulties: Counter[str]
    source_types: Counter[str]


def collect_stats(path: str | Path) -> DatasetStats:
    result = validate_jsonl_file(path)
    if not result.is_valid:
        issue_text = "; ".join(
            f"line {issue.line_number}: {issue.message}" for issue in result.issues
        )
        raise ValueError(f"cannot collect stats for invalid dataset: {issue_text}")

    example_types: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    source_types: Counter[str] = Counter()

    for example in result.examples:
        if isinstance(example, TrainingExample):
            example_types["training"] += 1
            source_types[example.source_type] += 1
        elif isinstance(example, EvalExample):
            example_types["eval"] += 1
        categories[example.category] += 1
        difficulties[example.difficulty] += 1

    return DatasetStats(
        total=len(result.examples),
        example_types=example_types,
        categories=categories,
        difficulties=difficulties,
        source_types=source_types,
    )


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "  none"
    return "\n".join(f"  {key}: {value}" for key, value in sorted(counter.items()))


def format_stats(stats: DatasetStats) -> str:
    return "\n".join(
        [
            f"total: {stats.total}",
            "types:",
            format_counter(stats.example_types),
            "categories:",
            format_counter(stats.categories),
            "difficulties:",
            format_counter(stats.difficulties),
            "source_types:",
            format_counter(stats.source_types),
        ]
    )
