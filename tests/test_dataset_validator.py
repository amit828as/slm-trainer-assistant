from pathlib import Path

from slm_trainer_assistant.dataset_stats import collect_stats
from slm_trainer_assistant.dataset_validator import validate_jsonl_file


def test_validator_catches_bad_jsonl_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(
        '{"id": "eval-001", "category": "debugging", "difficulty": "beginner", '
        '"question": "What broke?", "expected_traits": ["asks for logs"]}\n'
        '{"id": "broken"\n',
        encoding="utf-8",
    )

    result = validate_jsonl_file(dataset_path)

    assert not result.is_valid
    assert result.issues[0].line_number == 2
    assert "invalid JSON" in result.issues[0].message


def test_stats_counts_categories_correctly(tmp_path: Path) -> None:
    dataset_path = tmp_path / "evals.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id": "eval-001", "category": "debugging", "difficulty": "beginner", '
                    '"question": "What broke?", "expected_traits": ["asks for logs"]}'
                ),
                (
                    '{"id": "eval-002", "category": "debugging", "difficulty": "intermediate", '
                    '"question": "Loss down, quality down?", "expected_traits": ["checks eval"]}'
                ),
                (
                    '{"id": "eval-003", "category": "lora", "difficulty": "beginner", '
                    '"question": "What is LoRA?", "expected_traits": ["explains adapters"]}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = collect_stats(dataset_path)

    assert stats.total == 3
    assert stats.categories["debugging"] == 2
    assert stats.categories["lora"] == 1
    assert stats.difficulties["beginner"] == 2


def test_validator_accepts_image_eval_media(tmp_path: Path) -> None:
    dataset_path = tmp_path / "image-evals.jsonl"
    dataset_path.write_text(
        (
            '{"id": "image-001", "category": "debugging", "difficulty": "intermediate", '
            '"question": "What does the chart show?", '
            '"media": [{"type": "image", "path": "evals/media/debugging_loss_curve.png"}], '
            '"expected_traits": ["reads the chart"]}\n'
        ),
        encoding="utf-8",
    )

    result = validate_jsonl_file(dataset_path)

    assert result.is_valid
    assert len(result.examples) == 1
