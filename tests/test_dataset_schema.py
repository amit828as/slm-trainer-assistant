from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_trainer_assistant.schemas import EvalExample, EvalMedia, TrainingExample


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as image_file:
        header = image_file.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def test_valid_training_example() -> None:
    example = TrainingExample.model_validate(
        {
            "id": "example-001",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert assistant for training small language models.",
                },
                {"role": "user", "content": "Should I use LoRA or full fine-tuning?"},
                {"role": "assistant", "content": "Start with LoRA for a first small run."},
            ],
            "category": "lora",
            "difficulty": "beginner",
            "source_type": "manual",
            "source_ref": None,
            "notes": None,
        }
    )

    assert example.id == "example-001"
    assert example.messages[-1].role == "assistant"


def test_invalid_role() -> None:
    with pytest.raises(ValidationError):
        TrainingExample.model_validate(
            {
                "id": "example-002",
                "messages": [
                    {"role": "user", "content": "Can I train on evals?"},
                    {"role": "developer", "content": "No."},
                ],
                "category": "data_policy",
                "difficulty": "beginner",
                "source_type": "manual",
            }
        )


def test_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        TrainingExample.model_validate(
            {
                "id": "example-003",
                "messages": [
                    {"role": "user", "content": "What is QLoRA?"},
                    {"role": "assistant", "content": "It combines quantization with LoRA."},
                ],
                "difficulty": "beginner",
                "source_type": "manual",
            }
        )


def test_valid_eval_example() -> None:
    example = EvalExample.model_validate(
        {
            "id": "eval-001",
            "category": "debugging",
            "difficulty": "intermediate",
            "question": "Training loss falls but eval quality drops. What should I check?",
            "expected_traits": ["mentions overfitting", "checks data quality"],
            "anti_traits": ["recommends more epochs without context"],
        }
    )

    assert example.category == "debugging"
    assert example.expected_traits


def test_valid_eval_example_with_image_media() -> None:
    example = EvalExample.model_validate(
        {
            "id": "eval-image-001",
            "category": "debugging",
            "difficulty": "intermediate",
            "question": "What risk does this chart suggest?",
            "media": [
                {
                    "type": "image",
                    "path": "evals/media/debugging_loss_curve.png",
                    "description": "Synthetic loss and quality trend chart.",
                }
            ],
            "expected_traits": ["mentions visual trend"],
        }
    )

    assert example.media[0].type == "image"
    assert example.media[0].path == "evals/media/debugging_loss_curve.png"


def test_golden_image_eval_fixtures_are_readable() -> None:
    media_dir = Path(__file__).resolve().parents[1] / "evals" / "media"

    for image_path in media_dir.glob("*.png"):
        width, height = _png_dimensions(image_path)

        assert width >= 640, image_path
        assert height >= 360, image_path


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/evals/media/chart.png",
        "../private/chart.png",
        "evals/media/chart.gif",
        "https://example.com/chart.png",
    ],
)
def test_eval_media_rejects_unsafe_or_unsupported_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        EvalMedia.model_validate(
            {
                "type": "image",
                "path": path,
            }
        )
