import pytest
from pydantic import ValidationError

from slm_trainer_assistant.schemas import EvalExample, TrainingExample


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
