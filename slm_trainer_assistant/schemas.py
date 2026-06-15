"""Framework-independent dataset schemas.

The schema intentionally models data before any trainer-specific prompt rendering.
This keeps evals and datasets reusable across Unsloth, TRL + PEFT, and future backends.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant"]
Difficulty = Literal["beginner", "intermediate", "advanced"]
SourceType = Literal["manual", "synthetic", "public_doc_derived", "user_notes", "rejected"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = Field(min_length=1)


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=2)
    category: str = Field(min_length=1)
    difficulty: Difficulty
    source_type: SourceType
    source_ref: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def require_user_and_assistant(self) -> TrainingExample:
        roles = {message.role for message in self.messages}
        if "user" not in roles or "assistant" not in roles:
            raise ValueError(
                "training examples must include at least one user and assistant message"
            )
        return self


class EvalExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    difficulty: Difficulty
    question: str = Field(min_length=1)
    expected_traits: list[str] = Field(min_length=1)
    anti_traits: list[str] = Field(default_factory=list)


DatasetExample = TrainingExample | EvalExample


def parse_dataset_example(payload: dict) -> DatasetExample:
    """Parse a JSON object as either a training example or an eval example."""

    if "messages" in payload:
        return TrainingExample.model_validate(payload)
    if "question" in payload:
        return EvalExample.model_validate(payload)
    raise ValueError("example must include either 'messages' for training or 'question' for eval")
