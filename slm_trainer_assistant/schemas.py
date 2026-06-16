"""Framework-independent dataset schemas.

The schema intentionally models data before any trainer-specific prompt rendering.
This keeps evals and datasets reusable across Unsloth, TRL + PEFT, and future backends.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant"]
Difficulty = Literal["beginner", "intermediate", "advanced"]
SourceType = Literal["manual", "synthetic", "public_doc_derived", "user_notes", "rejected"]
MediaType = Literal["image"]
IMAGE_MEDIA_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


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


class EvalMedia(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: MediaType
    path: str = Field(min_length=1)
    description: str | None = None

    @field_validator("path")
    @classmethod
    def require_safe_repo_relative_image_path(cls, value: str) -> str:
        normalized = value.strip()
        path = PurePosixPath(normalized)
        if not normalized or path.is_absolute():
            raise ValueError("media path must be repo-relative")
        if "://" in normalized or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("media path must not contain empty, current, or parent segments")
        if path.suffix.lower() not in IMAGE_MEDIA_EXTENSIONS:
            allowed = ", ".join(sorted(IMAGE_MEDIA_EXTENSIONS))
            raise ValueError(f"image media path must end with one of: {allowed}")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class EvalExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    difficulty: Difficulty
    question: str = Field(min_length=1)
    media: list[EvalMedia] = Field(default_factory=list)
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
