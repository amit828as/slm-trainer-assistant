"""JSONL dataset validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from slm_trainer_assistant.schemas import DatasetExample, parse_dataset_example


@dataclass(frozen=True)
class ValidationIssue:
    line_number: int
    message: str


@dataclass
class ValidationResult:
    path: Path
    examples: list[DatasetExample] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _format_validation_error(error: ValidationError) -> str:
    parts: list[str] = []
    for detail in error.errors():
        location = ".".join(str(item) for item in detail["loc"])
        parts.append(f"{location}: {detail['msg']}")
    return "; ".join(parts)


def validate_record(payload: dict[str, Any]) -> DatasetExample:
    return parse_dataset_example(payload)


def validate_jsonl_file(path: str | Path) -> ValidationResult:
    dataset_path = Path(path)
    result = ValidationResult(path=dataset_path)

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                result.issues.append(
                    ValidationIssue(line_number, f"invalid JSON: {exc.msg}")
                )
                continue

            if not isinstance(payload, dict):
                result.issues.append(
                    ValidationIssue(line_number, "line must contain a JSON object")
                )
                continue

            try:
                result.examples.append(validate_record(payload))
            except ValidationError as exc:
                result.issues.append(ValidationIssue(line_number, _format_validation_error(exc)))
            except ValueError as exc:
                result.issues.append(ValidationIssue(line_number, str(exc)))

    return result
