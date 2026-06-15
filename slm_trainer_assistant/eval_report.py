"""Structured baseline evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from slm_trainer_assistant.schemas import Difficulty


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    difficulty: Difficulty
    question: str = Field(min_length=1)
    response: str = Field(min_length=1)
    expected_traits: list[str] = Field(min_length=1)
    anti_traits: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    backend_name: str = Field(min_length=1)
    eval_file: str = Field(min_length=1)
    total_questions: int = Field(ge=0)
    results: list[EvalResult]


def write_report(report: EvalReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
