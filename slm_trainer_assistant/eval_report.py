"""Structured baseline evaluation reports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    human_score: int | None = Field(default=None, ge=1, le=5)
    matched_traits: list[str] = Field(default_factory=list)
    missed_traits: list[str] = Field(default_factory=list)
    triggered_anti_traits: list[str] = Field(default_factory=list)
    failure_type: str | None = None
    review_notes: str | None = None

    @field_validator("failure_type")
    @classmethod
    def normalize_failure_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("review_notes")
    @classmethod
    def normalize_review_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    backend_name: str = Field(min_length=1)
    eval_file: str = Field(min_length=1)
    total_questions: int = Field(ge=0)
    results: list[EvalResult]


@dataclass(frozen=True)
class ReportSummary:
    total_questions: int
    reviewed: int
    average_score: float | None
    failure_types: Counter[str]


def load_report(path: str | Path) -> EvalReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalReport.model_validate(payload)


def write_report(report: EvalReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def summarize_report(report: EvalReport) -> ReportSummary:
    scored_results = [result for result in report.results if result.human_score is not None]
    failure_types: Counter[str] = Counter(
        result.failure_type for result in report.results if result.failure_type is not None
    )
    average_score = None
    if scored_results:
        average_score = sum(result.human_score for result in scored_results) / len(scored_results)

    return ReportSummary(
        total_questions=report.total_questions,
        reviewed=len(scored_results),
        average_score=average_score,
        failure_types=failure_types,
    )


def format_report_summary(summary: ReportSummary) -> str:
    average = "n/a" if summary.average_score is None else f"{summary.average_score:.1f}"
    failure_lines = ["Failure types:"]
    if summary.failure_types:
        failure_lines.extend(
            f"- {failure_type}: {count}"
            for failure_type, count in sorted(summary.failure_types.items())
        )
    else:
        failure_lines.append("- none: 0")

    return "\n".join(
        [
            f"Total questions: {summary.total_questions}",
            f"Reviewed: {summary.reviewed}",
            f"Average score: {average} / 5",
            "",
            *failure_lines,
        ]
    )
