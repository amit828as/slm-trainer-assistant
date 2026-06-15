"""Baseline eval runner.

This module collects model responses for golden eval questions. It does not grade
answers yet; early reports are meant for human review and later scoring design.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from slm_trainer_assistant.dataset_validator import validate_jsonl_file
from slm_trainer_assistant.eval_report import EvalReport, EvalResult
from slm_trainer_assistant.model_backends import ModelBackend
from slm_trainer_assistant.schemas import EvalExample


def load_eval_examples(eval_file: str | Path) -> list[EvalExample]:
    result = validate_jsonl_file(eval_file)
    if not result.is_valid:
        issue_text = "; ".join(
            f"line {issue.line_number}: {issue.message}" for issue in result.issues
        )
        raise ValueError(f"invalid eval file: {issue_text}")

    eval_examples: list[EvalExample] = []
    for example in result.examples:
        if not isinstance(example, EvalExample):
            raise ValueError("baseline eval files must contain only eval examples")
        eval_examples.append(example)
    return eval_examples


def run_baseline_eval(
    eval_file: str | Path,
    backend: ModelBackend,
    *,
    run_id: str | None = None,
    created_at: str | None = None,
) -> EvalReport:
    examples = load_eval_examples(eval_file)
    report_results = [
        EvalResult(
            eval_id=example.id,
            category=example.category,
            difficulty=example.difficulty,
            question=example.question,
            response=backend.generate(example),
            expected_traits=example.expected_traits,
            anti_traits=example.anti_traits,
        )
        for example in examples
    ]

    return EvalReport(
        run_id=run_id or f"baseline-{uuid4()}",
        created_at=created_at or datetime.now(UTC).isoformat(),
        backend_name=backend.name,
        eval_file=str(eval_file),
        total_questions=len(report_results),
        results=report_results,
    )
