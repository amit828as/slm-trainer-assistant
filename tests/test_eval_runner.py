import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slm_trainer_assistant.cli import app
from slm_trainer_assistant.eval_report import (
    EvalReport,
    format_report_summary,
    load_report,
    summarize_report,
    write_report,
)
from slm_trainer_assistant.eval_runner import run_baseline_eval
from slm_trainer_assistant.model_backends import StubBackend


def _write_eval_file(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                (
                    '{"id": "eval-001", "category": "debugging", '
                    '"difficulty": "intermediate", "question": "Loss is down but '
                    'answers are worse. What should I check?", '
                    '"expected_traits": ["checks eval quality"], '
                    '"anti_traits": ["blindly recommends more epochs"]}'
                ),
                (
                    '{"id": "eval-002", "category": "lora", "difficulty": "beginner", '
                    '"question": "Explain LoRA simply.", '
                    '"expected_traits": ["mentions adapters"], "anti_traits": []}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_baseline_eval_creates_deterministic_stub_report(tmp_path: Path) -> None:
    eval_file = tmp_path / "evals.jsonl"
    _write_eval_file(eval_file)

    report = run_baseline_eval(
        eval_file,
        StubBackend(),
        run_id="test-run",
        created_at="2026-06-16T00:00:00+00:00",
    )

    assert report.run_id == "test-run"
    assert report.created_at == "2026-06-16T00:00:00+00:00"
    assert report.backend_name == "stub"
    assert report.eval_file == str(eval_file)
    assert report.total_questions == 2
    assert report.results[0].eval_id == "eval-001"
    assert report.results[0].response == (
        "[stub:eval-001] Baseline placeholder for debugging/intermediate: "
        "Loss is down but answers are worse. What should I check?"
    )
    assert report.results[0].expected_traits == ["checks eval quality"]
    assert report.results[0].anti_traits == ["blindly recommends more epochs"]
    assert report.results[0].human_score is None
    assert report.results[0].matched_traits == []
    assert report.results[0].missed_traits == []
    assert report.results[0].triggered_anti_traits == []
    assert report.results[0].failure_type is None
    assert report.results[0].review_notes is None


def test_write_report_outputs_json(tmp_path: Path) -> None:
    eval_file = tmp_path / "evals.jsonl"
    output_file = tmp_path / "reports" / "baseline.json"
    _write_eval_file(eval_file)
    report = run_baseline_eval(eval_file, StubBackend(), run_id="test-run")

    write_report(report, output_file)

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test-run"
    assert payload["backend_name"] == "stub"
    assert payload["total_questions"] == 2
    assert payload["results"][1]["eval_id"] == "eval-002"
    assert payload["results"][1]["human_score"] is None
    assert payload["results"][1]["matched_traits"] == []


def test_run_baseline_rejects_training_examples(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        '{"id": "train-001", "messages": [{"role": "user", "content": "Hi"}, '
        '{"role": "assistant", "content": "Hello"}], "category": "demo", '
        '"difficulty": "beginner", "source_type": "manual"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only eval examples"):
        run_baseline_eval(train_file, StubBackend())


def test_cli_run_baseline_writes_report(tmp_path: Path) -> None:
    eval_file = tmp_path / "evals.jsonl"
    output_file = tmp_path / "baseline.json"
    _write_eval_file(eval_file)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-baseline",
            str(eval_file),
            "--backend",
            "stub",
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "wrote baseline report" in result.output
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["backend_name"] == "stub"
    assert payload["total_questions"] == 2


def test_report_summary_counts_reviewed_scores_and_failure_types(tmp_path: Path) -> None:
    eval_file = tmp_path / "evals.jsonl"
    _write_eval_file(eval_file)
    report = run_baseline_eval(eval_file, StubBackend(), run_id="test-run")
    report.results[0].human_score = 5
    report.results[0].matched_traits = ["checks eval quality"]
    report.results[0].failure_type = "good_answer"
    report.results[1].human_score = 2
    report.results[1].missed_traits = ["mentions adapters"]
    report.results[1].triggered_anti_traits = ["too vague"]
    report.results[1].failure_type = "too_vague"
    report.results[1].review_notes = "Needs a more concrete explanation."

    summary = summarize_report(report)

    assert summary.total_questions == 2
    assert summary.reviewed == 2
    assert summary.average_score == 3.5
    assert summary.failure_types["good_answer"] == 1
    assert summary.failure_types["too_vague"] == 1
    assert format_report_summary(summary) == (
        "Total questions: 2\n"
        "Reviewed: 2\n"
        "Average score: 3.5 / 5\n"
        "\n"
        "Failure types:\n"
        "- good_answer: 1\n"
        "- too_vague: 1"
    )


def test_load_report_defaults_review_fields_for_existing_reports(tmp_path: Path) -> None:
    report_file = tmp_path / "old-report.json"
    report_file.write_text(
        json.dumps(
            {
                "run_id": "old-run",
                "created_at": "2026-06-16T00:00:00+00:00",
                "backend_name": "stub",
                "eval_file": "evals.jsonl",
                "total_questions": 1,
                "results": [
                    {
                        "eval_id": "eval-001",
                        "category": "debugging",
                        "difficulty": "intermediate",
                        "question": "What should I check?",
                        "response": "Check the eval.",
                        "expected_traits": ["checks eval"],
                        "anti_traits": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = load_report(report_file)

    assert isinstance(report, EvalReport)
    assert report.results[0].human_score is None
    assert report.results[0].matched_traits == []
    assert report.results[0].failure_type is None


def test_cli_summarize_report_outputs_review_summary(tmp_path: Path) -> None:
    eval_file = tmp_path / "evals.jsonl"
    report_file = tmp_path / "baseline.json"
    _write_eval_file(eval_file)
    report = run_baseline_eval(eval_file, StubBackend(), run_id="test-run")
    report.results[0].human_score = 4
    report.results[0].failure_type = "good_answer"
    report.results[1].human_score = 2
    report.results[1].failure_type = "missed_risk"
    write_report(report, report_file)
    runner = CliRunner()

    result = runner.invoke(app, ["summarize-report", str(report_file)])

    assert result.exit_code == 0
    assert "Total questions: 2" in result.output
    assert "Reviewed: 2" in result.output
    assert "Average score: 3.0 / 5" in result.output
    assert "- good_answer: 1" in result.output
    assert "- missed_risk: 1" in result.output
