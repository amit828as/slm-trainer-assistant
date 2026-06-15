import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slm_trainer_assistant.cli import app
from slm_trainer_assistant.eval_report import write_report
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
