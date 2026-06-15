from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from slm_trainer_assistant.dataset_stats import collect_stats, format_stats
from slm_trainer_assistant.dataset_validator import validate_jsonl_file
from slm_trainer_assistant.eval_report import (
    format_report_summary,
    load_report,
    summarize_report,
    write_report,
)
from slm_trainer_assistant.eval_runner import run_baseline_eval
from slm_trainer_assistant.model_backends import get_backend

app = typer.Typer(help="Tools for SLM trainer datasets and evals.")


@app.command()
def validate(path: Path) -> None:
    """Validate a JSONL dataset or eval file."""

    result = validate_jsonl_file(path)
    if result.is_valid:
        typer.echo(f"valid: {path} ({len(result.examples)} examples)")
        return

    typer.echo(f"invalid: {path}", err=True)
    for issue in result.issues:
        typer.echo(f"line {issue.line_number}: {issue.message}", err=True)
    raise typer.Exit(code=1)


@app.command()
def stats(path: Path) -> None:
    """Show simple category, difficulty, and source counts."""

    try:
        dataset_stats = collect_stats(path)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_stats(dataset_stats))


@app.command("run-baseline")
def run_baseline(
    eval_file: Path,
    output: Annotated[Path, typer.Option("--output", "-o", help="JSON report output path.")],
    backend_name: Annotated[
        str, typer.Option("--backend", help="Model backend to use.")
    ] = "stub",
) -> None:
    """Collect baseline model responses for an eval JSONL file."""

    try:
        backend = get_backend(backend_name)
        report = run_baseline_eval(eval_file, backend)
        report_path = write_report(report, output)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"wrote baseline report: {report_path}")


@app.command("summarize-report")
def summarize_report_command(report_path: Path) -> None:
    """Summarize human review fields in a baseline report."""

    report = load_report(report_path)
    summary = summarize_report(report)
    typer.echo(format_report_summary(summary))


if __name__ == "__main__":
    app()
