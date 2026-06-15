from __future__ import annotations

from pathlib import Path

import typer

from slm_trainer_assistant.dataset_stats import collect_stats, format_stats
from slm_trainer_assistant.dataset_validator import validate_jsonl_file

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


if __name__ == "__main__":
    app()
