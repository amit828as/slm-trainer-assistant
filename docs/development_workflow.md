# Development Workflow

This repo grows in small, testable steps. Deterministic software should use TDD or test-alongside development. Model behavior should use eval-driven development.

## Local Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

If your machine only exposes `python3`, use that for the virtualenv command.

## Tests

```bash
pytest
```

Run targeted tests while developing, then the full suite before finishing.

```bash
pytest tests/test_dataset_validator.py
pytest tests/test_dataset_schema.py
```

## Lint

```bash
ruff check .
```

Use `ruff check --fix .` only for straightforward formatting and import cleanup.

## Dataset Validation

Validate JSONL files before treating them as usable training or eval data.

```bash
python -m slm_trainer_assistant.cli validate data/eval/sample_eval.jsonl
python -m slm_trainer_assistant.cli stats data/eval/sample_eval.jsonl
```

With the package installed, the console script is also available:

```bash
slm-trainer validate data/eval/sample_eval.jsonl
slm-trainer stats data/eval/sample_eval.jsonl
```

## Expected Development Loop

1. Read the relevant docs, tests, and implementation files.
2. Write or update the smallest useful test for deterministic behavior.
3. Implement the smallest change that satisfies the test.
4. Run targeted tests.
5. Update docs, examples, or sample data when behavior changes.
6. Run full tests and lint.
7. Review the diff for scope, data hygiene, and accidental generated files.
8. Commit a coherent unit of work after checks pass.

## TDD Scope

Use TDD or test-alongside development for schemas, validators, CLI behavior, report generation, file parsing, stats, and deterministic prompt rendering. These components should fail loudly and predictably.

## Eval-Driven Scope

Use eval-driven development for model quality. Before changing training data or model behavior, define what better behavior means in eval examples, expected traits, anti-traits, and failure reports. Training loss is useful telemetry, but it is not a substitute for behavior evaluation.

## Data Hygiene

Keep train, eval, stress, and rejected data separate. Never copy golden eval examples into training data. If an eval reveals a gap, create new training examples that teach the same skill without duplicating the eval item.

## Commit Messages

Prefer short imperative commit messages:

- `Add dataset source validation`
- `Document Codex task workflow`
- `Add baseline eval runner scaffold`

Use the body when reviewers need context, tradeoffs, or follow-up notes.

## Done Means

A task is done when the requested behavior or documentation exists, relevant tests pass, lint passes, docs are updated, generated files are excluded, and the diff is small enough to review. If any check could not run, record the reason clearly.
