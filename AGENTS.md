# Repository Working Notes

This repo is building eval-first tooling for an SLM trainer assistant. Keep the
work practical, narrow, and measurable: improve the evaluation loop before
making training or fine-tuning changes.

## Working Style

- Prefer small, readable changes over broad rewrites.
- Keep repo logic in scripts and package modules, not in long notebook cells.
- Do not commit generated reports, model weights, adapters, or local progress
  notes.
- If a change affects behavior, add or update focused tests.
- Use existing schema, report, and CLI patterns before adding new abstractions.

## Verification

Run the project check before committing:

```bash
PATH=.venv/bin:$PATH make PYTHON=.venv/bin/python check
```

That command runs:

- `ruff check .`
- `pytest`
- validation for the sample eval file and every golden eval file

The repo has a local Git hook at `.githooks/pre-commit`. Enable it in a checkout
with:

```bash
git config core.hooksPath .githooks
```

The hook runs the same `make check` target and prefers `.venv/bin/python` when
the virtualenv exists.

## Kaggle Baselines

The normal baseline path is all-category batch inference:

```bash
RUN_NAME=baseline_v1_1_adaptive_depth_1024
python scripts/run_kaggle_baseline.py \
  --eval-dir evals/golden \
  --output-dir "evals/reports/${RUN_NAME}" \
  --text-only \
  --model google/gemma-4-E4B-it \
  --report-suffix gemma4_e4b_it \
  --max-new-tokens 1024
```

Single-file baseline runs are for smoke tests or targeted reruns, not for the
main baseline record.

Use explicit run names for Kaggle baseline records. Generated reports are not
committed, so Kaggle cannot reliably auto-infer the next version after a fresh
pull.

Reports under `evals/reports/` are runtime artifacts. Review and summarize them
locally, but do not commit them unless the project policy changes.
