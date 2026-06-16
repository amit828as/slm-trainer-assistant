# slm-trainer-assistant

`slm-trainer-assistant` is an open-source training lab for building one expert Small Language Model (SLM): an assistant that helps developers fine-tune, evaluate, debug, and release other SLMs.

This is not a generic chatbot project. The goal is to grow a focused model that understands practical SLM work: instruction dataset formatting, supervised fine-tuning (SFT), LoRA and QLoRA tradeoffs, evaluation design, failed-run debugging, model cards, dataset cards, and release hygiene.

## Why Expert SLMs

Small Language Models are cheaper to run, easier to deploy privately, and often good enough when the task is narrow. The catch is that they need a sharper training loop than large general assistants: clearer data, more targeted evals, and careful release checks. This repo exists to make that loop explicit and repeatable.

## Why Start With Evals And Data

Fine-tuning first is tempting, but it makes improvement hard to measure. This project starts with schemas, validators, sample evals, and reporting structure so every training run can answer a basic question: did the model become better at the expert behavior we actually care about?

## Planned Workflow

1. Define expert behavior.
2. Build evals.
3. Run a baseline model.
4. Create a seed dataset.
5. Fine-tune with LoRA or QLoRA.
6. Evaluate.
7. Inspect failures.
8. Improve the dataset.
9. Retrain.
10. Release the model with clear model and dataset cards.

## Local Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Validate A Dataset

The validator reads JSONL files containing either training examples with `messages` or eval examples with `question`, `expected_traits`, and `anti_traits`.

```bash
python -m slm_trainer_assistant.cli validate data/eval/sample_eval.jsonl
slm-trainer validate data/eval/sample_eval.jsonl
```

## Inspect Dataset Stats

```bash
python -m slm_trainer_assistant.cli stats data/eval/sample_eval.jsonl
slm-trainer stats data/eval/sample_eval.jsonl
```

## Baseline Example Model

The current real-model baseline example uses `google/gemma-4-E4B-it` in Kaggle.
That model choice is interchangeable: use another compatible instruct model by
changing the `--model` value and report filename, then adjust Kaggle hardware or
dependencies if that model needs a different runtime.

Gemma 4 E4B-it is multimodal, so the repo also includes a small image eval file
under `evals/golden/multimodal_image_questions.jsonl`. Text evals remain the
first smoke test; image evals check whether the model actually uses visual
evidence.

## Run Tests

```bash
pytest
```

## Development

The Makefile wraps the common local workflow:

```bash
make install
make test
make lint
make validate-samples
make check
```

`make check` runs lint, tests, and sample eval validation.

## Current Scope

The current repository intentionally does not run fine-tuning yet. Training backend support should come after the dataset schema, validator, eval sets, and baseline comparison workflow are stable.
