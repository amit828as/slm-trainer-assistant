.PHONY: install test lint validate-samples check

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

validate-samples:
	$(PYTHON) -m slm_trainer_assistant.cli validate data/eval/sample_eval.jsonl
	$(PYTHON) -m slm_trainer_assistant.cli validate evals/golden/beginner_questions.jsonl
	$(PYTHON) -m slm_trainer_assistant.cli validate evals/golden/debugging_questions.jsonl
	$(PYTHON) -m slm_trainer_assistant.cli validate evals/golden/dataset_formatting_questions.jsonl
	$(PYTHON) -m slm_trainer_assistant.cli validate evals/golden/eval_design_questions.jsonl

check: lint test validate-samples
