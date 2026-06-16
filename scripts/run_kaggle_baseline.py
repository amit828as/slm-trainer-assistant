#!/usr/bin/env python
"""Run real baseline inference inside a Kaggle notebook.

This script intentionally keeps Hugging Face dependencies out of the normal
project install. Install them inside Kaggle before running this file.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from slm_trainer_assistant.eval_report import EvalReport, EvalResult, write_report
from slm_trainer_assistant.eval_runner import load_eval_examples

SYSTEM_PROMPT = (
    "You are an expert assistant for training small language models. "
    "Give practical, careful advice about datasets, evals, fine-tuning, "
    "debugging, and model release hygiene. Ask for missing context when needed "
    "and do not bluff about facts you cannot verify."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Kaggle baseline eval with HF models.")
    parser.add_argument("--eval-file", required=True, type=Path, help="Golden eval JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="JSON report output path.")
    parser.add_argument("--model", required=True, help="Hugging Face instruct model id.")
    parser.add_argument(
        "--max-new-tokens",
        default=512,
        type=int,
        help="Maximum tokens to generate per eval question.",
    )
    return parser.parse_args()


def render_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{question}\n\nAssistant:\n"


def load_hf_model(model_name: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing notebook dependencies. In Kaggle, run: "
            "python -m pip install transformers accelerate torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def generate_response(
    tokenizer: Any,
    model: Any,
    question: str,
    *,
    max_new_tokens: int,
) -> str:
    prompt = render_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_kaggle_baseline(
    eval_file: Path,
    output: Path,
    *,
    model_name: str,
    max_new_tokens: int,
) -> Path:
    examples = load_eval_examples(eval_file)
    tokenizer, model = load_hf_model(model_name)

    results = []
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example.id}", flush=True)
        response = generate_response(
            tokenizer,
            model,
            example.question,
            max_new_tokens=max_new_tokens,
        )
        results.append(
            EvalResult(
                eval_id=example.id,
                category=example.category,
                difficulty=example.difficulty,
                question=example.question,
                response=response,
                expected_traits=example.expected_traits,
                anti_traits=example.anti_traits,
            )
        )

    report = EvalReport(
        run_id=f"kaggle-baseline-{uuid4()}",
        created_at=datetime.now(UTC).isoformat(),
        backend_name=f"huggingface:{model_name}",
        eval_file=str(eval_file),
        total_questions=len(results),
        metadata={
            "model": model_name,
            "max_new_tokens": max_new_tokens,
            "system_prompt": SYSTEM_PROMPT,
        },
        results=results,
    )
    return write_report(report, output)


def main() -> None:
    args = parse_args()
    report_path = run_kaggle_baseline(
        args.eval_file,
        args.output,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"wrote Kaggle baseline report: {report_path}")


if __name__ == "__main__":
    main()
