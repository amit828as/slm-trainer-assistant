#!/usr/bin/env python
"""Run real baseline inference inside a Kaggle notebook.

This script intentionally keeps Hugging Face dependencies out of the normal
project install. Install them inside Kaggle before running this file.
"""

from __future__ import annotations

import argparse
import gc
import re
from datetime import UTC, datetime
from inspect import Parameter, signature
from pathlib import Path
from typing import Any
from uuid import uuid4

from slm_trainer_assistant.eval_report import EvalReport, EvalResult, write_report
from slm_trainer_assistant.eval_runner import load_eval_examples
from slm_trainer_assistant.schemas import EvalMedia

SYSTEM_PROMPT = (
    "You are an expert assistant for training small language models. "
    "Give practical, careful advice about datasets, evals, fine-tuning, "
    "debugging, and model release hygiene. Ask for missing context when needed "
    "and do not bluff about facts you cannot verify."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kaggle baseline evals with HF models.")
    parser.add_argument("--eval-file", type=Path, help="Golden eval JSONL file.")
    parser.add_argument("--output", type=Path, help="JSON report output path.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        help="Directory of golden eval JSONL files to run as a batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where batch JSON reports should be written.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face instruct model id.")
    parser.add_argument(
        "--hf-token-secret",
        default="HF_TOKEN",
        help=(
            "Kaggle secret label containing a Hugging Face token. "
            "Use an empty string to skip Kaggle secret login."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        default=512,
        type=int,
        help="Maximum tokens to generate per eval question.",
    )
    parser.add_argument(
        "--report-suffix",
        help=(
            "Suffix for batch report filenames. Defaults to a slug from --model, "
            "for example google/gemma-4-E4B-it becomes gemma_4_e4b_it."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In batch mode, skip reports that already exist.",
    )

    args = parser.parse_args()
    single_mode = args.eval_file is not None or args.output is not None
    batch_mode = args.eval_dir is not None or args.output_dir is not None
    if single_mode == batch_mode:
        parser.error("use either --eval-file/--output or --eval-dir/--output-dir")
    if single_mode and (args.eval_file is None or args.output is None):
        parser.error("--eval-file and --output must be provided together")
    if batch_mode and (args.eval_dir is None or args.output_dir is None):
        parser.error("--eval-dir and --output-dir must be provided together")
    if args.skip_existing and not batch_mode:
        parser.error("--skip-existing is only valid with --eval-dir/--output-dir")
    if args.report_suffix and not batch_mode:
        parser.error("--report-suffix is only valid with --eval-dir/--output-dir")
    return args


def _supports_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        parameters = signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def render_prompt(text_processor: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    apply_chat_template = getattr(text_processor, "apply_chat_template", None)
    if callable(apply_chat_template):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if _supports_keyword(apply_chat_template, "enable_thinking"):
            kwargs["enable_thinking"] = False
        return apply_chat_template(messages, **kwargs)
    return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{question}\n\nAssistant:\n"


def _config_architectures(config: Any) -> list[str]:
    return list(getattr(config, "architectures", None) or [])


def _uses_processor_for_text(model_name: str, config: Any) -> bool:
    model_type = getattr(config, "model_type", "")
    architectures = _config_architectures(config)
    return (
        model_type == "gemma4"
        or "gemma-4" in model_name.lower()
        or any(architecture.startswith("Gemma4") for architecture in architectures)
    )


def _load_causal_lm(auto_model_for_causal_lm: Any, model_name: str) -> Any:
    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    try:
        return auto_model_for_causal_lm.from_pretrained(
            model_name,
            dtype="auto",
            **kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return auto_model_for_causal_lm.from_pretrained(
            model_name,
            torch_dtype="auto",
            **kwargs,
        )


def _load_multimodal_lm(auto_model_for_multimodal_lm: Any, model_name: str) -> Any:
    try:
        return auto_model_for_multimodal_lm.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            trust_remote_code=True,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return auto_model_for_multimodal_lm.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )


def load_hf_model(model_name: str, *, use_multimodal: bool = False) -> tuple[Any, Any]:
    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoProcessor,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing notebook dependencies. In Kaggle, run: "
            "python -m pip install -U transformers accelerate huggingface_hub pillow torch"
        ) from exc

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if use_multimodal or _uses_processor_for_text(model_name, config):
        text_processor = AutoProcessor.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )
    else:
        text_processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if use_multimodal:
        model = _load_multimodal_lm(AutoModelForMultimodalLM, model_name)
    else:
        model = _load_causal_lm(AutoModelForCausalLM, model_name)
    return text_processor, model


def login_from_kaggle_secret(secret_name: str | None) -> bool:
    if not secret_name:
        return False

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return False

    try:
        token = UserSecretsClient().get_secret(secret_name)
    except Exception as exc:  # pragma: no cover - depends on Kaggle runtime services.
        print(
            f"Hugging Face token secret {secret_name!r} was not available; "
            f"continuing without login ({exc}).",
            flush=True,
        )
        return False

    if not token:
        print(f"Hugging Face token secret {secret_name!r} was empty; continuing without login.")
        return False

    try:
        from huggingface_hub import login
    except ImportError as exc:
        raise SystemExit(
            "Missing notebook dependency. In Kaggle, run: "
            "python -m pip install -U huggingface_hub"
        ) from exc

    login(token=token, add_to_git_credential=False)
    print(f"loaded Hugging Face token from Kaggle secret {secret_name!r}", flush=True)
    return True


def _move_inputs_to_model(inputs: Any, model: Any, *, include_dtype: bool = False) -> Any:
    model_device = getattr(model, "device", None)
    if model_device is None:
        return inputs

    move_to = getattr(inputs, "to", None)
    if callable(move_to):
        if include_dtype and getattr(model, "dtype", None) is not None:
            try:
                return move_to(model_device, dtype=model.dtype)
            except TypeError:
                return move_to(model_device)
        return move_to(model_device)

    return {
        name: tensor.to(model_device) if hasattr(tensor, "to") else tensor
        for name, tensor in inputs.items()
    }


def _encode_prompt(text_processor: Any, prompt: str) -> Any:
    try:
        return text_processor(text=prompt, return_tensors="pt")
    except TypeError as exc:
        if "text" not in str(exc):
            raise
        return text_processor(prompt, return_tensors="pt")


def _load_image(media: EvalMedia) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing notebook dependency for image evals. In Kaggle, run: "
            "python -m pip install -U pillow"
        ) from exc

    return Image.open(media.path).convert("RGB")


def _render_multimodal_messages(question: str, media: list[EvalMedia]) -> list[dict[str, Any]]:
    user_content = [
        {"type": "image", "image": _load_image(item)}
        for item in media
        if item.type == "image"
    ]
    user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def _encode_multimodal_prompt(
    text_processor: Any,
    *,
    question: str,
    media: list[EvalMedia],
) -> Any:
    apply_chat_template = getattr(text_processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise SystemExit("The selected model processor does not support multimodal chat templates.")

    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if _supports_keyword(apply_chat_template, "enable_thinking"):
        kwargs["enable_thinking"] = False
    return apply_chat_template(_render_multimodal_messages(question, media), **kwargs)


def _eos_token_id(text_processor: Any) -> int | None:
    eos_token_id = getattr(text_processor, "eos_token_id", None)
    if eos_token_id is not None:
        return eos_token_id

    tokenizer = getattr(text_processor, "tokenizer", None)
    return getattr(tokenizer, "eos_token_id", None)


def _coerce_response_text(parsed_response: Any) -> str:
    if isinstance(parsed_response, str):
        return parsed_response

    if isinstance(parsed_response, dict):
        for key in ("content", "text", "response"):
            value = parsed_response.get(key)
            if value:
                return _coerce_response_text(value)
        return str(parsed_response)

    if isinstance(parsed_response, list):
        parts = [_coerce_response_text(part) for part in parsed_response]
        return "\n".join(part for part in parts if part)

    for attribute in ("content", "text"):
        value = getattr(parsed_response, attribute, None)
        if value:
            return _coerce_response_text(value)

    return str(parsed_response)


def _decode_response(text_processor: Any, generated_ids: Any) -> str:
    parse_response = getattr(text_processor, "parse_response", None)
    should_parse = callable(parse_response)
    response = text_processor.decode(
        generated_ids,
        skip_special_tokens=not should_parse,
    )
    if should_parse:
        return _coerce_response_text(parse_response(response))
    return response


def generate_response(
    text_processor: Any,
    model: Any,
    question: str,
    *,
    media: list[EvalMedia] | None = None,
    max_new_tokens: int,
) -> str:
    media = media or []
    if media:
        inputs = _move_inputs_to_model(
            _encode_multimodal_prompt(text_processor, question=question, media=media),
            model,
            include_dtype=True,
        )
    else:
        prompt = render_prompt(text_processor, question)
        inputs = _move_inputs_to_model(_encode_prompt(text_processor, prompt), model)

    generate_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }
    pad_token_id = _eos_token_id(text_processor)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id

    output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return _decode_response(text_processor, generated_ids).strip()


def _has_multimodal_examples(examples: list[Any]) -> bool:
    return any(example.media for example in examples)


def _write_kaggle_baseline_report(
    eval_file: Path,
    output: Path,
    *,
    examples: list[Any],
    text_processor: Any,
    model: Any,
    model_name: str,
    max_new_tokens: int,
    hf_login_used: bool,
    uses_multimodal: bool,
) -> Path:
    results = []
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example.id}", flush=True)
        response = generate_response(
            text_processor,
            model,
            example.question,
            media=example.media,
            max_new_tokens=max_new_tokens,
        )
        results.append(
            EvalResult(
                eval_id=example.id,
                category=example.category,
                difficulty=example.difficulty,
                question=example.question,
                media=example.media,
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
            "hf_login": "kaggle_secret" if hf_login_used else "not_used",
            "text_processor": text_processor.__class__.__name__,
            "multimodal": uses_multimodal,
        },
        results=results,
    )
    return write_report(report, output)


def run_kaggle_baseline(
    eval_file: Path,
    output: Path,
    *,
    model_name: str,
    max_new_tokens: int,
    hf_token_secret: str | None = "HF_TOKEN",
) -> Path:
    examples = load_eval_examples(eval_file)
    uses_multimodal = _has_multimodal_examples(examples)
    hf_login_used = login_from_kaggle_secret(hf_token_secret)
    text_processor, model = load_hf_model(model_name, use_multimodal=uses_multimodal)
    return _write_kaggle_baseline_report(
        eval_file,
        output,
        examples=examples,
        text_processor=text_processor,
        model=model,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        hf_login_used=hf_login_used,
        uses_multimodal=uses_multimodal,
    )


def _default_report_suffix(model_name: str) -> str:
    model_leaf = model_name.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "_", model_leaf.lower()).strip("_")


def _report_prefix_for_eval_file(eval_file: Path) -> str:
    prefix = eval_file.stem
    if prefix.endswith("_questions"):
        prefix = prefix.removesuffix("_questions")
    if prefix == "proactive_risk_detection":
        return "proactive"
    return prefix


def _batch_report_path(eval_file: Path, output_dir: Path, report_suffix: str) -> Path:
    return output_dir / f"{_report_prefix_for_eval_file(eval_file)}_{report_suffix}.json"


def _release_model_memory(text_processor: Any, model: Any) -> None:
    del text_processor
    del model
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_kaggle_baselines(
    eval_dir: Path,
    output_dir: Path,
    *,
    model_name: str,
    max_new_tokens: int,
    hf_token_secret: str | None = "HF_TOKEN",
    report_suffix: str | None = None,
    skip_existing: bool = False,
) -> list[Path]:
    report_suffix = report_suffix or _default_report_suffix(model_name)
    eval_files = sorted(eval_dir.glob("*.jsonl"))
    if not eval_files:
        raise ValueError(f"no eval JSONL files found in {eval_dir}")

    pending_by_modality: dict[bool, list[tuple[Path, Path, list[Any]]]] = {
        False: [],
        True: [],
    }
    for eval_file in eval_files:
        output = _batch_report_path(eval_file, output_dir, report_suffix)
        if skip_existing and output.exists():
            print(f"skipping existing report: {output}", flush=True)
            continue
        examples = load_eval_examples(eval_file)
        pending_by_modality[_has_multimodal_examples(examples)].append(
            (eval_file, output, examples)
        )

    pending_count = sum(len(items) for items in pending_by_modality.values())
    if pending_count == 0:
        print("no pending Kaggle baseline reports to run", flush=True)
        return []

    hf_login_used = login_from_kaggle_secret(hf_token_secret)
    report_paths: list[Path] = []
    for uses_multimodal in (False, True):
        pending = pending_by_modality[uses_multimodal]
        if not pending:
            continue
        mode_name = "multimodal" if uses_multimodal else "text"
        print(f"loading {mode_name} model once for {len(pending)} report(s)", flush=True)
        text_processor, model = load_hf_model(model_name, use_multimodal=uses_multimodal)
        try:
            for eval_file, output, examples in pending:
                print(f"running {eval_file} -> {output}", flush=True)
                report_paths.append(
                    _write_kaggle_baseline_report(
                        eval_file,
                        output,
                        examples=examples,
                        text_processor=text_processor,
                        model=model,
                        model_name=model_name,
                        max_new_tokens=max_new_tokens,
                        hf_login_used=hf_login_used,
                        uses_multimodal=uses_multimodal,
                    )
                )
        finally:
            _release_model_memory(text_processor, model)

    return report_paths


def main() -> None:
    args = parse_args()
    if args.eval_dir is not None and args.output_dir is not None:
        report_paths = run_kaggle_baselines(
            args.eval_dir,
            args.output_dir,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            hf_token_secret=args.hf_token_secret,
            report_suffix=args.report_suffix,
            skip_existing=args.skip_existing,
        )
        print(f"wrote {len(report_paths)} Kaggle baseline report(s)")
        for report_path in report_paths:
            print(f"- {report_path}")
    else:
        report_path = run_kaggle_baseline(
            args.eval_file,
            args.output,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            hf_token_secret=args.hf_token_secret,
        )
        print(f"wrote Kaggle baseline report: {report_path}")


if __name__ == "__main__":
    main()
