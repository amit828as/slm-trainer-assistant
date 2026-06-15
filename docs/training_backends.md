# Training Backends

This repo should support more than one training path, but backend support comes after schemas and evals are stable.

## Unsloth

Unsloth is useful for practical, faster, lower-memory LoRA and QLoRA fine-tuning. It is a good fit when the project wants fast local or single-GPU experiments.

## TRL + PEFT

Hugging Face TRL plus PEFT is the standard ecosystem path for supervised fine-tuning and adapter-based training. It is useful for compatibility, examples, and broader community support.

## Framework Independence

Datasets, evals, and reports should stay independent of either backend. A JSONL training example should not need to know whether it will be consumed by Unsloth or TRL. Backend-specific prompt rendering and trainer configuration can be added later as adapters.

## Timing

Add training backend support after schema validation, sample evals, baseline comparison, and failure reporting are strong enough to make training results meaningful.
