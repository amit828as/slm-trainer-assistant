# Roadmap

## Milestone 0: Repo Scaffold

Create the project structure, documentation, packaging metadata, sample data folders, and agent guidance.

## Milestone 1: Schema + Validator

Define framework-independent JSONL schemas for training and eval examples. Add validation tooling and tests.

## Milestone 2: Eval Set

Build a small golden eval set across dataset formatting, LoRA/QLoRA, SFT workflow, debugging, eval design, refusal to bluff, and release hygiene.

## Milestone 3: Baseline Runner

Add a simple runner that asks a baseline model the golden eval questions and records responses for review.

## Milestone 4: Manual Seed Dataset

Create a small, high-quality instruction dataset from manual examples and carefully sourced public-doc-derived examples.

## Milestone 5: First LoRA SFT Run

Run the first lightweight LoRA or QLoRA supervised fine-tuning experiment through either Unsloth or TRL + PEFT.

## Milestone 6: Evaluation Report

Compare baseline and fine-tuned model outputs. Record strengths, regressions, and failure modes.

## Milestone 7: Failure-Driven Dataset Improvement

Convert observed failures into targeted data improvements while keeping eval examples out of training.

## Milestone 8: CLI Demo

Expose a small local demo flow for validation, stats, baseline evaluation, and report generation.

## Milestone 9: Model Release

Prepare model cards, dataset cards, license notes, known limitations, and reproducibility instructions.
