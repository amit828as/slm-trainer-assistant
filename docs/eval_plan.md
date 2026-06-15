# Eval Plan

## Eval-First Development

The project starts with evals because fine-tuning without measurement is guesswork. A baseline model should answer the same questions before and after training so changes can be inspected.

## Baseline Model Comparison

Before the first LoRA run, choose one or more baseline models and collect their answers on the golden eval set. The goal is not a perfect numeric score at first. The goal is to understand the model's failure patterns.

## Eval Question Categories

- Dataset formatting.
- LoRA and QLoRA understanding.
- SFT workflow.
- Training debugging.
- Eval design.
- Refusal to bluff.
- Model release hygiene.

## Failure Modes To Track

- Gives exact hyperparameters without enough context.
- Ignores prompt-template mismatch.
- Confuses training loss with real task quality.
- Suggests training on eval examples.
- Bluffs about current leaderboard facts.
- Omits source, license, or release limitations.
- Recommends more epochs as a default fix.

## Why Generic Benchmarks Are Not Enough

Generic benchmarks can show broad model capability, but this assistant needs expert behavior in a narrow workflow. The eval set should test the exact habits we want: careful data handling, practical debugging, and honest uncertainty.

## Initial Score Categories

- Dataset formatting.
- LoRA/QLoRA understanding.
- SFT workflow.
- Training debugging.
- Eval design.
- Refusal to bluff.
- Model release hygiene.
