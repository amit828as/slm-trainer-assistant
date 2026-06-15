# Eval Plan

## Eval-First Development

The project starts with evals because fine-tuning without measurement is guesswork. A baseline model should answer the same questions before and after training so changes can be inspected.

## Baseline Model Comparison

Before the first LoRA run, choose one or more baseline models and collect their answers on the golden eval set. The goal is not a perfect numeric score at first. The goal is to understand the model's failure patterns.

Use `slm-trainer run-baseline <eval-file> --backend stub --output <report.json>` to exercise the report pipeline locally. The `stub` backend is deterministic and does not call a real model; provider backends should be added only after the baseline report shape is stable.

After reviewing responses by hand, fill in `human_score`, matched/missed traits, triggered anti-traits, `failure_type`, and `review_notes` in the report JSON. Then run `slm-trainer summarize-report <report.json>` to turn the review into a feedback summary before changing data or training settings.

## Eval Question Categories

- Dataset formatting.
- LoRA and QLoRA understanding.
- SFT workflow.
- Training debugging.
- Eval design.
- Proactive risk detection.
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
- Proactive risk detection.
- Refusal to bluff.
- Model release hygiene.
