# Eval Plan

## Eval-First Development

The project starts with evals because fine-tuning without measurement is guesswork. A baseline model should answer the same questions before and after training so changes can be inspected.

## Baseline Model Comparison

Before the first LoRA run, choose one or more baseline models and collect their answers on the golden eval set. The goal is not a perfect numeric score at first. The goal is to understand the model's failure patterns.

Use `slm-trainer run-baseline <eval-file> --backend stub --output <report.json>` to exercise the report pipeline locally. The `stub` backend is deterministic and does not call a real model; provider backends should be added only after the baseline report shape is stable.

For real model baselines, use the Kaggle workflow in `docs/notebook_baseline.md` so model weights and heavy inference dependencies stay outside the normal local and CI setup.

The current Kaggle example uses `google/gemma-4-E4B-it`, but the eval workflow is
not tied to Gemma. Other compatible instruct models can be substituted by
changing the model id and output report filename. Keep the model id in the
report metadata so comparisons are traceable.

After reviewing responses by hand, fill in `human_score`, matched/missed traits, triggered anti-traits, `failure_type`, and `review_notes` in the report JSON. Then run `slm-trainer summarize-report <report.json>` to turn the review into a feedback summary before changing data or training settings.

## Eval Quality Criteria

Good eval questions should test behavior in realistic situations, not ask for definitions alone. Prefer prompts where the assistant must make a judgment: ask for missing context, warn about a likely risk, preserve train/eval separation, identify a workflow mismatch, or refuse to bluff. `expected_traits` should describe observable pieces of a good answer, and `anti_traits` should name risky failure modes such as confident guessing, vague advice, unsafe shortcuts, or ignoring data provenance.

For multimodal models, text-only evals are not enough. They can show whether a
model gives good generic advice, but they cannot prove the model reads visual
evidence correctly. Image evals should include owned or generated media assets,
ask questions that require inspecting the image, and use anti-traits for answers
that ignore the visual input or infer facts not present in it.

## Eval Question Categories

- Dataset formatting.
- LoRA and QLoRA understanding.
- SFT workflow.
- Training debugging.
- Eval design.
- Proactive risk detection.
- Refusal to bluff.
- Model release hygiene.
- Multimodal debugging and data hygiene.

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
