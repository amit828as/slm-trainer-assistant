# Kaggle Baseline Workflow

This project keeps local development lightweight. The repo stores evals, schemas, report formats, and review tools. Kaggle is where we can temporarily install ML dependencies, download a model, run baseline inference, and export a report.

For now, this workflow is only for baseline inference. Do not use it for training, LoRA, QLoRA, Unsloth, PEFT, or TRL yet.

## Why Kaggle

Running even a small instruct model can require large downloads and GPU memory. Keeping that work inside a Kaggle notebook avoids adding heavy dependencies to the normal local install or CI.

Local repo:

- stores golden eval JSONL files
- validates eval files
- defines the report schema
- summarizes reviewed reports

Kaggle runtime:

- installs notebook-only ML dependencies
- downloads temporary model weights
- generates baseline responses
- writes a JSON report

## Notebook Setup

Create a Kaggle notebook, then enable GPU if the model needs it. In the first cells, clone and install the repo:

```bash
git clone https://github.com/amit828as/slm-trainer-assistant.git
cd slm-trainer-assistant
python -m pip install -e ".[dev]"
```

Install notebook-only model dependencies inside Kaggle:

```bash
python -m pip install transformers accelerate torch
```

These dependencies are intentionally not part of `pyproject.toml`.

## Run One Baseline

Start with one golden eval file before running the whole suite. For a quick smoke test, use a very small instruct model that the Kaggle runtime can load quickly. For the first meaningful baseline, use the candidate base model we may fine-tune later.

For the first serious baseline, use:

```text
google/gemma-3-4b-it
```

If the model requires Hugging Face access approval or a token, add the token through Kaggle secrets rather than writing it into the notebook.

```bash
python scripts/run_kaggle_baseline.py \
  --eval-file evals/golden/proactive_risk_detection_questions.jsonl \
  --output evals/reports/proactive_gemma3_4b_it.json \
  --model google/gemma-3-4b-it \
  --max-new-tokens 512
```

The script intentionally requires `--model` so the baseline report always records an explicit model choice. The report also writes a small `metadata` block with the model id, `max_new_tokens`, and the system prompt used for generation.

## Bring The Report Back

Download the generated JSON report from Kaggle and place it locally under `evals/reports/` for review. Reports under `evals/reports/*.json` are ignored by git by default.

Then review the report manually:

- fill `human_score`
- fill `matched_traits`
- fill `missed_traits`
- fill `triggered_anti_traits`
- fill `failure_type`
- fill `review_notes`

Summarize it locally:

```bash
slm-trainer summarize-report evals/reports/proactive_gemma3_4b_it.json
```

## Do Not Commit Runtime Artifacts

Do not commit:

- generated reports
- model weights
- adapters
- checkpoints
- caches
- notebook outputs
- Kaggle working directories

The report can be shared manually while we are still shaping the evaluation workflow.

## Next Step After One Report

After one baseline report is reviewed and summarized, decide whether the report format is sufficient. Only then run the same workflow on the rest of the golden eval files.
