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

Create a Kaggle notebook, then enable GPU if the model needs it. For Gemma 4
E4B, use **GPU T4 x2** in the current Kaggle runtime. P100 can fail with newer
PyTorch/CUDA builds because the runtime may not include kernels for that older
GPU architecture.

Keep the notebook as a reusable runner instead of a scratchpad. A clean notebook
should have these cells, in this order:

1. repo/dependency setup
2. Hugging Face auth from Kaggle secrets
3. GPU smoke check
4. one baseline script invocation
5. optional report preview/download

In the first cell, clone and install the repo:

```bash
%%bash
set -e
cd /kaggle/working
rm -rf slm-trainer-assistant
git clone https://github.com/amit828as/slm-trainer-assistant.git
cd slm-trainer-assistant
python -m pip install -e ".[dev]"
```

Install notebook-only model dependencies inside Kaggle:

```bash
%%bash
set -e
python -m pip install -U transformers accelerate huggingface_hub pillow
```

These dependencies are intentionally not part of `pyproject.toml`. Keep
Kaggle's preinstalled PyTorch/CUDA build unless the GPU smoke check below shows
Torch itself is missing or broken. Do not install `torchvision`; the baseline
script does not use it, and upgrading it can spend time reinstalling PyTorch.

If the model needs a Hugging Face token, save it in Kaggle secrets as
`HF_TOKEN`. Do not paste the token into the notebook. Add a small auth smoke
check cell:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=hf_token, add_to_git_credential=False)
print("HF token loaded from Kaggle secrets")
```

The baseline script also tries to load `HF_TOKEN` automatically, so this cell is
mainly a quick confirmation that the secret is wired correctly.

Then confirm the GPU is usable:

```python
import subprocess
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print(torch.ones(1, device="cuda") + 1)
subprocess.run(["nvidia-smi"], check=False)
```

## Run The Baseline Suite

Run the text golden eval suite when collecting the next baseline. For a quick
smoke test, you can still run one file with `--eval-file` and `--output`, but
the real text baseline should cover every text category so later comparisons
have a complete starting point.

For the first serious baseline, use:

```text
google/gemma-4-E4B-it
```

Use the instruction-tuned model for assistant-style evals. The base checkpoint
can run, but it is not the right first baseline for instruction-following
answers. This is the current worked example, not a permanent requirement; swap
another compatible Hugging Face model by changing `--model` and using a matching
report filename.

```bash
%%bash
set -e
cd /kaggle/working/slm-trainer-assistant
RUN_NAME=baseline_v1_1_adaptive_depth_1024
python scripts/run_kaggle_baseline.py \
  --eval-dir evals/golden \
  --output-dir "evals/reports/${RUN_NAME}" \
  --text-only \
  --model google/gemma-4-E4B-it \
  --report-suffix gemma4_e4b_it \
  --max-new-tokens 1024
```

The script intentionally requires `--model` so every baseline report records an
explicit model choice. The report also writes a small `metadata` block with the
model id, `max_new_tokens`, and the system prompt used for generation.

Use an explicit `RUN_NAME` for Kaggle baseline records. Generated reports are
local artifacts and are not committed, so a fresh Kaggle checkout cannot infer
the next baseline version from existing report folders. For prompt ablations,
change only the run name and the prompt, then keep the model, evals, backend,
and `max_new_tokens` fixed. `--auto-version-output-dir` is still available as a
local convenience, but explicit run names are better for Kaggle records.
`--text-only` skips eval files that contain media while the project is focused
on text behavior.

The script supports Gemma 4 text evals by using the model's `AutoProcessor`
chat template and `AutoModelForCausalLM` load path. It keeps thinking disabled
for deterministic baseline answers. In batch mode, it loads the model once for
all text eval files and only loads the multimodal path when media evals are not
skipped.

At model load time, the script prints CUDA availability and the model device map.
If CUDA is unavailable, or if `device_map="auto"` offloads any layers to CPU or
disk, the script stops immediately instead of continuing with very slow CPU-bound
generation.

## Run One Baseline

Single-file mode is useful for a smoke test or a rerun of one category:

```bash
%%bash
set -e
cd /kaggle/working/slm-trainer-assistant
python scripts/run_kaggle_baseline.py \
  --eval-file evals/golden/proactive_risk_detection_questions.jsonl \
  --output evals/reports/proactive_gemma4_e4b_it.json \
  --model google/gemma-4-E4B-it \
  --max-new-tokens 512
```

When an eval file contains `media`, the script switches to the multimodal model
load path and stores the media metadata in the JSON report.

If you want to run without Hugging Face login, pass an empty secret name:

```bash
python scripts/run_kaggle_baseline.py \
  --eval-file evals/golden/proactive_risk_detection_questions.jsonl \
  --output evals/reports/proactive_gemma4_e4b_it_no_login.json \
  --model google/gemma-4-E4B-it \
  --hf-token-secret "" \
  --max-new-tokens 512
```

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
slm-trainer summarize-report evals/reports/proactive_gemma4_e4b_it.json
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

Keep one clean reusable notebook for this project/model family. Do not create a
new notebook for every eval unless you intentionally want an immutable
experiment snapshot. The repeatable artifact is the JSON report, so give each
run a descriptive output filename under `evals/reports/`.

## Next Step After One Report

After one baseline report is reviewed and summarized, decide whether the report format is sufficient. Only then run the same workflow on the rest of the golden eval files.
