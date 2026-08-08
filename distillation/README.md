This subdirectory contains source code for distilling gpt-oss-20b into neuromancer-GPT. The goal is to get a good enough transformer model using only the Sprawl corpus, without requiring additional text.

## Setup

Add the Hugging Face and W&B credentials to `distillation/.env`:

```dotenv
HF_TOKEN=<your_hugging_face_token>
WANDB_API_KEY=<your_wandb_api_key>
```

The teacher model must be available in the `hf` Modal Volume at `/openai/gpt-oss-20b`, run download_gptoss.py to achieve that.

## Run distillation

```bash
uv run modal run distillation/src.py \
  --checkpoint-name <your_checkpoint_name> \
  --wandb-run-name <your_wandb_run> \
  --wandb-entity-name <your_wandb_entity> \
  --epochs 5
```

Use `--corpus sprawl` to train only on `sprawl.txt`; the default is `corpus.txt` (as of right now it contains sprawl + snow crasher).

## Checkpoints and restart

Restart distillation by rerunning the command with the same `--checkpoint-name`. The student model, optimizer, and scheduler are restored before the requested additional epochs run.

Run inference from a saved checkpoint with:

```bash
uv run modal run distillation/inference.py --checkpoint-name <your_checkpoint_name>
```
