# Transformer model trained on the Sprawl trilogy

Decoder-only multi-head attention transformer.

\* neuromancer-GPT because it borrows elements from the GPT-2 architecture, obviously it is not a pre-trained one :D.

This is a pet project. I put this on Github only to showcase it. Everything here can pretty much be found in other sources, but maybe it is helpful for people starting out as there are just 2 files and it is easy to follow.

I trained the model on [Modal](https://modal.com/).

## Usage

Add your W&B API key to a `.env` file in the project root before running training:

```dotenv
WANDB_API_KEY=<your_wandb_api_key>
```

```bash
# Train and run the base model
uv run modal run main.py --checkpoint-name <your_checkpoint_name> --wandb-run-name <your_wandb_run> --wandb-entity-name <your_wandb_entity> --epochs 5
uv run modal run inference.py --checkpoint-name <your_checkpoint_name>
```

## Checkpoints

To restart training, run the same command again with the same `--checkpoint-name`. If that checkpoint exists, training restores its model, optimizer, and scheduler states before starting the requested additional epochs. Use a new checkpoint name to start from scratch.

Here is a 100 token paragraph the model generated.

> it was a dark rainy afternoon, by the glow of the invisible casino. the thing was a kind of pilotless biplane of gossamer polymer, its wings silkscreened to resemble a giant butterfly. then it was gone, beyond the mesa's edge. he'd seen a wink of reflected neon off glass, either lenses or the turrets of lasers. the drones were part

This is obviously text from the first book. The model is too big (~115M params) for the sprawl corpus (~250k tokens). As per Chinchilla Scaling Laws, the corpus should be bigger (20tokens/param).

As such, this project is a work in progress. My goal is to get a model that is only able to write fiction, not a general chatbot.

# distillation

I also distilled gpt-oss:20b on the small current dataset. check the distillation/ dir
