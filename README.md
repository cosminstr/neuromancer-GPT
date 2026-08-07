# Transformer model trained on the Sprawl trilogy

Decoder-only multi-head attention transformer.

\* neuromancer-GPT because it borrows elements from the GPT-2 architecture, obviously it is not a pre-trained one :D.

This is a pet project. I put this on Github only to showcase it. Everything here can pretty much be found in other sources, but maybe it is helpful for people starting out as there are just 2 files and it is easy to follow.

I trained the model on [Modal](https://modal.com/).

## Usage

```bash
# Train and run the base model
uv run modal run main.py --checkpoint-name <your_checkpoint_name> --epochs 5
uv run modal run inference.py --checkpoint-name <your_checkpoint_name>

# Train and run the distilled model
uv run modal run distillation/src.py --checkpoint-name <your_checkpoint_name> --wandb-run-name <your_wandb_run> --wandb-entity-name <your_wandb_entity> --epochs 5
uv run modal run distillation/inference.py --checkpoint-name <your_checkpoint_name>
```

Here is a 100 token paragraph the model generated.

> it was a dark rainy afternoon, by the glow of the invisible casino. the thing was a kind of pilotless biplane of gossamer polymer, its wings silkscreened to resemble a giant butterfly. then it was gone, beyond the mesa's edge. he'd seen a wink of reflected neon off glass, either lenses or the turrets of lasers. the drones were part

This is obviously text from the first book. The model is too big (~115M params) for the sprawl corpus (~250k tokens). As per Chinchilla Scaling Laws, the corpus should be bigger (20tokens/param).

As such, this project is a work in progress. My goal is to get a model that is only able to write fiction, not a general chatbot.

# distillation

I also distilled gpt-oss:20b on the small current dataset. check the distillation/ dir
