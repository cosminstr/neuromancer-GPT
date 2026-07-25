import modal
import torch
from main import GPTConfig, GPT, sample_sprawl

app = modal.App("gpt-neuromancer-inference")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "tqdm")
    .add_local_python_source("main")
    .add_local_python_source("utils")
)

checkpoint_volume = modal.Volume.from_name("gpt-neuromancer", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    timeout=30,
    volumes={"/checkpoints": modal.Volume.from_name("gpt-neuromancer")},
)
def main():
    device = torch.device("cuda")
    checkpoint = torch.load("/checkpoints/neuromancer-gpt.pth")
    model = GPT(GPTConfig(vocab_size=len(checkpoint["stoi"]))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    sample_sprawl(model, 500, checkpoint["stoi"], checkpoint["itos"])
