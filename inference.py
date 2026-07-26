import modal
import torch
from main import GPTConfig, GPT, sample_sprawl

app = modal.App("gpt-neuromancer-inference")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "tqdm")
    .add_local_file(
        "checkpoints/neuromancer-gpt-128.pth",
        "/checkpoints/neuromancer-gpt-128.pth",
        copy=True,
    )
    .add_local_file(
        "checkpoints/neuromancer-gpt-512.pth",
        "/checkpoints/neuromancer-gpt-512.pth",
        copy=True,
    )
    .add_local_python_source("main")
    .add_local_python_source("utils")
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=30,
)
def main():
    device = torch.device("cuda")
    checkpoint = torch.load("/checkpoints/neuromancer-gpt-128.pth")
    model = GPT(GPTConfig(vocab_size=len(checkpoint["stoi"]))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    sample_sprawl(model, 100, checkpoint["stoi"], checkpoint["itos"])
