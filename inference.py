import modal
import torch

from main import GPT, GPTConfig, sample_sprawl

app = modal.App("gpt-neuromancer-inference")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "tqdm")
    .add_local_python_source("main")
    .add_local_python_source("utils")
)
checkpoint_volume = modal.Volume.from_name("gpt-neuromancer")


@app.function(
    image=image,
    gpu="L40S",
    timeout=30,
    volumes={"/checkpoints": checkpoint_volume},
)
def main(checkpoint_name: str):
    device = torch.device("cuda")
    checkpoint = torch.load(f"/checkpoints/{checkpoint_name}")
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    sample_sprawl(model, 100, checkpoint["stoi"], checkpoint["itos"])
