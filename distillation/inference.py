import os
from dataclasses import dataclass

from dotenv import load_dotenv
import modal
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from main import GPT

load_dotenv()


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 201088
    n_layer: int = 4
    n_head: int = 2
    n_embd: int = 64


def sample_sprawl(
    model, length, tokenizer, input_text="He was now inside the matrix, the grid,"
):
    model.eval()

    context_size = GPTConfig().block_size

    tokens = torch.tensor(tokenizer.encode(input_text), dtype=torch.long)
    tokens = tokens.unsqueeze(0).to(next(model.parameters()).device)

    while tokens.size(1) < length:
        with torch.no_grad():
            tokens_id = tokens if tokens.size(1) <= context_size else tokens[:, -context_size:] 
            logits, _ = model(tokens_id)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)

    text = tokens[0, :length].tolist()
    decoded_text = tokenizer.decode(text)
    print(">", decoded_text)


app = modal.App("gpt-neuromancer-inference")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "tqdm", "transformers", "python-dotenv", "wandb")
    .env(
        {
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        }
    )
    .add_local_python_source("main")
    .add_local_python_source("utils")
)
checkpoint_volume = modal.Volume.from_name("gpt-neuromancer")
hf_volume = modal.Volume.from_name("hf")


@app.function(
    image=image,
    gpu="L40S",
    timeout=30,
    volumes={"/checkpoints": checkpoint_volume, "/hf": hf_volume},
)
def main(checkpoint_name: str):
    device = torch.device("cuda")
    checkpoint = torch.load(f"/checkpoints/{checkpoint_name}")
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "/hf/openai/gpt-oss-20b", local_files_only=True
    )

    sample_sprawl(model, 500, tokenizer, input_text="It was a dark rainy afternoon,")
