import modal
import torch

from main import GPT

global_config = {
    "gpt-2": {  # this was trained on the original sprawl-only corpus.
        "block_size": 128,
        "vocab_size": 18413,  # 25497 for corpus.txt
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    },
    "lightweight": {
        "block_size": 256,
        "vocab_size": 25497,
        "n_layer": 4,
        "n_head": 2,
        "n_embd": 64,
    },
}


@dataclass
class GPTConfig:
    block_size: int = global_config["lightweight"]["block_size"]  # max sequence length
    vocab_size: int = global_config["lightweight"][
        "vocab_size"
    ]  # number of tokens in vocabulary
    n_layer: int = global_config["lightweight"][
        "n_layer"
    ]  # number of transformer blocks
    n_head: int = global_config["lightweight"]["n_head"]  # number of attention heads
    n_embd: int = global_config["lightweight"]["n_embd"]  # embedding dimension



def sample_sprawl(model, length, stoi, itos):
    model.eval()
    context_window = GPTConfig().vocab_size

    tokens = torch.tensor(
        encode("He was now inside the matrix, the grid,", stoi), dtype=torch.long
    )
    tokens = tokens.unsqueeze(0).to(next(model.parameters()).device)

    while tokens.size(1) < length:
        with torch.no_grad():
            tokens_id = tokens if tokens.size(1) <= context_window else tokens[:, -context_window:]
            logits, _ = model(tokens_id)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)

    text = tokens[0, :length].tolist()
    decoded_text = decode(text, itos)
    print(">", decoded_text)



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

    sample_sprawl(model, 500, checkpoint["stoi"], checkpoint["itos"])
