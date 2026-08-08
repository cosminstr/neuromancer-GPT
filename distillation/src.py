import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
import modal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from main import GPT

load_dotenv(Path(__file__).parent / ".env")


def set_seed(seed=42):
    print("setting seed")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("setting gpu seeds")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("finished setting seeds")


class sprawlDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=256):
        super().__init__()
        self.encoded_text = torch.tensor(
            tokenizer.encode(text, add_special_tokens=True, dtype=torch.long)
        )  # this returns a warning because the its ~500k tokens and the
        # tokenizer is for gpt-oss which has max context window ~130k
        # safely ignore the warning as the model will receive 256 token windows
        self.context_window = block_size

    def __len__(self):
        return len(self.encoded_text) - self.context_window

    def __getitem__(self, idx):
        return self.encoded_text[idx : idx + self.context_window], self.encoded_text[
            idx + 1 : idx + self.context_window + 1
        ]


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 201088
    n_layer: int = 4
    n_head: int = 2
    n_embd: int = 64


class DistillationLoss(torch.nn.Module):
    def __init__(self, temperature=2):
        super().__init__()
        self.temperature = temperature

    def forward(self, x, soft_targets, hard_targets, alfa=0.7):
        T = self.temperature

        soft_loss = F.kl_div(
            F.log_softmax(
                x.contiguous().view(-1, x.size(-1)) / T, dim=-1
            ),  # kldiv requires log-scaled inputs
            F.softmax(
                soft_targets.reshape(-1, soft_targets.size(-1)) / T, dim=-1
            ),  # view requires contiguous data in memory
            # reshape handles both cont and non-cont scenarios
            reduction="batchmean",
        )  # cross_entropy should not be used with soft targets in pytorch

        hard_loss = F.cross_entropy(
            x.view(-1, x.size(-1)), hard_targets.view(-1)
        )  # reshape for cross_entropy

        return alfa * (T**2) * soft_loss + (1 - alfa) * hard_loss


def build_gpt_input(tokens, tokenizer):
    """Place the original tokens in a teacher-forced Harmony assistant response."""
    source_marker = "<|teacher_source_tokens|>"
    messages = [
        {
            "role": "system",
            "content": "Write internally consistent William Gibson-like cyberpunk fiction.",
        },
        {"role": "user", "content": "Write a passage of fiction."},
        {"role": "assistant", "content": source_marker},
    ]
    formatted_message = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        reasoning_effort="low",
    )
    prefix, marker, _ = formatted_message.partition(source_marker)
    if not marker:
        raise RuntimeError("Could not locate source marker in formatted teacher input")

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    prefix_tokens = torch.tensor(prefix_ids, dtype=tokens.dtype).unsqueeze(0)
    prefix_tokens = prefix_tokens.expand(tokens.size(0), -1)
    input_ids = torch.cat((prefix_tokens, tokens), dim=1)

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }, len(prefix_ids)


def extract_source_logits(logits, source_start, source_length):
    """Return next-token logits produced at each original source position."""
    return logits[:, source_start : source_start + source_length]


app = modal.App("gpt-neuromancer-distillation")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "tqdm",
        "transformers",
        "tiktoken",
        "openai-harmony",
        "accelerate",
        "kernels>=0.12.0,<0.16.0",
        "bitsandbytes>=0.46.1",
        "wandb",
        "python-dotenv",
    )
    .add_local_file("corpus.txt", "/root/corpus.txt", copy=True)
    .add_local_file("sprawl.txt", "/root/sprawl.txt", copy=True)
    .env(
        {
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "HF_TOKEN": os.environ["HF_TOKEN"],
        }
    )
    .add_local_python_source("utils")
    .add_local_python_source("main")
)
checkpoint_volume = modal.Volume.from_name("gpt-neuromancer")
hf_volume = modal.Volume.from_name("hf")


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 24,
    volumes={"/checkpoints": checkpoint_volume, "/hf": hf_volume},
)
def distill(epochs, text, checkpoint_name, wandb_run_name, wandb_entity_name):
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained(
        "/hf/openai/gpt-oss-20b", local_files_only=True
    )
    tokenizer.padding_side = "left"
    device = torch.device("cuda:0")
    teacher = AutoModelForCausalLM.from_pretrained(
        "/hf/openai/gpt-oss-20b",
        device_map={"": device},
        torch_dtype="auto",
        local_files_only=True,
    )
    teacher.eval()

    print(f"Teacher on {device}; student on {device}")

    with open(f"/root/{text}.txt") as f:
        text = f.read()

    dataset = sprawlDataset(text, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    config = GPTConfig(vocab_size=teacher.get_output_embeddings().out_features)
    checkpoint_path = f"/checkpoints/{checkpoint_name}"
    checkpoint = (
        torch.load(checkpoint_path) if os.path.exists(checkpoint_path) else None
    )

    base_lr = 2.1e-4
    criterion = DistillationLoss()
    total_steps = epochs * len(loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    if checkpoint:  # continue a run
        config = GPTConfig(**checkpoint["config"])
        student = GPT(config).to(device)
        student.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.AdamW(student.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print("Loaded pre-saved model for further distillation")
        print(
            f"Loaded:\nConfig: {checkpoint['config']}\nEpoch: {checkpoint['epoch']}\nLoss: {checkpoint['loss']}\nlr:{optimizer.param_groups[0]['lr']:.2e}"
        )
    else:
        student = GPT(config).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    wandb_run = wandb.init(
        entity=wandb_entity_name, project="neuromancer-GPT", name=wandb_run_name
    )

    global_loss = float("inf")
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        steps = 0
        for x, y in tqdm(loader):
            steps += 1
            optimizer.zero_grad()
            gpt_input, source_start = build_gpt_input(x, tokenizer)
            gpt_input = {name: value.to(device) for name, value in gpt_input.items()}
            x, y = x.to(device), y.to(device)
            with torch.inference_mode():
                teacher_output = teacher(**gpt_input)
                teacher_logits = extract_source_logits(
                    teacher_output.logits,
                    source_start,
                    x.size(1),
                ).to(device)

            predicted_logits, _ = student(x, y)
            loss = criterion(predicted_logits, teacher_logits, y)
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()

            if steps % 1000 == 0:
                sample_sprawl(student, 200, tokenizer)

                torch.save(
                    {
                        "model_state_dict": student.state_dict(),
                        "config": config.__dict__,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "step": steps,  # total steps in current epoch (round to multiples of 1k)
                        "loss": total_loss / steps,
                    },
                    f"/checkpoints/{checkpoint_name}",
                )
                checkpoint_volume.commit()

            wandb_run.log({"loss": total_loss / steps})

        epoch_loss = total_loss / len(loader)
        print(
            f"Loss at epoch {epoch}/{epochs} -> {epoch_loss} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if epoch_loss < global_loss:
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "config": config.__dict__,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": steps,
                    "loss": total_loss / steps,
                },
                f"/checkpoints/{checkpoint_name}",
            )
            checkpoint_volume.commit()
            global_loss = epoch_loss


def sample_sprawl(
    model, length, tokenizer, input_text="He was now inside the matrix, the grid,"
):
    model.eval()

    tokens = torch.tensor(tokenizer.encode(input_text), dtype=torch.long)
    tokens = tokens.unsqueeze(0).to(next(model.parameters()).device)

    while tokens.size(1) < length:
        with torch.no_grad():
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)

    text = tokens[0, :length].tolist()
    decoded_text = tokenizer.decode(text)
    print(">", decoded_text)
    model.train()


@app.local_entrypoint()
def main(
    checkpoint_name: str,
    wandb_run_name: str,
    wandb_entity_name: str,
    print_params: bool = False,
    corpus: Literal["sprawl", "corpus"] = "corpus",
    epochs: int = 5,
):

    if print_params:
        config = GPTConfig()
        model = GPT(config)
        model.print_num_parameters()
    else:
        distill.remote(
            epochs,
            corpus,
            checkpoint_name,
            wandb_run_name,
            wandb_entity_name,
        )
