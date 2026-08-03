import os
from pathlib import Path
from dotenv import load_dotenv
import modal
import wandb
from tqdm import tqdm
import torch
import math
from torch.utils.data import Dataset, DataLoader
from main import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Literal
import torch.nn.functional as F

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
    """Apply harmony template to the paragraphs from sprawl.txt for gpt-oss usage"""
    text_batch = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    system_prompt = "You are a teacher model for a distillation task. Your job is to create 'soft targets' to train the student model. For a given sequence, create minimum 512 additional tokens. This is William Gibson-like fiction. Keep the style consistent."
    messages = []

    for text in text_batch:
        messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
        )

    formatted_messages = []

    for m in messages:
        formatted_messages.append(
            tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
        )

    inputs = tokenizer(
        formatted_messages,
        return_tensors="pt",
        padding=True,
    )  # not on device

    return inputs


def extract_source_logits(input_ids, logits, source_tokens):
    """Return logits at the formatted input positions matching source_tokens."""
    outputs = []

    for batch_index, token_tensor in enumerate(input_ids):
        token_ids = token_tensor.tolist()
        source_ids = source_tokens[batch_index].tolist()
        source_start = next(
            (
                index
                for index in range(len(token_ids) - len(source_ids), -1, -1)
                if token_ids[index : index + len(source_ids)] == source_ids
            ),
            None,
        )
        if source_start is None:
            raise RuntimeError(
                "Could not locate source tokens in formatted teacher input"
            )

        # The logit at each source position predicts the following source token.
        outputs.append(
            logits[batch_index, source_start : source_start + len(source_ids) - 1]
        )

    return torch.stack(outputs)


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
        "kernels>=0.12.0",
        "bitsandbytes>=0.46.1",
        "wandb",
        "python-dotenv",
    )
    .add_local_file("corpus.txt", "/root/corpus.txt", copy=True)
    .add_local_file("sprawl.txt", "/root/sprawl.txt", copy=True)
    .env({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]})
    .add_local_python_source("utils")
    .add_local_python_source("main")
)
checkpoint_volume = modal.Volume.from_name("gpt-neuromancer")
hf_volume = modal.Volume.from_name("hf")


@app.function(
    image=image,
    gpu="h100:2",
    timeout=60 * 60 * 24,
    volumes={"/checkpoints": checkpoint_volume, "/hf": hf_volume},
)
def distill(epochs, text):
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained(
        "/hf/openai/gpt-oss-20b", local_files_only=True
    )
    tokenizer.padding_side = "left"
    teacher_device = torch.device("cuda:0")
    student_device = torch.device("cuda:1")
    teacher = AutoModelForCausalLM.from_pretrained(
        "/hf/openai/gpt-oss-20b",
        device_map={"": teacher_device},
        torch_dtype="auto",
        local_files_only=True,
    )
    teacher.eval()

    global_config = {
        "checkpoint_name": "neuromancer-distilled-256-init.pth",
        "lightweight": {
            "block_size": 256,
            "vocab_size": teacher.get_output_embeddings().out_features,
            "n_layer": 4,
            "n_head": 2,
            "n_embd": 64,
        },
    }

    wandb_run = wandb.init(
        entity="personal-cosmin", project="neuromancer-GPT", name="distillation-1"
    )

    @dataclass
    class GPTConfig:
        block_size: int = global_config["lightweight"][
            "block_size"
        ]  # max sequence length
        vocab_size: int = global_config["lightweight"][
            "vocab_size"
        ]  # number of tokens in vocabulary
        n_layer: int = global_config["lightweight"][
            "n_layer"
        ]  # number of transformer blocks
        n_head: int = global_config["lightweight"][
            "n_head"
        ]  # number of attention heads
        n_embd: int = global_config["lightweight"]["n_embd"]  # embedding dimension

    print(f"Teacher on {teacher_device}; student on {student_device}")

    with open(f"/root/{text}.txt") as f:
        text = f.read()

    dataset = sprawlDataset(text, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    config = GPTConfig()
    student = GPT(config).to(student_device)
    base_lr = 3e-4
    optimizer = torch.optim.AdamW(student.parameters(), lr=base_lr)
    criterion = DistillationLoss()
    total_steps = epochs * len(loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    global_loss = float("inf")
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        steps = 0
        for x, y in tqdm(loader):
            steps += 1
            optimizer.zero_grad()
            teacher_source = torch.cat((x, y[:, -1:]), dim=1)
            gpt_input = build_gpt_input(teacher_source, tokenizer).to(teacher_device)
            x, y = x.to(student_device), y.to(student_device)
            with torch.inference_mode():
                teacher_output = teacher(**gpt_input)
                teacher_logits = extract_source_logits(
                    gpt_input.input_ids,
                    teacher_output.logits,
                    teacher_source,
                ).to(student_device)

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
                        "epoch": epoch,
                        "step": steps,
                        "loss": total_loss / steps,
                    },
                    f"/checkpoints/{global_config['checkpoint_name']}",
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
                    "epoch": epoch,
                    "loss": epoch_loss,
                },
                f"/checkpoints/{global_config['checkpoint_name']}",
            )
            checkpoint_volume.commit()
            global_loss = epoch_loss


def sample_sprawl(model, length, tokenizer):
    model.eval()

    tokens = torch.tensor(
        tokenizer.encode("He was now inside the matrix, the grid,"), dtype=torch.long
    )
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
def main(print_params=False, corpus: Literal["sprawl", "corpus"] = "corpus"):

    if print_params:
        global_config = {
            "checkpoint_name": "neuromancer-distilled-256.pth",
            "lightweight": {
                "block_size": 256,
                "vocab_size": 201088,
                "n_layer": 4,
                "n_head": 2,
                "n_embd": 64,
            },
        }

        @dataclass
        class GPTConfig:
            block_size: int = global_config["lightweight"][
                "block_size"
            ]  # max sequence length
            vocab_size: int = global_config["lightweight"][
                "vocab_size"
            ]  # number of tokens in vocabulary
            n_layer: int = global_config["lightweight"][
                "n_layer"
            ]  # number of transformer blocks
            n_head: int = global_config["lightweight"][
                "n_head"
            ]  # number of attention heads
            n_embd: int = global_config["lightweight"]["n_embd"]  # embedding dimension

        config = GPTConfig()
        model = GPT(config)
        model.print_num_parameters()
    else:
        distill.remote(5, corpus)
