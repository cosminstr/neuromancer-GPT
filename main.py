import math
from tqdm import tqdm
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from utils import generate_vocab, encode, decode

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working on {device}")


with open("sprawl.txt", "r") as f:
    sprawl = f.read()

sprawl_table, stoi, itos = generate_vocab(sprawl)


class sprawlDataset(Dataset):
    def __init__(self, text, stoi):
        super().__init__()
        self.encoded_text = torch.tensor(encode(text, stoi), dtype=torch.long)

    def __len__(self):
        return len(self.encoded_text)

    def __getitem__(self, idx):
        return self.encoded_text[idx : idx + 128], self.encoded_text[
            idx + 1 : idx + 129
        ]


sprawl_dataset = sprawlDataset(sprawl, stoi)
dataloader = DataLoader(sprawl_dataset, batch_size=32, drop_last=True)


@dataclass
class GPTConfig:
    block_size: int = 128  # max sequence length (reduced for demo)
    vocab_size: int = 18413  # number of tokens in vocabulary
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 6
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # 7
        att = F.softmax(att, dim=-1)
        y = att @ v  # 8
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 9
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        #         self.lm_head = self.gpt.wte.weight
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.size()
        assert T <= self.config.block_size, (
            f"Cannot forward seq of sequence of length {T}, block size is only {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.gpt.wpe(pos)
        tok_emb = self.gpt.wte(x)
        x = tok_emb + pos_emb

        for block in self.gpt.h:
            x = block(x)

        x = self.gpt.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


def train(model, loader, device, epochs):
    model.train()
    base_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    total_steps = epochs * len(loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            output, loss = model(x, y)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        print(
            f"Loss at epoch {epoch}/{epochs} -> {total_loss / len(loader)} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )


def sanity_check(model, loader, device):
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    print(y.min(), y.max())
    print(y.shape)

    output, loss = model(x, y)

    print(output.shape)
    print(loss.item())
    print(-torch.log(torch.tensor(1 / 18413)).item())


def sample_sprawl(model, length):
    model.eval()

    tokens = torch.tensor(
        encode("It was a dark rainy afternoon,", stoi), dtype=torch.long
    )
    tokens = tokens.to(device)
    tokens.unsqueeze(0)

    while tokens.size(1) < length:
        with torch.no_grad():
            output = model(tokens)
            logits = output[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)

    for i in range(length):
        text = tokens[i, :length].tolist()
        decoded_text = decode(text, itos)
        print(">", decoded_text)


net = GPT(GPTConfig)
net = net.to(device)


sanity_check(net, dataloader, device)
train(net, dataloader, device, 10)
sample_sprawl(net, 100)
