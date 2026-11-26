"""Train the SAME tiny GPT twice (same init, same data order):
   (1) with ordinary transformer blocks, (2) with tensor-parallel blocks (simulated shards).
If tensor parallelism is correct, both loss curves must overlay.

Run:  python train_demo.py        (about 5-8 minutes on an RTX-class GPU)
"""
import glob
import os
import sysconfig
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Block
from parallel import TPBlock

# ------------------------- config -------------------------
D, HEADS, LAYERS, SHARDS = 512, 8, 6, 4        # HEADS must be divisible by SHARDS
CTX, BATCH, STEPS, EVAL_EVERY, LR = 256, 32, 1000, 50, 3e-4
SEED = 1234
dev = "cuda"

# ------------------------- data ---------------------------
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text():
    if os.path.exists("input.txt"):
        return open("input.txt", encoding="utf-8").read(), "Tiny Shakespeare"
    try:
        urllib.request.urlretrieve(URL, "input.txt")
        return open("input.txt", encoding="utf-8").read(), "Tiny Shakespeare"
    except Exception:
        # offline fallback: Python's own standard-library source code
        files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))[:200]
        text = "".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)
        return text[:2_000_000], "Python stdlib source (offline fallback)"


text, corpus_name = load_text()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]
print(f"Corpus: {corpus_name}, {len(text):,} characters, vocabulary {V}")


def get_batch(split, gen):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - CTX - 1, (BATCH,), generator=gen)
    x = torch.stack([src[i:i + CTX] for i in ix]).to(dev)
    y = torch.stack([src[i + 1:i + CTX + 1] for i in ix]).to(dev)
    return x, y


# ------------------------- model --------------------------
class GPT(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss


torch.manual_seed(SEED)
base = GPT([Block(D, HEADS) for _ in range(LAYERS)]).to(dev)

tp_blocks = []
for b in base.blocks:
    t = TPBlock(D, HEADS, SHARDS).to(dev)
    t.load_from(b)                                   # identical starting weights
    tp_blocks.append(t)
tp = GPT(tp_blocks).to(dev)
for name in ["tok", "pos", "ln", "head"]:
    getattr(tp, name).load_state_dict(getattr(base, name).state_dict())

n_base = sum(p.numel() for p in base.parameters())
n_tp = sum(p.numel() for p in tp.parameters())
print(f"Parameters: normal {n_base / 1e6:.2f} M, tensor-parallel {n_tp / 1e6:.2f} M (should be equal)")

g_eval = torch.Generator().manual_seed(999)
val_batches = [get_batch("val", g_eval) for _ in range(10)]   # fixed validation set


# ------------------------- training -----------------------
def train(model, label):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    gen = torch.Generator().manual_seed(SEED)       # same batch order for both models
    train_losses, val_rows = [], []
    t0 = time.time()
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(model(x, y)[1].item() for x, y in val_batches) / len(val_batches)
            model.train()
            val_rows.append((step, vl))
            recent = np.mean(train_losses[-EVAL_EVERY:]) if train_losses else float("nan")
            print(f"[{label}] step {step:4d}  train {recent:.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)")
        if step == STEPS:
            break
        x, y = get_batch("train", gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_losses.append(loss.item())
    return np.array(train_losses), val_rows, time.time() - t0


@torch.no_grad()
def generate(model, n=300):
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    for _ in range(n):
        logits, _ = model(idx[:, -CTX:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1, generator=gen)], dim=1)
    return "".join(chars[i] for i in idx[0].tolist())


tl_base, val_base, t_base = train(base, "normal ")
tl_tp, val_tp, t_tp = train(tp, "tensor-parallel")

# ------------------------- results ------------------------
pd.DataFrame({"step": np.arange(1, STEPS + 1), "normal_train_loss": tl_base,
              "tp_train_loss": tl_tp}).to_csv("train_demo_losses.csv", index=False)
pd.DataFrame({"step": [s for s, _ in val_base], "normal_val_loss": [v for _, v in val_base],
              "tp_val_loss": [v for _, v in val_tp]}).to_csv("train_demo_val.csv", index=False)


def smooth(a, k=20):
    return np.convolve(a, np.ones(k) / k, mode="valid")


diff = np.abs(tl_base - tl_tp)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(np.arange(len(smooth(tl_base))) + 20, smooth(tl_base), label="Normal block (train)")
axes[0].plot(np.arange(len(smooth(tl_tp))) + 20, smooth(tl_tp), "--", label=f"Tensor-parallel, {SHARDS} shards (train)")
axes[0].plot([s for s, _ in val_base], [v for _, v in val_base], "o", ms=4, label="Normal (val)")
axes[0].plot([s for s, _ in val_tp], [v for _, v in val_tp], "x", ms=5, label="Tensor-parallel (val)")
axes[0].set_xlabel("Training step")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].set_title("Loss curves overlay")
axes[0].legend()
axes[1].semilogy(np.arange(1, STEPS + 1), diff + 1e-9)
axes[1].set_xlabel("Training step")
axes[1].set_ylabel("|loss normal - loss tensor-parallel|")
axes[1].set_title("Per-step difference between the two runs")
plt.tight_layout()
plt.savefig("train_demo.png", dpi=150)

print("\n================ SUMMARY ================")
print(f"Final val loss: normal {val_base[-1][1]:.4f} | tensor-parallel {val_tp[-1][1]:.4f}")
print(f"Max  |train loss diff| in first 100 steps: {diff[:100].max():.2e}")
print(f"Mean |train loss diff| over all steps    : {diff.mean():.2e}")
print(f"Wall time: normal {t_base:.0f}s, tensor-parallel (simulated, sequential shards) {t_tp:.0f}s")
print("\n--- sample from NORMAL model ---\n" + generate(base))
print("\n--- sample from TENSOR-PARALLEL model ---\n" + generate(tp))
print("\nSaved train_demo.png, train_demo_losses.csv, train_demo_val.csv")
