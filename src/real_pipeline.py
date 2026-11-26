"""REAL pipeline parallelism on 2 GPUs (GPipe-style micro-batching), measured not modelled.

Stage 0 (first half of the layers) lives on cuda:0, stage 1 on cuda:1. CUDA kernels are
asynchronous, so while GPU1 works on micro-batch m, GPU0 can already start micro-batch m+1.

Run on a machine with 2+ GPUs (e.g. Kaggle "GPU T4 x2"):
    python real_pipeline.py
"""
import copy
import gc
import time

import torch
import torch.nn as nn

from model import Block

if torch.cuda.device_count() < 2:
    raise SystemExit("This script needs at least 2 GPUs (e.g. Kaggle 'GPU T4 x2').")

D0, D1 = torch.device("cuda:0"), torch.device("cuda:1")
print("GPUs:", torch.cuda.get_device_name(0), "|", torch.cuda.get_device_name(1))


def sync_all():
    torch.cuda.synchronize(D0)
    torch.cuda.synchronize(D1)


def free_all():
    gc.collect()
    for dv in (D0, D1):
        with torch.cuda.device(dv):
            torch.cuda.empty_cache()


def pipe_forward(s0, s1, x, M):
    """Split x into M micro-batches; each goes stage0 (cuda:0) -> copy -> stage1 (cuda:1)."""
    return [s1(s0(xm).to(D1, non_blocking=True)) for xm in x.chunk(M)]


def pipe_step(s0, s1, x, M):
    outs = pipe_forward(s0, s1, x, M)
    sum(o.sum() for o in outs).backward()


def timed(fn, iters=3):
    fn()
    sync_all()
    t0 = time.time()
    for _ in range(iters):
        fn()
    sync_all()
    return (time.time() - t0) / iters


# ------------------------------------------------------------------
# 1) correctness: pipeline over 2 GPUs == ordinary model on 1 GPU
# ------------------------------------------------------------------
def check_correctness():
    torch.manual_seed(0)
    d, heads, L = 256, 4, 8
    layers = [Block(d, heads) for _ in range(L)]
    single = nn.Sequential(*copy.deepcopy(layers)).to(D0)
    s0 = nn.Sequential(*copy.deepcopy(layers[:L // 2])).to(D0)
    s1 = nn.Sequential(*copy.deepcopy(layers[L // 2:])).to(D1)
    x, w = torch.randn(8, 64, d), torch.randn(8, 64, d)

    xa = x.to(D0).requires_grad_(True)
    y0 = single(xa)
    (y0 * w.to(D0)).sum().backward()
    g0 = xa.grad.cpu()

    xb = x.to(D0).requires_grad_(True)
    y1 = torch.cat(pipe_forward(s0, s1, xb, 4))
    (y1 * w.to(D1)).sum().backward()
    g1 = xb.grad.cpu()

    dy = (y0.cpu() - y1.cpu()).abs().max().item()
    dg = (g0 - g1).abs().max().item()
    ok = dy < 1e-3 and dg < 1e-3
    print(f"[correctness] 2-GPU pipeline vs single GPU: max|output diff|={dy:.2e}  "
          f"max|input-grad diff|={dg:.2e}  PASS={ok}")


check_correctness()
free_all()

# ------------------------------------------------------------------
# 2) measured speedup vs number of micro-batches
# ------------------------------------------------------------------
D, L, B, T = 1024, 16, 32, 256
HEADS = D // 64
S = 2
print(f"\nBenchmark model: {L} layers, hidden {D}, batch {B}, sequence {T}")

# baseline: the whole model on ONE GPU
torch.cuda.reset_peak_memory_stats(D0)
with torch.device(D0):
    single = nn.Sequential(*[Block(D, HEADS) for _ in range(L)])
x = torch.randn(B, T, D, device=D0)


def single_step():
    single(x).sum().backward()


t_single = timed(single_step)
single_peak = torch.cuda.max_memory_allocated(D0) / 1e9
print(f"Single GPU: {t_single:.3f} s per step, peak memory {single_peak:.2f} GB")
del single
free_all()

# pipeline: half the layers per GPU
with torch.device(D0):
    s0 = nn.Sequential(*[Block(D, HEADS) for _ in range(L // 2)])
with torch.device(D1):
    s1 = nn.Sequential(*[Block(D, HEADS) for _ in range(L // 2)])

rows = []
for M in [1, 2, 4, 8, 16, 32]:
    torch.cuda.reset_peak_memory_stats(D0)
    torch.cuda.reset_peak_memory_stats(D1)

    def step():
        pipe_step(s0, s1, x, M)

    t = timed(step)
    rows.append(dict(microbatches=M, time_s=round(t, 3),
                     measured_speedup=round(t_single / t, 2),
                     formula_speedup=round(S * M / (M + S - 1), 2),
                     peak_GB_gpu0=round(torch.cuda.max_memory_allocated(D0) / 1e9, 2),
                     peak_GB_gpu1=round(torch.cuda.max_memory_allocated(D1) / 1e9, 2)))
    print(rows[-1])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(rows)
df["single_gpu_s"] = round(t_single, 3)
df["single_gpu_peak_GB"] = round(single_peak, 2)
df.to_csv("real_pipeline_results.csv", index=False)

plt.figure(figsize=(7, 4))
plt.plot(df.microbatches, df.measured_speedup, marker="o", label="Measured (2 real GPUs)")
plt.plot(df.microbatches, df.formula_speedup, "--", marker="x", label="GPipe formula S*M/(M+S-1)")
plt.axhline(S, color="gray", ls=":", label=f"Ideal ({S}x)")
plt.xscale("log", base=2)
plt.xlabel("Number of micro-batches")
plt.ylabel("Speedup vs single GPU")
plt.title("Pipeline parallelism: measured vs formula")
plt.legend()
plt.tight_layout()
plt.savefig("real_pipeline_speedup.png", dpi=150)
print("\nSaved real_pipeline_results.csv and real_pipeline_speedup.png")
