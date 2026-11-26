import time
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Block

torch.cuda.set_per_process_memory_fraction(0.92)

# Step 1: measure the REAL time of one 6-layer stage on one micro-batch
D, T, LAYERS_MEASURED, TOTAL_LAYERS = 4096, 512, 6, 24
with torch.device("cuda"):
    stage = torch.nn.Sequential(*[Block(D, D // 64) for _ in range(LAYERS_MEASURED)])
    x = torch.randn(1, T, D, requires_grad=True)

def step():
    stage(x).sum().backward()

step(); torch.cuda.synchronize()                 # warm-up
t0 = time.time()
for _ in range(5):
    step()
torch.cuda.synchronize()
t_layer = (time.time() - t0) / 5 / LAYERS_MEASURED
print(f"Measured time per layer (fwd+bwd, 1 micro-batch): {t_layer * 1000:.1f} ms")

# Step 2: GPipe schedule math. Stage s starts micro-batch m at time slot s + m.
rows = []
for S in [2, 4, 8]:
    t_stage = t_layer * (TOTAL_LAYERS // S)
    for M in [1, 2, 4, 8, 16, 32]:
        slots = M + S - 1
        rows.append(dict(stages=S, microbatches=M,
                         bubble_pct=round(100 * (S - 1) / slots, 1),
                         sequential_s=round(M * S * t_stage, 2),
                         pipelined_s=round(slots * t_stage, 2),
                         speedup=round(M * S / slots, 2)))
df = pd.DataFrame(rows)
df.to_csv("pipeline_schedule_results.csv", index=False)
print(df[df.stages == 4].to_string(index=False))

# Chart 1: schedule diagram (who works on what, and when)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
S = 4
for ax, M in zip(axes, [1, 8]):
    colors = plt.cm.tab10.colors
    for s in range(S):
        for m in range(M):
            ax.broken_barh([(s + m, 1)], (s - 0.4, 0.8), facecolors=colors[m % 10], edgecolor="black")
    ax.set_yticks(range(S)); ax.set_yticklabels([f"Device {i}" for i in range(S)])
    ax.invert_yaxis()
    ax.set_xlabel("Time slot")
    ax.set_title(f"{S} stages, {M} micro-batch(es): bubble = {100 * (S - 1) / (M + S - 1):.0f}%")
plt.tight_layout()
plt.savefig("pipeline_gantt.png", dpi=150)

# Chart 2: speedup vs number of micro-batches
plt.figure(figsize=(7, 4))
for S in [2, 4, 8]:
    sub = df[df.stages == S]
    plt.plot(sub.microbatches, sub.speedup, marker="o", label=f"{S} stages")
plt.xscale("log", base=2)
plt.xlabel("Number of micro-batches"); plt.ylabel("Speedup vs one device doing all stages")
plt.title("Pipeline speedup (from measured stage time + GPipe schedule)")
plt.legend(); plt.tight_layout()
print("Saved pipeline_gantt.png, pipeline_speedup.png, pipeline_schedule_results.csv")
