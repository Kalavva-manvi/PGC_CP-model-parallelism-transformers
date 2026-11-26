import gc
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Block

torch.cuda.set_per_process_memory_fraction(0.92)

D, B, T, TOTAL_LAYERS = 4096, 4, 512, 24

def one_stage(S):
    n = TOTAL_LAYERS // S                         # layers held by ONE device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.device("cuda"):
        stage = torch.nn.Sequential(*[Block(D, D // 64) for _ in range(n)])
        x = torch.randn(B, T, D, requires_grad=True)
    stage(x).sum().backward()
    torch.cuda.synchronize()
    return n, sum(p.numel() for p in stage.parameters()) / 1e6, torch.cuda.max_memory_allocated() / 1e9

rows = []
for S in [1, 2, 4, 8]:
    try:
        n, p, peak = one_stage(S)
        rows.append(dict(stages=S, layers_per_device=n, params_M_per_device=round(p),
                         peak_GB_per_device=round(peak, 2), status="OK"))
    except torch.cuda.OutOfMemoryError:
        rows.append(dict(stages=S, layers_per_device=TOTAL_LAYERS // S,
                         params_M_per_device=round(TOTAL_LAYERS // S * 12 * D * D / 1e6),
                         peak_GB_per_device=None, status="OOM"))
    gc.collect(); torch.cuda.empty_cache()
    print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv("pipeline_memory_results.csv", index=False)

ok, oom = df[df.status == "OK"], df[df.status == "OOM"]
limit = torch.cuda.get_device_properties(0).total_memory / 1e9
plt.figure(figsize=(7, 4))
plt.bar(ok.stages.astype(str), ok.peak_GB_per_device, label="Peak memory per device")
if len(oom):
    plt.bar(oom.stages.astype(str), [limit] * len(oom), color="none", edgecolor="red",
            hatch="//", label="Out of memory")
plt.axhline(limit, color="r", ls="--", label=f"GPU limit ({limit:.1f} GB)")
plt.xlabel("Number of pipeline stages (devices)")
plt.ylabel("Peak GPU memory per device (GB)")
plt.title(f"{TOTAL_LAYERS}-layer model, hidden size {D}")
plt.legend(); plt.tight_layout()
plt.savefig("pipeline_memory.png", dpi=150)
print("Saved pipeline_memory_results.csv and pipeline_memory.png")
