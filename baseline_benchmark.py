import gc, time
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Block

# Hard cap: PyTorch raises OOM at ~92% of GPU memory instead of
# letting Windows spill the overflow into system RAM.
torch.cuda.set_per_process_memory_fraction(0.92)

def run(d, B=4, T=512):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.device("cuda"):            # build weights directly on the GPU
        block = Block(d, d // 64)
        x = torch.randn(B, T, d, requires_grad=True)
    torch.cuda.synchronize()
    t0 = time.time()
    block(x).sum().backward()             # forward + backward pass
    torch.cuda.synchronize()
    sec = time.time() - t0
    params = sum(p.numel() for p in block.parameters())
    peak = torch.cuda.max_memory_allocated() / 1e9
    return params / 1e6, peak, sec

run(512)  # warm-up so timings are fair

rows = []
for d in [1024, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]:
    try:
        params_m, peak, sec = run(d)
        rows.append(dict(hidden=d, params_M=round(params_m), peak_GB=round(peak, 2),
                         time_s=round(sec, 3), status="OK"))
    except torch.cuda.OutOfMemoryError:
        rows.append(dict(hidden=d, params_M=round(12 * d * d / 1e6), peak_GB=None,
                         time_s=None, status="OOM"))
    gc.collect()
    torch.cuda.empty_cache()
    print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv("baseline_results.csv", index=False)

ok = df[df.status == "OK"]
oom = df[df.status == "OOM"]
limit = torch.cuda.get_device_properties(0).total_memory / 1e9

plt.figure(figsize=(7, 4))
plt.plot(ok.hidden, ok.peak_GB, marker="o", label="Peak memory (fits)")
plt.axhline(limit, color="r", ls="--", label=f"GPU limit ({limit:.1f} GB)")
if len(oom) > 0:
    plt.scatter(oom.hidden, [limit] * len(oom), color="red", marker="x",
                s=80, label="Out of memory", zorder=5)
plt.xlabel("Hidden size (d)")
plt.ylabel("Peak GPU memory (GB)")
plt.title("One transformer block: memory vs size (single GPU)")
plt.legend()
plt.tight_layout()
plt.savefig("baseline_memory.png", dpi=150)
print("Saved baseline_results.csv and baseline_memory.png")