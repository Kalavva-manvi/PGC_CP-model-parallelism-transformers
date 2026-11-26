import gc, time
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parallel import Shard

torch.cuda.set_per_process_memory_fraction(0.92)   # real limit, no spilling to RAM

D, B, T = 16384, 4, 512
HEADS = D // 64

def one_device(n_shards):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.device("cuda"):
        shard = Shard(D, HEADS, n_shards)            # only THIS device's slice
        x = torch.randn(B, T, D, requires_grad=True)
    torch.cuda.synchronize(); t0 = time.time()
    h = x + shard.attn_partial(x)                    # stand-in for "after all-reduce"
    out = shard.mlp_partial(h)
    out.sum().backward()
    torch.cuda.synchronize()
    return sum(p.numel() for p in shard.parameters()) / 1e6, torch.cuda.max_memory_allocated() / 1e9, time.time() - t0

one_device(8)   # warm-up so the first timed run isn't inflated
rows = []
for N in [1, 2, 4, 8]:
    try:
        p, peak, sec = one_device(N)
        rows.append(dict(shards=N, params_M_per_device=round(p), peak_GB_per_device=round(peak, 2),
                         time_s=round(sec, 3), status="OK"))
    except torch.cuda.OutOfMemoryError:
        rows.append(dict(shards=N, params_M_per_device=round(12 * D * D / N / 1e6),
                         peak_GB_per_device=None, time_s=None, status="OOM"))
    gc.collect(); torch.cuda.empty_cache()
    print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv("tp_memory_results.csv", index=False)

ok = df[df.status == "OK"]
oom = df[df.status == "OOM"]
limit = torch.cuda.get_device_properties(0).total_memory / 1e9
plt.figure(figsize=(7, 4))
plt.bar(ok.shards.astype(str), ok.peak_GB_per_device, label="Peak memory per device")
if len(oom):
    plt.bar(oom.shards.astype(str), [limit] * len(oom), color="none", edgecolor="red",
            hatch="//", label="Out of memory")
plt.axhline(limit, color="r", ls="--", label=f"GPU limit ({limit:.1f} GB)")
plt.xlabel("Number of devices (tensor-parallel shards)")
plt.ylabel("Peak GPU memory per device (GB)")
plt.title(f"Block with hidden size {D}: memory per device")
plt.legend(); plt.tight_layout()
plt.savefig("tp_memory.png", dpi=150)
print("Saved tp_memory_results.csv and tp_memory.png")