import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Figure A: scaling relative to the smallest config that fits ----------
df = pd.read_csv("comm_model_results.csv")
base_n = int(df.shards.min())                  # 2 devices (1 device runs out of memory)
Ns = sorted(df.shards.unique())

plt.figure(figsize=(6.5, 4))
plt.plot(Ns, [n / base_n for n in Ns], "k--", label="Ideal")
first = df[df.link == df.link.iloc[0]].sort_values("shards")
plt.plot(first.shards, first.compute_s.iloc[0] / first.compute_s,
         marker="o", label="Compute only (measured)")
rows = []
for link, sub in df.groupby("link"):
    sub = sub.sort_values("shards")
    sp = sub.total_s.iloc[0] / sub.total_s
    plt.plot(sub.shards, sp, marker="o", label=link)
    for n, s in zip(sub.shards, sp):
        rows.append(dict(link=link, shards=int(n), speedup=round(s, 2),
                         efficiency_pct=round(100 * s / (n / base_n), 1)))
plt.xlabel("Number of devices")
plt.ylabel(f"Speedup vs {base_n} devices")
plt.title("Tensor-parallel scaling (compute measured, comms modelled)")
plt.legend(); plt.tight_layout()
plt.savefig("tp_scaling.png", dpi=150)
out = pd.DataFrame(rows)
out.to_csv("tp_scaling_results.csv", index=False)
print(out.to_string(index=False))

# ---------- Figure B: where does the memory go? ----------
tp = pd.read_csv("tp_memory_results.csv")
tp = tp[tp.status == "OK"]
w = tp.params_M_per_device * 1e6 * 4 / 1e9          # fp32 weights in GB
act = tp.peak_GB_per_device - 2 * w                  # whatever is left over
x = tp.shards.astype(str)
plt.figure(figsize=(6.5, 4))
plt.bar(x, w, label="Weights")
plt.bar(x, w, bottom=w, label="Gradients")
plt.bar(x, act, bottom=2 * w, label="Activations + workspace")
plt.xlabel("Number of devices")
plt.ylabel("Peak memory per device (GB)")
plt.title("Memory breakdown, hidden size 16384")
plt.legend(); plt.tight_layout()
plt.savefig("tp_memory_breakdown.png", dpi=150)
print("Saved tp_scaling.png and tp_memory_breakdown.png")