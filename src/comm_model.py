import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, B, T, BYTES = 16384, 4, 512, 4          # same block that OOMed on one device
N_ALLREDUCE = 4                            # 2 forward + 2 backward per block
LATENCY = 20e-6                            # assumed 20 microseconds per hop
LINKS = {"PCIe 3.0 (~10 GB/s)": 10e9,
         "PCIe 4.0 (~25 GB/s)": 25e9,
         "NVLink (~300 GB/s)": 300e9}

tp = pd.read_csv("tp_memory_results.csv")
tp = tp[tp.status == "OK"]                  # measured compute time per device

msg = B * T * D * BYTES                     # bytes in ONE all-reduce
print(f"One all-reduce moves a {msg / 1e6:.0f} MB tensor")

rows = []
for _, r in tp.iterrows():
    N, compute = int(r.shards), float(r.time_s)
    for name, bw in LINKS.items():
        traffic = 2 * (N - 1) / N * msg     # ring all-reduce bytes per device
        comm = N_ALLREDUCE * (traffic / bw + LATENCY * 2 * (N - 1))
        rows.append(dict(shards=N, link=name, compute_s=round(compute, 3),
                         comm_s=round(comm, 4),
                         total_s=round(compute + comm, 3),
                         comm_pct=round(100 * comm / (compute + comm), 1)))
df = pd.DataFrame(rows)
df.to_csv("comm_model_results.csv", index=False)
print(df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, (name, _) in zip(axes, LINKS.items()):
    sub = df[df.link == name]
    x = sub.shards.astype(str)
    ax.bar(x, sub.compute_s, label="Compute (measured)")
    ax.bar(x, sub.comm_s, bottom=sub.compute_s, label="All-reduce (modelled)", color="orange")
    ax.set_title(name)
    ax.set_xlabel("Devices")
axes[0].set_ylabel("Time per training step (s)")
axes[0].legend()
plt.suptitle(f"Tensor parallelism, hidden size {D}: compute vs communication")
plt.tight_layout()
plt.savefig("comm_model.png", dpi=150)
print("Saved comm_model_results.csv and comm_model.png")