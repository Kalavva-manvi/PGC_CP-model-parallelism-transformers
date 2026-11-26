"""REAL tensor parallelism with torch.distributed (one process per GPU).

Run on a machine with 2+ GPUs (e.g. Kaggle "GPU T4 x2"):
    torchrun --nproc_per_node=2 real_tp.py --mode all

Modes:  correct   real all-reduce output + gradients == ordinary single-device block
        allreduce measured all-reduce time / bandwidth vs message size
        bench     measured step time: single device vs tensor-parallel (compute vs communication split)
        big       a block that goes OOM on ONE GPU runs when split across the GPUs
        all       everything above

Optional tiny CPU test on any machine (real processes, gloo backend, no GPU needed):
    torchrun --nproc_per_node=2 real_tp.py --mode correct --backend gloo --device cpu
"""
import argparse
import copy
import gc
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from model import Block
from parallel import Shard, TPBlock

COMM = True     # set False to time compute only (numerically wrong - timing experiments only)
RANK = 0


def set_comm(flag):
    global COMM
    COMM = flag


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def reset_peak(dev):
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)


def peak_gb(dev):
    return torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0.0


def free(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()


# ------------------------------------------------------------------
# Megatron-style communication operators (autograd aware)
#   f: identity in forward, all-reduce in backward  (before a column-parallel layer)
#   g: all-reduce in forward, identity in backward  (after a row-parallel layer)
# ------------------------------------------------------------------
class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        g = g.clone()
        if COMM:
            dist.all_reduce(g)
        return g


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        if COMM:
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, g):
        return g


class DistTPBlock(nn.Module):
    """The part of a transformer block that lives on THIS rank (one shard)."""

    def __init__(self, d, heads, world):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.shard = Shard(d, heads, world)
        self.attn_bias = nn.Parameter(torch.zeros(d))
        self.mlp_bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        h = _CopyToTP.apply(self.ln1(x))
        x = x + _ReduceFromTP.apply(self.shard.attn_partial(h)) + self.attn_bias
        h = _CopyToTP.apply(self.ln2(x))
        return x + _ReduceFromTP.apply(self.shard.mlp_partial(h)) + self.mlp_bias


def make_dist_block(base_block, d, heads, rank, world, dev):
    """Slice this rank's shard out of an ordinary block (reuses the tested load_from)."""
    full = TPBlock(d, heads, world)
    full.load_from(base_block)
    blk = DistTPBlock(d, heads, world)
    blk.ln1.load_state_dict(full.ln1.state_dict())
    blk.ln2.load_state_dict(full.ln2.state_dict())
    blk.shard.load_state_dict(full.shards[rank].state_dict())
    with torch.no_grad():
        blk.attn_bias.copy_(full.attn_bias)
        blk.mlp_bias.copy_(full.mlp_bias)
    return blk.to(dev)


# ------------------------------------------------------------------
# timing helpers
# ------------------------------------------------------------------
def timed(fn, dev, iters=3):
    fn()                                   # warm-up
    sync(dev)
    dist.barrier()
    t0 = time.time()
    for _ in range(iters):
        fn()
    sync(dev)
    sec = (time.time() - t0) / iters
    t = torch.tensor([sec], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)   # report the slowest rank
    return t.item()


def time_allreduce(nbytes, dev, iters=10):
    t = torch.ones(nbytes // 4, device=dev)
    for _ in range(3):
        dist.all_reduce(t)
    sync(dev)
    dist.barrier()
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(t)
    sync(dev)
    return (time.time() - t0) / iters


# ------------------------------------------------------------------
# Mode 1: correctness with REAL communication
# ------------------------------------------------------------------
def run_correct(rank, world, dev):
    d, heads = 512, 8
    torch.manual_seed(0)
    base = Block(d, heads)                              # CPU, identical on every rank
    x_cpu, w_cpu = torch.randn(2, 64, d), torch.randn(2, 64, d)

    ref = copy.deepcopy(base).to(dev)
    x = x_cpu.to(dev).requires_grad_(True)
    w = w_cpu.to(dev)
    y0 = ref(x)
    (y0 * w).sum().backward()
    g0 = x.grad.clone()
    x.grad = None

    blk = make_dist_block(base, d, heads, rank, world, dev)
    y1 = blk(x)
    (y1 * w).sum().backward()
    g1 = x.grad.clone()

    dy = (y0 - y1).abs().max().item()
    dg = (g0 - g1).abs().max().item()
    ln_ref = ref.ln1.weight.grad
    dl = (ln_ref - blk.ln1.weight.grad).abs().max().item()
    ok = dy < 1e-3 and dg < 1e-3 and dl < 1e-3 * max(1.0, ln_ref.abs().max().item())

    flag = torch.tensor([1.0 if ok else 0.0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    log(f"[correctness] world={world}  max|output diff|={dy:.2e}  max|input-grad diff|={dg:.2e}  "
        f"max|LayerNorm-grad diff|={dl:.2e}  PASS={bool(flag.item())}")


# ------------------------------------------------------------------
# Mode 2: measured all-reduce bandwidth
# ------------------------------------------------------------------
def run_allreduce(rank, world, dev):
    import pandas as pd
    rows = []
    for mb in [1, 4, 16, 64, 128, 256]:
        nbytes = mb * 1024 * 1024
        sec = time_allreduce(nbytes, dev)
        bus = 2 * (world - 1) / world * nbytes / sec / 1e9      # ring all-reduce bus bandwidth
        rows.append(dict(size_MB=mb, time_ms=round(sec * 1000, 2), bus_GBps=round(bus, 2)))
        log(f"[all-reduce] {mb:4d} MB: {sec * 1000:8.2f} ms   bus bandwidth {bus:6.2f} GB/s")
    if rank == 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.DataFrame(rows)
        df.to_csv("real_allreduce.csv", index=False)
        plt.figure(figsize=(6.5, 4))
        plt.plot(df.size_MB, df.bus_GBps, marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Message size (MB)")
        plt.ylabel("Measured all-reduce bus bandwidth (GB/s)")
        plt.title(f"Real all-reduce between {world} GPUs")
        plt.tight_layout()
        plt.savefig("real_allreduce.png", dpi=150)
        print("Saved real_allreduce.csv and real_allreduce.png", flush=True)


# ------------------------------------------------------------------
# Mode 3: measured step time, single device vs tensor parallel
# ------------------------------------------------------------------
def run_bench(rank, world, dev):
    import pandas as pd
    B, T = 4, 512
    rows = []
    for d in [4096, 8192]:
        heads = d // 64
        torch.manual_seed(0)

        # (a) ordinary block on ONE device
        reset_peak(dev)
        with torch.device(dev):
            blk = Block(d, heads)
            x = torch.randn(B, T, d, requires_grad=True)

        def step_single():
            blk(x).sum().backward()

        single = timed(step_single, dev)
        single_peak = peak_gb(dev)
        del blk, x
        free(dev)

        # (b) tensor parallel: compute only, then compute + real all-reduce
        reset_peak(dev)
        with torch.device(dev):
            tpb = DistTPBlock(d, heads, world)
            xt = torch.randn(B, T, d, requires_grad=True)

        def step_tp():
            tpb(xt).sum().backward()

        set_comm(False)
        t_compute = timed(step_tp, dev)
        set_comm(True)
        t_total = timed(step_tp, dev)
        tp_peak = peak_gb(dev)
        t_ar = time_allreduce(B * T * d * 4, dev)
        del tpb, xt
        free(dev)

        comm = max(t_total - t_compute, 0.0)
        speed = single / t_total
        rows.append(dict(hidden=d, single_s=round(single, 3), tp_compute_s=round(t_compute, 3),
                         tp_total_s=round(t_total, 3), comm_s=round(comm, 3),
                         four_isolated_allreduces_s=round(4 * t_ar, 3),
                         comm_pct=round(100 * comm / t_total, 1),
                         speedup_vs_single=round(speed, 2),
                         efficiency_pct=round(100 * speed / world, 1),
                         single_peak_GB=round(single_peak, 2), tp_peak_GB_per_gpu=round(tp_peak, 2)))
        log(f"[bench] {rows[-1]}")

    if rank == 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.DataFrame(rows)
        df.to_csv("real_tp_bench.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = range(len(df))
        ax.bar([i - 0.2 for i in xs], df.single_s, 0.4, label="Single GPU")
        ax.bar([i + 0.2 for i in xs], df.tp_compute_s, 0.4, label="Tensor parallel: compute")
        ax.bar([i + 0.2 for i in xs], df.comm_s, 0.4, bottom=df.tp_compute_s,
               color="orange", label="Tensor parallel: all-reduce (measured)")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"hidden {h}" for h in df.hidden])
        ax.set_ylabel("Time per training step (s)")
        ax.set_title(f"Measured on {world} real GPUs")
        ax.legend()
        plt.tight_layout()
        plt.savefig("real_tp_bench.png", dpi=150)
        print("Saved real_tp_bench.csv and real_tp_bench.png", flush=True)


# ------------------------------------------------------------------
# Mode 4: a block that does not fit on one GPU, split across the GPUs
# ------------------------------------------------------------------
def run_big(rank, world, dev, d):
    import pandas as pd
    B, T = 4, 512
    heads = d // 64
    params_m = 12 * d * d / 1e6

    reset_peak(dev)
    blk = x = None
    try:
        with torch.device(dev):
            blk = Block(d, heads)
            x = torch.randn(B, T, d, requires_grad=True)
        blk(x).sum().backward()
        single_status = f"fits (peak {peak_gb(dev):.2f} GB)"
    except torch.cuda.OutOfMemoryError:
        single_status = "OUT OF MEMORY"
    blk = x = None
    free(dev)
    dist.barrier()
    log(f"[big] hidden {d} ({params_m:.0f} M parameters), ordinary block on ONE GPU: {single_status}")

    reset_peak(dev)
    with torch.device(dev):
        tpb = DistTPBlock(d, heads, world)
        xt = torch.randn(B, T, d, requires_grad=True)

    def step():
        tpb(xt).sum().backward()

    sec = timed(step, dev, iters=2)
    stats = torch.tensor([peak_gb(dev)], device=dev)
    allp = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(allp, stats)
    peaks = [round(p.item(), 2) for p in allp]
    log(f"[big] tensor-parallel over {world} GPUs: step {sec:.2f} s, peak memory per GPU {peaks} GB")
    if rank == 0:
        pd.DataFrame([dict(hidden=d, params_M=round(params_m), single_gpu=single_status,
                           gpus=world, tp_step_s=round(sec, 2), tp_peak_GB_per_gpu=str(peaks))]
                     ).to_csv("real_tp_big.csv", index=False)
        print("Saved real_tp_big.csv", flush=True)


# ------------------------------------------------------------------
def setup(backend, device):
    local = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        torch.cuda.set_device(local)
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", local) if device == "cuda" else torch.device("cpu")
    return rank, world, dev


def main():
    global RANK
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["correct", "allreduce", "bench", "big", "all"])
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--big-d", type=int, default=14336,
                    help="hidden size for the OOM demo (lower it, e.g. 12288, if the split block still OOMs)")
    a = ap.parse_args()
    if a.device == "cpu" and a.mode not in ("correct",):
        raise SystemExit("--device cpu only supports --mode correct")

    rank, world, dev = setup(a.backend, a.device)
    RANK = rank
    log(f"world size {world}, backend {a.backend}, device {dev}")
    if dev.type == "cuda":
        log("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    if a.mode in ("correct", "all"):
        run_correct(rank, world, dev)
    if a.mode in ("allreduce", "all"):
        run_allreduce(rank, world, dev)
    if a.mode in ("bench", "all"):
        run_bench(rank, world, dev)
    if a.mode in ("big", "all"):
        run_big(rank, world, dev, a.big_d)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
