import torch
from model import Block

torch.manual_seed(0)
d, heads, L = 256, 4, 8
layers = [Block(d, heads).cuda() for _ in range(L)]
full = torch.nn.Sequential(*layers)

x = torch.randn(8, 64, d, device="cuda", requires_grad=True)
w = torch.randn(8, 64, d, device="cuda")

y0 = full(x)
(y0 * w).sum().backward()
g0 = x.grad.clone()
x.grad = None

for S, M in [(2, 2), (4, 4), (4, 8)]:          # S = stages, M = micro-batches
    k = L // S
    stages = [torch.nn.Sequential(*layers[i * k:(i + 1) * k]) for i in range(S)]
    outs = []
    for xm in x.chunk(M):                       # split the batch into micro-batches
        h = xm
        for st in stages:                       # pass through the stages in order
            h = st(h)
        outs.append(h)
    y1 = torch.cat(outs)
    (y1 * w).sum().backward()
    g1 = x.grad.clone()
    x.grad = None
    ok = torch.allclose(y0, y1, atol=1e-4, rtol=1e-4) and torch.allclose(g0, g1, atol=1e-4, rtol=1e-4)
    print(f"stages={S} micro-batches={M}: output diff={(y0 - y1).abs().max().item():.2e}  "
          f"grad diff={(g0 - g1).abs().max().item():.2e}  match={ok}")