import torch
from model import Block
from parallel import TPBlock

torch.manual_seed(0)
d, heads = 512, 8
base = Block(d, heads).cuda()

x = torch.randn(2, 64, d, device="cuda", requires_grad=True)
w = torch.randn(2, 64, d, device="cuda")      # random weights so the loss is not trivial

y0 = base(x)
(y0 * w).sum().backward()
g0 = x.grad.clone()
x.grad = None

for N in [1, 2, 4, 8]:
    tp = TPBlock(d, heads, N).cuda()
    tp.load_from(base)
    y1 = tp(x)
    (y1 * w).sum().backward()
    g1 = x.grad.clone()
    x.grad = None
    ok = torch.allclose(y0, y1, atol=1e-4, rtol=1e-4) and torch.allclose(g0, g1, atol=1e-4, rtol=1e-4)
    print(f"N={N}: output diff={(y0 - y1).abs().max().item():.2e}  "
          f"grad diff={(g0 - g1).abs().max().item():.2e}  match={ok}")