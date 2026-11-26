import torch
import torch.nn as nn
import torch.nn.functional as F


class Shard(nn.Module):
    """The slice of one transformer block that lives on ONE device."""
    def __init__(self, d, n_heads, n_shards):
        super().__init__()
        assert n_heads % n_shards == 0 and (4 * d) % n_shards == 0
        self.hp = n_heads // n_shards          # heads on this device
        self.dh = d // n_heads                 # size of each head
        dl = self.hp * self.dh                 # local attention width
        m = 4 * d // n_shards                  # local MLP hidden width
        self.qkv = nn.Linear(d, 3 * dl)                # column-parallel (by heads)
        self.attn_out = nn.Linear(dl, d, bias=False)   # row-parallel (bias added once later)
        self.fc1 = nn.Linear(d, m)                     # column-parallel
        self.fc2 = nn.Linear(m, d, bias=False)         # row-parallel (bias added once later)

    def attn_partial(self, h):
        B, T, _ = h.shape
        qkv = self.qkv(h).view(B, T, 3, self.hp, self.dh).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, self.hp * self.dh)
        return self.attn_out(y)                # PARTIAL result, needs all-reduce

    def mlp_partial(self, h):
        return self.fc2(F.gelu(self.fc1(h)))   # PARTIAL result, needs all-reduce


class TPBlock(nn.Module):
    """A transformer block split across n_shards simulated devices."""
    def __init__(self, d, n_heads, n_shards):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)   # tiny, so replicated
        self.shards = nn.ModuleList([Shard(d, n_heads, n_shards) for _ in range(n_shards)])
        self.attn_bias = nn.Parameter(torch.zeros(d))
        self.mlp_bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        h = self.ln1(x)
        # all-reduce #1 = sum of the partial outputs from every device
        x = x + sum(s.attn_partial(h) for s in self.shards) + self.attn_bias
        h = self.ln2(x)
        # all-reduce #2
        x = x + sum(s.mlp_partial(h) for s in self.shards) + self.mlp_bias
        return x

    @torch.no_grad()
    def load_from(self, block):
        """Copy the weights of a normal Block into the shards (for the correctness test)."""
        N = len(self.shards)
        d = block.ln1.weight.numel()
        h, dh = block.attn.h, block.attn.dh
        hp, m = h // N, 4 * d // N
        self.ln1.load_state_dict(block.ln1.state_dict())
        self.ln2.load_state_dict(block.ln2.state_dict())
        Wqkv = block.attn.qkv.weight.view(3, h, dh, d)
        bqkv = block.attn.qkv.bias.view(3, h, dh)
        for r, s in enumerate(self.shards):
            hs = slice(r * hp, (r + 1) * hp)
            s.qkv.weight.copy_(Wqkv[:, hs].reshape(-1, d))
            s.qkv.bias.copy_(bqkv[:, hs].reshape(-1))
            s.attn_out.weight.copy_(block.attn.out.weight[:, r * hp * dh:(r + 1) * hp * dh])
            s.fc1.weight.copy_(block.mlp.fc1.weight[r * m:(r + 1) * m])
            s.fc1.bias.copy_(block.mlp.fc1.bias[r * m:(r + 1) * m])
            s.fc2.weight.copy_(block.mlp.fc2.weight[:, r * m:(r + 1) * m])
        self.attn_bias.copy_(block.attn.out.bias)
        self.mlp_bias.copy_(block.mlp.fc2.bias)