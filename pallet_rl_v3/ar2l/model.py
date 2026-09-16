"""Transformer policies for the packer, the attacker and the mixture model.

AR2L Sec. A.4: heterogeneous state elements are first projected by independent
element-wise FC layers, mixed by one scaled dot-product attention layer with a
skip connection and an element-wise feed-forward layer, and turned into a
distribution by a pointer head

    pi(.) = softmax(c_temp * tanh(xbar^T x / sqrt(d)))          (Eq. 28)

over the candidate nodes -- the feasible positions L for the packer, the
observable items B for the attacker and the mixture-dynamics model.  Critics
read (C, B) only, which is what Algorithm 2 needs to evaluate a state under
both the nominal and the permuted item sequence.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


def positional_encoding(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(d, device=device, dtype=torch.float32)[None, :]
    ang = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    pe = torch.where(i.long() % 2 == 0, torch.sin(ang), torch.cos(ang))
    return pe


class Encoder(nn.Module):
    """Independent projections + `n_layer` attention blocks over the union."""

    def __init__(self, in_dims, d=64, n_head=1, n_layer=1):
        super().__init__()
        self.embed = nn.ModuleList([nn.Linear(k, d) for k in in_dims])
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(d, n_head, batch_first=True),
                "n1": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)),
                "n2": nn.LayerNorm(d),
            }))
        self.d = d

    def forward(self, parts, masks, pos_on=None):
        """parts/masks: lists of (B, N_k, dim_k) and (B, N_k) bool."""
        xs = []
        for k, (p, m) in enumerate(zip(parts, masks)):
            h = self.embed[k](p)
            if pos_on is not None and k in pos_on:
                h = h + positional_encoding(h.shape[1], self.d, h.device)
            xs.append(h * m.unsqueeze(-1))
        x = torch.cat(xs, 1)
        pad = ~torch.cat(masks, 1)
        # a bin with nothing packed and nothing feasible would be all-padding
        safe = pad.clone()
        safe[:, 0] = False
        for blk in self.blocks:
            a, _ = blk["attn"](x, x, x, key_padding_mask=safe, need_weights=False)
            x = blk["n1"](x + a)
            x = blk["n2"](x + blk["ff"](x))
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        keep = (~pad).sum(1, keepdim=True).clamp(min=1)
        return x, x.sum(1) / keep, pad


class Pointer(nn.Module):
    """Eq. 28: compatibility of every candidate with the global feature."""

    def __init__(self, d=64, c_temp=10.0):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.c_temp, self.d = c_temp, d

    def forward(self, xbar, nodes, mask):
        y = torch.einsum("bd,bnd->bn", self.q(xbar), self.k(nodes)) / math.sqrt(self.d)
        return self.c_temp * torch.tanh(y) + torch.where(mask, 0.0, NEG)


class Critic(nn.Module):
    """V(C, B) -- no dependence on the candidate list."""

    def __init__(self, d=64, n_head=1, n_layer=1):
        super().__init__()
        self.enc = Encoder([6, 3], d, n_head, n_layer)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, c, c_mask, b, b_mask):
        _, xbar, _ = self.enc([c, b], [c_mask, b_mask], pos_on={1})
        return self.head(xbar).squeeze(-1)


class PackNet(nn.Module):
    """Packing policy pi_pack(l | C, B, L) and its value function."""

    def __init__(self, d=64, n_head=1, n_layer=1, c_temp=10.0):
        super().__init__()
        self.enc = Encoder([6, 3, 6], d, n_head, n_layer)
        self.ptr = Pointer(d, c_temp)
        self.critic = Critic(d, n_head, n_layer)

    def logits(self, o):
        x, xbar, _ = self.enc([o["c"], o["b"], o["l"]],
                              [o["c_mask"], o["b_mask"], o["l_mask"]], pos_on={1})
        nl = o["l"].shape[1]
        return self.ptr(xbar, x[:, -nl:], o["l_mask"])

    def value(self, o):
        return self.critic(o["c"], o["c_mask"], o["b"], o["b_mask"])

    def forward(self, o):
        return self.logits(o), self.value(o)


class PermNet(nn.Module):
    """pi(b_i | C, B): pick the observable item to move to the conveyor front.

    Used both for the permutation-based attacker and for the mixture-dynamics
    model; they differ only in the loss they are trained with.
    """

    def __init__(self, d=64, n_head=1, n_layer=1, c_temp=10.0):
        super().__init__()
        self.enc = Encoder([6, 3], d, n_head, n_layer)
        self.ptr = Pointer(d, c_temp)
        self.critic = Critic(d, n_head, n_layer)

    def logits(self, o):
        x, xbar, _ = self.enc([o["c"], o["b"]], [o["c_mask"], o["b_mask"]],
                              pos_on={1})
        nb = o["b"].shape[1]
        return self.ptr(xbar, x[:, -nb:], o["b_mask"])

    def value(self, o):
        return self.critic(o["c"], o["c_mask"], o["b"], o["b_mask"])

    def forward(self, o):
        return self.logits(o), self.value(o)


def sample(logits, greedy=False):
    # a non-finite logit would make Categorical return an out-of-range index,
    # which surfaces much later as an illegal memory access in the gather
    logits = torch.nan_to_num(logits, nan=NEG, posinf=NEG, neginf=NEG)
    logp = F.log_softmax(logits, -1)
    if greedy:
        a = logp.argmax(-1)
    else:
        a = torch.distributions.Categorical(logits=logp).sample()
    ent = -(logp.exp() * logp).sum(-1)
    return a, logp.gather(1, a[:, None]).squeeze(1), ent
