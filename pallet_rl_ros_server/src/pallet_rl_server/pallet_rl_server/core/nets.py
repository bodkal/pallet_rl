"""The trained policies, inference only (AR2L Sec. A.4).

Each node is projected on its own, mixed by attention, and a pointer head
scores the candidates:  logits = c_temp * tanh(score(xbar, x_i)).

  PackNet  (C, B, L) -> which candidate placement L for the front box
  PermNet  (C, B)    -> which station box B to bring to the front

C = boxes on the pallet, B = conveyor window, L = feasible placements.  The
critics in the checkpoint are not needed to act and are not loaded.
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
    return torch.where(i.long() % 2 == 0, torch.sin(ang), torch.cos(ang))


class Encoder(nn.Module):
    """Per-part projections (+ type embedding), then attention blocks."""

    def __init__(self, in_dims, d, n_head, n_layer, n_types, type_embed):
        super().__init__()
        # one row per box type, plus the pallet floor
        self.type_embedding = nn.Embedding(n_types + 1, type_embed)
        self.embed = nn.ModuleList([nn.Linear(k + type_embed, d)
                                    for k in in_dims])
        self.blocks = nn.ModuleList(nn.ModuleDict({
            "attn": nn.MultiheadAttention(d, n_head, batch_first=True),
            "n1": nn.LayerNorm(d),
            "ff": nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)),
            "n2": nn.LayerNorm(d),
        }) for _ in range(n_layer))
        self.d = d

    def forward(self, parts, masks, types, pos_on):
        xs = []
        for k, (p, m, t) in enumerate(zip(parts, masks, types)):
            h = self.embed[k](torch.cat([p, self.type_embedding(t)], -1))
            if k in pos_on:      # only the conveyor order matters
                h = h + positional_encoding(h.shape[1], self.d, h.device)
            xs.append(h * m.unsqueeze(-1))
        x = torch.cat(xs, 1)
        pad = ~torch.cat(masks, 1)
        safe = pad.clone()
        safe[:, 0] = False       # an all-padding row would give NaN
        for blk in self.blocks:
            a, _ = blk["attn"](x, x, x, key_padding_mask=safe,
                               need_weights=False)
            x = blk["n1"](x + a)
            x = blk["n2"](x + blk["ff"](x))
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        keep = (~pad).sum(1, keepdim=True).clamp(min=1)
        return x, x.sum(1) / keep


class Pointer(nn.Module):
    """Eq. 28.  `norm` > 0 bounds the score as norm * cos(q, k); `center`
    subtracts the mean (1) or the max (2) over live candidates first."""

    def __init__(self, d, c_temp, norm, center):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.c_temp, self.d = c_temp, d
        self.norm, self.center = float(norm), int(center)

    def forward(self, xbar, nodes, mask):
        q, k = self.q(xbar), self.k(nodes)
        if self.norm > 0:
            y = self.norm * torch.einsum("bd,bnd->bn", F.normalize(q, dim=-1),
                                         F.normalize(k, dim=-1))
        else:
            y = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(self.d)
        if self.center == 1:
            live = mask.to(y.dtype)
            y = y - ((y * live).sum(-1, keepdim=True)
                     / live.sum(-1, keepdim=True).clamp(min=1))
        elif self.center == 2:
            y = y - y.masked_fill(~mask, NEG).amax(-1, keepdim=True)
        return self.c_temp * torch.tanh(y) + torch.where(mask, 0.0, NEG)


class PackNet(nn.Module):
    def __init__(self, d, n_head, n_layer, c_temp, n_types, type_embed,
                 ptr_norm, ptr_center):
        super().__init__()
        self.enc = Encoder([6, 3, 6], d, n_head, n_layer, n_types, type_embed)
        self.ptr = Pointer(d, c_temp, ptr_norm, ptr_center)

    def forward(self, o):
        x, xbar = self.enc([o["c"], o["b"], o["l"]],
                           [o["c_mask"], o["b_mask"], o["l_mask"]],
                           [o["c_type"], o["b_type"], o["l_type"]], pos_on={1})
        return self.ptr(xbar, x[:, -o["l"].shape[1]:], o["l_mask"])


class PermNet(nn.Module):
    def __init__(self, d, n_head, n_layer, c_temp, n_types, type_embed,
                 ptr_norm, ptr_center):
        super().__init__()
        self.enc = Encoder([6, 3], d, n_head, n_layer, n_types, type_embed)
        self.ptr = Pointer(d, c_temp, ptr_norm, ptr_center)

    def forward(self, o):
        # every visible box is attended to; only reachable ones can be chosen
        x, xbar = self.enc([o["c"], o["b"]], [o["c_mask"], o["b_mask"]],
                           [o["c_type"], o["b_type"]], pos_on={1})
        return self.ptr(xbar, x[:, -o["b"].shape[1]:], o["b_pick"])


def greedy(logits):
    """(index, probability) of the best choice."""
    logits = torch.nan_to_num(logits, nan=NEG, posinf=NEG, neginf=NEG)
    logp = F.log_softmax(logits, -1)
    a = logp.argmax(-1)
    return int(a[0]), float(logp[0, a[0]].exp())


def load_nets(path, selector, device):
    """-> (pack, select or None, checkpoint args) from a training checkpoint.

    `selector` names the station picker stored in it: `mixer` (algo
    `select`), `attacker`, or None for no picker (always the front box).
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    kw = dict(d=a["width"], n_head=a["heads"], n_layer=a["layers"],
              c_temp=a["c_temp"], n_types=a["n_types"],
              type_embed=a["type_embed"], ptr_norm=a.get("ptr_norm", 0.0),
              ptr_center=a.get("ptr_center", 0))

    def build(cls, state):
        net = cls(**kw).to(device)
        net.load_state_dict({k: v for k, v in state.items()
                             if not k.startswith("critic.")})
        return net.eval()

    pack = build(PackNet, ck["pack"])
    if selector and selector not in ck:
        raise ValueError(f"{path} has no '{selector}' net "
                         f"(it has {sorted(k for k in ck if k != 'args')})")
    select = build(PermNet, ck[selector]) if selector else None
    return pack, select, a
