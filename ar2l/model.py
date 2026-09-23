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

Every node also carries a box type: a packed item and an observable item carry
their own, and a candidate placement carries the type of whatever it would rest
on, the bin floor included.  One trainable embedding table per encoder turns
that index into `type_embed` numbers, which are concatenated onto the node's
physical features before the element-wise projection -- so the projection is
wider by `type_embed` and nothing downstream of it changes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG

NEG = -1e9


def positional_encoding(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(d, device=device, dtype=torch.float32)[None, :]
    ang = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    pe = torch.where(i.long() % 2 == 0, torch.sin(ang), torch.cos(ang))
    return pe


class Encoder(nn.Module):
    """Independent projections + `n_layer` attention blocks over the union."""

    def __init__(self, in_dims, d=None, n_head=None, n_layer=None,
                 n_types=None, type_embed=None):
        super().__init__()
        m = CFG["model"]
        d = m["width"] if d is None else d
        n_head = m["heads"] if n_head is None else n_head
        n_layer = m["layers"] if n_layer is None else n_layer
        n_types = CFG["env"]["n_types"] if n_types is None else n_types
        type_embed = m["type_embed"] if type_embed is None else type_embed
        self.n_types, self.type_embed = int(n_types), int(type_embed)
        # `n_types + 1` rows: one per box type, plus the bin floor -- what a
        # candidate resting on nothing reports, and a state the packer has to
        # be able to tell apart from resting on a box, since the floor accepts
        # every type and a box accepts only its own
        self.type_embedding = nn.Embedding(self.n_types + 1, self.type_embed)
        self.embed = nn.ModuleList([nn.Linear(k + self.type_embed, d)
                                    for k in in_dims])
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(d, n_head, batch_first=True),
                "n1": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)),
                "n2": nn.LayerNorm(d),
            }))
        self.d = d

    def forward(self, parts, masks, types, pos_on=None):
        """parts/masks/types: lists of (B, N_k, dim_k), (B, N_k) bool, (B, N_k) long.

        A padding slot's type index is whatever the env left there -- in range,
        but meaningless -- and the mask below zeroes its whole token, so it
        reaches the attention as padding exactly as it did before.
        """
        xs = []
        for k, (p, m, t) in enumerate(zip(parts, masks, types)):
            h = self.embed[k](torch.cat([p, self.type_embedding(t)], -1))
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

    def __init__(self, d=None, c_temp=None):
        super().__init__()
        d = CFG["model"]["width"] if d is None else d
        c_temp = CFG["model"]["c_temp"] if c_temp is None else c_temp
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.c_temp, self.d = c_temp, d

    def forward(self, xbar, nodes, mask):
        y = torch.einsum("bd,bnd->bn", self.q(xbar), self.k(nodes)) / math.sqrt(self.d)
        return self.c_temp * torch.tanh(y) + torch.where(mask, 0.0, NEG)


class Critic(nn.Module):
    """V(C, B) -- no dependence on the candidate list.

    Takes the observation dict rather than loose tensors: the state a value is
    asked for now names six arrays, and the robust targets of Algorithm 2 build
    theirs by hand from a rollout, where one silently missing key would be a
    value evaluated on the wrong state rather than an error.
    """

    def __init__(self, d=None, n_head=None, n_layer=None, n_types=None,
                 type_embed=None):
        super().__init__()
        self.enc = Encoder([6, 3], d, n_head, n_layer, n_types, type_embed)
        d = self.enc.d
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, o):
        _, xbar, _ = self.enc([o["c"], o["b"]], [o["c_mask"], o["b_mask"]],
                              [o["c_type"], o["b_type"]], pos_on={1})
        return self.head(xbar).squeeze(-1)


class PackNet(nn.Module):
    """Packing policy pi_pack(l | C, B, L) and its value function."""

    def __init__(self, d=None, n_head=None, n_layer=None, c_temp=None,
                 n_types=None, type_embed=None):
        super().__init__()
        self.enc = Encoder([6, 3, 6], d, n_head, n_layer, n_types, type_embed)
        self.ptr = Pointer(self.enc.d, c_temp)
        self.critic = Critic(d, n_head, n_layer, n_types, type_embed)

    def logits(self, o):
        x, xbar, _ = self.enc([o["c"], o["b"], o["l"]],
                              [o["c_mask"], o["b_mask"], o["l_mask"]],
                              [o["c_type"], o["b_type"], o["l_type"]],
                              pos_on={1})
        nl = o["l"].shape[1]
        return self.ptr(xbar, x[:, -nl:], o["l_mask"])

    def value(self, o):
        return self.critic(o)

    def forward(self, o):
        return self.logits(o), self.value(o)


class PermNet(nn.Module):
    """pi(b_i | C, B): pick the observable item to move to the conveyor front.

    Used both for the permutation-based attacker and for the mixture-dynamics
    model; they differ only in the loss they are trained with.

    Visibility and reachability are separate masks.  The encoder is given
    `b_mask`, so every observable item is attended to, while the pointer is
    given `b_pick`, the `n_pick` items the cell can reach -- the preview tail
    informs the choice without being choosable, and under the stacking rule a
    reachable box with nowhere legal to go is dropped from `b_pick` too, so the
    permuter is pointed at the boxes it can actually hand over.

    The observable items carry their type, so the reordering this learns is a
    reordering over types as well as over shapes: which box is worst to hand
    the packer next depends on what the packer is already stacking.
    """

    def __init__(self, d=None, n_head=None, n_layer=None, c_temp=None,
                 n_types=None, type_embed=None):
        super().__init__()
        self.enc = Encoder([6, 3], d, n_head, n_layer, n_types, type_embed)
        self.ptr = Pointer(self.enc.d, c_temp)
        self.critic = Critic(d, n_head, n_layer, n_types, type_embed)

    def logits(self, o):
        x, xbar, _ = self.enc([o["c"], o["b"]], [o["c_mask"], o["b_mask"]],
                              [o["c_type"], o["b_type"]], pos_on={1})
        nb = o["b"].shape[1]
        # a (C, B) dict built by hand -- the `approx` bootstrap -- may carry no
        # `b_pick`, and then every observable item is selectable, as before
        return self.ptr(xbar, x[:, -nb:], o.get("b_pick", o["b_mask"]))

    def value(self, o):
        return self.critic(o)

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
