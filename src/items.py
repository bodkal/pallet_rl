"""Item set and benchmark sequence generators (paper Sec. 4 + supplemental C).

Three benchmarks:
  RS     - items sampled uniformly at random from the pre-defined set I until the
           accumulated volume reaches the bin volume.
  CUT-1  - the bin is recursively "cut" into items of the pre-defined types
           (cutting stock, Gilmore & Gomory 1961), then sorted by the Z coordinate
           of each item's front-left-bottom (FLB) corner, ties broken randomly.
  CUT-2  - same cutting, but sorted by *stacking dependency*: an item may enter
           the sequence only once all of its supporting items are already there.
           (a random topological order of the support DAG)

A CUT-1 / CUT-2 sequence can by construction be packed back into the bin with
100% space utilisation, which is what makes those benchmarks measurable.
"""
from __future__ import annotations

import numpy as np


def item_set(lo: int = 2, hi: int = 5):
    """The |I| pre-defined item dimensions.  Paper: lo=2, hi=5 -> 4^3 = 64 types."""
    return [(l, w, h)
            for l in range(lo, hi + 1)
            for w in range(lo, hi + 1)
            for h in range(lo, hi + 1)]


# ---------------------------------------------------------------------------
# RS
# ---------------------------------------------------------------------------
def gen_rs(rng: np.random.Generator, L, W, H, lo=2, hi=5):
    types = item_set(lo, hi)
    target = L * W * H
    seq, vol = [], 0
    while vol < target:
        d = types[rng.integers(len(types))]
        seq.append(d)
        vol += d[0] * d[1] * d[2]
    return seq


# ---------------------------------------------------------------------------
# cutting stock
# ---------------------------------------------------------------------------
def _cut(rng, L, W, H, lo=2, hi=5):
    """Recursively cut the bin into boxes whose every dimension lies in [lo, hi].

    Returns a list of (x, y, z, l, w, h) placed boxes that exactly tile the bin.
    """
    boxes = [(0, 0, 0, L, W, H)]
    while True:
        # indices of boxes that still have an over-sized dimension
        over = [i for i, b in enumerate(boxes)
                if b[3] > hi or b[4] > hi or b[5] > hi]
        if not over:
            break
        i = over[rng.integers(len(over))]
        x, y, z, l, w, h = boxes.pop(i)
        dims = (l, w, h)
        axes = [a for a in range(3) if dims[a] > hi]
        a = axes[rng.integers(len(axes))]
        s = dims[a]
        # split so that both parts are at least `lo` thick
        p = int(rng.integers(lo, s - lo + 1))
        if a == 0:
            boxes += [(x, y, z, p, w, h), (x + p, y, z, l - p, w, h)]
        elif a == 1:
            boxes += [(x, y, z, l, p, h), (x, y + p, z, l, w - p, h)]
        else:
            boxes += [(x, y, z, l, w, p), (x, y, z + p, l, w, h - p)]
    return boxes


def _order_cut1(rng, boxes):
    """Sort by FLB z, ties broken randomly."""
    keys = [(b[2], rng.random()) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: keys[i])
    return [boxes[i] for i in order]


def _supporters(boxes):
    """supports[j] = set of box indices that j rests on (must precede j)."""
    n = len(boxes)
    sup = [set() for _ in range(n)]
    for j, (xj, yj, zj, lj, wj, hj) in enumerate(boxes):
        if zj == 0:
            continue
        for i, (xi, yi, zi, li, wi, hi) in enumerate(boxes):
            if i == j:
                continue
            if zi + hi != zj:
                continue
            if xi < xj + lj and xj < xi + li and yi < yj + wj and yj < yi + wi:
                sup[j].add(i)
    return sup


def _order_cut2(rng, boxes):
    """Random topological order of the support DAG."""
    n = len(boxes)
    sup = _supporters(boxes)
    remaining = [len(s) for s in sup]
    children = [[] for _ in range(n)]
    for j, s in enumerate(sup):
        for i in s:
            children[i].append(j)
    ready = [i for i in range(n) if remaining[i] == 0]
    out = []
    while ready:
        k = int(rng.integers(len(ready)))
        i = ready.pop(k)
        out.append(i)
        for j in children[i]:
            remaining[j] -= 1
            if remaining[j] == 0:
                ready.append(j)
    assert len(out) == n, "support graph is not a DAG"
    return [boxes[i] for i in out]


def gen_cut(rng, L, W, H, lo=2, hi=5, variant="CUT-2"):
    boxes = _cut(rng, L, W, H, lo, hi)
    boxes = _order_cut1(rng, boxes) if variant == "CUT-1" else _order_cut2(rng, boxes)
    return [(b[3], b[4], b[5]) for b in boxes]


# ---------------------------------------------------------------------------
def gen_sequence(rng, L, W, H, lo=2, hi=5, dataset="CUT-2"):
    if dataset == "RS":
        return gen_rs(rng, L, W, H, lo, hi)
    if dataset in ("CUT-1", "CUT-2"):
        return gen_cut(rng, L, W, H, lo, hi, dataset)
    raise KeyError(f"unknown dataset {dataset!r}")


def make_testset(n, L, W, H, lo=2, hi=5, dataset="CUT-2", seed=12345):
    """Deterministic held-out benchmark (the paper uses 2,000 sequences)."""
    rng = np.random.default_rng(seed)
    return [gen_sequence(rng, L, W, H, lo, hi, dataset) for _ in range(n)]
