"""Real pallet orders from a CSV, as the instance table the env plays.

One row per box, in the order the boxes arrive at the cell:

    pallet_id,seq,length_cm,width_cm,height_cm,type
    P001,1,24,14,12,0
    P001,2,16,12,10,1

`pallet_id`, `length_cm`, `width_cm` and `height_cm` are required.  `seq`
orders the boxes within a pallet (file order when it is absent), `type` is the
box type for the stacking rule (0 when absent), and `qty` repeats a row that
many times in a row.  Height is the vertical side; length and width lie along
the pallet's x and y, and the env yaws the box by itself when `rot` is 2.

Sides are divided by `box_scale` (0 or 1 keeps them; 2 halves every side),
then converted to grid cells of `cell_cm` and rounded by `box_round`: `up`
never models a box smaller than it is, `down` never bigger, `nearest` either
way; a side is never less than one cell.  The pallet is `pallet_cm` rounded *down*,
so it is never modelled bigger; both default to `eval:` in `config.yaml`, and
with no `pallet_cm` the boxes go onto `env.bin`.  A pallet_id may be spread
over the file (a cell building two pallets at once from one conveyor): its
boxes are gathered in file order.  Pallets of different lengths share one
`(n_pallets, max_boxes, 4)` array, padded at the end with all-zero rows, which
`BPPBatch` reads as the end of that pallet.

    python -m ar2l.orders orders.csv data/orders.npy --pallet_cm 120 80 180
    python -m ar2l.evaluate --data orders.csv --pallet_cm 120 80 180 ...
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

from .config import CFG

REQUIRED = ("pallet_id", "length_cm", "width_cm", "height_cm")
# a float division like 24 / 2 can land a hair above the integer; this keeps
# ceil from turning an exact 12 cells into 13
EPS = 1e-6


ROUNDING = {
    # EPS keeps a float hair off an exact cell boundary from moving it a cell
    "up": lambda x: math.ceil(x - EPS),
    "down": lambda x: math.floor(x + EPS),
    # an exact half goes down: 1.5 -> 1, 1.51 -> 2
    "nearest": lambda x: math.ceil(x - 0.5 - EPS),
}


def _cells(v, cell_cm, where, scale=1.0, rounding="up"):
    try:
        cm = float(v)
    except ValueError:
        raise ValueError(f"{where}: {v!r} is not a number") from None
    if not cm > 0:
        raise ValueError(f"{where}: a side must be positive, got {v!r}")
    return max(1, ROUNDING[rounding](cm / scale / cell_cm))


def box_rounding(box_round=None):
    """`box_round`, default `eval.box_round`, checked."""
    r = CFG["eval"]["box_round"] if box_round is None else box_round
    if r not in ROUNDING:
        raise ValueError(f"box_round is one of {sorted(ROUNDING)}, got {r!r}")
    return r


def box_divisor(box_scale=None):
    """What every box side is divided by: `box_scale`, with 0 meaning 1."""
    s = CFG["eval"]["box_scale"] if box_scale is None else box_scale
    s = float(s)
    # 0.5 would *grow* every box, which reads too easily as "halve it"
    if s < 0 or 0 < s < 1:
        raise ValueError(f"box_scale is 0 (sizes as in the file) or the "
                         f"number >= 1 to divide every side by, got {s:g}")
    return 1.0 if s == 0 else s


def orders_bin(pallet_cm=None, cell_cm=None):
    """The bin, in cells, that real orders are packed onto.

    `pallet_cm` is rounded *down* to cells so the pallet is never modelled
    bigger than it is; `None` for either reads `eval:` in `config.yaml`, and
    no pallet at all falls back to `env.bin`.
    """
    ev = CFG["eval"]
    cell_cm = ev["cell_cm"] if cell_cm is None else cell_cm
    pallet_cm = ev["pallet_cm"] if pallet_cm is None else pallet_cm
    if pallet_cm is None:
        return tuple(int(v) for v in np.broadcast_to(np.asarray(CFG["env"]["bin"]), (3,)))
    cm = [float(v) for v in np.broadcast_to(np.asarray(pallet_cm, float), (3,))]
    cells = tuple(int(math.floor(v / cell_cm + EPS)) for v in cm)
    if min(cells) < 1:
        raise ValueError(f"pallet {cm} cm is under one {cell_cm} cm cell")
    return cells


def load_orders(path, cell_cm=None, S=None, rot=None, box_scale=None,
                box_round=None):
    """Read an orders CSV.  Returns `(seqs, pallet_ids)`.

    `seqs` is int16 `(n_pallets, max_boxes, 4)` of `(sx, sy, sz, type_id)` in
    cells; `pallet_ids` lists the pallets in order of first appearance.  `S`
    (the bin in cells, default `orders_bin()`) is checked against every box:
    one that fits in no allowed orientation could never be placed and would
    silently end its pallet, so it is an error here instead.  `box_scale`
    (default `eval.box_scale`) divides every side before gridding, and
    `box_round` (default `eval.box_round`) rounds it to cells.
    """
    cell_cm = CFG["eval"]["cell_cm"] if cell_cm is None else cell_cm
    scale = box_divisor(box_scale)
    rounding = box_rounding(box_round)
    S = orders_bin(cell_cm=cell_cm) if S is None else S
    rot = CFG["env"]["rot"] if rot is None else rot
    Lx, Ly, Lz = (int(v) for v in np.broadcast_to(np.asarray(S), (3,)))
    pallets = {}                              # dicts keep insertion order
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = [c.strip() for c in (rd.fieldnames or [])]
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; "
                             f"have {cols}")
        for n, raw in enumerate(rd, start=2):          # line 1 is the header
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            if not any(row.values()):
                continue
            where = f"{path}:{n}"
            pid = row["pallet_id"]
            if not pid:
                raise ValueError(f"{where}: empty pallet_id")
            box = [_cells(row[c], cell_cm, f"{where} {c}", scale, rounding)
                   for c in ("length_cm", "width_cm", "height_cm")]
            t = int(row.get("type") or 0)
            qty = int(row.get("qty") or 1)
            if qty < 1:
                raise ValueError(f"{where}: qty must be at least 1")
            sx, sy, sz = box
            flat = (sx <= Lx and sy <= Ly) or (rot >= 2 and sy <= Lx and sx <= Ly)
            if not (flat and sz <= Lz):
                raise ValueError(
                    f"{where}: box {row['length_cm']}x{row['width_cm']}x"
                    f"{row['height_cm']} cm"
                    + (f" / {scale:g}" if scale != 1 else "")
                    + f" is {sx}x{sy}x{sz} cells and does "
                    f"not fit the {Lx}x{Ly}x{Lz}-cell bin "
                    f"({Lx * cell_cm:g}x{Ly * cell_cm:g}x{Lz * cell_cm:g} cm); "
                    f"set the real pallet with --pallet_cm or eval.pallet_cm")
            seq = float(row["seq"]) if row.get("seq") else n
            pallets.setdefault(pid, []).extend([(seq, n, sx, sy, sz, t)] * qty)
    if not pallets:
        raise ValueError(f"{path}: no boxes")
    ids = list(pallets)
    width = max(len(v) for v in pallets.values())
    out = np.zeros((len(ids), width, 4), np.int16)
    for i, pid in enumerate(ids):
        # stable on the file line, so equal `seq` values keep file order
        rows = sorted(pallets[pid], key=lambda r: (r[0], r[1]))
        out[i, : len(rows)] = [r[2:] for r in rows]
    return out, ids


def randomize_order(seqs, amount, seed=0):
    """Randomise each instance's box order by `amount` in [0, 1].

    A noisy sort: box `i` of an `n`-box instance is keyed by
    `(1 - amount) * i / n + amount * U(0, 1)` and the boxes are re-sorted on
    that key.  At 0 the order is untouched, at 1 the key is pure noise and the
    order is a uniform random permutation, and in between a box drifts from
    its place by about `amount / (1 - amount)` of the instance -- local
    swaps, the way a real conveyor jitters, rather than a few boxes thrown to
    the far end.  Padding rows stay at the end and the table is not modified.
    """
    amount = float(amount)
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"order_random is 0 (data order) to 1 (fully "
                         f"random), got {amount}")
    seqs = np.asarray(seqs)
    if amount == 0.0:
        return seqs.copy()
    n_items = seqs.shape[-2]
    length = (seqs[..., :3] > 0).all(-1).sum(-1, keepdims=True)
    i = np.arange(n_items)
    key = ((1.0 - amount) * i / np.maximum(length, 1)
           + amount * np.random.default_rng(seed).random(seqs.shape[:-1]))
    key = np.where(i < length, key, np.inf)          # padding sorts last
    order = np.argsort(key, axis=-1, kind="stable")
    return np.take_along_axis(seqs, order[..., None], axis=-2)


def load_instances(path, cell_cm=None, S=None, rot=None, box_scale=None,
                   box_round=None):
    """An instance table from a `.npy` or an orders `.csv`.

    `box_scale` applies to a `.csv` only: a `.npy` is already in cells.
    """
    if str(path).lower().endswith(".csv"):
        return load_orders(path, cell_cm, S, rot, box_scale, box_round)[0]
    return np.load(path)


def add_cm_args(p):
    """`--cell_cm` and `--pallet_cm`, for every CLI that reads an orders CSV."""
    ev = CFG["eval"]
    p.add_argument("--cell_cm", type=float, default=ev["cell_cm"],
                   help="grid cell size in cm for an orders .csv (%(default)s)")
    p.add_argument("--pallet_cm", type=float, nargs=3, default=ev["pallet_cm"],
                   metavar=("L", "W", "H"),
                   help="the real pallet for an orders .csv, in cm; default "
                        "eval.pallet_cm, else env.bin")
    p.add_argument("--box_scale", type=float, default=ev["box_scale"],
                   help="divide every box side of an orders .csv by this; "
                        "0 keeps the file's sizes (%(default)s)")
    p.add_argument("--box_round", choices=sorted(ROUNDING),
                   default=ev["box_round"],
                   help="how a box side is rounded to cells (%(default)s)")


def add_order_args(p):
    """`--order_random` and `--order_seed`, for every CLI that plays instances."""
    ev = CFG["eval"]
    p.add_argument("--order_random", type=float, default=ev["order_random"],
                   help="randomise the box order: 0 = as in the data, "
                        "1 = fully random (%(default)s)")
    p.add_argument("--order_seed", type=int, default=ev["order_seed"],
                   help="seed of the order randomisation (%(default)s)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert an orders CSV to the "
                                "(n_pallets, max_boxes, 4) .npy the env plays.")
    p.add_argument("csv")
    p.add_argument("out", help="the .npy to write; pallet ids go beside it "
                               "as <out>.ids.txt, one per row of the array")
    add_cm_args(p)
    a = p.parse_args(argv)
    S = orders_bin(a.pallet_cm, a.cell_cm)
    seqs, ids = load_orders(a.csv, a.cell_cm, S, box_scale=a.box_scale,
                            box_round=a.box_round)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out, seqs)
    with open(os.path.splitext(a.out)[0] + ".ids.txt", "w") as f:
        f.write("\n".join(ids) + "\n")
    n = (seqs[..., :3] > 0).all(-1).sum(1)
    print(f"{len(ids)} pallets, {int(n.sum())} boxes "
          f"({int(n.min())}-{int(n.max())} per pallet), "
          f"types {sorted(set(seqs[..., 3][seqs[..., 0] > 0].tolist()))} "
          f"on a {S[0]}x{S[1]}x{S[2]}-cell pallet -> {a.out}")


if __name__ == "__main__":
    main()
