#!/usr/bin/env python3
"""Turn a folder of pallet-day files into a merged orders csv for training.

    python3 scripts/make_orders.py                  # use_data -> data/train_data/orders_all.csv
    python3 scripts/make_orders.py --src data/box_pallet_data/generated_data \\
                                   --out data/train_data/orders_all_generated_data

Reads every box of every day file, in file order (day files sorted by date,
numbered files numerically) and row order inside a file.  TYPE 1/2/3 becomes
type 0/1/2 and a side that is a whole number loses its ".0".

Writes <out>.csv: the boxes regrouped into ceil(boxes / --max_boxes) pallets as
even as possible, the bigger ones first.  The source pallets are laid end to
end in the order they first appear, each with its boxes in their own order,
and cut into consecutive chunks.  A chunk takes the id of its first source
pallet whose id is not taken yet.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "train_data")
SRC_DIMS = ["V_BOX_DEPTH", "V_BOX_WIDTH", "V_BOX_HEIGHT"]
COLS = ["pallet_id", "length_cm", "width_cm", "height_cm", "type"]


def _num_key(path):
    s = os.path.basename(path)[:-4]
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def _cm(x):
    v = float(x)
    return str(int(v)) if v.is_integer() else repr(v)


def read_boxes(src):
    """[pallet_id, length, width, height, type] for every box, in file order."""
    boxes = []
    for f in sorted(glob.glob(os.path.join(src, "*.csv")), key=_num_key):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                boxes.append([r["PALLET_ID"], *(_cm(r[c]) for c in SRC_DIMS),
                              str(int(r["TYPE"]) - 1)])
    return boxes


def merge(boxes, max_boxes):
    """(merged boxes, rows of [pallet id, boxes, sources]): the source pallets
    end to end, cut even."""
    by = {}
    for b in boxes:
        by.setdefault(b[0], []).append(b)
    stream = [b for g in by.values() for b in g]
    n = math.ceil(len(stream) / max_boxes)
    q, r = divmod(len(stream), n)
    merged, rows, used, pos = [], [], set(), 0
    for k in range(n):
        chunk = stream[pos:pos + q + (k < r)]
        pos += len(chunk)
        src = {}
        for b in chunk:
            src[b[0]] = src.get(b[0], 0) + 1
        pid = next((p for p in src if p not in used), None)
        if pid is None:                   # only when a source pallet spans 3+ chunks
            pid = next(f"{chunk[0][0]}_{i}" for i in range(2, len(stream))
                       if f"{chunk[0][0]}_{i}" not in used)
        used.add(pid)
        merged += [[pid, *b[1:]] for b in chunk]
        rows.append([pid, len(chunk), " ".join(f"{p}:{c}" for p, c in src.items())])
    return merged, rows


def write(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def make(src, out, max_boxes):
    boxes = read_boxes(src)
    merged, rows = merge(boxes, max_boxes)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write(f"{out}.csv", COLS, merged)
    sizes = [r[1] for r in rows]
    print(f"{len(boxes)} boxes in {len({b[0] for b in boxes})} pallets "
          f"-> {len(rows)} pallets of {min(sizes)}-{max(sizes)} boxes")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", default=os.path.join(DATA, "box_pallet_data", "use_data"))
    p.add_argument("--out", default=os.path.join(OUT, "orders_all"),
                   help="path prefix; .csv is added")
    p.add_argument("--max_boxes", type=int, default=75)
    a = p.parse_args()
    make(a.src, a.out, a.max_boxes)
    print(f"wrote {os.path.relpath(a.out, ROOT)}.csv")


if __name__ == "__main__":
    main()
