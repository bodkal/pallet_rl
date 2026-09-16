#!/usr/bin/env python3
"""Attribute utilisation to the two action-space changes, without learning.

    python3 scripts/ablate_env.py --n 512

The six heuristics never train, so running them under each combination of

    rot   1 = the item is offered in one orientation, 2 = also yawed 90 deg
    ems   1 = candidates are the bottom corners of the empty maximal spaces,
          0 = every loading position on the grid is a candidate

separates what the action space is worth from what the policy is worth.  The
paper's Table 1 column is printed alongside as the target.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ar2l.env import BPPBatch                     # noqa: E402
from ar2l.heuristics import NAMES, act            # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# AR2L Table 1, beta = 0 column: (utilisation, items)
PAPER = {"dbl": (63.6, 25.8), "bmf": (62.0, 24.8), "lsah": (60.9, 24.6),
         "onlinebph": (64.1, 25.8), "hmm": (56.1, 22.6), "macs": (53.0, 21.5)}
CONFIGS = [(1, 1), (2, 1), (1, 0), (2, 0)]


def play(name, rot, ems, n, seed, batch=128):
    u, k = [], []
    for s in range(0, n, batch):
        env = BPPBatch(min(batch, n - s), nb=1, seed=seed + s, rot=rot,
                       ems=bool(ems))
        while not env.done.all():
            env.step(act(env, name))
        u.append(env.utilization()); k.append(env.n_packed)
    return float(np.concatenate(u).mean() * 100), float(np.concatenate(k).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default=os.path.join(ROOT, "results/ablate_env.json"))
    a = p.parse_args()

    head = "".join(f"  rot{r} ems{e}" for r, e in CONFIGS)
    print(f"{'heuristic':11s}{head}{'     paper':>11s}")
    res = {}
    for h in NAMES:
        res[h] = {}
        row = ""
        for rot, ems in CONFIGS:
            uti, num = play(h, rot, ems, a.n, a.seed)
            res[h][f"rot{rot}_ems{ems}"] = {"uti": uti, "num": num}
            row += f"{uti:10.1f}"
        print(f"{h:11s}{row}{PAPER[h][0]:11.1f}", flush=True)

    print(f"\n{'mean':11s}" + "".join(
        f"{np.mean([res[h][f'rot{r}_ems{e}']['uti'] for h in NAMES]):10.1f}"
        for r, e in CONFIGS)
        + f"{np.mean([PAPER[h][0] for h in NAMES]):11.1f}")
    print(f"{'mean |gap|':11s}" + "".join(
        f"{np.mean([abs(res[h][f'rot{r}_ems{e}']['uti'] - PAPER[h][0]) for h in NAMES]):10.1f}"
        for r, e in CONFIGS))
    res["_paper"] = {h: {"uti": PAPER[h][0], "num": PAPER[h][1]} for h in PAPER}
    res["_n"] = a.n
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
