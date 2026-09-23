#!/usr/bin/env python3
"""Identify the stability rule from the heuristic baselines AR2L reports.

The paper says stability follows Zhao et al.; implemented as those papers write
it (60% of the base supported *and* four corners, or 80%/3, or 95%), every
method here caps out near 36% -- some 25 points below every number AR2L
reports.  The six heuristics of Table 1 are fully specified algorithms, so they
pin the simulator down: this script scores them under each candidate rule and
prints the gap.

    python3 scripts/calibrate.py --n 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import ar2l.env as E                                  # noqa: E402
from ar2l.env import BPPBatch                         # noqa: E402
from ar2l import heuristics as H                      # noqa: E402

# AR2L Table 1, "w/o attack" column: (Uti., Num.)
PAPER = {"dbl": (63.6, 25.8), "bmf": (62.0, 24.8), "lsah": (60.9, 24.6),
         "onlinebph": (64.1, 25.8), "hmm": (56.1, 22.6), "macs": (53.0, 21.5)}

# (stability mode, SUPPORT_RULES override, contact-area floor).  The floor is
# pinned per row rather than left at the env default: this table is the
# identification argument for the *paper's* rule, and must not move when the
# simulator's own default does.
RULES = {
    "60% + 4 corners (as written)": ("cdrl", ((0.60, 4), (0.80, 3), (0.95, 0)), 0.0),
    "4 corners only":               ("cdrl", ((0.0, 4),), 0.0),
    "support area >= 60%":          ("cdrl", ((0.599, 0),), 0.0),
    "support area >= 40%":          ("cdrl", ((0.399, 0),), 0.0),
    "support area >= 20%":          ("cdrl", ((0.199, 0),), 0.0),
    "no stability check":           ("cdrl", ((0.0, 0),), 0.0),
    "centre of mass over support":  ("com", None, 0.0),
    # not a candidate for the paper's rule -- the simulator's own default,
    # scored here for the cost of the area floor over the bare CoM rule
    "centre of mass + 80% area":    ("com", None, 0.80),
}


def score(rule, n, seed=11, rot=2):
    mode, rules, floor = RULES[rule]
    if rules:
        E.SUPPORT_RULES = rules
    out = {}
    for h in PAPER:
        # the paper's rule is calibrated on the paper's problem, which has
        # no box types; the stacking rule would move every number here
        env = BPPBatch(n, nb=1, seed=seed, stability=mode, rot=rot,
                       min_support=floor, n_types=1, types=False)
        while not env.done.all():
            env.step(H.act(env, h))
        out[h] = {"uti": float(env.utilization().mean() * 100),
                  "num": float(env.n_packed.mean())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--out", default=os.path.join(ROOT, "results/calibration.json"))
    p.add_argument("--rot", type=int, default=2,
                   help="orientations offered to the heuristics")
    a = p.parse_args()

    res = {}
    names = list(PAPER)
    print(f"{'stability rule':30s}" + "".join(f"{h:>10s}" for h in names)
          + f"{'mean|gap|':>11s}")
    print(f"{'paper':30s}" + "".join(f"{PAPER[h][0]:10.1f}" for h in names))
    for rule in RULES:
        r = score(rule, a.n, rot=a.rot)
        gap = float(np.mean([abs(r[h]["uti"] - PAPER[h][0]) for h in names]))
        res[rule] = {"per_heuristic": r, "mean_abs_gap": gap}
        print(f"{rule:30s}" + "".join(f"{r[h]['uti']:10.1f}" for h in names)
              + f"{gap:11.1f}", flush=True)

    best = min(res, key=lambda k: res[k]["mean_abs_gap"])
    res["_chosen"] = best
    res["_rot"] = a.rot
    res["_paper"] = {h: {"uti": PAPER[h][0], "num": PAPER[h][1]} for h in PAPER}
    print(f"\nclosest to the paper: {best}")
    print("item counts under it: " + ", ".join(
        f"{h} {res[best]['per_heuristic'][h]['num']:.1f} (paper {PAPER[h][1]})"
        for h in names))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
