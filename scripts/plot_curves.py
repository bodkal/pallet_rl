#!/usr/bin/env python3
"""AR2L Figure 3(a)(b): nominal learning curves of the robust algorithms.

    python3 scripts/plot_curves.py --nbs 10 20 --out results/training_curves.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ar2l.viz.dashboard import PAPER, read_log          # noqa: E402
from ar2l.viz.report import LABEL                       # noqa: E402

ORDER = ["pct", "cppo", "rarl", "rfmdp", "ex05", "ex10", "ap05", "ap10"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbs", type=int, nargs="+", default=[10, 20])
    p.add_argument("--methods", nargs="*", default=ORDER)
    p.add_argument("--out", default=os.path.join(ROOT, "results/training_curves.png"))
    a = p.parse_args()

    fig, axes = plt.subplots(1, len(a.nbs), figsize=(5.6 * len(a.nbs), 3.8),
                             squeeze=False)
    for ax, nb in zip(axes[0], a.nbs):
        for m in a.methods:
            rows = [r for r in read_log(f"{m}_nb{nb}") if r.get("nom_util")]
            if rows:
                ax.plot([r["it"] for r in rows],
                        [r["nom_util"] * 100 for r in rows], lw=1.7,
                        label=LABEL.get(m, m))
        t = PAPER.get(("pct", nb))
        if t:                                   # (nominal, fully attacked)
            ax.axhline(t[0], ls="--", lw=1, color="#888",
                       label=f"paper PCT ({t[0]:.1f}%)")
        ax.set_title(f"$N_B$ = {nb}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("held-out nominal utilisation (%)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=135)
    print("->", a.out)


if __name__ == "__main__":
    main()
