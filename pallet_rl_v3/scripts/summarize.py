#!/usr/bin/env python3
"""Markdown tables for the README: ours next to the paper's, with the gap.

    python3 scripts/summarize.py            # both tables
    python3 scripts/summarize.py --which table2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ar2l.viz.report import LABEL, PAPER_T1, PAPER_T2, BETAS   # noqa: E402


def fmt2(res):
    out = []
    for nb in sorted({v["nb"] for v in res.values()}):
        cell = {v["method"]: v for v in res.values() if v["nb"] == nb}
        out += [f"\n**N_B = {nb}**\n",
                "| method | " + " | ".join(f"β={b} ours | β={b} paper"
                                           for b in BETAS) + " |",
                "|---" * (1 + 2 * len(BETAS)) + "|"]
        for m in LABEL:
            if m not in cell:
                continue
            p = PAPER_T2.get((nb, m))
            cells = []
            for i, b in enumerate(BETAS):
                cells.append(f"{cell[m][str(b)]['uti']:.1f}")
                cells.append(f"{p[i][0]:.1f}" if p else "-")
            out.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def gap2(res):
    d = []
    for v in res.values():
        p = PAPER_T2.get((v["nb"], v["method"]))
        if not p:
            continue
        for i, b in enumerate(BETAS):
            d.append(v[str(b)]["uti"] - p[i][0])
    d = np.array(d)
    return (f"\n{len(d)} cells: mean gap to the paper {d.mean():+.1f} points, "
            f"mean |gap| {np.abs(d).mean():.1f}, worst {d.min():+.1f} / "
            f"{d.max():+.1f}\n")


def fmt1(res):
    cols = ["5", "10", "15", "20", "none"]
    head = ["| method "] + [f"| N_B={c} ours | N_B={c} paper "
                            if c != "none" else "| no attack ours | no attack paper "
                            for c in cols] + ["|"]
    out = ["".join(head), "|---" * (1 + 2 * len(cols)) + "|"]
    for name, row in res.items():
        p = PAPER_T1.get(name)
        cells = []
        for i, c in enumerate(cols):
            ov = row.get(c, {}).get("uti")
            cells.append(f"{ov:.1f}" if ov is not None else "-")
            cells.append(f"{p[4 if c == 'none' else i][0]:.1f}" if p else "-")
        out.append(f"| {name.upper()} | " + " | ".join(cells) + " |")
    return "\n".join(out)


CLAIMS = [
 ("ExactAR2L(1.0) packs at least as many items as PCT",
  lambda c: [(c("ex10", nb, b)["num"] >= c("pct", nb, b)["num"] - 1e-9)
             for nb, b in c.cells()]),
 ("ExactAR2L(1.0) has a smaller Std. than PCT",
  lambda c: [(c("ex10", nb, b)["std"] <= c("pct", nb, b)["std"])
             for nb, b in c.cells()]),
 ("ExactAR2L(1.0) Std. lies between RARL's and PCT's",
  lambda c: [(min(c("rarl", nb, b)["std"], c("pct", nb, b)["std"]) - 1e-9
              <= c("ex10", nb, b)["std"] <=
              max(c("rarl", nb, b)["std"], c("pct", nb, b)["std"]) + 1e-9)
             for nb, b in c.cells()]),
 ("RARL gives up nominal utilisation relative to PCT (beta=0)",
  lambda c: [(c("rarl", nb, 0)["uti"] < c("pct", nb, 0)["uti"])
             for nb, _ in c.cells() if _ == 0]),
 ("ExactAR2L(1.0) beats PCT once a quarter of the set is attacked",
  lambda c: [(c("ex10", nb, b)["uti"] > c("pct", nb, b)["uti"])
             for nb, b in c.cells() if b >= 25]),
 ("ApproxAR2L(0.5) packs more items than RfMDP",
  lambda c: [(c("ap05", nb, b)["num"] > c("rfmdp", nb, b)["num"])
             for nb, b in c.cells()]),
 ("ApproxAR2L(0.5) beats RfMDP on utilisation",
  lambda c: [(c("ap05", nb, b)["uti"] > c("rfmdp", nb, b)["uti"])
             for nb, b in c.cells()]),
 ("the attacker hurts: utilisation falls monotonically in beta",
  lambda c: [all(c(m, nb, BETAS[i])["uti"] >= c(m, nb, BETAS[i + 1])["uti"] - .3
                 for i in range(len(BETAS) - 1))
             for m in LABEL for nb, b in c.cells() if b == 0
             if c.has(m, nb)]),
]


class Cells:
    """Lookup of (method, N_B, beta) -> metrics, over whatever was trained."""

    def __init__(self, res):
        self.d = {(v["method"], v["nb"]): v for v in res.values()}
        self.nbs = sorted({v["nb"] for v in res.values()})

    def has(self, m, nb):
        return (m, nb) in self.d

    def __call__(self, m, nb, b):
        return self.d[(m, nb)][str(b)]

    def cells(self):
        return [(nb, b) for nb in self.nbs for b in BETAS]


def claims(res):
    c = Cells(res)
    out = ["", "| claim in the paper | holds here |", "|---|---|"]
    for text, fn in CLAIMS:
        try:
            v = fn(c)
        except KeyError:
            out.append(f"| {text} | (not trained) |")
            continue
        if not v:
            out.append(f"| {text} | (no cells) |")
            continue
        out.append(f"| {text} | {sum(v)}/{len(v)} |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="*", default=["table1", "table2"])
    a = p.parse_args()
    for w in a.which:
        f = os.path.join(ROOT, "results", w + ".json")
        if not os.path.exists(f):
            print(f"({w}.json not built yet)"); continue
        res = json.load(open(f))
        print(f"\n### {w}")
        if w == "table2":
            print(fmt2(res) + gap2(res))
            print("\n### the paper's comparative claims, on our numbers")
            print(claims(res))
        else:
            print(fmt1(res))


if __name__ == "__main__":
    main()
