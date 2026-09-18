#!/usr/bin/env python3
"""Build AR2L Table 1 and Table 2 from the trained checkpoints.

    python3 scripts/eval_all.py table2 --nbs 10 20
    python3 scripts/eval_all.py table1 --nbs 5 10 15 20

Table 1 attacks the heuristics and a 1-item PCT policy with the attacker
trained against each of them.  Table 2 evaluates every robust method on mixture
datasets in which beta% of the instances are reordered by that policy's own
attacker.  Results land in results/table{1,2}.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ar2l.evaluate import load_nets, metrics, run          # noqa: E402
from ar2l.heuristics import NAMES as HEURISTICS            # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METHODS = ["pct", "cppo", "rarl", "rfmdp", "ex05", "ex10", "ap05", "ap10"]


def ckpt(run, which="best"):
    p = os.path.join(ROOT, "runs", run, which + ".pt")
    return p if os.path.exists(p) else None


# algorithms whose own training loop fits an attacker; for the rest the
# checkpoint carries a freshly initialised one that must never be used
CO_TRAINED = {"rarl", "exact", "approx", "attack"}


def attacker_for(run, device):
    """The dedicated attacker trained against this frozen policy.

    Falls back to the co-trained attacker only for the algorithms that
    actually fit one; PCT, CPPO and RfMDP never do, and using their unused
    randomly-initialised head would silently report a harmless attack.
    """
    p = ckpt(f"att_{run}")
    if p:
        return load_nets(p, device, ("attacker",))["attacker"], f"att_{run}"
    p = ckpt(run)
    if p:
        algo = json.load(open(os.path.join(ROOT, "runs", run, "args.json")))["algo"]
        if algo in CO_TRAINED:
            return load_nets(p, device, ("attacker",))["attacker"], run + ":co"
    print(f"  !! {run}: no trained attacker -- attacked columns are meaningless")
    return None, None


def flags(n, betas, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return {b: np.isin(np.arange(n), order[: int(round(n * b / 100.0))])
            for b in betas}


def table2(a):
    seqs = np.load(a.data)[: a.n_inst]
    fl = flags(len(seqs), a.betas)
    out = {}
    for nb in a.nbs:
        for m in METHODS:
            runname = f"{m}_nb{nb}"
            p = ckpt(runname)
            if not p:
                print("skip", runname, "(no checkpoint)"); continue
            pack = load_nets(p, a.device, ("pack",))["pack"]
            att, asrc = attacker_for(runname, a.device)
            row = {}
            for b in a.betas:
                u, k = run(seqs, pack, nb, att, fl[b], a.batch, device=a.device,
                           min_support=a.min_support)
                row[str(b)] = metrics(u, k)
            out[runname] = {"nb": nb, "method": m, "attacker": asrc, **row}
            print(f"{runname:12s} att={asrc or '-':16s} " + "  ".join(
                f"b{b}:{row[str(b)]['uti']:5.1f}/{row[str(b)]['std']:4.1f}/"
                f"{row[str(b)]['num']:4.1f}" for b in a.betas), flush=True)
    return out


def table1(a):
    seqs = np.load(a.data)[: a.n_inst]
    on = np.ones(len(seqs), bool)
    out = {}
    packers = list(HEURISTICS) + ["pct"]
    for name in packers:
        policy = name if name in HEURISTICS else None
        if policy is None:
            p = ckpt("pct_nb1")
            if not p:
                print("skip pct (no pct_nb1 checkpoint)"); continue
            policy = load_nets(p, a.device, ("pack",))["pack"]
        row = {}
        u, k = run(seqs, policy, 1, None, None, a.batch, device=a.device,
                   min_support=a.min_support)
        row["none"] = metrics(u, k)
        for nb in a.nbs:
            att, asrc = attacker_for(f"h{name}_nb{nb}", a.device)
            if att is None:
                print(f"  {name} N_B={nb}: no attacker"); continue
            u, k = run(seqs, policy, nb, att, on, a.batch, device=a.device,
                       min_support=a.min_support)
            row[str(nb)] = metrics(u, k)
        out[name] = row
        print(f"{name:10s} " + "  ".join(
            f"{kk}:{vv['uti']:5.1f}/{vv['num']:4.1f}" for kk, vv in row.items()),
            flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["table1", "table2"])
    p.add_argument("--nbs", type=int, nargs="+", default=[10, 20])
    p.add_argument("--betas", type=int, nargs="+", default=[0, 25, 50, 75, 100])
    p.add_argument("--data", default=os.path.join(ROOT, "data/discrete_test.npy"))
    p.add_argument("--n_inst", type=int, default=3000)
    p.add_argument("--batch", type=int, default=375)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min_support", type=float, default=None,
                   help="contact-area floor; must match the one the policies "
                        "were trained under (0 for the published tables)")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    res = table2(a) if a.which == "table2" else table1(a)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    f = os.path.join(ROOT, "results", a.which + ".json")
    json.dump(res, open(f, "w"), indent=1)
    print("->", f)


if __name__ == "__main__":
    main()
