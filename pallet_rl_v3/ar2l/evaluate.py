"""Held-out evaluation: nominal, attacked, and the mixture datasets of Table 2.

`beta` is the percentage of test instances whose conveyor is reordered by the
policy's own trained permutation-based attacker, exactly as in AR2L Sec. 5.2.
Reported metrics are the paper's: mean space utilisation, its standard
deviation across instances, and the mean number of packed items.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .env import BPPBatch
from .heuristics import NAMES as HEURISTICS, act as heur_act
from .model import PackNet, PermNet, sample
from .ppo import to_torch


def make_dataset(n_inst=3000, n_items=150, size_lo=1, size_hi=5, seed=12345):
    rng = np.random.default_rng(seed)
    return rng.integers(size_lo, size_hi + 1, (n_inst, n_items, 3), dtype=np.int16)


def load_nets(path, device, what=("pack", "attacker")):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    kw = dict(d=a["width"], n_head=a["heads"], n_layer=a["layers"])
    out = {}
    if "pack" in what:
        net = PackNet(**kw).to(device); net.load_state_dict(ck["pack"]); net.eval()
        out["pack"] = net
    for k in ("attacker", "mixer"):
        if k in what:
            net = PermNet(**kw).to(device); net.load_state_dict(ck[k]); net.eval()
            out[k] = net
    out["args"] = a
    return out


@torch.no_grad()
def run(seqs, policy, nb, attacker=None, attacked=None, batch=256, S=10,
        device="cuda", greedy=True, stability="com", ems=True, rot=2):
    """Play every instance once.  `policy` is a net or a heuristic name."""
    n = len(seqs)
    util = np.zeros(n, np.float32); items = np.zeros(n, np.float32)
    attacked = np.zeros(n, bool) if attacked is None else attacked
    for s in range(0, n, batch):
        chunk = seqs[s:s + batch]
        env = BPPBatch(len(chunk), S=S, nb=nb, n_items=seqs.shape[1],
                       size_hi=int(seqs.max()), stability=stability, ems=ems,
                       rot=rot)
        env.reset(chunk)
        on = torch.as_tensor(attacked[s:s + batch]).to(device)
        while not env.done.all():
            if attacker is not None and bool(on.any()):
                o = to_torch(env.obs_cb(), device)
                idx, _, _ = sample(attacker.logits(o), greedy)
                env.permute(torch.where(on, idx, torch.zeros_like(idx)).cpu().numpy())
            if isinstance(policy, str):
                a = heur_act(env, policy)
            else:
                o = to_torch(env.obs(), device)
                a, _, _ = sample(policy.logits(o), greedy)
                a = a.cpu().numpy()
            env.step(a)
        util[s:s + batch] = env.utilization()
        items[s:s + batch] = env.n_packed
    return util, items


def metrics(util, items):
    return {"uti": float(util.mean() * 100), "std": float(util.std() * 100),
            "num": float(items.mean())}


@torch.no_grad()
def nominal_score(pack, nb, n_inst=256, n_items=150, seed=999, device="cuda",
                  stability="com", rot=2, S=10, size_hi=5):
    """Quick greedy score on a fixed held-out slice, for training curves."""
    seqs = make_dataset(n_inst, n_items, size_hi=size_hi, seed=seed)
    u, k = run(seqs, pack, nb, batch=n_inst, device=device, stability=stability,
               rot=rot, S=S)
    return float(u.mean()), float(k.mean())


@torch.no_grad()
def attack_score(policy, attacker, nb, n_inst=256, n_items=150, seed=998,
                 device="cuda", stability="com", rot=2, S=10, size_hi=5):
    """Greedy utilisation of `policy` with every conveyor reordered.

    The attacker is selected and reported under the conditions it will be
    tested in -- both sides greedy -- rather than by the rolling average of
    training rollouts, where the packer it faces is still sampling.
    """
    seqs = make_dataset(n_inst, n_items, size_hi=size_hi, seed=seed)
    u, k = run(seqs, policy, nb, attacker=attacker,
               attacked=np.ones(n_inst, bool), batch=n_inst, device=device,
               stability=stability, rot=rot, S=S)
    return float(u.mean()), float(k.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="packing policy checkpoint")
    p.add_argument("--heuristic", choices=HEURISTICS)
    p.add_argument("--attacker", help="checkpoint holding the attacker to use")
    p.add_argument("--nb", type=int, default=10)
    p.add_argument("--beta", type=float, nargs="*", default=[0, 25, 50, 75, 100])
    p.add_argument("--data", default="data/discrete_test.npy")
    p.add_argument("--n_inst", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rot", type=int, default=2)
    a = p.parse_args()

    seqs = np.load(a.data)[: a.n_inst]
    policy = a.heuristic if a.heuristic else load_nets(a.ckpt, a.device)["pack"]
    attacker = load_nets(a.attacker, a.device, ("attacker",))["attacker"] \
        if a.attacker else None

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(seqs))
    res = {}
    for b in a.beta:
        flag = np.zeros(len(seqs), bool)
        flag[order[: int(round(len(seqs) * b / 100.0))]] = True
        u, k = run(seqs, policy, a.nb, attacker, flag, a.batch,
                   device=a.device, rot=a.rot)
        res[str(int(b))] = metrics(u, k)
        print(f"beta={b:5.0f}  Uti {res[str(int(b))]['uti']:5.1f}  "
              f"Std {res[str(int(b))]['std']:4.1f}  Num {res[str(int(b))]['num']:5.1f}",
              flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
