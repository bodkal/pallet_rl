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

from .config import CFG, load as load_config
from .env import BPPBatch, MIN_SUPPORT
from .heuristics import NAMES as HEURISTICS, act as heur_act
from .model import PackNet, PermNet, sample
from .ppo import to_torch


def make_dataset(n_inst=3000, n_items=150, size_lo=1, size_hi=5, seed=12345):
    """`size_lo`/`size_hi` may be an int or a per-axis triple."""
    from .env import sample_items
    return sample_items(np.random.default_rng(seed), (n_inst, n_items),
                        size_lo, size_hi)


def load_nets(path, device, what=("pack", "attacker")):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    kw = dict(d=a["width"], n_head=a["heads"], n_layer=a["layers"])
    if "c_temp" in a:                    # runs trained before --c_temp existed
        kw["c_temp"] = a["c_temp"]       # have none, and take the config value
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
def run(seqs, policy, nb, attacker=None, attacked=None, batch=None, S=None,
        device=None, greedy=True, stability=None, ems=None, rot=None,
        max_l=None, min_support=None, n_pick=None):
    """Play every instance once.  `policy` is a net or a heuristic name.

    Anything left `None` comes from `config.yaml`; the env arguments are
    passed straight through to `BPPBatch`, which resolves them the same way.
    """
    batch = CFG["eval"]["batch"] if batch is None else batch
    device = CFG["run"]["device"] if device is None else device
    n = len(seqs)
    util = np.zeros(n, np.float32); items = np.zeros(n, np.float32)
    attacked = np.zeros(n, bool) if attacked is None else attacked
    for s in range(0, n, batch):
        chunk = seqs[s:s + batch]
        env = BPPBatch(len(chunk), S=S, nb=nb, n_items=seqs.shape[1],
                       size_hi=seqs.reshape(-1, 3).max(0), stability=stability,
                       ems=ems, rot=rot, max_l=max_l, min_support=min_support,
                       n_pick=n_pick)
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
                  stability="com", rot=2, S=10, size_hi=5, max_l=120,
                  min_support=None, n_pick=None):
    """Quick greedy score on a fixed held-out slice, for training curves."""
    seqs = make_dataset(n_inst, n_items, size_hi=size_hi, seed=seed)
    u, k = run(seqs, pack, nb, batch=n_inst, device=device, stability=stability,
               rot=rot, S=S, max_l=max_l, min_support=min_support,
               n_pick=n_pick)
    return float(u.mean()), float(k.mean())


@torch.no_grad()
def attack_score(policy, attacker, nb, n_inst=256, n_items=150, seed=998,
                 device="cuda", stability="com", rot=2, S=10, size_hi=5,
                 max_l=120, min_support=None, n_pick=None):
    """Greedy utilisation of `policy` with every conveyor reordered.

    The attacker is selected and reported under the conditions it will be
    tested in -- both sides greedy -- rather than by the rolling average of
    training rollouts, where the packer it faces is still sampling.
    """
    seqs = make_dataset(n_inst, n_items, size_hi=size_hi, seed=seed)
    u, k = run(seqs, policy, nb, attacker=attacker,
               attacked=np.ones(n_inst, bool), batch=n_inst, device=device,
               stability=stability, rot=rot, S=S, max_l=max_l,
               min_support=min_support, n_pick=n_pick)
    return float(u.mean()), float(k.mean())


def get_parser():
    """Defaults from `config.yaml`; `--config` is applied before this runs."""
    e, ev, r = CFG["env"], CFG["eval"], CFG["run"]
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="packing policy checkpoint")
    p.add_argument("--heuristic", choices=HEURISTICS)
    p.add_argument("--attacker", help="checkpoint holding the permuter to use")
    p.add_argument("--permuter", choices=["attacker", "mixer"], default="attacker",
                   help="which permutation net inside --attacker to run: the "
                        "adversary, or `mixer`, the cooperative selector that "
                        "--algo select trains")
    p.add_argument("--config", default=None,
                   help="YAML file of defaults to merge over config.yaml")
    p.add_argument("--nb", type=int, default=CFG["train"]["nb"])
    p.add_argument("--n_pick", type=int, default=CFG["train"]["n_pick"],
                   help="how many of the N_B are within reach; must match "
                        "what the policy was trained with")
    p.add_argument("--beta", type=float, nargs="*", default=ev["beta"])
    p.add_argument("--data", default=ev["data"])
    p.add_argument("--n_inst", type=int, default=ev["n_inst"])
    p.add_argument("--batch", type=int, default=ev["batch"])
    p.add_argument("--device", default=r["device"])
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=ev["seed"])
    p.add_argument("--rot", type=int, default=e["rot"])
    p.add_argument("--min_support", type=float, default=None,
                   help="contact-area floor; the config.yaml value (%.2f) "
                        "unless set" % e["min_support"])
    return p


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        load_config(known.config)
    a = get_parser().parse_args(argv)

    seqs = np.load(a.data)[: a.n_inst]
    policy = a.heuristic if a.heuristic else load_nets(a.ckpt, a.device)["pack"]
    attacker = load_nets(a.attacker, a.device, (a.permuter,))[a.permuter] \
        if a.attacker else None

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(seqs))
    res = {}
    for b in a.beta:
        flag = np.zeros(len(seqs), bool)
        flag[order[: int(round(len(seqs) * b / 100.0))]] = True
        u, k = run(seqs, policy, a.nb, attacker, flag, a.batch,
                   device=a.device, rot=a.rot, min_support=a.min_support,
                   n_pick=a.n_pick)
        res[str(int(b))] = metrics(u, k)
        print(f"beta={b:5.0f}  Uti {res[str(int(b))]['uti']:5.1f}  "
              f"Std {res[str(int(b))]['std']:4.1f}  Num {res[str(int(b))]['num']:5.1f}",
              flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
