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
from .orders import (add_cm_args, add_order_args, load_orders, orders_bin,
                     randomize_order)
from .ppo import to_torch


def make_dataset(n_inst=3000, n_items=150, size_lo=1, size_hi=5, seed=12345,
                 n_types=None, types=None):
    """An instance set as `(n_inst, n_items, 4)`: sizes and a type per box.

    `size_lo`/`size_hi` may be an int or a per-axis triple.  `types`/`n_types`
    default to `config.yaml`, so the held-out sets a run is scored on are drawn
    from the same size classes the run was trained on; pass `types=False` for
    the single untyped class.
    """
    from .env import sample_items, type_classes
    if types is None:
        types = CFG["env"]["types"]
    elif types is False:
        types = None
    n_types = CFG["env"]["n_types"] if n_types is None else n_types
    classes = type_classes(types, n_types, size_lo, size_hi)
    return sample_items(np.random.default_rng(seed), (n_inst, n_items),
                        size_lo, size_hi, classes)


def _fit_state(net, sd):
    """A checkpoint's weights onto `net`, widening the pre-type projections.

    Before box types every node was projected straight from its physical
    features; now the type embedding is concatenated on first, so each
    element-wise `fc_in` is `type_embed` columns wider.  Padding the old
    columns with zeros is the identity on the features the run was trained
    with -- the loaded packer computes exactly what it used to -- and leaves
    the embedding at its fresh initialisation, which is the honest state for a
    policy that has never seen a type.  Everything else must match as before.
    """
    tgt = net.state_dict()
    stray = set(sd) - set(tgt)
    if stray:
        raise ValueError(f"checkpoint holds weights this network has no place "
                         f"for: {sorted(stray)}")
    out = {}
    for k, want in tgt.items():
        have = sd.get(k)
        if have is None:
            # only the embedding table may be absent -- a pre-type checkpoint
            # has none.  Anything else missing is a different network, not an
            # older one, and silently keeping the fresh initialisation there
            # would report a trained policy that is partly untrained.
            if not k.endswith("type_embedding.weight"):
                raise ValueError(f"checkpoint is missing {k}")
            out[k] = want
            continue
        if (have.shape != want.shape and have.dim() == 2
                and have.shape[0] == want.shape[0]
                and have.shape[1] < want.shape[1]):
            grown = torch.zeros_like(want)
            grown[:, : have.shape[1]] = have.to(want.dtype)
            out[k] = grown
        else:
            out[k] = have
    net.load_state_dict(out)
    return net


def load_nets(path, device, what=("pack", "attacker")):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    kw = dict(d=a["width"], n_head=a["heads"], n_layer=a["layers"])
    if "c_temp" in a:                    # runs trained before --c_temp existed
        kw["c_temp"] = a["c_temp"]       # have none, and take the config value
    # a run trained before box types recorded neither, and is rebuilt at the
    # current type count so it can be run against a typed env at all
    kw["n_types"] = a.get("n_types", CFG["env"]["n_types"])
    kw["type_embed"] = a.get("type_embed", CFG["model"]["type_embed"])
    out = {}
    if "pack" in what:
        net = _fit_state(PackNet(**kw).to(device), ck["pack"]); net.eval()
        out["pack"] = net
    for k in ("attacker", "mixer"):
        if k in what:
            net = _fit_state(PermNet(**kw).to(device), ck[k]); net.eval()
            out[k] = net
    out["args"] = a
    return out


@torch.no_grad()
def run(seqs, policy, nb, attacker=None, attacked=None, batch=None, S=None,
        device=None, greedy=True, stability=None, ems=None, rot=None,
        max_l=None, min_support=None, n_pick=None, n_types=None,
        type_constraint=None):
    """Play every instance once.  `policy` is a net or a heuristic name.

    Anything left `None` comes from `config.yaml`; the env arguments are
    passed straight through to `BPPBatch`, which resolves them the same way.
    The instances carry their own box types, so the env is told how many types
    to expect but draws nothing itself -- `types=False` keeps it from
    re-deriving the size envelope from the config classes when the dataset on
    disk was drawn from different ones.
    """
    batch = CFG["eval"]["batch"] if batch is None else batch
    device = CFG["run"]["device"] if device is None else device
    seqs = np.asarray(seqs)
    n = len(seqs)
    util = np.zeros(n, np.float32); items = np.zeros(n, np.float32)
    attacked = np.zeros(n, bool) if attacked is None else attacked
    for s in range(0, n, batch):
        chunk = seqs[s:s + batch]
        env = BPPBatch(len(chunk), S=S, nb=nb, n_items=seqs.shape[1],
                       size_hi=seqs[..., :3].reshape(-1, 3).max(0),
                       stability=stability,
                       ems=ems, rot=rot, max_l=max_l, min_support=min_support,
                       n_pick=n_pick, n_types=n_types, types=False,
                       type_constraint=type_constraint)
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


def metrics(util, items, length=None):
    out = {"uti": float(util.mean() * 100), "std": float(util.std() * 100),
           "num": float(items.mean())}
    if length is not None:
        # a real order is finished only when every one of its boxes is on
        out["all"] = float((items >= length).mean() * 100)
    return out


@torch.no_grad()
def nominal_score(pack, nb, n_inst=256, n_items=150, seed=999, device="cuda",
                  stability="com", rot=2, S=10, size_hi=5, max_l=120,
                  min_support=None, n_pick=None, n_types=None,
                  type_constraint=None, seqs=None):
    """Quick greedy score on a fixed held-out slice, for training curves.

    `seqs` scores those instances (a run's held-out real pallets) instead of
    a slice drawn from the size classes."""
    if seqs is None:
        seqs = make_dataset(n_inst, n_items, size_hi=size_hi, seed=seed,
                            n_types=n_types)
    u, k = run(seqs, pack, nb, batch=len(seqs), device=device, stability=stability,
               rot=rot, S=S, max_l=max_l, min_support=min_support,
               n_pick=n_pick, n_types=n_types, type_constraint=type_constraint)
    return float(u.mean()), float(k.mean())


@torch.no_grad()
def attack_score(policy, attacker, nb, n_inst=256, n_items=150, seed=998,
                 device="cuda", stability="com", rot=2, S=10, size_hi=5,
                 max_l=120, min_support=None, n_pick=None, n_types=None,
                 type_constraint=None, seqs=None):
    """Greedy utilisation of `policy` with every conveyor reordered.

    The attacker is selected and reported under the conditions it will be
    tested in -- both sides greedy -- rather than by the rolling average of
    training rollouts, where the packer it faces is still sampling.  `seqs`
    as in `nominal_score`.
    """
    if seqs is None:
        seqs = make_dataset(n_inst, n_items, size_hi=size_hi, seed=seed,
                            n_types=n_types)
    u, k = run(seqs, policy, nb, attacker=attacker,
               attacked=np.ones(len(seqs), bool), batch=len(seqs), device=device,
               stability=stability, rot=rot, S=S, max_l=max_l,
               min_support=min_support, n_pick=n_pick, n_types=n_types,
               type_constraint=type_constraint)
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
    p.add_argument("--data", default=ev["data"],
                   help="instances: a .npy, or an orders .csv (see ar2l/orders.py)")
    add_cm_args(p)
    add_order_args(p)
    p.add_argument("--per_pallet", default=None,
                   help="write one CSV row per instance and beta here")
    p.add_argument("--n_inst", type=int, default=ev["n_inst"])
    p.add_argument("--batch", type=int, default=ev["batch"])
    p.add_argument("--device", default=r["device"])
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=ev["seed"])
    p.add_argument("--rot", type=int, default=e["rot"])
    p.add_argument("--min_support", type=float, default=None,
                   help="contact-area floor; the config.yaml value (%.2f) "
                        "unless set" % e["min_support"])
    p.add_argument("--n_types", type=int, default=e["n_types"],
                   help="box types the dataset carries; must match the "
                        "policy's")
    p.add_argument("--type_constraint", type=int, default=e["type_constraint"],
                   help="0 scores the policy with the stacking rule lifted")
    return p


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        load_config(known.config)
    a = get_parser().parse_args(argv)

    if a.data.lower().endswith(".csv"):
        S = orders_bin(a.pallet_cm, a.cell_cm)
        seqs, ids = load_orders(a.data, a.cell_cm, S, rot=a.rot,
                                box_scale=a.box_scale, box_round=a.box_round)
        print(f"{len(seqs)} pallets from {a.data} on a {S[0]}x{S[1]}x{S[2]}-cell "
              f"pallet ({a.cell_cm:g} cm cells"
              + (f", boxes / {a.box_scale:g}" if a.box_scale not in (0, 1) else "")
              + ")", flush=True)
    else:
        S, (seqs, ids) = None, (np.load(a.data), None)
    # the whole table, then the slice: pallet k gets the same order here as
    # in a replay of it, whatever --n_inst is
    seqs = randomize_order(seqs, a.order_random, a.order_seed)[: a.n_inst]
    if a.order_random:
        print(f"box order randomised: {a.order_random:g} "
              f"(seed {a.order_seed})", flush=True)
    ids = [str(i) for i in range(len(seqs))] if ids is None else ids[: a.n_inst]
    length = (seqs[..., :3] > 0).all(-1).sum(-1)
    rows = []
    policy = a.heuristic if a.heuristic else load_nets(a.ckpt, a.device)["pack"]
    attacker = load_nets(a.attacker, a.device, (a.permuter,))[a.permuter] \
        if a.attacker else None

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(seqs))
    res = {}
    for b in a.beta:
        flag = np.zeros(len(seqs), bool)
        flag[order[: int(round(len(seqs) * b / 100.0))]] = True
        u, k = run(seqs, policy, a.nb, attacker, flag, a.batch, S=S,
                   device=a.device, rot=a.rot, min_support=a.min_support,
                   n_pick=a.n_pick, n_types=a.n_types,
                   type_constraint=bool(a.type_constraint))
        m = res[str(int(b))] = metrics(u, k, length)
        print(f"beta={b:5.0f}  Uti {m['uti']:5.1f}  Std {m['std']:4.1f}  "
              f"Num {m['num']:5.1f}  All {m['all']:5.1f}%", flush=True)
        rows += [(ids[i], int(b), int(length[i]), int(k[i]),
                  round(float(u[i]) * 100, 2)) for i in range(len(seqs))]
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    if a.per_pallet:
        os.makedirs(os.path.dirname(a.per_pallet) or ".", exist_ok=True)
        with open(a.per_pallet, "w") as f:
            f.write("pallet_id,beta,boxes,placed,util_pct\n")
            f.writelines(",".join(map(str, r)) + "\n" for r in rows)


if __name__ == "__main__":
    main()
