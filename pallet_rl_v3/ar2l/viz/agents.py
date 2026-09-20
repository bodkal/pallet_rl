"""Packing policies, attackers, and the episode driver the viewers share.

Everything here plays through `ar2l.env.BPPBatch`, so the bins the viewers draw
are produced by exactly the simulator the agent was trained in, under the same
stability rule and the same candidate list.

Policy ids
----------
  run:<name>[:best|last]   a trained packing policy from runs/<name>
  heur:<dbl|bmf|lsah|onlinebph|hmm|macs>
  random                   uniform over the feasible positions

Attacker ids are `run:<name>` (the permutation-based attacker stored in that
checkpoint), `mix:<name>` (its mixture-dynamics model), or None for the
nominal conveyor.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import torch

from ..env import BPPBatch
from ..evaluate import load_nets
from ..heuristics import NAMES as HEURISTICS, scores as heur_scores  # noqa: F401
from ..ppo import to_torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_NETS: dict = {}
_LOCK = threading.Lock()


def run_dir(run):
    return os.path.join(ROOT, "runs", run)


def list_runs():
    d = os.path.join(ROOT, "runs")
    if not os.path.isdir(d):
        return []
    out = [r for r in sorted(os.listdir(d))
           if os.path.exists(os.path.join(d, r, "log.jsonl"))]
    out.sort(key=lambda r: -os.path.getmtime(os.path.join(d, r, "log.jsonl")))
    return out


def run_info(run):
    """Configuration and latest logged numbers for one run, or None."""
    d = run_dir(run)
    ck = next((c for c in ("best.pt", "last.pt")
               if os.path.exists(os.path.join(d, c))), None)
    if ck is None:
        return None
    args = json.load(open(os.path.join(d, "args.json")))
    rows = read_log(run)
    return {"run": run, "ckpt": os.path.join(d, ck), "args": args,
            "algo": args["algo"], "nb": args["nb"], "alpha": args["alpha"],
            # runs trained before --n_pick existed recorded none, and are
            # unrestricted, which is what `None` means downstream
            "n_pick": args.get("n_pick"),
            "it": rows[-1]["it"] if rows else 0,
            "nom_util": rows[-1].get("nom_util", 0.0) if rows else 0.0}


def read_log(run):
    p = os.path.join(run_dir(run), "log.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _nets(path, device):
    key = (path, str(device))
    with _LOCK:
        if key not in _NETS:
            _NETS[key] = load_nets(path, device, ("pack", "attacker", "mixer"))
    return _NETS[key]


def load_policy(spec, device="cuda"):
    """-> (callable(env) -> (action, probs_over_candidates), label, nb)."""
    if spec == "random":
        def fn(env):
            m = env.obs()["l_mask"]
            p = m / np.maximum(m.sum(1, keepdims=True), 1)
            return np.array([np.random.choice(len(r), p=r) if r.sum() > 0 else 0
                             for r in p]), p
        return fn, "random", None

    if spec.startswith("heur:"):
        name = spec.split(":", 1)[1]
        assert name in HEURISTICS, name

        def fn(env, _n=name):
            s = heur_scores(env, _n)
            p = (s == s.max(1, keepdims=True)).astype(np.float64)
            return s.argmax(1), p / p.sum(1, keepdims=True)
        return fn, name.upper(), 1

    run = spec.split(":")[1]
    which = spec.split(":")[2] if spec.count(":") > 1 else "best"
    info = run_info(run)
    if info is None:
        raise FileNotFoundError(f"no checkpoint in runs/{run}")
    path = os.path.join(run_dir(run), which + ".pt")
    nets = _nets(path if os.path.exists(path) else info["ckpt"], device)

    def fn(env, _net=nets["pack"]):
        o = to_torch(env.obs(), device)
        with torch.no_grad():
            lg = _net.logits(o)
            p = torch.softmax(lg, -1)
        return p.argmax(-1).cpu().numpy(), p.cpu().numpy()
    return fn, f"{info['algo']}({run})", info["nb"]


def load_attacker(spec, device="cuda"):
    if not spec or spec == "none":
        return None, "nominal", None
    kind, run = spec.split(":")[0], spec.split(":")[1]
    info = run_info(run)
    if info is None:
        raise FileNotFoundError(f"no checkpoint in runs/{run}")
    nets = _nets(info["ckpt"], device)
    net = nets["mixer" if kind == "mix" else "attacker"]

    def fn(env, _net=net):
        o = to_torch(env.obs_cb(), device)
        with torch.no_grad():
            p = torch.softmax(_net.logits(o), -1)
        return p.argmax(-1).cpu().numpy(), p.cpu().numpy()
    return fn, f"{kind}({run})", info["nb"]


def spec_n_pick(spec):
    """The reach the run behind `spec` was trained with, or `None`.

    A viewer that replays a restricted policy on an unrestricted window would
    let the permuter reach items the cell never could, so the reach is read
    back from the run rather than left to the env default.
    """
    if not spec or spec in ("none", "random") or spec.startswith("heur:"):
        return None
    info = run_info(spec.split(":")[1])
    return info["n_pick"] if info else None


def extent(S):
    """A bin size -- an int for a cube, or an (Lx, Ly, Lz) triple -- as 3 ints."""
    a = np.broadcast_to(np.asarray(S, np.int64), (3,))
    return int(a[0]), int(a[1]), int(a[2])


def play(seq, policy, attacker=None, nb=1, S=10, stability="com", record=True,
         min_support=None,
         size_hi=None, max_l=None, n_pick=None, rot=None, ems=None):
    """One episode; returns the trace the viewers draw.

    `S` is an int for a cube or an (Lx, Ly, Lz) triple.  `size_hi` bounds the
    items the env draws for itself once this sequence runs out, so it defaults
    to the sequence's own per-axis maximum rather than the env's cube default,
    which a big bin's sequence would otherwise overshoot.

    `rot` and `ems` are here for the same reason the rest are: a viewer that
    lets the geometry be chosen has to be able to hand the agent *that*
    geometry, and a policy trained with one orientation replayed with two is
    not the policy that was trained.
    """
    seq = np.asarray(seq, np.int16)
    env = BPPBatch(1, S=S, nb=nb, n_items=len(seq), stability=stability,
                   min_support=min_support, n_pick=n_pick, rot=rot, ems=ems,
                   size_hi=seq.reshape(-1, 3).max(0) if size_hi is None else size_hi,
                   **({} if max_l is None else {"max_l": max_l}))
    env.reset(seq[None])
    scale = env.scale        # one divisor for every axis, as in the env itself
    trace = []
    while not env.done[0]:
        win, wmask = env.window()
        pidx, pprob = (attacker(env) if attacker is not None else (None, None))
        if pidx is not None:
            env.permute(pidx)
        o = env.obs()
        act, prob = policy(env)
        if not o["l_mask"][0].any():
            break
        rec = None
        if record:
            k = int(o["l_mask"][0].sum())
            item = env.seq[0, min(int(env.head[0]), env.n_items - 1)]
            rec = {
                "t": len(trace),
                "hmap": env.hmap[0].copy(),
                "window": win[0][wmask[0]].tolist(),
                "perm_idx": None if pidx is None else int(pidx[0]),
                "perm_probs": None if pprob is None else pprob[0].tolist(),
                "item": item.tolist(),
                "cands": env.candidates(0, k),
                "probs": prob[0][:k].tolist(),
                "choice": int(act[0]),
            }
        env.step(act)
        if record:
            rec["placed"] = (env.packed[0, env.n_packed[0] - 1] * scale).round().astype(int).tolist()
            rec["util"] = float(env.utilization()[0])
            trace.append(rec)
    return {"trace": trace, "util": float(env.utilization()[0]),
            "items": int(env.n_packed[0]), "S": (env.Lx, env.Ly, env.Lz),
            "placed": (env.packed[0, : env.n_packed[0]] * scale)
                      .round().astype(int).tolist()}
