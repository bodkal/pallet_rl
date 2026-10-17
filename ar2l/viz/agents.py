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

from ..env import BPPBatch, CORNER_TYPES
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


def prefer_big(picker, tol=0.01):
    """`picker`, but where boxes tie with its top choice -- a probability at
    most `tol` below the top one's -- the biggest of them goes instead, by
    x * y + z in cells: the footprint first, the height a lesser say.  An
    equal size goes to the likelier box.  The probabilities stay the
    picker's own, so the page still shows what it scored."""
    def fn(env):
        _, p = picker(env)
        win, _ = env.window()
        d = win[..., :3].astype(np.int64)
        size = d[..., 0] * d[..., 1] + d[..., 2]
        tie = (p > 0) & (p >= p.max(1, keepdims=True) - tol - 1e-9)
        return np.where(tie, size + p, -1.0).argmax(1), p
    return fn


def prefer_stack(policy, tol=0.30):
    """`policy`, but a blue or white box it would put on the floor (z = 0)
    goes up instead when a legal placement with z > 0 has a probability at
    most `tol` below the floor one's -- the likeliest such placement.  The
    probabilities stay the policy's own."""
    def fn(env):
        act, p = policy(env)
        z = env._lz                       # the candidate list `policy` scored
        q = p[:, :z.shape[1]]
        ar = np.arange(len(act))
        floor = np.isin(env.head_item()[1], CORNER_TYPES) & (z[ar, act] == 0)
        ok = ((z > 0) & (q > 0)
              & (q >= q[ar, act][:, None] - tol - 1e-9) & floor[:, None])
        up = np.where(ok, q, -1.0).argmax(1)
        return np.where(ok.any(1), up, act), p
    return fn


def floor_x0(policy):
    """`policy`, but a blue or white box it would put on the floor (z = 0)
    off the x = 0 edge, while the box has any legal placement at x = 0, goes
    to the likeliest placement that is left once every floor placement with
    x > 0 is dropped.  The probabilities stay the policy's own."""
    def fn(env):
        act, p = policy(env)
        x, z, lm = env._lxy[..., 0], env._lz, env._lmask
        q = p[:, :z.shape[1]]
        ar = np.arange(len(act))
        hit = (np.isin(env.head_item()[1], CORNER_TYPES)
               & (z[ar, act] == 0) & (x[ar, act] > 0)
               & (lm & (x == 0)).any(1))
        keep = lm & ~((z == 0) & (x > 0))
        return np.where(hit, np.where(keep, q, -1.0).argmax(1), act), p
    return fn


def stack_rule(policy, label, p):
    """`policy` and its label under a parameter set's `prefer_stack` and
    `floor_x0`: the stack first, then the floor edge on what it chose."""
    if p and p.get("prefer_stack"):
        pct = float(p.get("prefer_stack_pct", 30.0))
        policy = prefer_stack(policy, pct / 100)
        label = f"{label}, stacks within {pct:g} points"
    if p and p.get("floor_x0"):
        policy = floor_x0(policy)
        label = f"{label}, floor only at x = 0"
    return policy, label


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


def episode_env(seq, nb=1, S=10, stability="com", min_support=None,
                size_hi=None, max_l=None, n_pick=None, rot=None, ems=None,
                n_types=None, type_constraint=None, arm_collision=None,
                arm_cell_m=None, arm_moves=None, stack_cap=None,
                stack_types=None, stack_side_cm=None, stack_allow_cm=None,
                type_rule=None, soft_mix=None, outline_resort=None,
                outline_when=None, outline_area=None, corner_first=None):
    """The one-bin env `play` packs `seq` in, reset to its first box.

    `S` is an int for a cube or an (Lx, Ly, Lz) triple.  `size_hi` bounds the
    items the env draws for itself once this sequence runs out, so it defaults
    to the sequence's own per-axis maximum rather than the env's cube default,
    which a big bin's sequence would otherwise overshoot.

    `rot` and `ems` are here for the same reason the rest are: a viewer that
    lets the geometry be chosen has to be able to hand the agent *that*
    geometry, and a policy trained with one orientation replayed with two is
    not the policy that was trained.  A viewer that re-plays an episode
    step by step builds its env here too, so it is the same env.
    """
    seq = np.asarray(seq, np.int16)
    # the sequence carries its own types, so the env is told how many to expect
    # and draws no classes of its own
    env = BPPBatch(1, S=S, nb=nb, n_items=len(seq), stability=stability,
                   min_support=min_support, n_pick=n_pick, rot=rot, ems=ems,
                   n_types=n_types, types=False,
                   type_constraint=type_constraint, type_rule=type_rule,
                   soft_mix=soft_mix, outline_resort=outline_resort,
                   outline_when=outline_when, outline_area=outline_area,
                   corner_first=corner_first,
                   arm_collision=arm_collision, arm_cell_m=arm_cell_m,
                   arm_moves=arm_moves, stack_cap=stack_cap,
                   stack_types=stack_types, stack_side_cm=stack_side_cm,
                   stack_allow_cm=stack_allow_cm,
                   size_hi=(seq[..., :3].reshape(-1, 3).max(0)
                            if size_hi is None else size_hi),
                   **({} if max_l is None else {"max_l": max_l}))
    env.reset(seq[None])
    return env


def play(seq, policy, attacker=None, record=True, **kw):
    """One episode; returns the trace the viewers draw.

    `kw` is the env, as `episode_env` takes it.
    """
    env = episode_env(seq, **kw)
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
            item = env.seq[0, min(int(env.head[0]), int(env.length[0]) - 1)]
            rec = {
                "t": len(trace),
                "hmap": env.hmap[0].copy(),
                "tmap": env.tmap[0].copy(),
                # sizes and types stay apart, so a viewer that draws a box has
                # a plain (w, l, h) and never a fourth number in the middle of
                # its geometry
                "window": win[0][wmask[0], :3].tolist(),
                "window_types": win[0][wmask[0], 3].tolist(),
                "perm_idx": None if pidx is None else int(pidx[0]),
                "perm_probs": None if pprob is None else pprob[0].tolist(),
                "item": item[:3].tolist(),
                "item_type": int(item[3]),
                "cands": [c[:6] for c in env.candidates(0, k)],
                "cand_types": [c[6] for c in env.candidates(0, k)],
                # which orientation each candidate turns the item to
                "cand_rot": env._lxy[0, :k, 2].tolist(),
                "probs": prob[0][:k].tolist(),
                "choice": int(act[0]),
            }
        env.step(act)
        if record:
            rec["placed"] = (env.packed[0, env.n_packed[0] - 1] * scale).round().astype(int).tolist()
            rec["placed_type"] = int(env.ptype[0, env.n_packed[0] - 1])
            rec["util"] = float(env.utilization()[0])
            trace.append(rec)
    # what was left on the conveyor when the episode ended: the boxes nothing
    # more could be done with, which no step of the trace records
    win, wmask = env.window()
    return {"trace": trace, "util": float(env.utilization()[0]),
            "final_window": win[0][wmask[0], :3].tolist(),
            "final_window_types": win[0][wmask[0], 3].tolist(),
            "items": int(env.n_packed[0]), "S": (env.Lx, env.Ly, env.Lz),
            "placed": (env.packed[0, : env.n_packed[0]] * scale)
                      .round().astype(int).tolist(),
            "placed_types": env.ptype[0, : env.n_packed[0]].tolist()}
