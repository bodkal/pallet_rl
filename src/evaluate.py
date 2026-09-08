"""Evaluate BPP-1 / BPP-k / baselines on the fixed benchmarks.

    python -m src.evaluate --run bpp1_cut2 --datasets RS CUT-1 CUT-2 --episodes 500
    python -m src.evaluate --run bpp1_cut2 --bppk 1 3 5 --episodes 100
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .baselines import BASELINES
from .config import Config
from .env import PackingEnv, build_obs_tensor
from .items import make_testset
from .mcts import PermutationMCTS
from .model import PackNet


def load_run(run, ckpt="best.pt", device="cuda"):
    d = os.path.join("runs", run)
    cfg = Config.from_json(os.path.join(d, "config.json"))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    path = os.path.join(d, ckpt)
    if not os.path.exists(path):
        path = os.path.join(d, "latest.pt")
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = PackNet(cfg).to(dev)
    net.load_state_dict(ck["net"])
    net.eval()
    return cfg, net, dev, d, ck.get("step", 0)


class NetPolicy:
    """Greedy BPP-1 policy using the *predicted* feasibility mask (as deployed)."""
    name = "BPP-1 (ours)"

    def __init__(self, cfg, net, device, deterministic=True):
        self.cfg, self.net, self.device, self.det = cfg, net, device, deterministic

    @torch.no_grad()
    def __call__(self, bin_, obs):
        cfg = self.cfg
        if obs["mask"].sum() == 0:
            return None
        x = build_obs_tensor(obs["hmap"][None], obs["item"][None], cfg)
        logits, _, mask_logits = self.net(torch.as_tensor(x, device=self.device))
        used = (torch.sigmoid(mask_logits) > 0.5).float()
        pl = self.net.projected_logits(logits, used)
        if self.det:
            return int(pl.argmax(-1).item())
        return int(torch.distributions.Categorical(logits=pl).sample().item())


class MCTSPolicy:
    def __init__(self, cfg, net, device, k, n_sim):
        self.cfg, self.k = cfg, k
        self.mcts = PermutationMCTS(cfg, net, device)
        self.n_sim = n_sim
        self.name = f"BPP-{k} (MCTS)"
        self.env = None

    def __call__(self, bin_, obs):
        if obs["mask"].sum() == 0:
            return None
        last = self.env.peek_next_after_lookahead() if self.env is not None else None
        return self.mcts.search(bin_, obs["lookahead"], last, self.n_sim)


def rollout(env, policy, record=False):
    obs = env.reset()
    if hasattr(policy, "env"):
        policy.env = env
    trace = []
    while True:
        a = policy(env.bin, obs)
        if a is None:
            break
        item = tuple(int(v) for v in obs["item"])
        if record:
            x, y, l, w, h = env.bin.decode(a, item, env.cfg.orientations)
            z = int(env.bin.hmap[x:x + l, y:y + w].max())
            trace.append(dict(action=int(a), item=list(item), pos=[x, y, z],
                              dims=[l, w, h],
                              hmap=env.bin.hmap.tolist(),
                              mask=obs["mask"].astype(np.int8).tolist(),
                              valid=bool(env.bin.is_feasible(a, item, env.cfg.orientations))))
        obs, r, d, info = env.step(a)
        if d:
            break
    return dict(utilization=env.bin.utilization, n_items=len(env.bin.placed),
                reason=env.done_reason, placed=env.bin.placed,
                seq_len=len(env.seq), trace=trace)


def evaluate(policy, cfg, dataset, sequences, n, record_first=0, tag=""):
    env = PackingEnv(cfg, dataset=dataset, sequences=sequences, seed=0)
    us, its, reasons, traces = [], [], {}, []
    t0 = time.time()
    for i in range(n):
        rec = rollout(env, policy, record=(i < record_first))
        us.append(rec["utilization"]); its.append(rec["n_items"])
        reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1
        if i < record_first:
            traces.append(rec)
    dt = time.time() - t0
    return dict(policy=policy.name, dataset=dataset, n=n, tag=tag,
                space_util=float(np.mean(us)), space_util_std=float(np.std(us)),
                n_items=float(np.mean(its)), reasons=reasons,
                utils=[float(u) for u in us],
                items=[int(v) for v in its],
                sec_per_episode=dt / n,
                sec_per_item=dt / max(1, sum(its))), traces


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--datasets", nargs="+", default=["RS", "CUT-1", "CUT-2"])
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--bppk", nargs="*", type=int, default=[])
    p.add_argument("--bppk-episodes", type=int, default=100)
    p.add_argument("--mcts-sims", type=int, default=100)
    p.add_argument("--baselines", nargs="*", default=["boundary", "dbl", "random"])
    p.add_argument("--baseline-episodes", type=int, default=200)
    p.add_argument("--record", type=int, default=3, help="episodes to record for replay")
    p.add_argument("--out", default=None)
    p.add_argument("--use-true-mask", default="inherit",
                   choices=["inherit", "yes", "no"],
                   help="project with the ground-truth mask at TEST time. "
                        "'no' scores a net trained with it using only its own "
                        "predictor -- the honest number, since the paper says "
                        "the ground-truth mask is 'only used in the training "
                        "processing'. 'yes' measures the ceiling instead.")
    p.add_argument("--invalid-action-mode", default=None,
                   choices=[None, "terminate", "resample"],
                   help="override the run's setting. A run trained with "
                        "'resample' stores it in config.json, and inheriting it "
                        "here rescues illegal choices at TEST time and inflates "
                        "the numbers -- pass 'terminate' to score it fairly")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    cfg, net, dev, d, step = load_run(a.run, a.ckpt, a.device)
    if a.use_true_mask != "inherit":
        cfg.use_true_mask_for_policy = (a.use_true_mask == "yes")
    elif cfg.use_true_mask_for_policy:
        # Training with the true mask is legitimate -- the paper says the mask
        # is "only used in the training processing". Scoring with it is not:
        # the deployed agent has only its predictor.
        print("WARNING: scoring with the GROUND-TRUTH mask, inherited from the "
              "run's config. That is a ceiling, not a comparable result -- pass "
              "--use-true-mask no for the number the paper's setting implies.",
              flush=True)
    if a.invalid_action_mode:
        cfg.invalid_action_mode = a.invalid_action_mode
    elif cfg.invalid_action_mode != "terminate":
        # A run trained with "resample" carries it in config.json. Scoring with
        # it substitutes a random feasible LP whenever the policy picks an
        # illegal one, so the agent is never punished for a mistake and the
        # utilisation is not comparable to the paper -- which terminates.
        print(f"WARNING: scoring with invalid_action_mode="
              f"{cfg.invalid_action_mode!r} inherited from the run's config. "
              f"Illegal choices will be rescued and utilisation inflated. "
              f"Pass --invalid-action-mode terminate for a comparable number.",
              flush=True)
    out = {"run": a.run, "step": step, "config": cfg.__dict__, "results": []}
    all_traces = {}

    for ds in a.datasets:
        seqs = make_testset(max(a.episodes, a.baseline_episodes), cfg.L, cfg.W, cfg.H,
                            cfg.item_min, cfg.item_max, ds, seed=999)
        # --- our BPP-1 ---
        cfg.lookahead_k = 1
        pol = NetPolicy(cfg, net, dev)
        r, tr = evaluate(pol, cfg, ds, seqs, a.episodes, record_first=a.record)
        out["results"].append(r); all_traces[f"{ds}|BPP-1"] = tr
        print(f"{ds:6s} {r['policy']:18s} util {r['space_util']*100:5.2f}%  "
              f"items {r['n_items']:5.2f}  {r['sec_per_item']*1000:.2f} ms/item")

        # --- baselines ---
        for bn in a.baselines:
            n = a.baseline_episodes if bn != "boundary" else min(a.baseline_episodes, 100)
            bp = BASELINES[bn](cfg)
            r, tr = evaluate(bp, cfg, ds, seqs, n, record_first=a.record if bn == "boundary" else 0)
            out["results"].append(r)
            if tr:
                all_traces[f"{ds}|{bp.name}"] = tr
            print(f"{ds:6s} {r['policy']:18s} util {r['space_util']*100:5.2f}%  "
                  f"items {r['n_items']:5.2f}  {r['sec_per_item']*1000:.2f} ms/item")

        # --- BPP-k ---
        for k in a.bppk:
            if k <= 1:
                continue
            cfg.lookahead_k = k
            mp = MCTSPolicy(cfg, net, dev, k, a.mcts_sims)
            r, tr = evaluate(mp, cfg, ds, seqs, a.bppk_episodes,
                             record_first=min(a.record, 2))
            out["results"].append(r); all_traces[f"{ds}|BPP-{k}"] = tr
            print(f"{ds:6s} {r['policy']:18s} util {r['space_util']*100:5.2f}%  "
                  f"items {r['n_items']:5.2f}  {r['sec_per_item']*1000:.1f} ms/item")
        cfg.lookahead_k = 1

    path = a.out or os.path.join(d, "eval.json")
    with open(path, "w") as f:
        json.dump(out, f)
    with open(os.path.join(d, "traces.json"), "w") as f:
        json.dump(all_traces, f)
    print("wrote", path)
    return out


if __name__ == "__main__":
    main()
