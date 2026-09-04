"""Train the BPP-1 constrained-DRL agent.

    python -m src.train --preset paper --dataset CUT-2 --run bpp1_cut2
    python -m src.train --preset smoke                    # ~20 min pipeline check
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch

from .config import build, Config
from .env import VecPackingEnv
from .model import PackNet
from .ppo import PPOTrainer


def run_dir(name):
    d = os.path.join("runs", name)
    os.makedirs(d, exist_ok=True)
    return d


def save_ckpt(path, net, opt, step, update, cfg, elapsed=0.0):
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                "step": step, "update": update, "elapsed": elapsed,
                "cfg": cfg.__dict__}, path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="paper")
    p.add_argument("--run", default=None)
    p.add_argument("--dataset", default=None, choices=[None, "RS", "CUT-1", "CUT-2"])
    p.add_argument("--orientations", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    # ablation switches (paper Table 1)
    p.add_argument("--no-mp", action="store_true", help="disable mask prediction")
    p.add_argument("--no-mc", action="store_true", help="disable mask projection")
    p.add_argument("--no-fe", action="store_true", help="disable feasibility entropy")
    a = p.parse_args(argv)

    cfg = build(a.preset, run_name=a.run, dataset=a.dataset,
                orientations=a.orientations, total_steps=a.total_steps,
                max_hours=a.max_hours, seed=a.seed, lr=a.lr,
                num_envs=a.num_envs, device=a.device)
    if a.no_mp: cfg.use_mask_prediction = False
    if a.no_mc: cfg.use_mask_constraint = False
    if a.no_fe: cfg.use_feasibility_entropy = False

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    d = run_dir(cfg.run_name)
    cfg.to_json(os.path.join(d, "config.json"))
    metrics_path = os.path.join(d, "metrics.jsonl")

    envs = VecPackingEnv(cfg, cfg.num_envs, seed=cfg.seed)
    net = PackNet(cfg).to(device)
    tr = PPOTrainer(cfg, net, envs, device)

    step = update = 0
    prior_elapsed = 0.0
    if a.resume and os.path.exists(os.path.join(d, "latest.pt")):
        ck = torch.load(os.path.join(d, "latest.pt"), map_location=device, weights_only=False)
        net.load_state_dict(ck["net"]); tr.opt.load_state_dict(ck["opt"])
        step, update = ck["step"], ck["update"]
        prior_elapsed = ck.get("elapsed", 0.0)
        print(f"resumed from step {step:,} (update {update}, "
              f"{prior_elapsed/3600:.2f}h already spent)", flush=True)
    else:
        open(metrics_path, "w").close()

    batch = cfg.num_envs * cfg.num_steps
    n_updates = cfg.total_steps // batch
    util_win, item_win = deque(maxlen=400), deque(maxlen=400)
    reasons = {}
    best_util, t0 = 0.0, time.time() - prior_elapsed

    print(f"run={cfg.run_name} dataset={cfg.dataset} orient={cfg.orientations} "
          f"MP={cfg.use_mask_prediction} MC={cfg.use_mask_constraint} "
          f"FE={cfg.use_feasibility_entropy} device={device}")
    print(f"batch={batch}  updates={n_updates}  action_dim={cfg.action_dim}")

    while step < cfg.total_steps:
        if cfg.lr_schedule == "linear":
            tr.set_lr(cfg.lr * max(0.05, 1.0 - step / cfg.total_steps))

        (adv, ret), invalid_rate = tr.collect()
        logs = tr.update(adv, ret)
        step += batch; update += 1

        for u, n, why in tr.drain_episode_stats():
            util_win.append(u); item_win.append(n)
            reasons[why] = reasons.get(why, 0) + 1

        if update % cfg.log_interval == 0 and util_win:
            el = time.time() - t0
            tot = max(1, sum(reasons.values()))
            rec = dict(update=update, step=step, time=el, fps=step / el,
                       space_util=float(np.mean(util_win)),
                       space_util_std=float(np.std(util_win)),
                       n_items=float(np.mean(item_win)),
                       ep_reward=float(np.mean(util_win)) * 10.0,
                       invalid_rate=invalid_rate,
                       lr=tr.opt.param_groups[0]["lr"],
                       frac_seq_end=reasons.get("sequence_end", 0) / tot,
                       frac_no_feasible=reasons.get("no_feasible", 0) / tot,
                       frac_invalid_end=reasons.get("invalid_action", 0) / tot,
                       **logs)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if update % 10 == 0 or update < 5:
                print(f"[{update:6d}] step {step/1e6:6.2f}M  util {rec['space_util']:.4f} "
                      f"items {rec['n_items']:5.2f}  invalid {invalid_rate:.4f}  "
                      f"maskacc {logs['mask_acc']:.4f}  {rec['fps']:6.0f} fps  "
                      f"{el/3600:.2f}h", flush=True)
            reasons = {}

        if update % cfg.save_interval == 0:
            el = time.time() - t0
            save_ckpt(os.path.join(d, "latest.pt"), net, tr.opt, step, update, cfg, el)
            if util_win and np.mean(util_win) > best_util:
                best_util = float(np.mean(util_win))
                save_ckpt(os.path.join(d, "best.pt"), net, tr.opt, step, update, cfg, el)

        if (time.time() - t0) / 3600.0 > cfg.max_hours:
            print("time budget reached"); break

    save_ckpt(os.path.join(d, "latest.pt"), net, tr.opt, step, update, cfg,
              time.time() - t0)
    print(f"done: {step:,} steps in {(time.time()-t0)/3600:.2f}h, best util {best_util:.4f}")
    return d


if __name__ == "__main__":
    main()
