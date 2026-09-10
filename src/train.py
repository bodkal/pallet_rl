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
from .env import make_vec_env
from .items import make_testset
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
    p.add_argument("--bin", default=None, metavar="N|LxWxH",
                   help="bin size, e.g. 15 (cubic) or 15x15x20. Item sizes follow "
                        "the paper's l<=L/2 rule unless --item-max is given, and a "
                        "checkpoint from a different bin size cannot be resumed")
    p.add_argument("--item-min", type=int, default=None)
    p.add_argument("--item-max", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None,
                   help="trunk / mask-head width (paper 256)")
    p.add_argument("--cnn-channels", type=int, default=None,
                   help="conv width (paper 64)")
    p.add_argument("--cnn-layers", type=int, default=None,
                   help="3x3 conv layers before the 1x1 bottleneck (paper 2)")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--workers", type=int, default=None,
                   help="processes to run the env loop across (default 0 = one "
                        "core, as before). Gradients are identical either way; "
                        "this only stops env.step being pinned to a single core")
    p.add_argument("--seq-pool", type=int, default=None,
                   help="pre-generate N training sequences and cycle them "
                        "instead of generating one per episode reset (0 = off)")
    p.add_argument("--target-kl", type=float, default=None,
                   help="abandon the rest of an update once approx_kl exceeds "
                        "this (0 = off). ~0.02 is the usual PPO target; the "
                        "20^3 runs measured 0.174 early with peaks of 2.92")
    p.add_argument("--invalid-action-mode", default=None,
                   choices=[None, "terminate", "resample"],
                   help="terminate = the paper's setting; resample replaces an "
                        "illegal choice with a random feasible LP")
    p.add_argument("--w-mask", type=float, default=None,
                   help="weight on the mask-prediction loss (paper 0.5)")
    p.add_argument("--use-true-mask", action="store_true",
                   help="DEVIATES from the paper: project with the ground-truth "
                        "mask instead of the predicted one")
    p.add_argument("--mask-eps", type=float, default=None,
                   help="infeasible actions keep prob * eps (paper 1e-3). 0 is "
                        "a hard projection; combined with --use-true-mask it "
                        "removes the last source of illegal moves, since eps is "
                        "then the only one left")
    p.add_argument("--epochs", type=int, default=None,
                   help="PPO epochs per update (paper/default 4). Lowering this "
                        "raises steps/s but takes fewer gradient steps per sample")
    p.add_argument("--minibatches", type=int, default=None,
                   help="PPO minibatches per epoch (paper/default 8). Same "
                        "trade-off as --epochs")
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    # ablation switches (paper Table 1)
    p.add_argument("--no-mp", action="store_true", help="disable mask prediction")
    p.add_argument("--no-mc", action="store_true", help="disable mask projection")
    p.add_argument("--no-fe", action="store_true", help="disable feasibility entropy")
    a = p.parse_args(argv)

    L = W = H = None
    if a.bin:
        try:
            parts = [int(v) for v in a.bin.lower().split("x")]
        except ValueError:
            p.error(f"--bin: expected N or LxWxH, got {a.bin!r}")
        if len(parts) == 1:
            L = W = H = parts[0]
        elif len(parts) == 3:
            L, W, H = parts
        else:
            p.error(f"--bin: expected N or LxWxH, got {a.bin!r}")
        if min(parts) < 2:
            p.error("--bin: every dimension must be >= 2")

    cfg = build(a.preset, run_name=a.run, dataset=a.dataset, L=L, W=W, H=H,
                item_min=a.item_min, item_max=a.item_max,
                orientations=a.orientations, total_steps=a.total_steps,
                max_hours=a.max_hours, seed=a.seed, lr=a.lr,
                num_envs=a.num_envs, device=a.device, env_workers=a.workers,
                hidden=a.hidden, cnn_channels=a.cnn_channels,
                cnn_layers=a.cnn_layers, target_kl=a.target_kl,
                invalid_action_mode=a.invalid_action_mode, w_mask=a.w_mask,
                seq_pool=a.seq_pool, epochs=a.epochs,
                minibatches=a.minibatches, mask_eps=a.mask_eps)
    if a.no_mp: cfg.use_mask_prediction = False
    if a.no_mc: cfg.use_mask_constraint = False
    if a.no_fe: cfg.use_feasibility_entropy = False
    if a.use_true_mask: cfg.use_true_mask_for_policy = True

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    # Fail loudly rather than silently dropping to CPU. A wedged driver makes
    # torch.cuda.is_available() return False ("CUDA unknown error" from
    # cuda_getDeviceCount), and the old fallback quietly restarted a 10M-step
    # run on CPU at ~5 cores and a fraction of the speed -- it looked like it
    # had resumed fine. Pass --device cpu to ask for CPU on purpose.
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {cfg.device} was requested but torch.cuda.is_available() "
            f"is False, so this run would silently train on CPU.\n"
            f"  If nvidia-smi works, the driver's compute stack is usually "
            f"wedged after a GPU fault; reloading the UVM module normally "
            f"clears it without a reboot:\n"
            f"      sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm\n"
            f"  Re-check with: python3 -c 'import torch; "
            f"print(torch.cuda.is_available())'\n"
            f"  Or pass --device cpu to train on CPU deliberately.")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    d = run_dir(cfg.run_name)
    cfg.to_json(os.path.join(d, "config.json"))
    metrics_path = os.path.join(d, "metrics.jsonl")

    pool = None
    if cfg.seq_pool > 0:
        # NOTE: seed must not collide with the held-out benchmark, which
        # src.evaluate builds with seed=999 — training on it would be leakage.
        pool = make_testset(cfg.seq_pool, cfg.L, cfg.W, cfg.H, cfg.item_min,
                            cfg.item_max, cfg.dataset, seed=100_000 + cfg.seed)
        print(f"training on a pre-generated pool of {cfg.seq_pool:,} sequences")
    # built before the net moves to the GPU: the workers are forked, and they
    # must not inherit an initialised CUDA context
    envs = make_vec_env(cfg, cfg.num_envs, seed=cfg.seed, sequences=pool,
                        workers=cfg.env_workers)
    if cfg.env_workers > 1:
        print(f"env loop split over {cfg.env_workers} worker processes")
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
