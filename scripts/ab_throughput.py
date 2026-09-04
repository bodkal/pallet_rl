"""A/B the throughput knobs: do they actually buy utilisation, or just steps/s?

    python3 -m scripts.ab_throughput --steps 3000000     # run + render
    python3 -m scripts.ab_throughput --render-only       # re-draw from runs/old_run/ab_*

Each arm adds ONE change to the one before it, so the deltas are attributable.
Arms run sequentially -- concurrent runs contend for the GPU and would make the
timing meaningless (TODO.md D.1: 3 concurrent runs drop to 2,447 steps/s each).

The point is that steps/s is NOT the answer. --num-envs and --epochs both take
fewer gradient steps per sample, so an arm can be faster per step and worse per
step. Three questions, three panels:

    utilisation vs step   -- did the change cost sample efficiency?
    utilisation vs hour   -- did it pay for that cost in wall-clock?
    steps/s               -- the raw speed, for reference
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.make_readme_figures import (SURFACE, INK, INK2, INK3, GRID, strip)

OUT = "docs"
# dataviz reference palette, categorical slots 1-5 (adjacent pairlist: lines)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# each arm = the previous one plus exactly one change
ARMS = [
    ("paper (32 envs)", "old_run/ab_paper",   []),
    ("+ seq pool",      "old_run/ab_pool",    ["--seq-pool", "40000"]),
    ("+ 128 envs",      "old_run/ab_envs128", ["--seq-pool", "40000", "--num-envs", "128"]),
    ("+ PPO 2x4",       "old_run/ab_ppo24",   ["--seq-pool", "40000", "--num-envs", "128",
                                       "--epochs", "2", "--minibatches", "4"]),
]


def load(run):
    """metrics.jsonl -> (steps, hours, utilisation). Empty if the run is missing."""
    p = os.path.join("runs", run, "metrics.jsonl")
    if not os.path.exists(p):
        return None
    step, hrs, util = [], [], []
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            step.append(r["step"]); hrs.append(r["time"] / 3600); util.append(r["space_util"])
    if not step:
        return None
    return np.array(step), np.array(hrs), np.array(util)


def train(run, extra, steps, dataset, seed, force):
    d = os.path.join("runs", run)
    have = load(run)
    if have is not None and have[0][-1] >= steps * 0.98 and not force:
        print(f"  {run}: already at {have[0][-1]:,} steps -- skipping (use --force)")
        return
    if os.path.exists(d):
        import shutil; shutil.rmtree(d)
    cmd = [sys.executable, "-m", "src.train", "--preset", "paper",
           "--dataset", dataset, "--run", run, "--seed", str(seed),
           "--total-steps", str(steps)] + extra
    print(f"  $ {' '.join(cmd)}")
    t = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    print(f"  {run}: done in {(time.time()-t)/60:.1f} min")


def at(x, y, xq):
    """y interpolated at xq, or nan if the arm never got that far."""
    return float(np.interp(xq, x, y)) if xq <= x[-1] else float("nan")


def grad_steps(run, n_updates):
    """PPO takes epochs x minibatches gradient steps per update."""
    with open(os.path.join("runs", run, "config.json")) as f:
        c = json.load(f)
    return n_updates * c["epochs"] * c["minibatches"], c["num_envs"] * c["num_steps"]


def summarise(data, steps):
    """One row per arm, all deltas against the first arm."""
    rows = []
    base_step_util = base_hour_util = None
    # compare wall-clock at the point the SLOWEST arm reached, so every arm has a value
    t_common = min(h[-1] for _, h, _ in data.values())
    for i, (label, _, _) in enumerate(ARMS):
        if label not in data:
            continue
        s, h, u = data[label]
        run = ARMS[i][1]
        gsteps, batch = grad_steps(run, len(s))
        sps = s[-1] / (h[-1] * 3600)
        u_step = u[-1]                       # utilisation at the full step budget
        u_hour = at(s * 0 + h, u, t_common)  # utilisation at matched wall-clock
        if i == 0:
            base_step_util, base_hour_util, base_sps = u_step, u_hour, sps
        rows.append(dict(
            label=label, steps=int(s[-1]), hours=h[-1], sps=sps,
            batch=batch, updates=len(s), grad_steps=gsteps,
            speedup=sps / base_sps,
            util_at_steps=u_step, d_step=(u_step - base_step_util) * 100,
            util_at_hours=u_hour, d_hour=(u_hour - base_hour_util) * 100,
        ))
    return rows, t_common


def smooth(x, y, target=70):
    """Rolling mean to ~target points -- the raw per-update series is noisy."""
    w = max(1, len(y) // target)
    if w < 2:
        return x, y
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def render(data, rows, t_common, steps, dataset, seed):
    colors = {lab: PALETTE[i] for i, (lab, _, _) in enumerate(ARMS)}
    fig = plt.figure(figsize=(12.4, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.15, 0.85),
                          wspace=0.28, top=0.70, bottom=0.13, left=0.055, right=0.985)
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))

    for label, (st, h, u) in data.items():
        c = colors[label]
        a1.plot(*smooth(st / 1e6, u * 100), lw=1.9, color=c, solid_capstyle="round")
        a2.plot(*smooth(h, u * 100), lw=1.9, color=c, solid_capstyle="round")

    a1.set_xlabel("environment steps (millions)")
    a1.set_title("Utilisation per step", fontsize=11.5, color=INK, loc="left", pad=13)
    a1.text(0, 1.015, "same x = same experience: does the change cost sample efficiency?",
            transform=a1.transAxes, fontsize=8.5, color=INK3, va="bottom")

    a2.axvline(t_common, color=INK3, lw=0.9, ls=(0, (3, 3)), zorder=2)
    a2.set_xlabel("wall-clock (hours)")
    a2.set_title("Utilisation per hour", fontsize=11.5, color=INK, loc="left", pad=13)
    a2.text(0, 1.015, "same x = same GPU time: is the speed worth what it cost?",
            transform=a2.transAxes, fontsize=8.5, color=INK3, va="bottom")

    for ax in (a1, a2):
        ax.set_ylabel("space utilisation")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        strip(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
    a2.text(t_common, a2.get_ylim()[0] + 0.02 * (a2.get_ylim()[1] - a2.get_ylim()[0]),
            "compared here ", fontsize=8, color=INK3, va="bottom", ha="right")

    y = np.arange(len(rows))[::-1]
    a3.barh(y, [r["sps"] for r in rows], height=0.6,
            color=[colors[r["label"]] for r in rows], zorder=3)
    for yy, r in zip(y, rows):
        a3.text(r["sps"] * 1.03, yy, f"{r['sps']:,.0f}   {r['speedup']:.2f}x",
                va="center", fontsize=8.5, color=INK2)
    a3.set_yticks(y, [r["label"] for r in rows], fontsize=8.5)
    a3.set_xlim(0, max(r["sps"] for r in rows) * 1.62)
    a3.set_xlabel("steps / second")
    a3.set_title("Raw throughput", fontsize=11.5, color=INK, loc="left", pad=13)
    a3.text(0, 1.015, "speed alone -- says nothing about quality",
            transform=a3.transAxes, fontsize=8.5, color=INK3, va="bottom")
    a3.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    strip(a3, keep=("bottom",))
    a3.tick_params(axis="y", length=0)

    fig.text(0.006, 0.985, "Throughput knobs: what each one costs and buys",
             ha="left", va="top", fontsize=13.5, color=INK)
    fig.text(0.006, 0.925,
             f"{dataset}, {steps/1e6:.1f}M steps per arm, seed {seed}, run sequentially "
             f"on one GPU. Each arm adds one change to the one above it; curves smoothed.",
             ha="left", va="top", fontsize=9, color=INK3)
    # shared legend row -- identity is never colour-alone, and it keeps the
    # line labels out of the panels, where 4 of them collided
    for i, (label, _, _) in enumerate(ARMS):
        if label not in data:
            continue
        x0 = 0.006 + i * 0.135
        fig.add_artist(plt.Line2D([x0, x0 + 0.022], [0.845, 0.845], lw=2.6,
                                  color=colors[label], transform=fig.transFigure,
                                  solid_capstyle="round"))
        fig.text(x0 + 0.028, 0.845, label, ha="left", va="center",
                 fontsize=9, color=colors[label], fontweight="bold")

    fig.savefig(f"{OUT}/ab_throughput.png", dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--dataset", default="CUT-2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="re-run arms already on disk")
    p.add_argument("--render-only", action="store_true")
    a = p.parse_args()

    if not a.render_only:
        print(f"running {len(ARMS)} arms x {a.steps:,} steps on {a.dataset}, sequentially")
        for label, run, extra in ARMS:
            print(f"[{label}]")
            train(run, extra, a.steps, a.dataset, a.seed, a.force)

    data = {}
    for label, run, _ in ARMS:
        d = load(run)
        if d is None:
            print(f"! no metrics for {run} -- skipped")
        else:
            data[label] = d
    if not data:
        sys.exit("nothing to render")

    # report what is actually on disk, not the CLI default -- --render-only
    # would otherwise label runs with whatever --steps happened to default to
    a.steps = int(max(st[-1] for st, _, _ in data.values()))
    rows, t_common = summarise(data, a.steps)
    os.makedirs(OUT, exist_ok=True)
    render(data, rows, t_common, a.steps, a.dataset, a.seed)
    with open(f"{OUT}/ab_throughput.json", "w") as f:
        json.dump(dict(steps=a.steps, dataset=a.dataset, seed=a.seed, rows=rows), f, indent=1)

    hdr = (f"{'arm':<18}{'batch':>7}{'grad steps':>11}{'steps/s':>9}{'speed':>7}"
           f"{'util@steps':>12}{'Δ':>7}{'util@hour':>11}{'Δ':>7}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<18}{r['batch']:7,}{r['grad_steps']:11,}"
              f"{r['sps']:9,.0f}{r['speedup']:6.2f}x"
              f"{r['util_at_steps']*100:11.2f}%{r['d_step']:+7.2f}"
              f"{r['util_at_hours']*100:10.2f}%{r['d_hour']:+7.2f}")
    print(f"\nutil@steps = at {a.steps/1e6:.1f}M steps (sample efficiency)")
    print(f"util@hour  = at {t_common:.2f} h, when the slowest arm finished (what you get)")
    print("single seed -- treat differences under ~1 pp as noise")
    print(f"\nwrote {OUT}/ab_throughput.png and {OUT}/ab_throughput.json")


if __name__ == "__main__":
    main()
