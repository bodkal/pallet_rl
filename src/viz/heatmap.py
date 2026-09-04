"""Live 2D view: height map | ground-truth mask | predicted mask | action probs.

This is the debugging loop for what the agent and the mask predictor are actually
doing at each step.

    python -m src.viz.heatmap --run bpp1_cut2 --dataset CUT-2 --gif
    python -m src.viz.heatmap --run bpp1_cut2 --live        # step through in a window
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import torch

from ..env import PackingEnv, build_obs_tensor
from ..evaluate import load_run


def _step_records(cfg, net, device, dataset, seed, max_steps=200):
    """Run one greedy episode, recording everything the panels need."""
    env = PackingEnv(cfg, dataset=dataset, seed=seed)
    obs = env.reset()
    recs = []
    for _ in range(max_steps):
        if obs["mask"].sum() == 0:
            break
        x = build_obs_tensor(obs["hmap"][None], obs["item"][None], cfg)
        with torch.no_grad():
            logits, value, mlog = net(torch.as_tensor(x, device=device))
            pred = (torch.sigmoid(mlog) > 0.5).float()
            probs = torch.softmax(net.projected_logits(logits, pred), -1)
        a = int(probs.argmax(-1).item())
        recs.append(dict(hmap=obs["hmap"].copy(),
                         item=tuple(int(v) for v in obs["item"]),
                         true=obs["mask"].copy(),
                         pred=pred[0].cpu().numpy(),
                         probs=probs[0].cpu().numpy(),
                         action=a, value=float(value.item()),
                         util=env.bin.utilization,
                         legal=bool(env.bin.is_feasible(a, tuple(int(v) for v in obs["item"]),
                                                        cfg.orientations))))
        obs, r, d, info = env.step(a)
        if d:
            break
    return recs, env


def _grid(v, cfg, o):
    """flat action vector -> (W, L) image for pose o (a = x + L*y)."""
    n = cfg.L * cfg.W
    return v[o * n:(o + 1) * n].reshape(cfg.W, cfg.L)


def draw_step(fig, rec, cfg):
    import matplotlib.pyplot as plt
    fig.clf()
    no = cfg.orientations
    ncol = 1 + 3 * no
    axes = fig.subplots(1, ncol, squeeze=False)[0]
    l, w, h = rec["item"]

    ax = axes[0]
    im = ax.imshow(rec["hmap"].T, origin="lower", cmap="viridis",
                   vmin=0, vmax=cfg.H)
    for xx in range(cfg.L):
        for yy in range(cfg.W):
            v = rec["hmap"][xx, yy]
            ax.text(xx, yy, str(int(v)), ha="center", va="center", fontsize=6,
                    color="w" if v < cfg.H * 0.6 else "k")
    ax.set_title(f"height map\nitem {l}x{w}x{h}  V={rec['value']:.2f}", fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    labels = ["true mask M", "predicted mask", "action probs P(a|s)"]
    cmaps = ["Greens", "Blues", "magma"]
    for o in range(no):
        for j, (key, lab, cm) in enumerate(zip(["true", "pred", "probs"], labels, cmaps)):
            ax = axes[1 + o * 3 + j]
            g = _grid(rec[key], cfg, o)
            if key == "probs":
                ax.imshow(g, origin="lower", cmap=cm)
            else:
                ax.imshow(g, origin="lower", cmap=cm, vmin=0, vmax=1)
            if key == "pred":   # mark disagreements with the ground truth
                dis = np.argwhere(_grid(rec["pred"], cfg, o) != _grid(rec["true"], cfg, o))
                if len(dis):
                    ax.scatter(dis[:, 1], dis[:, 0], s=14, marker="x", c="red", lw=1.2)
            n = cfg.L * cfg.W
            if rec["action"] // n == o:
                ay, ax_ = divmod(rec["action"] % n, cfg.L)
                ax.scatter([ax_], [ay], s=90, facecolors="none",
                           edgecolors="#ff2d55" if not rec["legal"] else "#ffffff", lw=2)
            extra = f" (pose {o})" if no > 1 else ""
            ax.set_title(lab + extra, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    acc = float((rec["pred"] == rec["true"]).mean())
    fig.suptitle(f"space utilisation {rec['util']*100:.1f}%   "
                 f"mask acc {acc*100:.1f}%   "
                 f"chosen LP {'LEGAL' if rec['legal'] else 'ILLEGAL'}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--dataset", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--live", action="store_true", help="interactive window")
    p.add_argument("--gif", action="store_true", default=True)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    if not a.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, net, dev, d, step = load_run(a.run, a.ckpt, a.device)
    recs, env = _step_records(cfg, net, dev, a.dataset or cfg.dataset, a.seed)
    print(f"{len(recs)} steps, final util {env.bin.utilization*100:.1f}%")

    fig = plt.figure(figsize=(3.1 * (1 + 3 * cfg.orientations), 3.4))
    if a.live:
        idx = [0]
        def on_key(e):
            if e.key in ("right", " "): idx[0] = min(idx[0] + 1, len(recs) - 1)
            elif e.key == "left":       idx[0] = max(idx[0] - 1, 0)
            elif e.key == "q":          plt.close(fig); return
            draw_step(fig, recs[idx[0]], cfg); fig.canvas.draw_idle()
        fig.canvas.mpl_connect("key_press_event", on_key)
        draw_step(fig, recs[0], cfg)
        print("left/right = step,  q = quit")
        plt.show()
        return

    outdir = os.path.join(d, "replays"); os.makedirs(outdir, exist_ok=True)
    frames = []
    for r in recs:
        draw_step(fig, r, cfg)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    import imageio.v2 as iio
    path = os.path.join(outdir, f"heightmap_{a.dataset or cfg.dataset}.gif")
    iio.mimsave(path, frames, duration=1.0 / a.fps, loop=0)
    import imageio
    imageio.imwrite(os.path.join(outdir, f"heightmap_{a.dataset or cfg.dataset}_last.png"),
                    frames[-1])
    print("wrote", path)


if __name__ == "__main__":
    main()
