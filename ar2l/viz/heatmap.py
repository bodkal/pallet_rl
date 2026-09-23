"""What the networks see at one decision: the board, and both distributions.

Five panels per step - the height map, the landing height of the item at every
loading position, which of those positions the EMS + stability filter keeps,
the packing policy's probability over them, and the conveyor with the
attacker's probability of promoting each observable item.

    python3 -m ar2l.viz.heatmap --policy run:pct_nb10 --gif
    python3 -m ar2l.viz.heatmap --policy run:ex10_a1 --attacker run:ex10_a1 --step 6
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..orders import (add_cm_args, add_order_args, load_instances, orders_bin,
                      randomize_order)
from . import agents as A


def panels(fig, rec, S, plabel, alabel):
    """`S` is an int for a cube or an (Lx, Ly, Lz) triple."""
    Lx, Ly, Lz = A.extent(S)
    fig.clf()
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.9], wspace=0.35)
    hm = rec["hmap"]
    sx, sy, sz = rec["item"]

    zmap = np.full((Lx, Ly), np.nan)
    fmap = np.zeros((Lx, Ly))
    pmap = np.full((Lx, Ly), np.nan)
    # two orientations can land on the same cell; keep the likelier one
    for (x, y, z, _a, _b, _c), p in zip(rec["cands"], rec["probs"]):
        zmap[x, y] = z; fmap[x, y] = 1.0
        pmap[x, y] = p if np.isnan(pmap[x, y]) else max(pmap[x, y], p)

    axes = []
    for k, (title, data, cmap) in enumerate([
            ("height map", hm.astype(float), "viridis"),
            ("landing height", zmap, "magma"),
            ("feasible", fmap, "Greys"),
            (r"$\pi_{pack}$", pmap, "inferno")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(data.T, origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)
        axes.append(ax)
    bx, by, bz, bsx, bsy, _ = rec["cands"][rec["choice"]]
    for ax in axes[2:]:                       # ring the chosen placement
        ax.add_patch(plt.Rectangle((bx - .5, by - .5), bsx, bsy,
                                   fill=False, ec="#e8453c", lw=1.8))

    ax = fig.add_subplot(gs[0, 4])
    win = np.array(rec["window"])
    vol = win.prod(1) / float(Lx * Ly * Lz)
    pr = rec["perm_probs"] or ([1.0] + [0.0] * (len(win) - 1))
    ax.barh(np.arange(len(win)), pr[: len(win)], color="#5b8ff9")
    ax.set_yticks(np.arange(len(win)))
    ax.set_yticklabels([f"{a}x{b}x{c}" for a, b, c in win], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(r"$\pi_{perm}$ over $B_t$", fontsize=9)
    if rec["perm_idx"] is not None:
        ax.get_yticklabels()[rec["perm_idx"]].set_color("#e8453c")

    fig.suptitle(f"{plabel} vs {alabel}   step {rec['t']}   "
                 f"item {sx}x{sy}x{sz} as {bsx}x{bsy}   packed at ({bx},{by},{bz})   "
                 f"{rec['util']*100:.1f}%  ({len(rec['cands'])} candidates, "
                 f"mean item volume {vol.mean()*1000:.1f})", fontsize=10)


def bin_for(a):
    if a.bin:
        return tuple(a.bin)
    return orders_bin(a.pallet_cm, a.cell_cm) if a.data.lower().endswith('.csv') else 10


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='heur:dbl',
                    help='run:<name>, heur:<dbl|bmf|lsah|onlinebph|hmm|macs>, random')
    ap.add_argument('--attacker', default=None, help='run:<name> or mix:<name>')
    ap.add_argument('--nb', type=int, default=None)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--step', type=int, default=None, help='render only this step')
    ap.add_argument('--data', default='data/discrete_test.npy')
    ap.add_argument('--root', default='.')
    ap.add_argument('--bin', type=int, nargs=3, default=None,
                    help='bin in cells; default 10^3 for a .npy (the discrete '
                         'test set), --pallet_cm for an orders .csv')
    add_cm_args(ap)
    add_order_args(ap)
    ap.add_argument('--gif', action='store_true')
    ap.add_argument('--fps', type=float, default=1.4)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)

    policy, plabel, pnb = A.load_policy(a.policy, a.device)
    attacker, alabel, anb = A.load_attacker(a.attacker, a.device)
    nb = a.nb or anb or pnb or 1
    n_pick = A.spec_n_pick(a.attacker) or A.spec_n_pick(a.policy)
    ep = A.play(randomize_order(load_instances(os.path.join(a.root, a.data), a.cell_cm, bin_for(a),
                                         box_scale=a.box_scale,
                                         box_round=a.box_round),
                          a.order_random, a.order_seed)[a.seq], policy, attacker, nb=nb, n_pick=n_pick, S=bin_for(a))

    tag = f"{plabel}_{alabel}_s{a.seq}".replace(":", "-").replace("/", "-")
    out = os.path.join(a.root, 'results', 'heatmaps', tag)
    os.makedirs(out, exist_ok=True)
    steps = [a.step] if a.step is not None else range(len(ep["trace"]))
    fig = plt.figure(figsize=(15, 3.4))
    files = []
    for t in steps:
        panels(fig, ep["trace"][t], ep["S"], plabel, alabel)
        f = os.path.join(out, f"step_{t:02d}.png")
        fig.savefig(f, dpi=105, bbox_inches="tight")
        files.append(f)
    print(f"{len(files)} panels -> {out}")
    if a.gif and len(files) > 1:
        from PIL import Image
        ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
        g = os.path.join(a.root, 'results', 'heatmaps', tag + '.gif')
        ims[0].save(g, save_all=True, append_images=ims[1:],
                    duration=int(1000 / a.fps), loop=0)
        print("gif ->", g)


if __name__ == "__main__":
    main()
