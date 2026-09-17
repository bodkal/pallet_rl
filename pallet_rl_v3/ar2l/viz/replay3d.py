"""3D packing replay: render a bin being built, step by step, as PNG / GIF.

    python3 -m ar2l.viz.replay3d --policy run:pct_nb10 --gif
    python3 -m ar2l.viz.replay3d --policy run:pct_nb10 --attacker run:att_pct_nb10 --gif

Writes results/replays/<tag>/step_XX.png and, with --gif, <tag>.gif.  With an
attacker the title also names the item it promoted, so you can watch the
conveyor being reordered against the policy.
"""
from __future__ import annotations

import argparse
import colorsys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import agents as A

FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))


def _polys(x, y, z, sx, sy, sz):
    v = np.array([[x, y, z], [x + sx, y, z], [x + sx, y + sy, z], [x, y + sy, z],
                  [x, y, z + sz], [x + sx, y, z + sz], [x + sx, y + sy, z + sz],
                  [x, y + sy, z + sz]], float)
    return [[v[i] for i in f] for f in FACES]


def box_color(i):
    """Golden-angle hue walk, so a box keeps its colour across every viewer."""
    return colorsys.hls_to_rgb(((i * 137.508 + 28) % 360) / 360.0, 0.54, 0.62)


def draw(ax, placed, S, upto=None, title=""):
    """`S` is an int for a cube or an (Lx, Ly, Lz) triple."""
    Lx, Ly, Lz = A.extent(S)
    ax.clear()
    upto = len(placed) if upto is None else upto
    for i, (x, y, z, sx, sy, sz) in enumerate(placed[:upto]):
        p = Poly3DCollection(_polys(x, y, z, sx, sy, sz), alpha=0.94)
        p.set_facecolor(box_color(i))
        p.set_edgecolor((0, 0, 0, 0.45))
        p.set_linewidth(0.5)
        ax.add_collection3d(p)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
    ax.set_box_aspect((Lx, Ly, Lz))
    ax.set_xticks([0, Lx]); ax.set_yticks([0, Ly]); ax.set_zticks([0, Lz])
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=22, azim=-58)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='heur:dbl',
                    help='run:<name>, heur:<dbl|bmf|lsah|onlinebph|hmm|macs>, random')
    ap.add_argument('--attacker', default=None, help='run:<name> or mix:<name>')
    ap.add_argument('--nb', type=int, default=None)
    ap.add_argument('--seq', type=int, default=0, help='index into the test set')
    ap.add_argument('--data', default='data/discrete_test.npy')
    ap.add_argument('--root', default='.')
    ap.add_argument('--gif', action='store_true')
    ap.add_argument('--fps', type=float, default=2.4)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)

    policy, plabel, pnb = A.load_policy(a.policy, a.device)
    attacker, alabel, anb = A.load_attacker(a.attacker, a.device)
    nb = a.nb or anb or pnb or 1
    seq = np.load(os.path.join(a.root, a.data))[a.seq]
    ep = A.play(seq, policy, attacker, nb=nb)

    tag = a.tag or f"{plabel}_{alabel}_s{a.seq}".replace(":", "-").replace("/", "-")
    out = os.path.join(a.root, 'results', 'replays', tag)
    os.makedirs(out, exist_ok=True)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    for i in range(len(ep["placed"]) + 1):
        u = ep["trace"][i - 1]["util"] if i else 0.0
        draw(ax, ep["placed"], ep["S"], i,
             f"{plabel} vs {alabel}   {i} items   {u*100:.1f}%")
        f = os.path.join(out, f"step_{i:02d}.png")
        fig.savefig(f, dpi=110, bbox_inches="tight")
        frames.append(f)
    print(f"{len(frames)} frames -> {out}   final {ep['util']*100:.1f}% "
          f"with {ep['items']} items")
    if a.gif:
        try:
            from PIL import Image
            ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
            g = os.path.join(a.root, 'results', 'replays', tag + '.gif')
            ims[0].save(g, save_all=True, append_images=ims[1:],
                        duration=int(1000 / a.fps), loop=0)
            print("gif ->", g)
        except ImportError:
            print("pillow not installed; skipping the gif")


if __name__ == "__main__":
    main()
