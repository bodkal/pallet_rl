"""3D packing replay: render a packed bin, step by step, as PNG / GIF.

    python3 -m bpp.viz.replay3d --ckpt runs/cut1_v2/last.pt --data cut1 --seq 0
    python3 -m bpp.viz.replay3d --ckpt runs/cut2_v2/last.pt --data cut2 --gif --sims 100
    python3 -m bpp.viz.replay3d --agent lowest --data rs --seq 3 --gif

Writes results/replays/<tag>/step_XX.png and, with --gif, <tag>.gif.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..env import Cfg
from . import agents as A

FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))


def _box_polys(x, y, z, w, l, h):
    v = np.array([[x, y, z], [x + w, y, z], [x + w, y + l, z], [x, y + l, z],
                  [x, y, z + h], [x + w, y, z + h], [x + w, y + l, z + h],
                  [x, y + l, z + h]], dtype=float)
    return [[v[i] for i in f] for f in FACES]


def box_color(i):
    """The same golden-angle hue walk the browser uses, so a box keeps its
    colour whichever viewer you look at it in."""
    import colorsys
    h = ((i * 137.508 + 28) % 360) / 360.0
    return colorsys.hls_to_rgb(h, 0.54, 0.62)


def draw(ax, placed, cfg, upto=None, title=''):
    ax.clear()
    upto = len(placed) if upto is None else upto
    for i, (x, y, z, w, l, h) in enumerate(placed[:upto]):
        c = box_color(i)
        p = Poly3DCollection(_box_polys(x, y, z, w, l, h), alpha=0.94)
        p.set_facecolor(c)
        p.set_edgecolor((0, 0, 0, 0.45))
        p.set_linewidth(0.5)
        ax.add_collection3d(p)
    ax.set_xlim(0, cfg.W); ax.set_ylim(0, cfg.L); ax.set_zlim(0, cfg.H)
    ax.set_xlabel('x  (W)'); ax.set_ylabel('y  (L)'); ax.set_zlabel('z  (H)')
    ax.set_box_aspect((cfg.W, cfg.L, cfg.H))
    ax.view_init(elev=24, azim=-58)
    ax.set_title(title, fontsize=10)


def main(argv=None):
    ap = argparse.ArgumentParser(description='render a packed bin step by step')
    ap.add_argument('--ckpt', default=None, help='checkpoint to pack with')
    ap.add_argument('--agent', default=None,
                    help="opponent id instead of --ckpt: 'lowest', 'random', "
                         "'policy:<run>', 'mcts:<run>:<sims>'")
    ap.add_argument('--data', default='cut1', choices=['cut1', 'cut2', 'rs'])
    ap.add_argument('--seq', type=int, default=0, help='index into the test set')
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--c_puct', type=float, default=4.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default=None)
    ap.add_argument('--gif', action='store_true')
    ap.add_argument('--fps', type=float, default=2.0)
    a = ap.parse_args(argv)

    if a.agent:
        opp = a.agent
    elif a.ckpt:
        run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
        opp = f'mcts:{run}:{a.sims}'
    else:
        ap.error('give --ckpt or --agent')

    cfg = A.config_for(opp, Cfg())
    te = np.load(os.path.join(a.root, 'data', f'{a.data}_test.npz'))
    seq, ln = te['seqs'][a.seq], int(te['lens'][a.seq])
    print(f'packing {a.data}[{a.seq}] ({ln} boxes) with {opp} ...')
    rec = A.play_opponent(opp, cfg, seq, ln, device=a.device, c_puct=a.c_puct)
    placed = rec['placed']
    print(f"  {rec['n_items']} items, {rec['utilization']*100:.1f}% "
          f"in {rec['seconds']:.1f}s ({rec['reason']})")

    tag = a.out or f"{a.data}_{a.seq}_{opp.replace(':', '-')}"
    d = os.path.join(a.root, 'results', 'replays', tag)
    os.makedirs(d, exist_ok=True)
    fig = plt.figure(figsize=(5.2, 5.0), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    frames = []
    for n in range(len(placed) + 1):
        vol = sum(w * l * h for _, _, _, w, l, h in placed[:n])
        draw(ax, placed, cfg, n,
             f'{a.data.upper()} #{a.seq} · {opp}\n'
             f'{n} / {ln} boxes · {100*vol/cfg.VB:.1f}% filled')
        fig.tight_layout()
        p = os.path.join(d, f'step_{n:03d}.png')
        fig.savefig(p)
        frames.append(p)
    plt.close(fig)
    print(f'  wrote {len(frames)} frames to {d}/')

    if a.gif:
        try:
            from PIL import Image
        except ImportError:
            print('  --gif needs Pillow (pip install --user Pillow); frames are on disk')
            return
        ims = [Image.open(p).convert('P', palette=Image.ADAPTIVE) for p in frames]
        g = os.path.join(a.root, 'results', 'replays', tag + '.gif')
        ims[0].save(g, save_all=True, append_images=ims[1:],
                    duration=int(1000 / a.fps), loop=0)
        print(f'  wrote {g}')


if __name__ == '__main__':
    main()
