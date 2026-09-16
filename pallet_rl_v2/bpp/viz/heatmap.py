"""Live 2D view: height map | landing height | feasibility | policy probabilities.

This is the debugging loop for what the network is actually doing at each step -
where it may put the box, where it wants to, and which item of the buffer it
reaches for.

    python3 -m bpp.viz.heatmap --ckpt runs/cut1_v2/last.pt --data cut1 --gif
    python3 -m bpp.viz.heatmap --ckpt runs/rs_v2/last.pt --data rs --seq 2 --step 7
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..env import Cfg, combo_dims, compute_masks, features, new_state, step_v
from . import agents as A


def step_records(cfg, ev, seq, ln, max_steps=200):
    """One greedy p_theta episode, recording everything the panels need."""
    seqs, lens = seq[None], np.array([ln], np.int32)
    hm, buf, nxt = new_state(seqs, lens, cfg)
    recs = []
    for _ in range(max_steps):
        dims = combo_dims(buf, cfg.k)
        mask, zmap = compute_masks(hm, dims, cfg.H)
        mf = np.ascontiguousarray(mask.reshape(1, -1))
        if not mf.any():
            break
        p = ev.probs(features(hm, buf, cfg.H), mf)[0]
        a = int(p.argmax())
        m, q = a // cfg.WL, a % cfg.WL
        recs.append(dict(hm=hm[0].copy(), buf=buf[0].copy(),
                         mask=mask[0].copy(), zmap=zmap[0].copy(),
                         p=p.reshape(cfg.M, cfg.W, cfg.L).copy(),
                         m=m, x=q // cfg.L, y=q % cfg.L, dims=dims[0].copy()))
        step_v(hm, buf, nxt, seqs, lens, dims, zmap, np.array([a], np.int64), cfg)
    return recs


def panels(fig, r, cfg, title):
    """Four panels for one decision.

    Every array is (W,L) with x the row and y the column, which is how imshow
    wants it.  Panel 1 is *spatial*, so the chosen box is outlined over the cells
    it will occupy; panels 2-4 are indexed by the FLB loading position, so there
    the choice is a single cell and is marked as one.
    """
    fig.clear()
    axs = fig.subplots(1, 4)
    m, x, y = r['m'], r['x'], r['y']
    j, o = m // (cfg.k + 1), m % (cfg.k + 1)
    w, l, h = (int(v) for v in r['dims'][m])

    def footprint(ax, c='w'):
        ax.add_patch(plt.Rectangle((y - .5, x - .5), l, w, fill=False, ec=c, lw=2))

    def flb(ax, c='c'):
        ax.add_patch(plt.Rectangle((y - .5, x - .5), 1, 1, fill=False, ec=c, lw=2))

    im = axs[0].imshow(r['hm'], vmin=0, vmax=cfg.H, cmap='viridis')
    axs[0].set_title('height map $H_t$\n(outline: where the box lands)', fontsize=9)
    footprint(axs[0])
    fig.colorbar(im, ax=axs[0], fraction=.046)

    cm = matplotlib.colormaps['magma'].with_extremes(bad='0.85')
    z = np.where(r['mask'][m], r['zmap'][m].astype(float), np.nan)
    im = axs[1].imshow(z, vmin=0, vmax=cfg.H, cmap=cm)
    axs[1].set_title(f'landing height per FLB\nchosen pose: slot {j+1}, orient {o}'
                     f' (grey = illegal)', fontsize=9)
    flb(axs[1])
    fig.colorbar(im, ax=axs[1], fraction=.046)

    # how many of the M combos may load at each (x,y) - the shape of the action space
    im = axs[2].imshow(r['mask'].sum(0), vmin=0, vmax=cfg.M, cmap='Blues')
    axs[2].set_title(f'feasible combos per FLB\n{int(r["mask"].sum())} legal actions of {cfg.A}',
                     fontsize=9)
    flb(axs[2], 'k')
    fig.colorbar(im, ax=axs[2], fraction=.046)

    im = axs[3].imshow(r['p'].max(0), cmap='inferno')
    axs[3].set_title(f'$p_\\theta$ per FLB (max over combos)\nargmax = ({x}, {y}), '
                     f'p = {r["p"][m, x, y]:.2f}', fontsize=9)
    flb(axs[3])
    fig.colorbar(im, ax=axs[3], fraction=.046)

    for ax in axs:
        ax.set_xlabel('y  (L)', fontsize=8)
        ax.set_ylabel('x  (W)', fontsize=8)
        ax.tick_params(labelsize=7)
    buf = ' | '.join('empty' if d[0] <= 0 else f'{d[0]}x{d[1]}x{d[2]}' for d in r['buf'])
    fig.suptitle(f'{title}   buffer: [{buf}]   ->  takes slot {j+1} '
                 f'({w}x{l}x{h}{", turned" if o else ""}) at ({x}, {y})', fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))


def main(argv=None):
    ap = argparse.ArgumentParser(description='what the policy sees, step by step')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='cut1', choices=['cut1', 'cut2', 'rs'])
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--step', type=int, default=None, help='render only this step')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--root', default='.')
    ap.add_argument('--gif', action='store_true')
    ap.add_argument('--fps', type=float, default=1.5)
    a = ap.parse_args(argv)

    cfg, ev = A.load_net(a.ckpt, a.device)
    te = np.load(os.path.join(a.root, 'data', f'{a.data}_test.npz'))
    seq, ln = te['seqs'][a.seq], int(te['lens'][a.seq])
    recs = step_records(cfg, ev, seq, ln)
    print(f'{a.data}[{a.seq}]: policy packed {len(recs)} boxes')

    tag = f'{a.data}_{a.seq}_{os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))}'
    d = os.path.join(a.root, 'results', 'heatmaps', tag)
    os.makedirs(d, exist_ok=True)
    want = range(len(recs)) if a.step is None else [a.step]
    fig = plt.figure(figsize=(15, 4.3), dpi=110)
    out = []
    for i in want:
        if not 0 <= i < len(recs):
            raise SystemExit(f'step {i} out of range (0..{len(recs)-1})')
        panels(fig, recs[i], cfg, f'step {i+1}/{len(recs)}')
        p = os.path.join(d, f'step_{i:03d}.png')
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    print(f'  wrote {len(out)} frames to {d}/')

    if a.gif and len(out) > 1:
        try:
            from PIL import Image
        except ImportError:
            print('  --gif needs Pillow; frames are on disk')
            return
        ims = [Image.open(p).convert('P', palette=Image.ADAPTIVE) for p in out]
        g = os.path.join(a.root, 'results', 'heatmaps', tag + '.gif')
        ims[0].save(g, save_all=True, append_images=ims[1:],
                    duration=int(1000 / a.fps), loop=0)
        print(f'  wrote {g}')


if __name__ == '__main__':
    main()
