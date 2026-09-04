"""3D packing replay: render a packed bin, step by step, as PNG / GIF / MP4.

    python -m src.viz.replay3d --run bpp1_cut2 --key "CUT-2|BPP-1" --episode 0
    python -m src.viz.replay3d --run bpp1_cut2 --all --gif
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))


def _box_polys(x, y, z, l, w, h):
    v = np.array([[x, y, z], [x + l, y, z], [x + l, y + w, z], [x, y + w, z],
                  [x, y, z + h], [x + l, y, z + h], [x + l, y + w, z + h],
                  [x, y + w, z + h]], dtype=float)
    return [[v[i] for i in f] for f in FACES]


def method_key(name):
    """Sort BPP-1, BPP-2, ... first (by k), then heuristics alphabetically."""
    if name.startswith("BPP-"):
        try:
            return (0, int(name.split("-")[1].split(" ")[0]))
        except ValueError:
            return (0, 99)
    return (1, name)


def safe(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")


def _color(i, n):
    return hsv_to_rgb([(0.62 + 0.61 * i / max(1, n)) % 1.0, 0.55, 0.95])


def draw_bin(ax, placed, L, W, H, upto=None, title=None, highlight_last=True,
             compact=False):
    ax.clear()
    n = len(placed)
    upto = n if upto is None else upto
    for i, (x, y, z, l, w, h) in enumerate(placed[:upto]):
        last = highlight_last and i == upto - 1
        pc = Poly3DCollection(_box_polys(x, y, z, l, w, h),
                              facecolors=_color(i, n),
                              edgecolors="#d92b2b" if last else "#2b2f38",
                              linewidths=2.0 if last else 0.6,
                              alpha=1.0 if last else 0.92)
        ax.add_collection3d(pc)
    # bin wireframe
    for s, e in (((0,0,0),(L,0,0)),((0,0,0),(0,W,0)),((L,0,0),(L,W,0)),((0,W,0),(L,W,0)),
                 ((0,0,0),(0,0,H)),((L,0,0),(L,0,H)),((0,W,0),(0,W,H)),((L,W,0),(L,W,H)),
                 ((0,0,H),(L,0,H)),((0,0,H),(0,W,H)),((L,0,H),(L,W,H)),((0,W,H),(L,W,H))):
        ax.plot(*zip(s, e), color="#7c8496", lw=0.8, alpha=0.7)
    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
    ax.set_box_aspect((L, W, H))
    if compact:
        # comparison sheets repeat the same axes in every cell -- drop the text
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(length=0)
    else:
        ax.set_xlabel("X (length)", fontsize=8); ax.set_ylabel("Y (width)", fontsize=8)
        ax.set_zlabel("Z (height)", fontsize=8)
        ax.tick_params(labelsize=7)
    ax.view_init(elev=24, azim=-58)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.04)
    if title:
        ax.set_title(title, fontsize=10, pad=2)


def render_episode(rec, cfg, out_prefix, gif=True, mp4=False, fps=3, dpi=110):
    placed = [tuple(p) for p in rec["placed"]]
    L, W, H = cfg["L"], cfg["W"], cfg["H"]
    vol = L * W * H
    frames = []
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    cum = 0
    for i in range(1, len(placed) + 1):
        x, y, z, l, w, h = placed[i - 1]
        cum += l * w * h
        draw_bin(ax, placed, L, W, H, upto=i,
                 title=f"item {i}/{len(placed)}  ({l}x{w}x{h}) @ ({x},{y},{z})"
                       f"\nspace utilisation {cum/vol*100:.1f}%")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
    # final still
    draw_bin(ax, placed, L, W, H, title=f"final: {len(placed)} items, "
                                        f"{rec['utilization']*100:.1f}% "
                                        f"({rec['reason']})",
             highlight_last=False)
    fig.tight_layout()
    fig.savefig(out_prefix + "_final.png", dpi=dpi)
    fig.canvas.draw()
    final = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    frames += [final] * 4
    plt.close(fig)

    if gif or mp4:
        import imageio.v2 as iio
        if gif:
            iio.mimsave(out_prefix + ".gif", frames, duration=1.0 / fps, loop=0)
        if mp4:
            try:
                iio.mimsave(out_prefix + ".mp4", frames, fps=fps)
            except Exception as e:
                print("  (mp4 skipped:", e, ")")
    return out_prefix + ".gif" if gif else out_prefix + "_final.png"


def grid_figure(recs, cfg, path, titles=None, ncol=3):
    """A contact sheet of several final packings (paper Figure 4 style)."""
    n = len(recs)
    ncol = min(ncol, n)
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(3.5 * ncol, 3.6 * nrow))
    for i, rec in enumerate(recs):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        t = titles[i] if titles else f"{rec['utilization']*100:.1f}%  |  {rec['n_items']} items"
        draw_bin(ax, [tuple(p) for p in rec["placed"]],
                 cfg["L"], cfg["W"], cfg["H"], title=t, highlight_last=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def comparison_figure(by_method, cfg, path, dataset, run=""):
    """All methods on one sheet: rows = method, columns = the SAME test sequence.

    Every policy is evaluated on the same held-out sequences, so column j is the
    identical item stream for every row -- the differences you see are entirely
    down to the packing policy.
    """
    methods = sorted(by_method, key=method_key)
    ncol = max(len(v) for v in by_method.values())
    nrow = len(methods)
    L, W, H = cfg["L"], cfg["W"], cfg["H"]

    fig = plt.figure(figsize=(3.3 * ncol + 0.6, 3.15 * nrow + 1.0))
    for r, m in enumerate(methods):
        recs = by_method[m]
        best = max(rec["utilization"] for rec in recs)
        for c in range(ncol):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            if c >= len(recs):
                ax.axis("off")
                ax.text2D(.5, .5, "not recorded", ha="center", va="center",
                          transform=ax.transAxes, fontsize=9, color="#9aa1af")
                continue
            rec = recs[c]
            mark = " *" if rec["utilization"] == best and len(recs) > 1 else ""
            draw_bin(ax, [tuple(p) for p in rec["placed"]], L, W, H,
                     title=f"{m}{mark}\nseq #{c}: {rec['utilization']*100:.1f}%  "
                           f"| {rec['n_items']} items",
                     highlight_last=False, compact=True)
    fig.suptitle(f"{dataset} benchmark — all methods on identical sequences"
                 + (f"   ({run})" if run else ""),
                 fontsize=14, fontweight="semibold", y=0.997)
    fig.subplots_adjust(top=1 - 0.55 / (3.15 * nrow + 1.0), bottom=0.015,
                        left=0.015, right=0.985, hspace=0.30, wspace=0.05)
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def summary_figure(by_method, cfg, path, dataset):
    """One row per method: its best recorded packing, side by side."""
    methods = sorted(by_method, key=method_key)
    n = len(methods)
    fig = plt.figure(figsize=(3.5 * n, 3.9))
    for i, m in enumerate(methods):
        rec = max(by_method[m], key=lambda r: r["utilization"])
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        draw_bin(ax, [tuple(p) for p in rec["placed"]],
                 cfg["L"], cfg["W"], cfg["H"],
                 title=f"{m}\n{rec['utilization']*100:.1f}%  |  {rec['n_items']} items",
                 highlight_last=False, compact=True)
    fig.suptitle(f"{dataset} — best recorded packing per method",
                 fontsize=13, fontweight="semibold")
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.01, right=0.99, wspace=0.05)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--key", default=None, help='e.g. "CUT-2|BPP-1"')
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--all", action="store_true", help="render every recorded key")
    p.add_argument("--gif", action="store_true", default=True)
    p.add_argument("--no-gif", dest="gif", action="store_false")
    p.add_argument("--mp4", action="store_true")
    p.add_argument("--fps", type=int, default=3)
    a = p.parse_args(argv)

    d = os.path.join("runs", a.run)
    traces = json.load(open(os.path.join(d, "traces.json")))
    cfg = json.load(open(os.path.join(d, "config.json")))
    root = os.path.join(d, "replays")
    os.makedirs(root, exist_ok=True)

    # group "<dataset>|<method>" -> {dataset: {method: [episodes]}}
    grouped = {}
    for k, recs in traces.items():
        if not recs:
            continue
        ds, _, method = k.partition("|")
        grouped.setdefault(ds, {})[method or k] = recs

    keys = None if a.all else [(a.key or list(traces)[0])]
    for ds, by_method in grouped.items():
        # one directory per benchmark, named after it
        outdir = os.path.join(root, ds)
        os.makedirs(outdir, exist_ok=True)

        # the combined sheet: every method, identical sequences
        print(f"[{ds}] comparison grid ({len(by_method)} methods)")
        comparison_figure(by_method, cfg,
                          os.path.join(outdir, "comparison_all_methods.png"),
                          ds, run=a.run)
        summary_figure(by_method, cfg,
                       os.path.join(outdir, "summary_best_per_method.png"), ds)

        for method, recs in sorted(by_method.items(), key=lambda kv: method_key(kv[0])):
            if keys is not None and f"{ds}|{method}" not in keys:
                continue
            idxs = range(len(recs)) if a.all else [a.episode]
            for i in idxs:
                if i >= len(recs):
                    continue
                pre = os.path.join(outdir, f"{safe(method)}_ep{i}")
                print("  ", os.path.relpath(pre, d))
                render_episode(recs[i], cfg, pre, gif=a.gif, mp4=a.mp4, fps=a.fps)
    print("wrote", root)


if __name__ == "__main__":
    main()
