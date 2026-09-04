"""Regenerate the figures embedded in README.md.

    python3 -m scripts.make_readme_figures      # -> docs/*.png

Reads what is already on disk: runs/<run>/eval.json for the benchmark bars and
the BPP-k curve, runs/<run>/metrics.jsonl for the training curves. Nothing here
trains or evaluates anything.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "docs"
# the 10^3 reproduction runs the README documents, archived under old_run/
RUNS = {"RS": "runs/old_run/bpp1_rs", "CUT-1": "runs/old_run/bpp1_cut1",
        "CUT-2": "runs/old_run/bpp1_cut2"}
PAPER_BPP1 = {"RS": 0.505, "CUT-1": 0.734, "CUT-2": 0.669}   # paper Table 3

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = {"RS": BLUE, "CUT-1": ORANGE, "CUT-2": AQUA}
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
# baselines are one neutral family; the BPP-k family is one blue ramp light->dark
BAR_COLOR = {
    "random feasible":     "#c9c8c2",
    "deepest-bottom-left": "#a8a7a1",
    "boundary rule":       "#87868027",
    "BPP-1 (ours)":        "#7fb3ea",
    "BPP-3 (MCTS)":        "#4a90e2",
    "BPP-5 (MCTS)":        "#1f5fae",
}
BAR_COLOR["boundary rule"] = "#878680"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
})


def strip(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def load_eval(run):
    with open(os.path.join(run, "eval.json")) as f:
        return json.load(f)


def load_metrics(run, keys):
    xs, ys = [], {k: [] for k in keys}
    with open(os.path.join(run, "metrics.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not all(k in r for k in keys):
                continue
            xs.append(r["step"])
            for k in keys:
                ys[k].append(r[k])
    return np.asarray(xs, float), {k: np.asarray(v, float) for k, v in ys.items()}


def smooth(y, w):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


# ---------------------------------------------------------------- figure 1
def fig_benchmarks():
    order = ["random feasible", "deepest-bottom-left", "boundary rule",
             "BPP-1 (ours)", "BPP-3 (MCTS)", "BPP-5 (MCTS)"]
    halo = dict(boxstyle="square,pad=0.14", fc=SURFACE, ec="none")
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.3), sharex=True)
    for ax, (ds, run) in zip(axes, RUNS.items()):
        res = {r["policy"]: r for r in load_eval(run)["results"]}
        vals = [res[p]["space_util"] for p in order]
        ypos = np.arange(len(order))[::-1]
        ax.barh(ypos, vals, height=0.62,
                color=[BAR_COLOR[p] for p in order], zorder=3)
        for y, v, p in zip(ypos, vals, order):
            ax.text(v + 0.015, y, f"{v*100:.1f}%", va="center", ha="left",
                    fontsize=9, zorder=6, bbox=halo,
                    color=INK if p.startswith("BPP") else INK2)
        pv = PAPER_BPP1[ds]
        ax.axvline(pv, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=5)
        ax.text(pv - 0.02, len(order) - 0.42, "paper BPP-1 ", fontsize=8.5,
                color=INK2, va="bottom", ha="right", zorder=6)
        ax.set_yticks(ypos, order if ds == "RS" else [""] * len(order), fontsize=9)
        ax.set_xlim(0, 1.1)
        ax.set_ylim(-0.6, 5.95)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
        ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_title(ds, fontsize=11, color=INK, pad=6, loc="left")
        strip(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Space utilisation on held-out sequences, 10x10x10 bin",
                 x=0.008, y=0.985, ha="left", va="top", fontsize=12.5, color=INK)
    fig.text(0.008, 0.905,
             "500 episodes for BPP-1, 100-200 for the search and heuristic rows; "
             "dashed line is the paper's reported BPP-1 result",
             ha="left", va="top", fontsize=9, color=INK3)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(f"{OUT}/benchmarks.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_training():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.9))
    for ds, run in RUNS.items():
        x, y = load_metrics(run, ["space_util", "mask_acc"])
        w = 400
        xs = x[w - 1:] / 1e6
        for ax, key in ((a1, "space_util"), (a2, "mask_acc")):
            ax.plot(xs, smooth(y[key], w), lw=2, color=SERIES[ds], zorder=3,
                    solid_capstyle="round", label=ds)
    for ax, ttl, sub in (
            (a1, "Space utilisation", "fraction of the bin filled, training rollouts"),
            (a2, "Mask-predictor accuracy", "agreement with the ground-truth feasibility mask")):
        ax.set_xlabel("environment steps (millions)")
        ax.set_xlim(0, 102)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        strip(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
        ax.set_title(ttl, fontsize=11.5, color=INK, loc="left", pad=14)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9,
                color=INK3, va="bottom")
    a1.set_ylim(0, 0.82)
    a1.set_yticks([0, .2, .4, .6, .8], ["0", "20%", "40%", "60%", "80%"])
    a1.annotate("run extended 30M -> 100M:\nthe LR schedule restarts",
                xy=(30.5, 0.475), xytext=(41, 0.24), fontsize=8.5, color=INK3,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8,
                                connectionstyle="angle3,angleA=0,angleB=70"))
    a2.set_ylim(0.9, 1.004)
    a2.set_yticks([.9, .95, 1], ["90%", "95%", "100%"])
    a2.text(52, 0.978, "all three streams sit at 99.6-99.8%\nfrom ~5M steps on",
            fontsize=8.5, color=INK3, ha="left", va="top")
    handles, labels = a1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 1.0),
                     ncol=3, frameon=False, handlelength=1.4, fontsize=9.5,
                     columnspacing=1.4, handletextpad=0.5)
    for t, ds in zip(leg.get_texts(), labels):
        t.set_color(SERIES[ds])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/training.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_lookahead():
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ks = [1, 3, 5]
    for ds, run in RUNS.items():
        res = {r["policy"]: r for r in load_eval(run)["results"]}
        ys = [res["BPP-1 (ours)"]["space_util"],
              res["BPP-3 (MCTS)"]["space_util"],
              res["BPP-5 (MCTS)"]["space_util"]]
        ax.plot(ks, ys, lw=2, marker="o", ms=7, color=SERIES[ds], zorder=3,
                solid_capstyle="round", markeredgecolor=SURFACE, markeredgewidth=1.6)
        ax.text(5.12, ys[-1], f" {ds}  {ys[-1]*100:.1f}%", color=SERIES[ds],
                fontsize=9.5, va="center", ha="left", fontweight="bold")
        ax.text(0.88, ys[0], f"{ys[0]*100:.1f}% ", color=SERIES[ds], fontsize=9,
                va="center", ha="right")
    ax.set_xticks(ks, ["k = 1\n(no search)", "k = 3", "k = 5"])
    ax.set_xlim(0.5, 7.4)
    ax.set_ylim(0.45, 0.92)
    ax.set_yticks([.5, .6, .7, .8, .9], ["50%", "60%", "70%", "80%", "90%"])
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    strip(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.set_title("Lookahead pays, with no retraining", fontsize=12, color=INK,
                 loc="left", pad=16)
    ax.text(0, 1.02, "same network; BPP-k only searches permutations of the k arriving boxes",
            transform=ax.transAxes, fontsize=9, color=INK3, va="bottom")
    fig.tight_layout()
    fig.savefig(f"{OUT}/lookahead.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_packing(run="runs/old_run/bpp1_cut2", ds="CUT-2", ep=1,
                pair=("boundary rule", "BPP-1 (ours)")):
    """The same box sequence, packed by the heuristic and by the policy."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    with open(os.path.join(run, "traces.json")) as f:
        traces = json.load(f)
    key = {"boundary rule": "boundary rule", "BPP-1 (ours)": "BPP-1"}
    ramp = LinearSegmentedColormap.from_list(
        "seq", ["#dce9f8", "#9cc2ec", "#4a90e2", "#1a4f95"])
    norm = Normalize(8, 100)          # box volume, cells

    fig = plt.figure(figsize=(9.4, 4.1))
    for i, pol in enumerate(pair):
        ep_data = traces[f"{ds}|{key[pol]}"][ep]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_facecolor(SURFACE)
        for x, y, z, dx, dy, dz in ep_data["placed"]:
            ax.bar3d(x, y, z, dx, dy, dz, shade=False,
                     color=ramp(norm(dx * dy * dz)),
                     edgecolor="#ffffff", linewidth=0.7)
        ax.set(xlim=(0, 10), ylim=(0, 10), zlim=(0, 10))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-58)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticks([0, 5, 10])
            axis.set_pane_color((1, 1, 1, 0))
            axis._axinfo["grid"]["color"] = GRID
        ax.tick_params(labelsize=7.5, colors=INK3, pad=-2)
        ax.set_title(f"{pol}\n{ep_data['utilization']*100:.1f}% filled  ·  "
                     f"{ep_data['n_items']} boxes",
                     fontsize=10.5, color=INK, pad=-2)
    fig.suptitle("One CUT-2 sequence, packed two ways",
                 x=0.02, y=0.985, ha="left", va="top", fontsize=12.5, color=INK)
    fig.text(0.02, 0.915, "identical boxes, identical arrival order; "
             "box shade encodes its volume",
             ha="left", va="top", fontsize=9, color=INK3)
    fig.tight_layout(rect=(0, -0.02, 1, 0.93))
    fig.savefig(f"{OUT}/packing.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_benchmarks()
    fig_training()
    fig_lookahead()
    fig_packing()
    print("wrote", ", ".join(f"{OUT}/{n}.png" for n in
                             ("benchmarks", "training", "lookahead", "packing")))
