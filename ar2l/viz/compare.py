"""Compare the experiments of every big-experiment report, one window per file.

Each report (see `ar2l.viz.experiment`) holds one block per run.  Every file
gets a window with two grouped bar plots over the pallet number: the number of
boxes on each pallet (top) and `boxes_volume_per_pallet [%]` (bottom).  Each
experiment keeps one colour in both plots and is named in the legend by the
name it was run under, or by its number in the file when it has none.  A last window sums up which experiment won in each file.

    python3 -m ar2l.viz.compare
    python3 -m ar2l.viz.compare results/big_experiment/PL33PR_2024-06-03.csv
    python3 -m ar2l.viz.compare --summary        # only the winners summary
    python3 -m ar2l.viz.compare --list           # the experiments in each file
    python3 -m ar2l.viz.compare --exp base --exp 3   # only these experiments
    python3 -m ar2l.viz.compare --save results/big_experiment/plots
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.text import Annotation
from matplotlib.colors import to_rgb
import numpy as np

BOXES = "total_number_of_boxes"
VOLUME = "boxes_volume_per_pallet [%]"


def read_report(path):
    """-> list of experiments: {"num", "time", "boxes", "params", "name", "rows": [dict per pallet]}.

    `num` is the experiment's number in the file, counted from 1.
    """
    exps, header = [], None
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            if len(row) > 1 and row[1].strip() == "current date:":
                params = " ".join(c.strip() for c in row[7:9])
                cells = [c.strip() for c in row]
                name = (cells[cells.index("name:") + 1]
                        if "name:" in cells[9:-1] else "")
                exps.append({"time": row[2].strip(), "boxes": int(row[4]),
                             "params": params, "name": name, "rows": []})
                header = None
            elif row[0].strip() == "pallet_id":
                header = [c.strip() for c in row if c.strip()]
            elif exps and header:
                exps[-1]["rows"].append(dict(zip(header, row)))
    exps = [e for e in exps if e["rows"]]
    for j, e in enumerate(exps):
        e["num"] = j + 1
    return exps


def key(e):
    """How an experiment is matched across files: its name, else `Exp <num>`."""
    return e["name"] or f"Exp {e['num']}"


def delete_named(path, names):
    """Drop every block whose `key` is one of `names` from the report.  -> how many.

    A named block goes by its name, an unnamed one by `Exp <num>`, numbered
    as `read_report` numbers it.  Names match exactly, case included, so a
    near namesake is never caught.  The other blocks are kept byte for byte;
    a report left with no block is removed.
    """
    want = {n.strip() for n in names if n.strip()}
    with open(path, newline="") as f:
        lines = f.read().splitlines(keepends=True)
    blocks, names_, rows = [[]], [None], [False]
    header = False
    for ln in lines:
        cells = [c.strip() for c in next(csv.reader([ln]), [])]
        if len(cells) > 1 and cells[1] == "current date:":
            blocks.append([])
            names_.append(cells[cells.index("name:") + 1]
                          if "name:" in cells[9:-1] else "")
            rows.append(False)
            header = False
        elif cells and cells[0] == "pallet_id":
            header = True
        elif header and cells and cells[0]:
            rows[-1] = True
        blocks[-1].append(ln)
    keep, num, gone = [], 0, 0
    for name, has_rows in zip(names_, rows):
        if name is None or not has_rows:   # the lines before any block; an empty block
            keep.append(True)
            continue
        num += 1
        keep.append((name or f"Exp {num}") not in want)
        gone += not keep[-1]
    if not gone:
        return 0
    text = "".join(ln for b, k in zip(blocks, keep) if k for ln in b)
    if text.strip():
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    else:
        os.remove(path)
    return gone


def select(exps, picks):
    """The experiments `picks` names, each by its name or its number; all when empty."""
    if not picks:
        return exps
    want = {p.strip().lower() for p in picks}
    return [e for e in exps
            if str(e["num"]) in want or (e["name"] and e["name"].lower() in want)]


# Apple system colours (light mode), in the order the Health / Stocks apps use
COLORS = ["#007AFF", "#FF9500", "#34C759", "#AF52DE", "#FF2D55",
          "#5AC8FA", "#FFCC00", "#5856D6", "#FF3B30", "#00C7BE"]
INK, MUTED, GRID, BG = "#1D1D1F", "#86868B", "#E5E5EA", "#FFFFFF"


def color(e):
    """An experiment's colour, by its number, so it holds when others are left out."""
    return COLORS[(e["num"] - 1) % len(COLORS)]


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("SF Pro Display", "SF Pro Text", "Helvetica Neue", "Inter",
                 "Lato", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def style():
    plt.rcParams.update({
        "font.family": font(),
        "font.size": 11,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 8,
        "ytick.major.pad": 8,
        "legend.frameon": False,
        "legend.labelcolor": INK,
    })


def data_aspect(ax):
    """y data units per x data unit on screen, so corners come out round."""
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return ((y1 - y0) / bbox.height) / ((x1 - x0) / bbox.width)


def rounded_bar(ax, x, w, v, color, aspect, zorder=3):
    """A bar from 0 to `v` whose top corners are rounded."""
    if v <= 0:
        return
    rx = w * 0.22
    ry = rx * aspect
    # start below the axis so only the top corners show rounded
    ax.add_patch(FancyBboxPatch(
        (x, -ry), w, v + ry, mutation_aspect=aspect,
        boxstyle=f"round,pad=0,rounding_size={rx}",
        facecolor=color, edgecolor="none", zorder=zorder))


def tint(color, k=0.35):
    """`color` mixed with white, `k` of the way from white."""
    return tuple(1 - k + k * c for c in to_rgb(color))


def bars(ax, exps, field, fmt):
    """Grouped bars with rounded tops; call once the axes are laid out."""
    n = len(exps)
    width = 0.8 / n
    gap = width * 0.12
    vmax = max(float(r[field]) for e in exps for r in e["rows"])
    ax.set_ylim(0, vmax * 1.18)
    ids_all = [int(r["pallet_id"]) for e in exps for r in e["rows"]]
    ax.set_xlim(min(ids_all) - 0.5, max(ids_all) + 0.5)
    aspect = data_aspect(ax)
    (y0, y1) = ax.get_ylim()
    for j, e in enumerate(exps):
        c = color(e)
        for r in e["rows"]:
            x = int(r["pallet_id"]) + (j - (n - 1) / 2) * width - (width - gap) / 2
            v = float(r[field])
            rounded_bar(ax, x, width - gap, v, c, aspect)
            ax.text(x + (width - gap) / 2, v + (y1 - y0) * 0.02, fmt(v),
                    ha="center", va="bottom", fontsize=9, color=INK,
                    fontweight="semibold", zorder=4)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(GRID)


def label(exps, j, short=False):
    """How experiment `j` is named: its own name, else its number."""
    n = exps[j]["num"]
    return exps[j].get("name") or (f"Exp {n}" if short else f"Experiment {n}")


def winners(exps):
    """Indices of the best runs: fewest pallets, then fewest boxes on the last.

    Empty pallets are not counted, and a run that left boxes unpacked cannot
    win (unless no run packed them all).
    """
    def score(e):
        used = [r for r in e["rows"] if int(r[BOXES]) > 0]
        last = max(used, key=lambda r: int(r["pallet_id"]))
        return len(used), int(last[BOXES])
    done = [j for j, e in enumerate(exps)
            if sum(int(r[BOXES]) for r in e["rows"]) >= e["boxes"]] or range(len(exps))
    best = min(score(exps[j]) for j in done)
    return [j for j in done if score(exps[j]) == best], best


def plot_file(path, picks=None):
    exps = select(read_report(path), picks)
    if not exps:
        return None
    style()
    fig, (top, bot) = plt.subplots(2, 1, sharex=True, figsize=(14, 8.5))
    fig.canvas.manager.set_window_title(os.path.basename(path))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.08, hspace=0.35)

    name = os.path.basename(path).split(".csv")[0]
    fig.text(0.07, 0.95, name, fontsize=22, fontweight="bold", color=INK)
    fig.text(0.07, 0.915, f"{len(exps)} experiments \u00b7 "
             f"{sum(int(r[BOXES]) for r in exps[0]['rows'])} boxes",
             fontsize=12, color=MUTED)

    win, (npal, last) = winners(exps)
    prev = fig.text(0.07, 0.868, "Winner" + ("s" if len(win) > 1 else ""),
                    fontsize=12, fontweight="bold", color=INK, va="center")
    # each piece sits right after the previous one
    for j in win:
        prev = fig.add_artist(Annotation(
            label(exps, j), xy=(1, 0.5), xycoords=prev,
            xytext=(16, 0), textcoords="offset points", va="center",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8",
                      facecolor=color(exps[j]), edgecolor="none")))
    fig.add_artist(Annotation(
        f"{npal} pallets \u00b7 {last} boxes on the last pallet",
        xy=(1, 0.5), xycoords=prev, xytext=(14, 0), textcoords="offset points",
        va="center", fontsize=12, color=MUTED))
    print(f"{os.path.basename(path)}: winner "
          + ", ".join(label(exps, j) for j in win)
          + f" ({npal} pallets, {last} boxes on the last)")

    handles = [Patch(facecolor=color(exps[j]), label=label(exps, j))
               for j in range(len(exps))]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.96),
               ncol=len(exps), handlelength=1.0, handleheight=1.0,
               columnspacing=1.6, fontsize=11)

    fig.canvas.draw()  # fix the axes sizes before the bars measure them
    bars(top, exps, BOXES, lambda v: f"{v:.0f}")
    top.set_title("Boxes per pallet")
    bars(bot, exps, VOLUME, lambda v: f"{v:.0f}%")
    bot.set_title("Volume per pallet")
    bot.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    bot.set_xlabel("Pallet")

    top_id = max(int(r["pallet_id"]) for e in exps for r in e["rows"])
    bot.set_xticks(range(top_id + 1))
    return fig


def plot_summary(files, picks=None):
    """One window over all files: how often each experiment won."""
    # experiments are matched across files by name; an unnamed one by its number
    unnamed = {}  # key of an unnamed experiment -> its number

    def key(e):
        if e["name"]:
            return e["name"]
        unnamed[f"Exp {e['num']}"] = e["num"]
        return f"Exp {e['num']}"

    table = []  # (file name, columns run, winning columns, (pallets, last))
    first = {}  # key -> earliest position it ran at
    for f in files:
        exps = select(read_report(f), picks)
        if not exps:
            continue
        keys = [key(e) for e in exps]
        for e, k in zip(exps, keys):
            first[k] = min(first.get(k, e["num"]), e["num"])
        win, best = winners(exps)
        table.append((os.path.basename(f).split(".csv")[0], keys,
                      [keys[j] for j in win], best))
    if not table:
        return None
    style()
    cols = sorted(first, key=lambda k: (first[k], k))
    col = {k: i for i, k in enumerate(cols)}
    table = [(name, [col[k] for k in keys], [col[k] for k in win], best)
             for name, keys, win, best in table]
    n_exp = len(cols)
    ccol = [COLORS[(first[k] - 1) % len(COLORS)] for k in cols]

    def exp_name(j, short=False):
        k = cols[j]
        return k if short or k not in unnamed else f"Experiment {unnamed[k]}"

    ran = [sum(j in t[1] for t in table) for j in range(n_exp)]
    sole = [sum(t[2] == [j] for t in table) for j in range(n_exp)]
    shared = [sum(j in t[2] and len(t[2]) > 1 for t in table) for j in range(n_exp)]

    fig = plt.figure(figsize=(15, max(8.5, 0.24 * len(table) + 2.4)))
    fig.canvas.manager.set_window_title("winners")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], left=0.06, right=0.97,
                          top=0.86, bottom=0.07, wspace=0.28)
    left, right = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    fig.text(0.06, 0.95, "Winners", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.06, 0.915, f"{len(table)} files \u00b7 fewest pallets, then fewest "
             "boxes on the last pallet \u00b7 a tie counts for every winner",
             fontsize=12, color=MUTED)

    # left: wins per experiment number, sole wins solid, shared wins tinted
    left.set_xlim(-0.5, n_exp - 0.5)
    top_v = max(a + b for a, b in zip(sole, shared))
    left.set_ylim(0, max(top_v, 1) * 1.25)
    fig.canvas.draw()
    aspect = data_aspect(left)
    w = 0.62
    for j in range(n_exp):
        c = ccol[j]
        x = j - w / 2
        rounded_bar(left, x, w, sole[j] + shared[j], tint(c), aspect)
        if shared[j]:
            left.add_patch(Rectangle((x, 0), w, sole[j], facecolor=c,
                                     edgecolor="none", zorder=3))
        else:
            rounded_bar(left, x, w, sole[j], c, aspect, zorder=4)
        total = sole[j] + shared[j]
        left.annotate(f"{total}", (j, total), xytext=(0, 20),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=16, fontweight="bold", color=INK, zorder=5)
        left.annotate(f"of {ran[j]} file" + ("s" if ran[j] != 1 else ""), (j, total), xytext=(0, 5),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=10, color=MUTED, zorder=5)
    left.set_xticks(range(n_exp), [exp_name(j) for j in range(n_exp)])
    left.set_title("Wins per experiment")
    left.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    left.set_axisbelow(True)
    left.spines["bottom"].set_color(GRID)
    left.legend(handles=[Patch(facecolor=MUTED, label="sole win"),
                         Patch(facecolor=tint(MUTED), label="shared win")],
                loc="upper right", handlelength=1.0, handleheight=1.0, fontsize=10)

    # right: one row per file, a coloured dot on each winner
    for i, (name, ran_cols, win, (npal, last)) in enumerate(table):
        y = len(table) - 1 - i
        for j in ran_cols:
            if j in win:
                right.scatter(j, y, s=150, color=ccol[j], zorder=3)
            else:
                right.scatter(j, y, s=28, color=GRID, zorder=2)
        right.text(n_exp - 0.45, y, f"{npal} pallets \u00b7 {last} on last",
                   va="center", fontsize=9, color=MUTED)
    right.set_xlim(-0.5, n_exp + 1.1)
    right.set_ylim(-0.7, len(table) - 0.3)
    right.set_yticks(range(len(table)), [t[0] for t in reversed(table)], fontsize=9)
    right.set_xticks(range(n_exp), [exp_name(j, True) for j in range(n_exp)])
    right.xaxis.tick_top()
    right.spines["bottom"].set_visible(False)
    right.set_title("Winner in each file", pad=30)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="*",
                   help="report files (default: results/big_experiment/*.csv)")
    p.add_argument("--save", metavar="DIR",
                   help="write one PNG per file to DIR instead of opening windows")
    p.add_argument("--summary", action="store_true",
                   help="only the winners summary, not the per-file plots")
    p.add_argument("-e", "--exp", action="append", metavar="NAME|NUM",
                   help="plot only this experiment, by its name or its number in "
                        "the file (repeat for more; default: all)")
    p.add_argument("--list", action="store_true",
                   help="print the experiments of each file and exit")
    a = p.parse_args()
    files = a.files or sorted(glob.glob("results/big_experiment/*.csv"))
    if a.list:
        for f in files:
            print(os.path.basename(f))
            for e in read_report(f):
                print(f"  {e['num']:>2}  {e['name'] or '(unnamed)'}")
        return
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    for f in [] if a.summary else files:
        fig = plot_file(f, a.exp)
        if fig is None:
            print(f"skip {f}: no experiments" + (" picked" if a.exp else ""))
        elif a.save:
            out = os.path.join(a.save, os.path.basename(f).rsplit(".", 1)[0] + ".png")
            fig.savefig(out, dpi=160)
            plt.close(fig)
            print(out)
    fig = plot_summary(files, a.exp)
    if fig is not None and a.save:
        out = os.path.join(a.save, "winners.png")
        fig.savefig(out, dpi=160)
        print(out)
    if not a.save:
        plt.show()


if __name__ == "__main__":
    main()
