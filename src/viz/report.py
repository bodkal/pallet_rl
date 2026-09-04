"""Self-contained HTML results report: our numbers next to the paper's.

    python -m src.viz.report --run bpp1_cut2
    python -m src.viz.report --run bpp1_cut2 --ablation abl_none abl_mp_fe ...
"""
from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np

# ---- reference numbers transcribed from the paper -------------------------
PAPER_T3 = {  # Table 3:  method -> dataset -> (# items, space util)
    "Boundary rule (Online)": {"RS": (8.7, .349), "CUT-1": (10.8, .412), "CUT-2": (11.1, .408)},
    "BPH (Online)":           {"RS": (8.7, .354), "CUT-1": (13.5, .519), "CUT-2": (13.1, .492)},
    "LBP (Offline)":          {"RS": (12.9, .547), "CUT-1": (14.9, .591), "CUT-2": (15.2, .595)},
    "Paper BPP-1 (Online)":   {"RS": (12.2, .505), "CUT-1": (19.1, .734), "CUT-2": (17.5, .669)},
}
PAPER_T1 = [  # Table 1 ablation on CUT-2: (MP, MC, FE, space util, # items)
    (0, 0, 0, .0782, 2.0), (1, 0, 1, .279, 7.5), (1, 1, 0, .637, 16.9),
    (0, 1, 1, .630, 16.7), (1, 1, 1, .669, 17.5),
]
PAPER_T4 = {"RS": (.621, .505), "CUT-1": (.762, .734), "CUT-2": (.702, .669)}  # w/ , w/o orient

CSS = """
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;
      --ok:#57cc99;--warn:#f7b32b;--bad:#ef476f;--acc:#4cc9f0}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--panel:#fff;--line:#e2e5ea;--fg:#1a1d23;--dim:#5f6673}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;padding:0 0 60px}
.wrap{max-width:1120px;margin:0 auto;padding:0 22px}
header{border-bottom:1px solid var(--line);padding:28px 0 20px;margin-bottom:26px}
h1{font-size:24px;margin:0 0 6px;letter-spacing:-.2px}
h2{font-size:16px;margin:38px 0 4px;letter-spacing:-.1px}
h3{font-size:13px;margin:22px 0 6px;color:var(--dim);text-transform:uppercase;letter-spacing:.6px}
p.note{color:var(--dim);margin:2px 0 14px;font-size:13px;max-width:74ch}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 4px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;
      font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
tr.ours{background:color-mix(in srgb,var(--acc) 11%,transparent);font-weight:600}
tr.paper td{color:var(--dim)}
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 4px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 15px;min-width:132px}
.kpi .v{font-size:21px;font-weight:650;font-variant-numeric:tabular-nums}
.kpi .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin:10px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h4{margin:0 0 8px;font-size:12px;color:var(--dim);font-weight:600}
img{max-width:100%;display:block;border-radius:6px}
.d{color:var(--ok)} .d.neg{color:var(--bad)}
code{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12px}
svg{width:100%;height:210px;overflow:visible}
"""


def b64img(path):
    if not os.path.exists(path):
        return None
    ext = "gif" if path.endswith(".gif") else "png"
    with open(path, "rb") as f:
        return f"data:image/{ext};base64," + base64.b64encode(f.read()).decode()


def delta(ours, ref):
    d = (ours - ref) * 100
    cls = "d" if d >= 0 else "d neg"
    return f'<span class="{cls}">{d:+.1f} pp</span>'


# ---- tiny SVG line / bar helpers ------------------------------------------
def line_chart(series, ylab="", xlab="", w=560, h=210, pct=False):
    ml, mr, mt, mb = 52, 12, 10, 30
    xs = [x for s in series for x in s["x"]]
    ys = [y for s in series for y in s["y"]]
    if not xs:
        return ""
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = (y1 - y0) * .12 or .05
    y0, y1 = y0 - pad, y1 + pad
    X = lambda v: ml + (w - ml - mr) * ((v - x0) / ((x1 - x0) or 1))
    Y = lambda v: mt + (h - mt - mb) * (1 - (v - y0) / ((y1 - y0) or 1))
    fmt = (lambda v: f"{v*100:.0f}%") if pct else (lambda v: f"{v:.3g}")
    o = [f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">']
    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        o.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{w-mr}" y2="{Y(v):.1f}" '
                 f'stroke="var(--line)"/><text x="{ml-6}" y="{Y(v)+4:.1f}" '
                 f'text-anchor="end" fill="var(--dim)" font-size="10">{fmt(v)}</text>')
    for s in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(s["x"], s["y"]))
        o.append(f'<polyline fill="none" stroke="{s["c"]}" stroke-width="2" '
                 f'stroke-linejoin="round" points="{pts}"/>')
        for a, b in zip(s["x"], s["y"]):
            o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="2.6" fill="{s["c"]}"/>')
        o.append(f'<text x="{X(s["x"][-1])-4:.1f}" y="{Y(s["y"][-1])-8:.1f}" '
                 f'fill="{s["c"]}" font-size="11" text-anchor="end">{s["n"]}</text>')
    o.append(f'<text x="{ml}" y="{h-8}" fill="var(--dim)" font-size="10">{fmt(0) and xlab}</text>')
    for v in (x0, x1):
        o.append(f'<text x="{X(v):.1f}" y="{h-8}" fill="var(--dim)" font-size="10" '
                 f'text-anchor="middle">{v:g}</text>')
    o.append("</svg>")
    return "".join(o)


def hist_chart(datasets, w=560, h=210):
    """Distribution of space utilisation (paper Fig. 7c style)."""
    bins = np.arange(0, 1.0001, 0.1)
    colors = ["#4cc9f0", "#f7b32b", "#57cc99", "#ef476f"]
    ml, mr, mt, mb = 42, 10, 10, 28
    o = [f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">']
    n = len(datasets)
    bw = (w - ml - mr) / (len(bins) - 1)
    mx = 0
    hs = []
    for name, utils in datasets:
        c, _ = np.histogram(utils, bins=bins)
        c = c / max(1, len(utils))
        hs.append((name, c)); mx = max(mx, c.max())
    for i, (name, c) in enumerate(hs):
        for j, v in enumerate(c):
            bh = (h - mt - mb) * (v / (mx or 1))
            x = ml + j * bw + i * bw / n
            o.append(f'<rect x="{x:.1f}" y="{h-mb-bh:.1f}" width="{bw/n-1:.1f}" '
                     f'height="{bh:.1f}" fill="{colors[i%4]}" rx="1"/>')
        o.append(f'<text x="{w-mr}" y="{mt+14+i*15}" text-anchor="end" '
                 f'fill="{colors[i%4]}" font-size="11">{name}</text>')
    for j in range(len(bins)):
        o.append(f'<text x="{ml+j*bw:.1f}" y="{h-8}" fill="var(--dim)" font-size="9" '
                 f'text-anchor="middle">{bins[j]*100:.0f}</text>')
    o.append(f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="var(--line)"/>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------------------
def build(run, ablation=(), out=None):
    d = os.path.join("runs", run)
    ev = json.load(open(os.path.join(d, "eval.json")))
    cfg = ev["config"]
    metrics = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if l.strip()]
    R = ev["results"]
    by = {}
    for r in R:
        by.setdefault(r["dataset"], {})[r["policy"]] = r
    datasets = list(by)
    H = []

    # ---- header / kpis ----
    last = metrics[-1] if metrics else {}
    main_ds = cfg["dataset"] if cfg["dataset"] in by else datasets[0]
    ours = by[main_ds].get("BPP-1 (ours)", {})
    H.append(f"""<header><div class="wrap">
      <h1>Online 3D Bin Packing with Constrained Deep RL &mdash; reproduction</h1>
      <p class="note">Zhao, She, Zhu, Yang &amp; Xu, AAAI 2021 (arXiv:2006.14978).
      Bin {cfg['L']}&times;{cfg['W']}&times;{cfg['H']}, |I| =
      {(cfg['item_max']-cfg['item_min']+1)**3} item types, {cfg['orientations']} orientation(s).
      Trained with PPO (paper uses ACKTR) keeping the full prediction-and-projection
      scheme: mask prediction, mask projection (&epsilon;={cfg['mask_eps']}),
      E<sub>inf</sub> penalty and feasibility-restricted entropy.</p>
      <div class="kpis">
        <div class="kpi"><div class="v">{ours.get('space_util',0)*100:.1f}%</div>
          <div class="k">{main_ds} space util</div></div>
        <div class="kpi"><div class="v">{ours.get('n_items',0):.1f}</div>
          <div class="k">items / bin</div></div>
        <div class="kpi"><div class="v">{ev['step']/1e6:.1f}M</div>
          <div class="k">training steps</div></div>
        <div class="kpi"><div class="v">{last.get('mask_acc',0)*100:.2f}%</div>
          <div class="k">mask accuracy</div></div>
        <div class="kpi"><div class="v">{(1-last.get('invalid_rate',0))*100:.2f}%</div>
          <div class="k">legit placements</div></div>
        <div class="kpi"><div class="v">{ours.get('sec_per_item',0)*1000:.2f} ms</div>
          <div class="k">decision time / item</div></div>
      </div></div></header><div class="wrap">""")

    # ---- Table 3 comparison ----
    H.append("<h2>Comparison with the paper (Table 3)</h2>"
             "<p class='note'>Space utilisation and items packed per bin. "
             "Rows in grey are the paper's reported numbers; ours are measured on "
             f"{ours.get('n', 0)} held-out sequences per benchmark.</p><table><thead><tr><th>Method</th>"
             + "".join(f"<th>{ds} util</th><th>{ds} items</th>" for ds in datasets)
             + "</tr></thead><tbody>")
    for m, vals in PAPER_T3.items():
        H.append(f'<tr class="paper"><td>{m}</td>' + "".join(
            (f"<td>{vals[ds][1]*100:.1f}%</td><td>{vals[ds][0]:.1f}</td>"
             if ds in vals else "<td>&mdash;</td><td>&mdash;</td>") for ds in datasets) + "</tr>")
    seen = []
    for ds in datasets:
        for pol in by[ds]:
            if pol not in seen:
                seen.append(pol)
    for pol in seen:
        cls = "ours" if "ours" in pol or "MCTS" in pol else ""
        H.append(f'<tr class="{cls}"><td>{pol}</td>')
        for ds in datasets:
            r = by[ds].get(pol)
            if r:
                extra = ""
                if pol == "BPP-1 (ours)" and ds in PAPER_T3["Paper BPP-1 (Online)"]:
                    extra = " " + delta(r["space_util"], PAPER_T3["Paper BPP-1 (Online)"][ds][1])
                H.append(f"<td>{r['space_util']*100:.1f}%{extra}</td><td>{r['n_items']:.1f}</td>")
            else:
                H.append("<td>&mdash;</td><td>&mdash;</td>")
        H.append("</tr>")
    H.append("</tbody></table>")

    # ---- BPP-k ----
    ks, ser = [], []
    for ds in datasets:
        pts = []
        for pol, r in by[ds].items():
            if pol == "BPP-1 (ours)":
                pts.append((1, r["space_util"]))
            elif pol.startswith("BPP-") and "MCTS" in pol:
                pts.append((int(pol.split("-")[1].split(" ")[0]), r["space_util"]))
        pts.sort()
        if len(pts) > 1:
            ser.append({"x": [p[0] for p in pts], "y": [p[1] for p in pts],
                        "n": ds, "c": ["#4cc9f0", "#f7b32b", "#57cc99"][len(ser) % 3]})
    if ser:
        H.append("<h2>BPP-k: Monte-Carlo permutation tree search (Sec. 3.3)</h2>"
                 "<p class='note'>The same trained BPP-1 network, no retraining. "
                 "Search over permutations of the k lookahead items with order-dependence "
                 "blocking and max-return backup. The paper's Figure 7(b) shows the same "
                 "monotone rise in utilisation with k.</p>"
                 f'<div class="card">{line_chart(ser, pct=True, xlab="lookahead k")}'
                 '<h4 style="text-align:center;margin-top:6px">avg. space utilisation vs lookahead k</h4></div>')
        H.append("<table><thead><tr><th>Benchmark</th>"
                 + "".join(f"<th>k={k}</th>" for k in ser[0]["x"])
                 + "<th>decision time</th></tr></thead><tbody>")
        for s in ser:
            ds = s["n"]
            tk = max((r["sec_per_item"] for p, r in by[ds].items() if "MCTS" in p), default=0)
            H.append(f"<tr><td>{ds}</td>" + "".join(f"<td>{y*100:.1f}%</td>" for y in s["y"])
                     + f"<td>{tk*1000:.0f} ms/item</td></tr>")
        H.append("</tbody></table>")

    # ---- distribution ----
    H.append("<h2>Distribution of space utilisation</h2>"
             "<p class='note'>Per-episode outcomes, not just the mean "
             "(the paper's Figure 7c).</p>"
             f'<div class="card">{hist_chart([(f"{p} · {ds}", by[ds][p]["utils"]) for ds in datasets for p in by[ds] if p in ("BPP-1 (ours)", "boundary rule")][:4])}'
             '<h4 style="text-align:center">episodes (%) per space-utilisation decile</h4></div>')

    # ---- training curves ----
    if metrics:
        def s(k, c, n, sm=25):
            y = [m.get(k, 0) for m in metrics]
            y = [float(np.mean(y[max(0, i - sm):i + 1])) for i in range(len(y))]
            return {"x": [m["step"] / 1e6 for m in metrics], "y": y, "n": n, "c": c}
        H.append("<h2>Training</h2><div class='grid'>")
        for title, series, pct in [
            ("space utilisation", [s("space_util", "#4cc9f0", "util")], True),
            ("items packed", [s("n_items", "#57cc99", "items")], False),
            ("mask predictor accuracy", [s("mask_acc", "#b892ff", "acc")], True),
            ("invalid-action rate", [s("invalid_rate", "#ef476f", "invalid")], True),
        ]:
            H.append(f'<div class="card"><h4>{title} &mdash; steps (M)</h4>'
                     f'{line_chart(series, pct=pct, w=430, h=180)}</div>')
        H.append("</div>")

    # ---- ablation ----
    H.append("<h2>Ablation: prediction-and-projection (Table 1)</h2>"
             "<p class='note'>MP = mask prediction, MC = mask projection onto the "
             "action distribution, FE = feasibility-restricted entropy.</p>"
             "<table><thead><tr><th>MP</th><th>MC</th><th>FE</th>"
             "<th>paper util</th><th>paper items</th><th>ours util</th><th>ours items</th>"
             "</tr></thead><tbody>")
    abl = {}
    for name in ablation:
        p = os.path.join("runs", name, "config.json")
        mp = os.path.join("runs", name, "metrics.jsonl")
        if not (os.path.exists(p) and os.path.exists(mp)):
            continue
        c = json.load(open(p))
        rows = [json.loads(l) for l in open(mp) if l.strip()][-20:]
        abl[(int(c["use_mask_prediction"]), int(c["use_mask_constraint"]),
             int(c["use_feasibility_entropy"]))] = (
            float(np.mean([r["space_util"] for r in rows])),
            float(np.mean([r["n_items"] for r in rows])))
    tick = lambda b: "&#10003;" if b else "&#10007;"
    for mp_, mc, fe, u, it in PAPER_T1:
        o = abl.get((mp_, mc, fe))
        cls = "ours" if (mp_, mc, fe) == (1, 1, 1) else ""
        H.append(f'<tr class="{cls}"><td>{tick(mp_)}</td><td>{tick(mc)}</td><td>{tick(fe)}</td>'
                 f"<td>{u*100:.1f}%</td><td>{it:.1f}</td>"
                 + (f"<td>{o[0]*100:.1f}%</td><td>{o[1]:.1f}</td>" if o
                    else "<td>&mdash;</td><td>&mdash;</td>") + "</tr>")
    H.append("</tbody></table>")

    # ---- replays ----
    rep = os.path.join(d, "replays")
    if os.path.isdir(rep):
        H.append("<h2>Packing replays</h2><p class='note'>Step-by-step placement; "
                 "the red-outlined box is the item just placed.</p><div class='grid'>")
        for f in sorted(os.listdir(rep)):
            if not f.endswith((".gif", ".png")):
                continue
            if f.endswith("_final.png") and f.replace("_final.png", ".gif") in os.listdir(rep):
                continue
            src = b64img(os.path.join(rep, f))
            if src:
                H.append(f'<div class="card"><h4>{f}</h4><img src="{src}"></div>')
        H.append("</div>")

    H.append("<h2>How to reproduce</h2><p class='note'>"
             "<code>python -m src.train --preset paper --dataset CUT-2 --run bpp1_cut2</code><br>"
             "<code>python -m src.viz.dashboard --port 8080</code> (live, during training)<br>"
             "<code>python -m src.evaluate --run bpp1_cut2 --bppk 2 3 5</code><br>"
             "<code>python -m src.viz.replay3d --run bpp1_cut2 --all</code><br>"
             "<code>python -m src.viz.report --run bpp1_cut2</code></p></div>")

    html = (f"<!doctype html><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'>"
            f"<title>3D-BPP reproduction &mdash; {run}</title><style>{CSS}</style>"
            + "".join(H))
    out = out or os.path.join(d, "report.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, f"({len(html)/1024:.0f} KB)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ablation", nargs="*", default=[])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    build(a.run, a.ablation, a.out)


if __name__ == "__main__":
    main()
