"""One self-contained HTML page with everything the run produced.

    python3 -m ar2l.viz.report --packing pct_nb10 ex10_nb10

Charts are inlined as SVG and rendered bins as base64 PNG, so the file can be
moved anywhere.  Reads results/table1.json, results/table2.json and the
runs/*/log.jsonl the trainer wrote.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dashboard import PAPER, read_log
from . import agents as A
from .replay3d import draw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# AR2L Table 2, discrete setting: Uti. / Std. / Num. per (method, N_B, beta)
PAPER_T2 = {
 (5,"pct"):[(76.2,6.8,29.9),(73.6,7.9,29.9),(70.6,9.2,28.5),(67.5,9.1,27.8),(64.9,8.2,27.1)],
 (5,"cppo"):[(75.5,6.9,29.6),(73.1,8.0,29.0),(70.7,8.4,28.5),(67.9,8.2,27.9),(65.2,7.3,27.4)],
 (5,"rarl"):[(74.6,6.2,29.2),(73.1,7.3,29.2),(70.8,8.4,29.0),(67.8,8.4,28.4),(66.7,8.1,28.5)],
 (5,"ex05"):[(76.5,7.1,30.1),(73.5,8.8,29.4),(70.5,10.4,28.8),(68.2,10.3,28.1),(65.3,10.0,27.6)],
 (5,"ex10"):[(77.4,6.5,30.2),(74.3,9.3,29.5),(72.0,9.5,29.0),(69.5,10.0,28.4),(66.5,8.7,27.9)],
 (5,"rfmdp"):[(75.5,6.7,29.6),(71.8,9.8,28.8),(69.6,9.4,28.5),(67.1,9.0,28.0),(63.9,8.1,27.3)],
 (5,"ap05"):[(76.1,5.8,29.8),(72.6,8.4,29.0),(69.8,9.9,28.4),(67.3,9.6,27.7),(64.7,8.8,27.4)],
 (5,"ap10"):[(76.5,6.5,30.1),(73.4,8.9,29.2),(70.8,9.3,28.6),(68.4,9.3,28.0),(65.7,8.5,27.5)],
 (10,"pct"):[(76.4,6.6,29.9),(70.6,12.4,28.5),(65.1,14.5,27.3),(61.4,14.8,26.4),(55.7,12.9,25.2)],
 (10,"cppo"):[(75.6,7.2,29.8),(70.7,11.6,28.6),(66.2,12.7,27.7),(62.3,12.9,26.9),(57.4,11.4,25.8)],
 (10,"rarl"):[(74.3,7.2,29.4),(71.1,8.7,29.0),(69.2,8.5,28.7),(65.6,8.6,28.0),(63.3,8.2,27.7)],
 (10,"ex05"):[(77.6,5.8,30.3),(73.1,10.2,29.5),(68.0,13.1,28.8),(64.0,12.7,28.0),(59.7,10.2,27.4)],
 (10,"ex10"):[(76.0,7.0,29.8),(72.4,9.7,30.0),(70.3,9.4,30.3),(66.7,9.3,30.3),(63.8,8.0,30.6)],
 (10,"rfmdp"):[(74.4,7.2,29.7),(70.5,11.4,28.7),(65.7,14.3,28.0),(60.8,14.4,26.8),(55.9,12.5,25.9)],
 (10,"ap05"):[(76.2,5.9,29.9),(72.1,11.7,29.2),(66.9,14.7,28.1),(62.1,15.0,26.9),(56.1,13.6,25.5)],
 (10,"ap10"):[(73.6,6.8,28.9),(69.3,10.7,29.2),(66.1,11.8,29.3),(61.9,11.6,29.4),(57.1,9.1,29.8)],
 (20,"pct"):[(77.0,5.5,30.1),(68.4,17.0,28.6),(59.7,19.3,27.0),(50.9,18.2,25.5),(41.9,12.2,24.2)],
 (20,"cppo"):[(74.1,7.5,29.2),(66.7,16.2,27.9),(59.1,18.5,26.5),(53.2,17.3,25.3),(45.8,13.6,24.0)],
 (20,"rarl"):[(72.0,6.4,28.4),(68.6,9.4,29.1),(64.6,10.6,29.3),(61.7,10.0,30.0),(58.7,8.4,30.4)],
 (20,"ex05"):[(76.8,6.2,30.1),(70.0,14.3,30.2),(64.7,15.8,30.5),(60.0,15.3,30.6),(54.4,13.3,30.8)],
 (20,"ex10"):[(76.1,7.3,30.0),(70.9,11.8,29.4),(66.7,12.7,29.0),(62.8,12.6,28.6),(58.5,10.3,28.2)],
 (20,"rfmdp"):[(73.8,7.0,29.0),(69.4,11.0,26.6),(64.7,13.3,24.2),(59.4,15.1,21.5),(54.4,13.0,19.2)],
 (20,"ap05"):[(75.0,7.6,29.5),(70.1,12.2,30.1),(63.9,15.4,30.5),(58.8,14.4,30.7),(53.1,11.7,31.1)],
 (20,"ap10"):[(73.4,8.2,28.9),(68.2,13.2,28.7),(65.6,13.9,29.0),(61.9,14.6,28.9),(57.6,12.9,28.8)],
}
PAPER_T1 = {  # (Uti., Std., Num.) for N_B = 5, 10, 15, 20 and no attack
 "dbl":[(40.4,14.3,18.4),(28.5,16.1,13.9),(26.0,14.9,14.9),(21.3,12.6,8.5),(63.6,11.9,25.8)],
 "bmf":[(46.9,11.5,21.0),(40.9,12.2,22.2),(38.6,12.9,21.1),(33.7,11.9,22.1),(62.0,9.2,24.8)],
 "lsah":[(46.0,10.1,20.3),(42.2,11.3,20.9),(38.6,11.0,19.4),(35.6,11.8,21.3),(60.9,10.9,24.6)],
 "onlinebph":[(47.1,21.0,19.8),(44.0,18.9,22.8),(29.8,18.5,14.6),(22.4,12.7,14.1),(64.1,8.9,25.8)],
 "hmm":[(49.1,11.1,22.5),(46.5,13.8,22.5),(43.4,13.0,21.4),(40.4,10.0,24.1),(56.1,10.4,22.6)],
 "macs":[(43.0,9.7,21.9),(40.8,9.0,24.8),(39.0,9.8,23.7),(38.3,9.0,27.0),(53.0,10.8,21.5)],
 "pct":[(63.6,9.9,27.3),(58.7,11.3,25.8),(50.9,13.1,25.6),(40.5,15.3,21.8),(76.6,6.0,30.0)],
}
LABEL = {"pct": "PCT", "cppo": "CPPO", "rarl": "RARL", "rfmdp": "RfMDP",
         "ex05": "ExactAR2L(0.5)", "ex10": "ExactAR2L(1.0)",
         "ap05": "ApproxAR2L(0.5)", "ap10": "ApproxAR2L(1.0)"}
BETAS = [0, 25, 50, 75, 100]

CSS = """
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;
      --ok:#57cc99;--warn:#f7b32b;--bad:#ef476f;--acc:#4cc9f0}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--panel:#fff;--line:#e2e5ea;--fg:#1a1d23;--dim:#6b7280}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:30px 22px 70px}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-.4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim);
  margin:34px 0 12px;font-weight:600}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);
  margin:22px 0 8px;font-weight:600}
p{margin:8px 0;max-width:78ch}
.sub{color:var(--dim);font-size:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
.kpis{display:flex;gap:11px;flex-wrap:wrap;margin:18px 0 4px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:11px 15px;flex:1;min-width:150px}
.kpi .v{font-size:25px;font-weight:660;font-variant-numeric:tabular-nums;line-height:1.15}
.kpi .k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.5px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:right;color:var(--dim);font-weight:500;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.5px;padding:0 12px 7px 0;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:6px 12px 6px 0;border-top:1px solid var(--line);white-space:nowrap;text-align:right}
td.us{font-weight:660}
.ok{color:var(--ok)} .bad{color:var(--bad)} .dim{color:var(--dim)}
td.p{color:var(--dim)}
.best{font-weight:700;color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:13px}
.grid img{width:100%;max-width:400px;margin:0 auto;border-radius:9px;display:block;background:#fff}
img{width:100%;display:block;border-radius:9px}
svg{width:100%;height:auto;display:block}
code{font:12px ui-monospace,SFMono-Regular,monospace;background:var(--bg);
  border:1px solid var(--line);border-radius:5px;padding:1px 5px}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);
  border-radius:9px;padding:11px 15px;margin:14px 0}
"""


def png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=118, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def curves(runs):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for r in runs:
        rows = [x for x in read_log(r) if x.get("nom_util")]
        if not rows:
            continue
        ax.plot([x["it"] for x in rows], [x["nom_util"] * 100 for x in rows],
                lw=1.6, label=r)
    for r in runs:
        m = r.rsplit("_nb", 1)
        t = PAPER.get((m[0], int(m[1]))) if len(m) == 2 and m[1].isdigit() else None
        if t:                                   # (nominal, fully attacked)
            ax.axhline(t[0], ls="--", lw=.8, color="#aab", zorder=0)
    ax.set_xlabel("iteration"); ax.set_ylabel("held-out nominal utilisation (%)")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return png(fig)


def bin_png(run, attacker=None, seq=0, device="cuda"):
    policy, plabel, nb = A.load_policy(f"run:{run}", device)
    att, alabel, anb = A.load_attacker(attacker, device)
    seqs = np.load(os.path.join(ROOT, "data/discrete_test.npy"))
    ep = A.play(seqs[seq], policy, att, nb=nb or 1)
    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, ep["placed"], ep["S"],
         title=f"{run} vs {alabel}\n{ep['util']*100:.1f}%  {ep['items']} items")
    return png(fig), ep


def kpi_html(t2):
    """The five numbers worth seeing before any table."""
    if not t2:
        return ""
    cells = [(v, PAPER_T2.get((v["nb"], v["method"]))) for v in t2.values()]
    hit = sum(1 for v, p in cells if p and v["0"]["uti"] >= p[0][0])
    gaps = [v["0"]["uti"] - p[0][0] for v, p in cells if p]
    best = max(t2.values(), key=lambda v: v["0"]["uti"])
    worst = min(t2.values(), key=lambda v: v["100"]["uti"] - v["0"]["uti"])
    rows = [("policies evaluated", len(t2)),
            ("at or above the paper, &beta;=0", f"{hit} / {len(gaps)}"),
            ("mean gap, &beta;=0", f"{sum(gaps) / max(len(gaps), 1):+.1f} pts"),
            ("best nominal", f"{best['0']['uti']:.1f}%"),
            ("largest drop under attack",
             f"{worst['0']['uti'] - worst['100']['uti']:.1f} pts")]
    return ('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in rows) + "</div>")


def t2_html(res):
    rows = []
    nbs = sorted({v["nb"] for v in res.values()})
    for nb in nbs:
        rows.append(f'<tr><td colspan="11" style="padding-top:12px">'
                    f'<b>N<sub>B</sub> = {nb}</b></td></tr>')
        cell = {v["method"]: v for v in res.values() if v["nb"] == nb}
        bestb = {b: max((cell[m][str(b)]["uti"] for m in cell), default=0)
                 for b in BETAS}
        for m in LABEL:
            if m not in cell:
                continue
            tds = []
            for i, b in enumerate(BETAS):
                o = cell[m][str(b)]
                cls = " class=best" if abs(o["uti"] - bestb[b]) < 1e-9 else ""
                p = PAPER_T2.get((nb, m))
                ref = f"<td class=p>{p[i][0]:.1f}</td>" if p else "<td class=p>-</td>"
                tds.append(f"<td{cls}>{o['uti']:.1f}</td>" + ref)
            rows.append(f"<tr><td>{LABEL[m]}</td>" + "".join(tds) + "</tr>")
    head = "".join(f"<th colspan=2>&beta;={b}</th>" for b in BETAS)
    sub = "".join("<th>ours</th><th>paper</th>" for _ in BETAS)
    return (f"<table><thead><tr><th>method</th>{head}</tr>"
            f"<tr><th></th>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def t2_detail(res):
    """Uti / Std / Num, ours only, so the other two metrics are visible too."""
    rows = []
    for nb in sorted({v["nb"] for v in res.values()}):
        rows.append(f'<tr><td colspan="16" style="padding-top:12px">'
                    f'<b>N<sub>B</sub> = {nb}</b></td></tr>')
        cell = {v["method"]: v for v in res.values() if v["nb"] == nb}
        for m in LABEL:
            if m not in cell:
                continue
            tds = "".join(f"<td>{cell[m][str(b)]['uti']:.1f}</td>"
                          f"<td>{cell[m][str(b)]['std']:.1f}</td>"
                          f"<td>{cell[m][str(b)]['num']:.1f}</td>" for b in BETAS)
            rows.append(f"<tr><td>{LABEL[m]}</td>{tds}</tr>")
    head = "".join(f"<th colspan=3>&beta;={b}</th>" for b in BETAS)
    sub = "".join("<th>Uti.</th><th>Std.</th><th>Num.</th>" for _ in BETAS)
    return (f"<table><thead><tr><th>method</th>{head}</tr>"
            f"<tr><th></th>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def t1_html(res):
    cols = ["5", "10", "15", "20", "none"]
    head = "".join(f"<th colspan=2>{'no attack' if c=='none' else 'N<sub>B</sub>='+c}</th>"
                   for c in cols)
    sub = "".join("<th>ours</th><th>paper</th>" for _ in cols)
    rows = []
    for name, row in res.items():
        tds = []
        for i, c in enumerate(cols):
            p = PAPER_T1.get(name)
            pv = p[4 if c == "none" else i][0] if p else None
            ov = row.get(c, {}).get("uti")
            tds.append(f"<td>{ov:.1f}</td>" if ov is not None else "<td>-</td>")
            tds.append(f"<td class=p>{pv:.1f}</td>" if pv is not None else "<td class=p>-</td>")
        rows.append(f"<tr><td>{name.upper()}</td>{''.join(tds)}</tr>")
    return (f"<table><thead><tr><th>method</th>{head}</tr>"
            f"<tr><th></th>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packing", nargs="*", default=[])
    p.add_argument("--curves", nargs="*", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "results/report.html"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    def load(f):
        p_ = os.path.join(ROOT, "results", f)
        return json.load(open(p_)) if os.path.exists(p_) else None

    t2, t1 = load("table2.json"), load("table1.json")
    runs = a.curves if a.curves is not None else \
        [r for r in A.list_runs() if not r.startswith(("att_", "h"))][:10]

    h = ["<!doctype html><meta charset=utf-8><title>AR2L reproduction</title>",
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         f"<style>{CSS}</style><main>",
         "<h1>Adjustable Robust Reinforcement Learning for Online 3D Bin Packing</h1>",
         "<p class=sub>Reproduction of arXiv:2310.04323 (NeurIPS 2023) &mdash; "
         "discrete setting, 10&times;10&times;10 bin, item sides 1&ndash;5, "
         "3000 held-out instances of 150 items.</p>",
         kpi_html(t2)]

    if t2:
        import subprocess
        claims = subprocess.run(
            ["python3", os.path.join(ROOT, "scripts/summarize.py"),
             "--which", "table2"], capture_output=True, text=True).stdout
        claims = claims.split("comparative claims, on our numbers")[-1]
        rows = [ln for ln in claims.splitlines() if ln.startswith("| ")][2:]
        claim_html = "".join(
            "<tr><td>" + ln.strip("| ").split(" | ")[0] + "</td><td>"
            + ln.strip("| ").split(" | ")[-1].strip() + "</td></tr>"
            for ln in rows)
        h += ["<h2>Table 2 &mdash; robust methods on mixture datasets</h2>",
              "<p>&beta; is the percentage of test instances reordered by that "
              "policy&rsquo;s own permutation-based attacker. Grey columns are "
              "the paper&rsquo;s numbers; blue is the best of ours in that "
              "column.</p>", t2_html(t2),
              "<h3>All three metrics, ours</h3>", t2_detail(t2),
              "<h3>The paper's comparative claims, tested on our numbers</h3>",
              "<p>How many of the (method, N<sub>B</sub>, &beta;) cells we "
              "trained agree with each claim the paper makes.</p>",
              "<table><thead><tr><th>claim</th><th>cells that agree</th></tr>"
              "</thead><tbody>" + claim_html + "</tbody></table>"]
    if t1:
        h += ["<h2>Table 1 &mdash; how far a trained attacker degrades each method</h2>",
              "<p>The packing policy observes one item; the attacker observes "
              "N<sub>B</sub>.</p>", t1_html(t1)]

    h += ["<h2>Training</h2>", f"<img src='{curves(runs)}'>"]

    if a.packing:
        h += ["<h2>Packings</h2><div class=grid>"]
        for spec in a.packing:
            run, att = (spec.split("@") + [None])[:2]
            try:
                src, ep = bin_png(run, att, device=a.device)
            except FileNotFoundError as e:
                print("skip packing", spec, "-", e)
                continue
            h.append(f"<div class=card><h4>{spec}</h4><img src='{src}'></div>")
        h.append("</div>")

    fig = os.path.join(ROOT, "results/attacker_behaviour.png")
    if os.path.exists(fig):
        b = base64.b64encode(open(fig, "rb").read()).decode()
        h += ["<h2>What the attacker learns</h2>",
              f"<img src='data:image/png;base64,{b}'>"]

    h.append("</main>")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(h))
    print("->", a.out)


if __name__ == "__main__":
    main()
