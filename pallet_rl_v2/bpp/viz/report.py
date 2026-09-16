"""One self-contained HTML page: our numbers next to the paper's.

    python3 -m bpp.viz.report                       # -> results/report.html
    python3 -m bpp.viz.report --packing cut1 cut2 rs

Everything is inlined - the charts are SVG, the rendered bins are base64 PNG -
so the file can be moved or mailed on its own.
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os

import numpy as np

from ..evaluate import NAME, PAPER
from . import agents as A

ROOT = A.ROOT
COLORS = ['#4cc9f0', '#57cc99', '#f7b32b', '#b892ff', '#ef476f', '#ff9770']

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
.sub{color:var(--dim);font-size:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
.kpis{display:flex;gap:11px;flex-wrap:wrap;margin:18px 0 4px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:11px 15px;flex:1;min-width:150px}
.kpi .v{font-size:25px;font-weight:660;font-variant-numeric:tabular-nums;line-height:1.15}
.kpi .k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.5px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--dim);font-weight:500;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.5px;padding:0 12px 7px 0;white-space:nowrap}
td{padding:6px 12px 6px 0;border-top:1px solid var(--line);white-space:nowrap}
td.us{font-weight:660}
.ok{color:var(--ok)} .bad{color:var(--bad)} .dim{color:var(--dim)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:13px}
.grid img{width:100%;border-radius:9px;display:block;background:#fff}
svg{width:100%;height:auto;display:block}
.ax{stroke:var(--line);stroke-width:1}.tk{fill:var(--dim);font-size:9px}
.bench{stroke:var(--warn);stroke-width:1.4;stroke-dasharray:5 4}
.lg{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--dim);margin-bottom:6px}
.lg i{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px}
code{font:12px ui-monospace,SFMono-Regular,monospace;background:var(--bg);
  border:1px solid var(--line);border-radius:5px;padding:1px 5px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:11px 13px;
  overflow-x:auto;font:12px ui-monospace,monospace;margin:0}
"""


# --------------------------------------------------------------------------
def collect():
    """Every finished evaluation, preferring the final (not eval-selected) one."""
    rows = [json.load(open(f))
            for f in sorted(glob.glob(f'{ROOT}/results/*_final_last.json'))]
    have = {(r['data'], r['b'], r['k']) for r in rows}
    best: dict = {}
    for f in sorted(glob.glob(f'{ROOT}/results/*_best.json')):
        r = json.load(open(f))
        key = (r['data'], r['b'], r['k'])
        if key in have:
            continue
        if key not in best or r['mcts_util'] > best[key]['mcts_util']:
            best[key] = r
    return rows + list(best.values())


def svg_plot(series, bench=None, W=460, H=210, fmt=lambda v: f'{v:.0f}',
             xfmt=lambda v: f'{v/1000:.0f}k', nt=4):
    """A line chart with no dependencies; `bench` draws the paper's target."""
    ml, mr, mt, mb = 44, 8, 8, 24
    xs = [x for s in series for x in s['xs']]
    ys = [y for s in series for y in s['ys']] + ([bench] if bench else [])
    if not xs:
        return ''
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = (y1 - y0 or 1) * 0.12
    y0, y1 = y0 - pad, y1 + pad
    X = lambda v: ml + (W - ml - mr) * ((v - x0) / ((x1 - x0) or 1))
    Y = lambda v: mt + (H - mt - mb) * (1 - (v - y0) / ((y1 - y0) or 1))
    o = [f'<line class="ax" x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}"/>',
         f'<line class="ax" x1="{ml}" y1="{mt}" x2="{ml}" y2="{H-mb}"/>']
    for i in range(nt + 1):
        v = y0 + (y1 - y0) * i / nt
        yy = mt + (H - mt - mb) * (1 - i / nt)
        o.append(f'<line class="ax" x1="{ml}" y1="{yy:.1f}" x2="{W-mr}" y2="{yy:.1f}" opacity=".45"/>'
                 f'<text class="tk" x="{ml-4}" y="{yy+3:.1f}" text-anchor="end">{fmt(v)}</text>')
    for i in range(5):
        v = x0 + (x1 - x0) * i / 4
        a = 'start' if i == 0 else 'end' if i == 4 else 'middle'
        o.append(f'<text class="tk" x="{X(v):.1f}" y="{H-7}" text-anchor="{a}">{xfmt(v)}</text>')
    if bench:
        o.append(f'<line class="bench" x1="{ml}" y1="{Y(bench):.1f}" x2="{W-mr}" y2="{Y(bench):.1f}"/>')
    for s in series:
        pts = ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in zip(s['xs'], s['ys']))
        o.append(f'<polyline fill="none" stroke="{s["c"]}" stroke-width="2" '
                 f'stroke-linejoin="round" points="{pts}"/>')
    return f'<svg viewBox="0 0 {W} {H}">' + ''.join(o) + '</svg>'


def render_bin(placed, cfg, title):
    """A packed bin as a base64 PNG, drawn by the same code as the replay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .replay3d import draw
    fig = plt.figure(figsize=(4.6, 4.4), dpi=115)
    ax = fig.add_subplot(111, projection='3d')
    draw(ax, placed, cfg, None, title)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format='png', facecolor='white')
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def run_for(row):
    """The run that produced a result row, named by the checkpoint it evaluated.

    Matching on (data, b, k) instead would also pull in abandoned smoke runs with
    the same configuration, and plot their curves next to the real one.
    """
    ck = row.get('ckpt', '')
    name = os.path.basename(os.path.dirname(os.path.abspath(ck))) if ck else ''
    info = A.run_info(name) if name else None
    if info:
        return info
    for r in A.runs_with_checkpoint():        # a row from before ckpt was recorded
        if (r['data'], r['b'], r['k']) == (row['data'], row['b'], row['k']):
            return r
    return None


# --------------------------------------------------------------------------
def build(rows, packings=(), device='cuda', sims=100, seq=0):
    from ..evaluate import BENCH
    from .dashboard import read_log

    rows = sorted(rows, key=lambda r: (r['k'], r['b'], r['data']))
    hit = [r for r in rows if r.get('pct_of_bench', 0) >= 100]
    o = [f'<!doctype html><meta charset="utf-8">'
         f'<meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>3D-BPP reproduction report</title><style>{CSS}</style><main>']
    o.append('<h1>Online 3D bin packing with a buffer &mdash; reproduction</h1>'
             '<div class="sub">Valero Puche &amp; Lee, arXiv:2208.07123. '
             'Average packed items and space utilisation over the 100 held-out '
             'test sequences of each stream, 10&times;10&times;10 bin.</div>')

    best = max(rows, key=lambda r: r.get('pct_of_bench', 0)) if rows else None
    o.append('<div class="kpis">')
    for k, v in [('configurations evaluated', len(rows)),
                 ('at or above the paper', f'{len(hit)} / {len(rows)}'),
                 ('best margin', f"+{best['gap_util']:.1f} pts" if best and 'gap_util' in best else '—'),
                 ('test sequences each', rows[0]['n'] if rows else '—'),
                 ('MCTS simulations', rows[0]['sims'] if rows else '—')]:
        o.append(f'<div class="kpi"><div class="v">{v}</div><div class="k">{k}</div></div>')
    o.append('</div>')

    # ---- the table ----
    o.append('<h2>Results against the paper</h2><div class="card"><table>'
             '<tr><th>stream</th><th>b</th><th>k</th><th>heuristics [8]</th>'
             '<th>model-free [2]</th><th>paper &ldquo;ours&rdquo;</th>'
             '<th>this run, MCTS</th><th>policy only</th><th>gap</th><th>% of paper</th></tr>')
    dash = '<span class="dim">&mdash;</span>'
    cell = lambda t: f'{t[0]:.1f} / {t[1]:.1f}%' if t else dash
    for r in rows:
        p = PAPER.get((r['data'], r['b'], r['k']), {})
        gap, pct = r.get('gap_util'), r.get('pct_of_bench')
        std = (f' <span class="dim">&plusmn;{r["mcts_util_std"]:.1f}</span>'
               if r.get('mcts_util_std') else '')
        # a configuration the paper does not tabulate has nothing to compare to
        gap_td = (f'<td class="{"ok" if gap > 0 else "bad"}">{gap:+.1f}</td>'
                  if gap is not None else f'<td>{dash}</td>')
        pct_td = (f'<td class="{"ok" if pct >= 100 else "bad"}">{pct:.0f}%</td>'
                  if pct is not None else f'<td>{dash}</td>')
        o.append(
            f"<tr><td>{NAME.get(r['data'], r['data'])}</td><td>{r['b']}</td><td>{r['k']}</td>"
            f"<td>{cell(p.get('heur'))}</td><td>{cell(p.get('mf'))}</td>"
            f"<td>{cell(p.get('ours'))}</td>"
            f'<td class="us">{r["mcts_items"]:.2f} / {r["mcts_util"]:.1f}%{std}</td>'
            f'<td>{r["greedy_util"]:.1f}%</td>{gap_td}{pct_td}</tr>')
    o.append('</table><div class="sub" style="margin-top:10px">'
             '&plusmn; is the spread over the evaluation repetitions. '
             '&ldquo;policy only&rdquo; is <i>p<sub>&theta;</sub></i> taken greedily with no '
             'search &mdash; the difference between the two columns is what MCTS buys.'
             '</div></div>')

    # ---- training curves ----
    o.append('<h2>Training</h2><div class="grid">')
    for r in rows:
        info = run_for(r)
        if info:
            log = read_log(info['run'])
            if not log:
                continue
            ev = [x for x in log if 'mcts_util' in x]
            series = [dict(xs=[x['ep'] for x in log], ys=[x['sp_util'] for x in log],
                           c=COLORS[0], n='self-play'),
                      dict(xs=[x['ep'] for x in log], ys=[x['base_util'] for x in log],
                           c=COLORS[3], n='greedy baseline')]
            if ev:
                series.append(dict(xs=[x['ep'] for x in ev], ys=[x['mcts_util'] for x in ev],
                                   c=COLORS[1], n='held-out MCTS'))
            bench = BENCH.get((r['data'], r['b'], r['k']))
            lg = ''.join(f'<span><i style="background:{s["c"]}"></i>{s["n"]}</span>'
                         for s in series)
            if bench:
                lg += (f'<span><i style="background:{COLORS[2]}"></i>'
                       f'paper = {bench[1]:.1f}%</span>')
            o.append(f'<div class="card"><b>{NAME.get(r["data"], r["data"])}</b> '
                     f'<span class="sub">&mdash; {info["run"]}, {log[-1]["ep"]:,} episodes, '
                     f'{log[-1].get("mins", 0):.0f} min</span>'
                     f'<div class="lg" style="margin-top:8px">{lg}</div>'
                     + svg_plot(series, bench[1] if bench else None,
                                fmt=lambda v: f'{v:.0f}%') + '</div>')
    o.append('</div>')

    # ---- example packings ----
    if packings:
        o.append('<h2>What it actually packs</h2><div class="grid">')
        te_cache = {}
        for r in rows:
            data = r['data']
            if data not in packings:
                continue
            info = run_for(r)
            if info is None:
                continue
            cfg = A.ckpt_cfg(info['ckpt'])
            if data not in te_cache:
                te_cache[data] = np.load(f'{ROOT}/data/{data}_test.npz')
            te = te_cache[data]
            s, ln = te['seqs'][seq], int(te['lens'][seq])
            rec = A.play_opponent(f"mcts:{info['run']}:{sims}", cfg, s, ln,
                                  device=device)
            t = (f"{NAME.get(data, data)} #{seq} - {rec['n_items']}/{ln} boxes, "
                 f"{rec['utilization']*100:.1f}% full")
            print(f'  rendered {t}')
            o.append(f'<div class="card"><img alt="{html.escape(t)}" '
                     f'src="{render_bin(rec["placed"], cfg, t)}"></div>')
        o.append('</div>')

    # ---- how to get here ----
    o.append('<h2>Reproduce</h2><div class="card"><pre>'
             'python3 -m bpp.datasets data\n'
             'python3 -m bpp.train --data cut1 --name cut1_v2 --iters 120 --games 512 \\\n'
             '    --sims 100 --c_puct 4.0 --dir_eps 0.05 --temp_moves 0\n'
             'python3 -m bpp.evaluate runs/cut1_v2/last.pt cut1 --reps 3 \\\n'
             '    --sims 100 --c_puct 4.0 --out results/cut1_final_last.json\n'
             'python3 report.py\n'
             'python3 -m bpp.viz.report --packing cut1 cut2 rs</pre>'
             '<div class="sub" style="margin-top:10px">Play the agent yourself with '
             '<code>python3 -m bpp.viz.game</code>, or watch a run train with '
             '<code>python3 -m bpp.viz.dashboard</code>.</div></div>')
    o.append('</main>')
    return ''.join(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description='self-contained HTML results report')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'report.html'))
    ap.add_argument('--packing', nargs='*', default=[],
                    help='render one MCTS packing per named stream (slow)')
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    rows = collect()
    if not rows:
        raise SystemExit('no results/*.json yet - run bpp.evaluate first')
    doc = build(rows, a.packing, a.device, a.sims, a.seq)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        f.write(doc)
    print(f'wrote {a.out}  ({len(doc)/1024:.0f} KB, {len(rows)} configurations)')


if __name__ == '__main__':
    main()
