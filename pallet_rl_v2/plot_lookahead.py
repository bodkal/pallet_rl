"""Utilisation vs. how far the search may see -> results/lookahead_curve.png"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bpp.evaluate import BENCH, NAME

X = [0, 1, 2, 3, 5, 10, 13]           # 13 is a stand-in position for "inf"
COL = {'cut1': '#4cc9f0', 'cut2': '#57cc99', 'rs': '#f7b32b'}
fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=130)
for d in ('cut1', 'cut2', 'rs'):
    r = json.load(open(f'results/curve_{d}.json'))
    pol = r['rows'][0]['util']
    ys = [x['util'] for x in r['rows'][1:]]
    ax.plot(X, ys, 'o-', color=COL[d], lw=2, ms=5, label=f'{NAME[d]} (MCTS)')
    ax.axhline(pol, color=COL[d], ls=':', lw=1.2, alpha=.8)
    ax.axhline(BENCH[(d, 1, 0)][1], color=COL[d], ls='--', lw=1.2, alpha=.6)
    ax.annotate(f'{NAME[d]} policy only', (0, pol), textcoords='offset points',
                xytext=(3, -11), fontsize=7.5, color=COL[d])
    ax.annotate(f'paper {NAME[d]}', (13, BENCH[(d, 1, 0)][1]), ha='right',
                textcoords='offset points', xytext=(-2, 4), fontsize=7.5, color=COL[d])
ax.set_xticks(X)
ax.set_xticklabels(['0', '1', '2', '3', '5', '10', '∞'])
ax.set_xlabel('boxes of lookahead available to the search  (N)')
ax.set_ylabel('space utilisation  (%)')
ax.set_title('The network never sees past the buffer; only the search horizon varies\n'
             'dotted = no search at all,  dashed = the paper (which has no lookahead)',
             fontsize=9.5)
ax.grid(alpha=.25)
ax.legend(fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig('results/lookahead_curve.png')
print('wrote results/lookahead_curve.png')
