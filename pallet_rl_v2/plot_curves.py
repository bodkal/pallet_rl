"""Training curves (cf. Fig. 4 of the paper) + benchmark reference lines."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = {'cut1': 83.4, 'cut2': 69.9, 'rs': 53.1}
MF = {'cut1': 73.4, 'cut2': 66.9, 'rs': 50.5}
NAME = {'cut1': 'CUT-1', 'cut2': 'CUT-2', 'rs': 'RS'}


def load(run):
    rows = []
    p = f'{ROOT}/runs/{run}/log.jsonl'
    if not os.path.exists(p):
        return rows
    for L in open(p):
        try:
            rows.append(json.loads(L))
        except Exception:
            pass
    return rows


def main(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.9), squeeze=False)
    for ax, (ds, run) in zip(axes[0], runs):
        rows = load(run)
        if not rows:
            continue
        ep = [r['ep'] for r in rows]
        ax.plot(ep, [r['sp_util'] for r in rows], lw=1.0, alpha=.55,
                color='tab:blue', label='self-play (MCTS, w/ noise)')
        ev = [(r['ep'], r['mcts_util']) for r in rows if 'mcts_util' in r]
        gr = [(r['ep'], r['greedy_util']) for r in rows if 'greedy_util' in r]
        if ev:
            ax.plot(*zip(*ev), 'o-', ms=3.5, lw=1.8, color='tab:red',
                    label='test: MCTS + rollout')
        if gr:
            ax.plot(*zip(*gr), 's--', ms=3, lw=1.2, color='tab:green',
                    label=r'test: policy $p_\theta$ only')
        ax.axhline(BENCH[ds], color='k', ls=':', lw=1.6,
                   label=f'paper Ours ({BENCH[ds]}%)')
        ax.axhline(MF[ds], color='gray', ls='-.', lw=1.0,
                   label=f'model-free [2] ({MF[ds]}%)')
        ax.set_title(NAME[ds]); ax.set_xlabel('episodes')
        ax.set_ylabel('space utilisation (%)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=6.5, loc='lower right')
    plt.tight_layout()
    out = f'{ROOT}/results/training_curves.png'
    plt.savefig(out, dpi=150)
    print(out)


if __name__ == '__main__':
    rs = [a.split('=') for a in sys.argv[1:]] or [
        ('cut1', 'cut1_v2'), ('cut2', 'cut2_v2'), ('rs', 'rs_v2')]
    main([(a, b) for a, b in rs])
