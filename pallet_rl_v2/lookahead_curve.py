"""Utilisation as a function of how far the search may see.

The agent's observation is unchanged throughout - the policy net always sees only
the bin and the buffer.  What varies is the horizon of the MCTS rollouts: the next
N items are the real ones, everything past N is resampled from the item
distribution.  N=0 is search that knows nothing past the buffer; N=inf is the
known-sequence oracle that evaluate.py has been reporting.

    python3 lookahead_curve.py --data cut2
"""
import argparse, json
import numpy as np
import torch

from bpp.env import Cfg
from bpp.model import Nets, Evaluator
from bpp.mcts import Runner
from bpp.train import setup_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--c_puct', type=float, default=4.0)
    ap.add_argument('--split', default='test')
    ap.add_argument('--ns', default='0,1,2,3,5,10,inf')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    setup_torch()
    ckpt = a.ckpt or f'runs/{a.data}_v2/last.pt'
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = Cfg(**ck['cfg'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Nets(cfg, with_value=any(k.startswith('value') for k in ck['net']))
    net.load_state_dict(ck['net'])
    ev = Evaluator(net.to(dev).eval(), cfg, dev)

    d = np.load(f'data/{a.data}_{a.split}.npz')
    S, Ln = d['seqs'][:a.n], d['lens'][:a.n]
    r = Runner(cfg, ev, n_sims=a.sims, c_puct=a.c_puct, dir_eps=0.0, seed=0)

    pol = float(np.mean(r.greedy_return(S, Ln))) / 10 * 100
    rows = [dict(n='policy', util=pol, items=None)]
    print(f'{a.data}: policy only (no search at all)   {pol:6.2f}%', flush=True)

    for tok in a.ns.split(','):
        N = None if tok in ('inf', 'none') else int(tok)
        rec = r.play(S, Ln, temp=0.0, train=False, collect=False, lookahead=N)
        u = float(np.mean([x['util'] for x in rec])) * 100
        it = float(np.mean([x['items'] for x in rec]))
        rows.append(dict(n=tok, util=u, items=it))
        print(f'{a.data}: search sees N={tok:<4}             {u:6.2f}%  ({it:.2f} items)',
              flush=True)

    out = dict(data=a.data, ckpt=ckpt, split=a.split, n=int(a.n), sims=a.sims,
               c_puct=a.c_puct, rows=rows)
    with open(a.out or f'results/curve_{a.data}.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
