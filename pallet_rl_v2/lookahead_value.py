"""How much of the MCTS gain is search, and how much is knowing the sequence?

At b=1 the agent has no choice of item - only of the 100 loading positions - so
every point MCTS adds over the bare policy comes from planning the future.  This
measures how much of that depends on the future being *known*.

  policy only        no search at all
  MCTS / true        the search plans against the real remaining stream (what
                     evaluate.py has been reporting - an oracle at inference)
  MCTS / shuffled    it knows which items are coming but not in what order
  MCTS / independent it knows only the item distribution - a fresh draw

    python3 lookahead_value.py --ckpt runs/cut2_v2/last.pt --data cut2
"""
import argparse, json, os
import numpy as np
import torch

from bpp.env import Cfg
from bpp.model import Nets, Evaluator
from bpp.mcts import Runner
from bpp.train import setup_torch
from bpp.datasets import gen_cut, gen_rs


def gen_like(data, n, T, seed, cfg):
    """A fresh draw from the same generator, shaped to the real array."""
    if data == 'rs':
        s, _ = gen_rs(n, seed, length=max(T, 1))
    else:
        s, ln = gen_cut(n, 1 if data == 'cut1' else 2, seed, cfg.W, cfg.L, cfg.H)
    out = np.zeros((n, T, 3), np.int8)
    for i in range(n):
        row = s[i][s[i][:, 0] > 0]
        if len(row) == 0:
            continue
        rep = np.tile(row, (T // len(row) + 1, 1))[:T]   # tile so it never runs dry
        out[i] = rep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--c_puct', type=float, default=4.0)
    ap.add_argument('--split', default='test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    setup_torch()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = Cfg(**ck['cfg'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Nets(cfg, with_value=any(k.startswith('value') for k in ck['net']))
    net.load_state_dict(ck['net'])
    ev = Evaluator(net.to(dev).eval(), cfg, dev)

    d = np.load(f'data/{a.data}_{a.split}.npz')
    S, Ln = d['seqs'][:a.n], d['lens'][:a.n]
    T = S.shape[1]
    rng = np.random.default_rng(a.seed)

    shuf = S.copy()                     # same multiset, order destroyed
    for i in range(len(Ln)):
        k = int(Ln[i])
        shuf[i, :k] = shuf[i, rng.permutation(k)]
    indep = gen_like(a.data, len(Ln), T, a.seed + 555, cfg)

    r = Runner(cfg, ev, n_sims=a.sims, c_puct=a.c_puct, dir_eps=0.0, seed=a.seed)
    out = {'ckpt': a.ckpt, 'data': a.data, 'split': a.split, 'n': int(a.n),
           'sims': a.sims, 'c_puct': a.c_puct}

    g = r.greedy_return(S, Ln)
    out['policy_only'] = float(np.mean(g)) / 10 * 100
    print(f"policy only         {out['policy_only']:6.2f}%")

    for tag, bel in (('true', None), ('shuffled', shuf), ('independent', indep)):
        rec = r.play(S, Ln, temp=0.0, train=False, collect=False,
                     belief=bel, belief_lens=None if bel is None else Ln)
        u = float(np.mean([x['util'] for x in rec])) * 100
        it = float(np.mean([x['items'] for x in rec]))
        out[f'mcts_{tag}'] = u
        out[f'items_{tag}'] = it
        print(f"MCTS / {tag:<12} {u:6.2f}%   ({it:.2f} items)")

    out['oracle_bonus'] = out['mcts_true'] - out['mcts_independent']
    out['search_gain'] = out['mcts_independent'] - out['policy_only']
    print(f"\nsearch without knowing the future : {out['search_gain']:+.2f} points")
    print(f"extra from knowing the real stream: {out['oracle_bonus']:+.2f} points")
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
