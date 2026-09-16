"""Final evaluation on the 100 held-out test sequences of a dataset.

Reports the two metrics of the paper (Table II/III): average number of packed
items and average space utilisation, for
  * the bare policy p_theta (argmax, no search)
  * the full method: MCTS with rollout evaluation on the known sequence.
"""
import argparse, json, os
import numpy as np
import torch

from .env import Cfg
from .model import Nets, Evaluator
from .mcts import Runner
from .train import setup_torch

BENCH = {   # Table II / III of the paper, "Ours"
    ('cut1', 1, 0): (21.3, 83.4), ('cut2', 1, 0): (18.0, 69.9), ('rs', 1, 0): (13.1, 53.1),
    ('cut1', 1, 1): (22.1, 85.6), ('cut2', 1, 1): (20.2, 73.9), ('rs', 1, 1): (15.7, 64.2),
    ('cut1', 2, 0): (22.0, 84.0), ('cut2', 2, 0): (20.2, 71.5), ('rs', 2, 0): (14.6, 57.6),
    ('cut1', 3, 0): (22.5, 85.7), ('cut2', 3, 0): (21.8, 77.1), ('rs', 3, 0): (15.8, 62.1),
}

PAPER = {   # the same rows with the two baselines the paper reports beside them
    ('cut1', 1, 0): dict(heur=(15.15, 59.8), mf=(19.1, 73.4)),
    ('cut2', 1, 0): dict(heur=(17.3, 61.19), mf=(17.5, 66.9)),
    ('rs',   1, 0): dict(heur=(13.8, 54.3), mf=(12.2, 50.5)),
    ('cut1', 1, 1): dict(heur=(15.99, 61.5), mf=(19.4, 76.2)),
    ('cut2', 1, 1): dict(heur=(17.62, 62.5), mf=(18.1, 70.2)),
    ('rs',   1, 1): dict(heur=(13.82, 56.9), mf=(15.2, 62.1)),
}
for _key, _v in BENCH.items():          # 'ours' is BENCH, kept in one place
    PAPER.setdefault(_key, {})['ours'] = _v

NAME = {'cut1': 'CUT-1', 'cut2': 'CUT-2', 'rs': 'RS'}


def run_eval(ckpt, data, root='.', n=100, sims=100, c_puct=1.5, scale=1.0,
             reps=1, seed=0, chunk=512, split='test'):
    setup_torch()
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = Cfg(**ck['cfg'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Nets(cfg, with_value=any(k.startswith('value') for k in ck['net']))
    net.load_state_dict(ck['net'])
    net = net.to(dev).eval()
    ev = Evaluator(net, cfg, dev)
    te = np.load(os.path.join(root, 'data', f'{data}_{split}.npz'))
    S, Ln = te['seqs'][:n], te['lens'][:n]

    items, utils = [], []
    for rep in range(reps):
        r = Runner(cfg, ev, n_sims=sims, c_puct=c_puct, scale=scale, seed=seed + rep)
        it, ut = [], []
        for s0 in range(0, len(Ln), chunk):
            rec = r.play(S[s0:s0 + chunk], Ln[s0:s0 + chunk], temp=0.0,
                         train=False, collect=False)
            it += [x['items'] for x in rec]
            ut += [x['util'] for x in rec]
        items.append(np.mean(it)); utils.append(np.mean(ut) * 100)
    r = Runner(cfg, ev, n_sims=sims, c_puct=c_puct, scale=scale, seed=seed)
    g = r.greedy_return(S, Ln)

    out = dict(ckpt=ckpt, data=data, split=split, b=cfg.b, k=cfg.k,
               n=int(n), sims=sims,
               c_puct=c_puct, reps=reps,
               mcts_items=float(np.mean(items)), mcts_util=float(np.mean(utils)),
               mcts_util_std=float(np.std(utils)),
               greedy_util=float(np.mean(g)) / 10 * 100)
    key = (data, cfg.b, cfg.k)
    if key in BENCH and split == 'test':
        bi, bu = BENCH[key]
        out['bench_items'], out['bench_util'] = bi, bu
        out['gap_util'] = round(out['mcts_util'] - bu, 2)
        out['pct_of_bench'] = round(100 * out['mcts_util'] / bu, 1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('data')
    ap.add_argument('--root', default='.'); ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--c_puct', type=float, default=1.5)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    res = run_eval(a.ckpt, a.data, a.root, a.n, a.sims, a.c_puct, reps=a.reps,
                   seed=a.seed, split=a.split)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=2)
