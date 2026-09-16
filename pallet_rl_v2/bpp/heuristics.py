"""Simple online heuristic baselines - used to sanity-check the simulator.
`lowest` places each item at the feasible position with the smallest landing
height (ties: smallest x then y), an online first-fit-decreasing-height rule.
"""
import numpy as np
from .env import Cfg, combo_dims, compute_masks, apply_actions, new_state


def run_heuristic(seqs, lens, cfg, kind='lowest'):
    B = seqs.shape[0]
    hm, buf, nxt = new_state(seqs, lens, cfg)
    done = np.zeros(B, bool)
    items = np.zeros(B, np.int32)
    util = np.zeros(B, np.float64)
    while not done.all():
        idx = np.nonzero(~done)[0]
        dims = combo_dims(buf[idx], cfg.k)
        mask, zmap = compute_masks(hm[idx], dims, cfg.H)
        mf = mask.reshape(len(idx), -1)
        any_ok = mf.any(1)
        done[idx[~any_ok]] = True
        keep = np.nonzero(any_ok)[0]
        if keep.size == 0:
            continue
        acts = np.empty(keep.size, np.int64)
        for t, q in enumerate(keep):
            z = zmap[q].astype(np.float64).ravel().copy()
            if kind == 'lowest':
                score = z * 1000.0 + np.arange(z.size)
            else:  # 'maxvol': prefer the lowest z, break ties by corner proximity
                score = z * 1000.0 + np.arange(z.size)
            score[~mf[q]] = np.inf
            acts[t] = int(score.argmin())
        sub = idx[keep]
        r = apply_actions(hm, buf, nxt, seqs, lens, dims[keep], zmap[keep], acts, sub, cfg)
        util[sub] += r / 10.0
        items[sub] += 1
    return items, util
