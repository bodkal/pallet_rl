"""How much would it help if the agent could CHOOSE which of the k visible boxes
to place, instead of being forced to take the first?

This is no longer online BPP -- it is BPP with a buffer of size k, a strictly
easier problem.  We reuse the trained BPP-1 network unchanged (so this is a
LOWER bound: the net was never trained for selection).

Two policies on identical sequences:
  forced(k)  -- must place buffer[0]; if it does not fit, the episode ends.
  choose(k)  -- may place any box in the buffer; ends only when NONE of them fit.
Both refill the buffer from the stream after each placement, so both see
exactly k boxes at a time.
"""
from __future__ import annotations

import argparse, copy, time
import numpy as np
import torch

from src.bin3d import Bin3D
from src.env import build_obs_tensor
from src.evaluate import load_run
from src.items import make_testset


class Net:
    def __init__(self, cfg, net, dev):
        self.cfg, self.net, self.dev = cfg, net, dev
        self.mean_item = np.array([(cfg.item_min + cfg.item_max) / 2.0] * 3, np.int32)

    @torch.no_grad()
    def act(self, bin_, item):
        """Greedy legal placement for `item`, or None. Same rule as NetPolicy."""
        cfg = self.cfg
        mask = bin_.feasibility_mask(item, cfg.orientations)
        if mask.sum() == 0:
            return None
        x = build_obs_tensor(bin_.hmap[None], np.asarray(item, np.int32)[None], cfg)
        logits, _, mlog = self.net(torch.as_tensor(x, device=self.dev))
        used = (torch.sigmoid(mlog) > 0.5).float()
        pl = self.net.projected_logits(logits, used)
        tm = torch.as_tensor(mask, device=self.dev)[None]
        pl = pl + torch.log(torch.where(tm > 0.5, torch.ones_like(tm), torch.zeros_like(tm)))
        return int(pl.argmax(-1).item())

    @torch.no_grad()
    def value(self, hmap, item):
        x = build_obs_tensor(hmap[None], np.asarray(item, np.int32)[None], self.cfg)
        _, v, _ = self.net(torch.as_tensor(x, device=self.dev))
        return float(v.item())


def rollout(pol: Net, seq, k, choose: bool):
    """Buffer of k boxes, refilled from the stream. Returns (utilisation, n_items)."""
    cfg = pol.cfg
    b = Bin3D(cfg.L, cfg.W, cfg.H)
    idx = min(k, len(seq))
    buf = list(seq[:idx])
    while buf:
        cands = []
        for i, it in enumerate(buf):
            a = pol.act(b, it)
            if a is None:
                continue
            if not choose:                      # forced: only buffer[0] counts
                cands = [(0.0, i, a, it)]
                break
            b2 = b.copy()
            b2.place(a, it, cfg.orientations)
            nxt = seq[idx] if idx < len(seq) else pol.mean_item
            score = 10.0 * (it[0]*it[1]*it[2]) / cfg.bin_volume + pol.value(b2.hmap, nxt)
            cands.append((score, i, a, it))
            continue
        if not choose:
            # forced order: the episode dies if the FRONT box cannot be placed
            a = pol.act(b, buf[0])
            if a is None:
                break
            b.place(a, buf[0], cfg.orientations)
            buf.pop(0)
        else:
            if not cands:
                break                            # nothing in the buffer fits
            _, i, a, it = max(cands, key=lambda c: c[0])
            b.place(a, it, cfg.orientations)
            buf.pop(i)
        if idx < len(seq):
            buf.append(seq[idx]); idx += 1
    return b.utilization, len(b.placed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="bpp1_cut2")
    ap.add_argument("--dataset", default="CUT-2")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    a = ap.parse_args()

    cfg, net, dev, _, _ = load_run(a.run, "best.pt", "cuda")
    pol = Net(cfg, net, dev)
    seqs = make_testset(a.episodes, cfg.L, cfg.W, cfg.H,
                        cfg.item_min, cfg.item_max, a.dataset, seed=999)

    print(f"{a.run} on {a.dataset}, {a.episodes} identical held-out sequences\n")
    print(f"  {'k':>2}  {'policy':22s} {'utilisation':>12s} {'boxes':>7s} {'s/ep':>7s}")
    base = None
    for k in a.ks:
        for choose in (False, True):
            if k == 1 and choose:
                continue                        # identical to forced at k=1
            t0 = time.time()
            us, its = [], []
            for s in seqs:
                u, n = rollout(pol, s, k, choose)
                us.append(u); its.append(n)
            lab = "choose from buffer" if choose else "forced first box"
            m = float(np.mean(us))
            if base is None:
                base = m
            print(f"  {k:2d}  {lab:22s} {m*100:11.2f}% {np.mean(its):7.2f} "
                  f"{(time.time()-t0)/len(seqs):7.3f}"
                  + (f"   {(m-base)*100:+.2f} pts vs BPP-1" if base is not None else ""))
    print(f"\n  (k=1 forced == plain BPP-1; reference eval says 59.73% on CUT-2)")


if __name__ == "__main__":
    main()
