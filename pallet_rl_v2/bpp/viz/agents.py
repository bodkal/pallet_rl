"""Opponents, and the single-episode drivers that record what they packed.

Every opponent obeys exactly the simulator of Sec. III/V-A - the mask comes from
`env.compute_masks`, so the agent and the human are held to the same stability
rules.  Each driver returns a trace of placements, which is what the viewers draw.

Opponent ids
------------
  mcts:<run>:<sims>   the paper's method: MCTS with rollout evaluation
  policy:<run>        the bare policy p_theta, argmax, no search
  lowest              lowest-landing-height heuristic
  random              uniform over the legal loading positions
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

from ..env import Cfg, combo_dims, compute_masks, features, new_state, step_v

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HEURISTICS = ('lowest', 'random')
_NETS: dict = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# what you can play against
# --------------------------------------------------------------------------
def run_dir(run):
    return os.path.join(ROOT, 'runs', run)


def run_info(run):
    """Geometry, dataset and training progress of one run, or None."""
    d = run_dir(run)
    ck = os.path.join(d, 'last.pt')
    if not os.path.exists(ck):
        ck = os.path.join(d, 'best.pt')
    if not os.path.exists(ck):
        return None
    try:
        args = json.load(open(os.path.join(d, 'args.json')))
    except Exception:
        args = {}
    it, ep = 0, 0
    p = os.path.join(d, 'log.jsonl')
    try:                                   # the tail is enough, and the file is append-only
        size = os.path.getsize(p)
        with open(p, 'rb') as f:
            f.seek(max(0, size - 8192))
            tail = f.read().decode('utf-8', 'replace')
        for line in reversed(tail.splitlines()):
            if line.strip():
                r = json.loads(line)
                it, ep = int(r.get('it', 0)) + 1, int(r.get('ep', 0))
                break
    except Exception:
        pass
    return dict(run=run, ckpt=ck, data=args.get('data', '?'),
                b=int(args.get('b', 1)), k=int(args.get('k', 0)),
                iters=int(args.get('iters', 0)), it=it, ep=ep,
                sims=int(args.get('sims', 100)),
                c_puct=float(args.get('c_puct', 4.0)))


def runs_with_checkpoint():
    d = os.path.join(ROOT, 'runs')
    if not os.path.isdir(d):
        return []
    out = []
    for r in sorted(os.listdir(d)):
        if not os.path.isdir(os.path.join(d, r)):
            continue
        info = run_info(r)
        if info:
            out.append(info)
    return out


def opponents(default_cfg=None):
    """Every playable opponent, best-trained first.

    Ordering matters: the list is a dropdown, so whatever lands on top is what
    most people will play.  Abandoned smoke runs share a directory with the real
    ones, so sort by how far each actually got rather than by name.
    """
    d = default_cfg or Cfg()
    out = []
    for r in sorted(runs_with_checkpoint(), key=lambda r: (-r['ep'], r['run'])):
        cfg = ckpt_cfg(r['ckpt'])
        geom = dict(W=cfg.W, L=cfg.L, H=cfg.H, b=cfg.b, k=cfg.k,
                    it=r['it'], iters=r['iters'], ep=r['ep'])
        for s in (r['sims'], 25, 400):
            if any(o.get('sims') == s and o['run'] == r['run'] for o in out):
                continue
            out.append(dict(id=f"mcts:{r['run']}:{s}",
                            label=f"MCTS x{s} ({r['run']})",
                            note=f"the paper's method - {s} simulations per move, "
                                 "each leaf evaluated by a rollout that samples p_theta",
                            run=r['run'], dataset=r['data'], sims=s,
                            searches=True, **geom))
        out.append(dict(id=f"policy:{r['run']}", label=f"policy only ({r['run']})",
                        note="the trained network alone, argmax, no search at all",
                        run=r['run'], dataset=r['data'], sims=0,
                        searches=False, **geom))
    geom = dict(W=d.W, L=d.L, H=d.H, b=d.b, k=d.k, it=0, iters=0, ep=0)
    out.append(dict(id='lowest', label='lowest landing height', run=None, dataset=None,
                    note='classic online placement rule - the simulator sanity check',
                    sims=0, searches=False, **geom))
    out.append(dict(id='random', label='random legal position', run=None, dataset=None,
                    note='uniform over every legal loading position - the floor',
                    sims=0, searches=False, **geom))
    return out


def opponent_meta(opp, default_cfg=None):
    for o in opponents(default_cfg):
        if o['id'] == opp:
            return o
    return None


def opponent_run(opp):
    p = opp.split(':')
    return p[1] if p[0] in ('mcts', 'policy') else None


def searches(opp):
    """True if the opponent rolls the real upcoming stream out in its search.

    MCTS does: `Runner.rollout` is handed the actual sequence, so the agent
    effectively knows what is coming.  The human is shown the same queue.
    """
    return opp.split(':')[0] == 'mcts'


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def ckpt_cfg(path):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return Cfg(**ck['cfg'])


def load_net(ckpt, device):
    """Cached (cfg, Evaluator) for a checkpoint."""
    import torch
    from ..model import Evaluator, Nets
    key = (ckpt, device)
    with _LOCK:
        if key in _NETS:
            return _NETS[key]
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        cfg = Cfg(**ck['cfg'])
        dev = torch.device(device if torch.cuda.is_available() or device == 'cpu'
                           else 'cpu')
        net = Nets(cfg, with_value=any(k.startswith('value') for k in ck['net']))
        net.load_state_dict(ck['net'])
        net = net.to(dev).eval()
        _NETS[key] = (cfg, Evaluator(net, cfg, dev))
        return _NETS[key]


def config_for(opp, default_cfg=None):
    """The action space the opponent was trained on - the human plays in it too."""
    run = opponent_run(opp)
    if run is None:
        return default_cfg or Cfg()
    return ckpt_cfg(run_info(run)['ckpt'])


# --------------------------------------------------------------------------
# the drivers
# --------------------------------------------------------------------------
def _record(trace, a, dims, zmap, buf, cfg):
    m, p = int(a) // cfg.WL, int(a) % cfg.WL
    x, y = p // cfg.L, p % cfg.L
    j, o = m // (cfg.k + 1), m % (cfg.k + 1)
    w, l, h = (int(v) for v in dims[0, m])
    trace.append(dict(x=int(x), y=int(y), z=int(zmap[0, m, x, y]),
                      w=w, l=l, h=h, slot=int(j), orient=int(o),
                      item=[int(v) for v in buf[0, j]]))


def run_simple(cfg, seq, ln, kind, ev=None, seed=0):
    """One episode of a policy that needs no search.  `kind` is 'policy',
    'lowest' or 'random'.  Returns the same dict shape as `run_mcts`."""
    rng = np.random.default_rng(seed)
    seqs = seq[None]
    lens = np.array([ln], np.int32)
    hm, buf, nxt = new_state(seqs, lens, cfg)
    trace, G = [], 0.0
    reason = 'no_feasible'
    while True:
        dims = combo_dims(buf, cfg.k)
        mask, zmap = compute_masks(hm, dims, cfg.H)
        mf = mask.reshape(1, -1)
        if not mf.any():
            reason = 'sequence_end' if int(nxt[0]) >= int(ln) and not buf.any() \
                else 'no_feasible'
            break
        if kind == 'policy':
            p = ev.probs(features(hm, buf, cfg.H), np.ascontiguousarray(mf))
            a = int(p[0].argmax())
        elif kind == 'lowest':
            z = zmap.reshape(-1).astype(np.float64) * 1000.0 + np.arange(cfg.A)
            z[~mf[0]] = np.inf
            a = int(z.argmin())
        else:
            a = int(rng.choice(np.nonzero(mf[0])[0]))
        _record(trace, a, dims, zmap, buf, cfg)
        G += float(step_v(hm, buf, nxt, seqs, lens, dims, zmap,
                          np.array([a], np.int64), cfg)[0])
    return dict(trace=trace, utilization=G / 10.0, n_items=len(trace), reason=reason)


def run_mcts(cfg, seq, ln, ev, sims=100, c_puct=4.0, seed=0):
    from ..mcts import Runner
    r = Runner(cfg, ev, n_sims=sims, c_puct=c_puct, dir_eps=0.0, seed=seed)
    rec = r.play(seq[None], np.array([ln], np.int32), temp=0.0,
                 train=False, collect=False, trace=True)[0]
    return dict(trace=rec['trace'], utilization=rec['util'],
                n_items=rec['items'],
                reason='sequence_end' if rec['items'] >= int(ln) else 'no_feasible')


def play_opponent(opp, cfg, seq, ln, device='cuda', seed=0, c_puct=4.0):
    """Run `opp` on one sequence; returns placements plus timing."""
    kind, *rest = opp.split(':')
    t0 = time.time()
    if kind in HEURISTICS:
        out = run_simple(cfg, seq, ln, kind, seed=seed)
    else:
        _, ev = load_net(run_info(rest[0])['ckpt'], device)
        if kind == 'policy':
            out = run_simple(cfg, seq, ln, 'policy', ev=ev, seed=seed)
        elif kind == 'mcts':
            out = run_mcts(cfg, seq, ln, ev, sims=int(rest[1]),
                           c_puct=c_puct, seed=seed)
        else:
            raise KeyError(f'unknown opponent {opp!r}')
    out['seconds'] = time.time() - t0
    out['placed'] = [[t['x'], t['y'], t['z'], t['w'], t['l'], t['h']]
                     for t in out['trace']]
    return out
