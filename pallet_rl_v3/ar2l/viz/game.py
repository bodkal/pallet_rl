"""Play the packing problem yourself, then hand the same instance to an agent.

    python3 -m ar2l.viz.game --port 8096

You pack a 10x10x10 bin under the simulator's own stability rule: every cell
the page lets you click is one the box would actually rest in.  That is a
superset of `env.obs()['l_mask']`, the EMS-corner candidate list the policy
scores - a human is not bound to the leaf nodes of a packing tree, so the
page marks which of your placements the packer would also have considered
rather than hiding the rest.

Pick an attacker and the same learned adversary reorders the conveyor for both
of you - which is the whole point of AR2L: a policy is only as good as its
worst ordering.  The instance is then handed to the opponent (any trained run,
or a heuristic) and the two bins are scored side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from ..env import BPPBatch, sample_items
from . import agents as A

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS_BIN = [10, 5, 120]  # (bin extent, item side cap, leaf cap); set from argv
ROOT = A.ROOT
_GAMES: dict = {}
_LOCK = threading.Lock()
DEVICE = 'cuda'


def free_positions(env):
    """(R, S, S) every placement the box would rest in, EMS filter off.

    The policy's action space is the corners of the empty maximal spaces, but
    a human packing by hand is not bound to it: anything that lands stably and
    stays inside the bin is a placement they can reach for.  Run the same
    sweep `_positions` runs, with `ems` off, so the square-footprint and
    terminal-state filters still apply.
    """
    env.ems = False
    env._invalidate(hmap=False)          # the height map is unchanged; only the filter
    try:
        return env._positions()[0][0].copy()
    finally:
        env.ems = True
        env._invalidate(hmap=False)


def grids(env):
    """Per-orientation legality and landing height for one position.

    The page is a grid you click, so it needs a value at *every* cell, not the
    sparse candidate list the network scores.  Three layers come back:

        cand    the cell is in the candidate list -- what the packer scores
        free    the box would rest there stably -- what *you* may do, a
                superset of `cand`
        z       where the box would land, legal or not, so hovering an illegal
                cell can still say why
    """
    k = int(env.obs()['l_mask'][0].sum())   # refreshes the candidate table first
    cand, z, odims = env._positions()
    return cand[0], free_positions(env), z[0], odims[0], k


def place(env, r, x, y):
    """Drop the leading item at (orientation r, x, y); False if it cannot rest.

    The cell may be outside the candidate list, so there is no action index to
    step.  Rather than duplicate `step`'s bookkeeping, rebuild the placement
    table with the EMS filter off and nothing truncated away, find the cell in
    it, and step that index normally.
    """
    ems, max_l = env.ems, env.max_l
    env.ems = False
    env.max_l = env.rot * env.Lx * env.Ly   # every (orientation, x, y), none dropped
    env._invalidate(hmap=False)
    try:
        o = env.obs()
        lxy = env._lxy[0]
        hit = np.flatnonzero((lxy[:, 2] == r) & (lxy[:, 0] == x) & (lxy[:, 1] == y)
                             & (o['l_mask'][0] > 0))
        if hit.size == 0:
            return False
        # `step` re-reads n_feasible() to decide the bin is finished, and does
        # it with ems still off - which is right: the game is over when *you*
        # have nowhere left, not when the packer would have run out.
        env.step(np.array([int(hit[0])]))
    finally:
        env.ems, env.max_l = ems, max_l
        env._invalidate(hmap=False)
    return True


def board(env):
    """Everything the page draws for one position."""
    cand, free, z, odims, k = grids(env)
    item = env.seq[0, min(int(env.head[0]), env.n_items - 1)]
    win, wmask = env.window()
    nfree = int(free.sum())
    return {
        'Lx': int(env.Lx), 'Ly': int(env.Ly), 'Lz': int(env.Lz),
        'rot': int(odims.shape[0]),
        'dims': odims.tolist(),                      # the footprint per orientation
        'hmap': env.hmap[0].tolist(),
        'item': item.tolist(),
        'mask': cand.astype(np.uint8).tolist(),      # the packer's candidate list
        'free': free.astype(np.uint8).tolist(),      # everything you may click
        'zmap': z.astype(int).tolist(),
        'ncand': k,
        'nfree': nfree,
        'window': win[0][wmask[0]].tolist(),
        'placed': (env.packed[0, : env.n_packed[0]] * env.scale)
                  .round().astype(int).tolist(),
        'util': float(env.utilization()[0]),
        'items': int(env.n_packed[0]),
        'done': bool(env.done[0]) or nfree == 0,
    }


def advance(st):
    """Let the attacker reorder the conveyor before the human sees it."""
    env = st['env']
    if st['att'] is not None and not env.done[0]:
        idx, _ = st['att'](env)
        st['promoted'] = int(idx[0])
        env.permute(idx)
    else:
        st['promoted'] = None


def new_game(nb, seed, attacker_spec, bin_size=10, size_hi=5, max_l=120):
    """`bin_size` is an int for a cube or an (Lx, Ly, Lz) triple."""
    att, alabel, anb = A.load_attacker(attacker_spec, DEVICE)
    nb = anb or nb
    # the attacker reorders only what its own run let it reach
    n_pick = A.spec_n_pick(attacker_spec)
    seq = sample_items(np.random.default_rng(seed), (150,), 1, size_hi)
    env = BPPBatch(1, S=bin_size, nb=nb, n_items=150, size_hi=size_hi,
                   max_l=max_l, n_pick=n_pick)
    env.reset(seq[None])
    st = {'env': env, 'seq': seq, 'nb': nb, 'att': att, 'alabel': alabel,
          'bin': bin_size, 'size_hi': size_hi, 'max_l': max_l,
          'promoted': None, 'opp_spec': None, 'opp': None}
    gid = uuid.uuid4().hex[:12]
    with _LOCK:
        _GAMES[gid] = st
        for old in list(_GAMES)[:-40]:      # keep the last few dozen sessions
            _GAMES.pop(old, None)
    advance(st)
    return gid, st


def opponent(st, spec):
    """Replay the identical instance with a chosen policy.

    Cached on the session: the page asks for the comparison once when the bin
    is full and again on every redraw of the result screen, and a trained
    policy is far too slow to replay on each of those.
    """
    if st.get('opp_spec') == spec:
        return st['opp']
    policy, label, _ = A.load_policy(spec, DEVICE)
    t0 = time.time()
    ep = A.play(st['seq'], policy, st['att'], nb=st['nb'], S=st['bin'],
                size_hi=st['size_hi'], max_l=st['max_l'])
    out = {'label': label, 'util': ep['util'], 'items': ep['items'],
           'placed': ep['placed'], 'seconds': time.time() - t0,
           'reason': 'ran out of room'}
    st['opp_spec'], st['opp'] = spec, out
    return out


class QuietServer(ThreadingHTTPServer):
    """A browser refresh aborts the in-flight response; that is not an error."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype='application/json', code=200):
        b = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(b)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass          # the page navigated away mid-response; nothing to do

    def _json(self, obj, code=200):
        self._send(json.dumps(obj), code=code)

    def do_GET(self):
        p = urlparse(self.path).path.rstrip('/') or '/'
        if p in ('/', '/index.html'):
            with open(os.path.join(HERE, 'game.html'), 'rb') as f:
                return self._send(f.read(), 'text/html; charset=utf-8')
        if p == '/api/setup':
            runs = [r for r in A.list_runs() if A.run_info(r)]
            policies = [r for r in runs if not r.startswith(('att_', 'h'))]
            attackers = [r for r in runs
                         if r.startswith(('att_', 'h')) or r in policies]
            return self._json({'policies': policies, 'attackers': attackers,
                               'heuristics': list(A.HEURISTICS)})
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            return self._json({'error': 'bad request'}, 400)
        p = urlparse(self.path).path.rstrip('/')
        if p == '/api/new':
            gid, st = new_game(body.get('nb', 10), body.get('seed', 0),
                               body.get('attacker'),
                               ARGS_BIN[0], ARGS_BIN[1], ARGS_BIN[2])
            return self._json({'gid': gid, 'board': board(st['env']),
                               'promoted': st['promoted'], 'alabel': st['alabel']})
        st = _GAMES.get(body.get('gid'))
        if st is None:
            return self._json({'error': 'that game has expired'}, 404)
        if p == '/api/place':
            if not place(st['env'], int(body['r']), int(body['x']), int(body['y'])):
                return self._json({'error': 'the box will not rest there'}, 400)
            advance(st)
            return self._json({'board': board(st['env']), 'promoted': st['promoted']})
        if p == '/api/agent':
            # a policy that fails to load must not take the result screen down
            # with it - the page needs a body it can show as an error
            try:
                return self._json(opponent(st, body['spec']))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
        self.send_error(404)

    def log_message(self, *a):
        pass


def main(argv=None):
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8096)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--bin', default='10',
                    help='bin extent: an int for a cube, or WxLxH')
    ap.add_argument('--size_hi', default='5',
                    help='item side cap: an int, or per-axis WxLxH')
    ap.add_argument('--max_l', type=int, default=120,
                    help='leaf cap; match the --max_l the opponent was trained with')
    a = ap.parse_args(argv)
    DEVICE = a.device

    def extent(v):
        q = str(v).lower().split('x')
        return int(q[0]) if len(q) == 1 else tuple(int(t) for t in q)

    ARGS_BIN[:] = [extent(a.bin), extent(a.size_hi), a.max_l]
    srv = QuietServer((a.host, a.port), Handler)
    runs = [r for r in A.list_runs() if not r.startswith(('att_', 'h'))]
    print(f"game: http://{a.host}:{a.port}/   bin {a.bin}   "
          f"(opponents: {', '.join(runs[:6]) or 'heuristics only'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
