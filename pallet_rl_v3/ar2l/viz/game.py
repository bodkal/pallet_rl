"""Play the packing problem yourself, then hand the same instance to an agent.

    python3 -m ar2l.viz.game --port 8096

The instance is built from the *opponent's own* training configuration: pick a
run and the setup form fills itself with the bin, the item bounds, the window
`nb` and the reach `n_pick` that run was trained under, so the duel is fought
on the geometry the agent actually knows.  Every field stays editable - the
server re-validates the whole set and refuses a combination the simulator
could not run - and anything you change away from the run's own value is
flagged, because an agent replayed off its training distribution is being
asked a question it was never trained on.

You pack that bin under the simulator's own stability rule: every cell the
page lets you click is one the box would actually rest in.  That is a superset
of `env.obs()['l_mask']`, the EMS-corner candidate list the policy scores - a
human is not bound to the leaf nodes of a packing tree, so the page marks
which of your placements the packer would also have considered rather than
hiding the rest.

With `n_pick > 1` the conveyor is a pick station rather than a queue: the
first `n_pick` boxes are within the cell's reach and you choose which of them
to pack, exactly as a `select`-trained agent's permuter does.  The rest of the
window is preview - seen, reasoned about, not reachable.

Pick an attacker and the same learned adversary reorders the conveyor for both
of you - which is the whole point of AR2L: a policy is only as good as its
worst ordering.  The instance is then handed to the opponent (any trained run,
or a heuristic) and the two bins are scored side by side.

The result screen can then recalculate: the same boxes in the same order, the
agent replayed under parameters you edit, one row per setting.  The three
fields that feed the draw itself - `n_items`, `size_lo`, `size_hi` - are held
at the game's own values there, because changing one deals a different
sequence and the row would no longer be measuring the field you moved.
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
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..config import CFG
from ..env import BPPBatch, sample_items
from . import agents as A

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = A.ROOT
_GAMES: dict = {}
_LOCK = threading.Lock()
DEVICE = 'cuda'

#: what the command line pinned, which then overrides config.yaml in the form
CLI: dict = {}

#: the editable fields, and the three that are per-axis triples
PARAM_KEYS = ('bin', 'n_items', 'size_lo', 'size_hi', 'nb', 'n_pick',
              'max_l', 'rot', 'ems', 'stability', 'min_support')
TRIPLES = ('bin', 'size_lo', 'size_hi')

#: the three that feed `sample_items`, and so fix the box stream a session was
#: dealt.  A recalculation holds them at the values the game was played under
#: -- move one and the same seed deals a *different* sequence, so the rerun
#: would no longer be about the parameter that was changed.
STREAM_KEYS = ('n_items', 'size_lo', 'size_hi')
#: everything else: geometry and rules, which the stream is indifferent to
RECALC_KEYS = tuple(k for k in PARAM_KEYS if k not in STREAM_KEYS)

# Caps that exist only to keep the *page* alive: the top view paints one
# rectangle per floor cell and the browser has to redraw all of them on every
# hover, so a 500-wide bin is not a configuration, it is a hang.
MAX_AXIS = 200
MAX_NB = 60
MAX_ITEMS = 5000


# --------------------------------------------------------------- parameters
def defaults():
    """config.yaml's env, plus anything the command line pinned over it."""
    e, t = CFG['env'], CFG['train']
    p = {'bin': list(A.extent(e['bin'])),
         'n_items': int(e['n_items']),
         'size_lo': list(A.extent(e['size_lo'])),
         'size_hi': list(A.extent(e['size_hi'])),
         'nb': int(t['nb']),
         'n_pick': int(t['n_pick'] or t['nb']),
         'max_l': int(e['max_l']),
         'rot': int(e['rot']),
         'ems': int(e['ems']),
         'stability': str(e['stability']),
         'min_support': float(e['min_support'])}
    p.update(CLI)
    return p


def run_params(spec):
    """The env `spec` was trained in, as the form's own fields.

    A heuristic or `random` was not trained in anything, so it falls back to
    the shipped defaults; those opponents run in whatever bin you give them.
    """
    p = defaults()
    if not spec or spec in ('none', 'random') or spec.startswith('heur:'):
        return p
    info = A.run_info(spec.split(':')[1])
    if info is None:
        return p
    a = info['args']
    for k in ('n_items', 'max_l', 'rot', 'ems', 'stability', 'min_support', 'nb'):
        if a.get(k) is not None:
            p[k] = a[k]
    for k in TRIPLES:
        if a.get(k) is not None:
            p[k] = list(A.extent(a[k]))
    # a run trained before --n_pick existed recorded `null`, i.e. the whole
    # window was within reach
    p['n_pick'] = int(a['n_pick'] or p['nb'])
    # A flag on the command line outranks the run, as it outranks config.yaml
    # everywhere else here; without this the preset would silently undo it the
    # moment an opponent was chosen, which is every time.  The field is then
    # marked as drift, which is exactly what it is.
    p.update(CLI)
    p['algo'] = a.get('algo')
    return p


def validate(raw):
    """-> (params, errors).  Every field the page sent, checked in one place.

    The page validates as you type for the immediate feedback, but nothing is
    trusted from it: a bin the sweeps cannot run, or an item bound no draw can
    satisfy, has to fail here rather than raise out of `BPPBatch.__init__`
    and take the request down with a 500.
    """
    p, err = defaults(), []
    raw = raw or {}

    def whole(k, lo, hi, what):
        v = raw.get(k, p[k])
        try:
            n = int(v)
        except (TypeError, ValueError):
            err.append(f'{what}: expected a whole number, got {v!r}')
            return p[k]
        if not lo <= n <= hi:
            err.append(f'{what}: must be between {lo} and {hi}, got {n}')
            return p[k]
        return n

    def triple(k, lo, hi, what):
        v = raw.get(k, p[k])
        v = list(v) if isinstance(v, (list, tuple)) else [v] * 3
        if len(v) != 3:
            err.append(f'{what}: expected three numbers, got {len(v)}')
            return p[k]
        out = []
        for ax, t in zip('xyz', v):
            try:
                n = int(t)
            except (TypeError, ValueError):
                err.append(f'{what} {ax}: expected a whole number, got {t!r}')
                return p[k]
            if not lo <= n <= hi:
                err.append(f'{what} {ax}: must be between {lo} and {hi}, got {n}')
                return p[k]
            out.append(n)
        return out

    p['bin'] = triple('bin', 1, MAX_AXIS, 'bin')
    p['size_lo'] = triple('size_lo', 1, MAX_AXIS, 'smallest item side')
    p['size_hi'] = triple('size_hi', 1, MAX_AXIS, 'largest item side')
    p['n_items'] = whole('n_items', 1, MAX_ITEMS, 'items in the sequence')
    p['nb'] = whole('nb', 1, MAX_NB, 'window N_B')
    p['n_pick'] = whole('n_pick', 1, MAX_NB, 'reach k')
    p['max_l'] = whole('max_l', 1, 1 << 20, 'candidate cap')
    p['rot'] = whole('rot', 1, 2, 'orientations')
    p['ems'] = whole('ems', 0, 1, 'EMS filter')

    s = str(raw.get('stability', p['stability']))
    if s not in ('com', 'cdrl'):
        err.append(f"stability rule: expected 'com' or 'cdrl', got {s!r}")
    else:
        p['stability'] = s
    try:
        ms = float(raw.get('min_support', p['min_support']))
        if not 0.0 <= ms <= 1.0:
            raise ValueError
        p['min_support'] = ms
    except (TypeError, ValueError):
        err.append('support floor: expected a fraction between 0 and 1')

    # ---- the combinations, once every field is individually sane -----------
    if p['n_pick'] > p['nb']:
        err.append(f"reach k = {p['n_pick']} is larger than the window "
                   f"N_B = {p['nb']}: you cannot reach a box you cannot see")
    for ax, (lo, hi, L) in enumerate(zip(p['size_lo'], p['size_hi'], p['bin'])):
        n = 'xyz'[ax]
        if lo > hi:
            err.append(f'item side {n}: the smallest ({lo}) is above the '
                       f'largest ({hi})')
        elif hi > L:
            # an item wider than the axis it is drawn for can never be placed
            # in its own pose, so the episode would die on the first such draw
            err.append(f'largest item side {n} = {hi} does not fit the bin '
                       f'({n} = {L})')
    cells = p['bin'][0] * p['bin'][1]
    if cells > 40000:
        err.append(f'a {p["bin"][0]}x{p["bin"][1]} floor is {cells} cells; '
                   f'the top view redraws all of them on every hover')
    return p, err


def diff_params(p, want, keys=PARAM_KEYS):
    """Which of `keys` differ between two parameter sets, formatted for the page."""
    out = []
    for k in keys:
        a, b = p[k], want[k]
        if isinstance(a, list):
            a, b = list(a), list(b)
        if a != b:
            fmt = (lambda v: 'x'.join(map(str, v))) if isinstance(a, list) else str
            out.append({'key': k, 'is': fmt(a), 'was': fmt(b)})
    return out


def mismatch(p, spec):
    """Which fields `p` moved away from the run behind `spec`."""
    if not spec or spec in ('none', 'random') or spec.startswith('heur:'):
        return []
    return diff_params(p, run_params(spec))


# -------------------------------------------------------------------- board
def free_positions(env):
    """(R, S, S) every placement the box would rest in, EMS filter off.

    The policy's action space is the corners of the empty maximal spaces, but
    a human packing by hand is not bound to it: anything that lands stably and
    stays inside the bin is a placement they can reach for.  Run the same
    sweep `_positions` runs, with `ems` off, so the square-footprint and
    terminal-state filters still apply.
    """
    ems = env.ems                        # restore what it *was*: the game can
    env.ems = False                      # itself be run with --ems 0
    env._invalidate(hmap=False)          # the height map is unchanged; only the filter
    try:
        return env._positions()[0][0].copy()
    finally:
        env.ems = ems
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


# -------------------------------------------------------------- pick station
def snapshot(st):
    """Remember the conveyor as the page is about to draw it.

    Every pick is applied to *this* order rather than to whatever the last
    pick left behind, so choosing slot 3 and then changing your mind to slot 1
    is one permutation of the window and not two composed ones -- which is
    what the trained permuter does, and the only reading under which the strip
    on screen can keep a stable order to click.
    """
    win, wmask = st['env'].window()
    st['win0'], st['wmask0'] = win[0].copy(), wmask[0].copy()
    st['sel'] = 0


def restore(st):
    """Put the remembered window back into the sequence."""
    env = st['env']
    off = env.head[0] + np.arange(env.nb)
    ok = off < env.n_items           # past the end the offsets clamp and would
    env.seq[0, off[ok]] = st['win0'][ok]     # otherwise write duplicates back
    env._invalidate(hmap=False)


def select(st, i):
    """Bring reachable item `i` of the remembered window to the front."""
    env = st['env']
    try:
        i = int(i)
    except (TypeError, ValueError):
        return False
    if not 0 <= i < min(env.n_pick, env.nb) or not st['wmask0'][i]:
        return False
    restore(st)
    if i:
        env.permute(np.array([i]))
    st['sel'] = i
    return True


def reach(st):
    """How many placements each box within reach still has.

    The bin is finished when *no* reachable box can be placed, not when the
    one at the front cannot: with a reach of five, four dead boxes and a live
    one is a live game.  One sweep per reachable box, at human pace.
    """
    env = st['env']
    keep, out = st['sel'], []
    for i in range(min(env.n_pick, env.nb)):
        if not st['wmask0'][i] or env.done[0]:
            out.append(0)
            continue
        select(st, i)
        out.append(int(free_positions(env).sum()))
    select(st, keep)
    return out


def board(st):
    """Everything the page draws for one position."""
    env = st['env']
    cand, free, z, odims, k = grids(env)
    item = env.seq[0, min(int(env.head[0]), env.n_items - 1)]
    picks = reach(st)
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
        'window': st['win0'][st['wmask0']].tolist(),
        'nb': int(env.nb),
        'n_pick': int(env.n_pick),
        'sel': int(st['sel']),
        'picks': picks,                # placements open to each reachable box
        'human_pick': bool(st['human_pick']),
        'placed': (env.packed[0, : env.n_packed[0]] * env.scale)
                  .round().astype(int).tolist(),
        'util': float(env.utilization()[0]),
        'items': int(env.n_packed[0]),
        # stuck means stuck with what you can actually take: with the pick
        # yours that is every reachable box, and without it only the one you
        # were handed
        'done': bool(env.done[0]) or not (any(picks) if st['human_pick']
                                          else picks[st['sel']]),
    }


def advance(st):
    """Let the attacker reorder the conveyor, then freeze it for the page."""
    env = st['env']
    if st['att'] is not None and not env.done[0]:
        idx, _ = st['att'](env)
        st['promoted'] = int(idx[0])
        env.permute(idx)
    else:
        st['promoted'] = None
    snapshot(st)


def new_game(seed, attacker_spec, params, human_pick=True):
    att, alabel, _ = A.load_attacker(attacker_spec, DEVICE)
    p = params
    seq = sample_items(np.random.default_rng(seed), (p['n_items'],),
                       p['size_lo'], p['size_hi'])
    env = BPPBatch(1, S=p['bin'], nb=p['nb'], n_items=p['n_items'],
                   size_lo=p['size_lo'], size_hi=p['size_hi'],
                   max_l=p['max_l'], rot=p['rot'], ems=p['ems'],
                   stability=p['stability'], min_support=p['min_support'],
                   n_pick=p['n_pick'])
    env.reset(seq[None])
    st = {'env': env, 'seq': seq, 'p': p, 'att': att, 'alabel': alabel,
          'human_pick': bool(human_pick) and p['n_pick'] > 1,
          'promoted': None, 'opp_spec': None, 'opp': None}
    gid = uuid.uuid4().hex[:12]
    with _LOCK:
        _GAMES[gid] = st
        for old in list(_GAMES)[:-40]:      # keep the last few dozen sessions
            _GAMES.pop(old, None)
    advance(st)
    return gid, st


def opp_permuter(st, spec, n_pick=None):
    """Who chooses the agent's box: the attacker, or its own selector.

    A `select` run is trained *with* a permuter that hands it one of the `k`
    boxes within reach; replayed without one it would silently take slot 0
    every step, which is not the policy that was trained.  So when no attacker
    is on, a select run gets its own mixer back.

    `n_pick` is the reach of the replay being set up rather than the session's,
    because a recalculation may have changed it: a select run recalculated down
    to k = 1 has nothing left to select, and must not be handed its mixer.
    """
    if st['att'] is not None:
        return st['att'], st['alabel']
    k = st['p']['n_pick'] if n_pick is None else n_pick
    if spec.startswith('run:'):
        run = spec.split(':')[1]
        info = A.run_info(run)
        if info and info['args'].get('algo') == 'select' and k > 1:
            fn, lab, _ = A.load_attacker(f'mix:{run}', DEVICE)
            return fn, lab
    return None, None


def hand_over(st, spec, p):
    """Play this session's own item stream with `spec`, under parameters `p`.

    `st['seq']` is what makes the comparison an instance rather than a sample:
    the boxes, and the order they arrive in, are the ones that were dealt once
    at `new_game`.  Everything the parameters touch -- the bin, the window, the
    reach, the action space, the stability rule -- is free to differ from the
    game that was played, which is what a recalculation varies.
    """
    policy, label, _ = A.load_policy(spec, DEVICE)
    perm, plabel = opp_permuter(st, spec, p['n_pick'])
    t0 = time.time()
    ep = A.play(st['seq'], policy, perm, nb=p['nb'], S=p['bin'],
                size_hi=p['size_hi'], max_l=p['max_l'], n_pick=p['n_pick'],
                rot=p['rot'], ems=p['ems'], stability=p['stability'],
                min_support=p['min_support'])
    return {'label': label, 'util': ep['util'], 'items': ep['items'],
            'placed': ep['placed'], 'seconds': time.time() - t0,
            'reason': 'ran out of room',
            # who picked the agent's box, so the result screen can say whether
            # the two sides had the same freedom
            'picker': plabel or ('first in reach' if p['n_pick'] > 1 else None)}


def opponent(st, spec):
    """Replay the identical instance with a chosen policy.

    Cached on the session: the page asks for the comparison once when the bin
    is full and again on every redraw of the result screen, and a trained
    policy is far too slow to replay on each of those.
    """
    if st.get('opp_spec') == spec:
        return st['opp']
    out = hand_over(st, spec, st['p'])
    st['opp_spec'], st['opp'] = spec, out
    return out


def recalc_params(st, raw):
    """-> (params, errors) for a recalculation of this session.

    The stream fields are taken from the session and not from the request, so
    "the same boxes" is a guarantee the server makes rather than a promise the
    page is trusted to keep.  Everything else is validated exactly as the setup
    form is, because it ends up in the same `BPPBatch`.
    """
    raw = raw or {}
    p = dict(st['p'])
    p.update({k: raw[k] for k in RECALC_KEYS if k in raw})
    p.update({k: st['p'][k] for k in STREAM_KEYS})
    return validate(p)


def recalc(st, spec, p):
    """One row of the recalculation table: the same instance, other rules."""
    out = hand_over(st, spec, p)
    out.pop('placed')          # the table scores the rerun, it does not draw it
    out['params'] = p
    # what moved from the game that was played, and what that leaves the agent
    # standing on: a rerun can walk a policy off its training configuration
    # just as the setup form can, and for the same reason it has to say so
    out['changed'] = diff_params(p, st['p'], RECALC_KEYS)
    out['drift'] = mismatch(p, spec)
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
        u = urlparse(self.path)
        p = u.path.rstrip('/') or '/'
        if p in ('/', '/index.html'):
            with open(os.path.join(HERE, 'game.html'), 'rb') as f:
                return self._send(f.read(), 'text/html; charset=utf-8')
        if p == '/api/setup':
            runs = [r for r in A.list_runs() if A.run_info(r)]
            policies = [r for r in runs if not r.startswith(('att_', 'h'))]
            attackers = [r for r in runs
                         if r.startswith(('att_', 'h')) or r in policies]
            return self._json({'policies': policies, 'attackers': attackers,
                               'heuristics': list(A.HEURISTICS),
                               'defaults': defaults(),
                               'limits': {'axis': MAX_AXIS, 'nb': MAX_NB,
                                          'items': MAX_ITEMS}})
        if p == '/api/params':
            q = parse_qs(u.query)
            spec = (q.get('spec') or [''])[0]
            try:
                return self._json({'params': run_params(spec)})
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            return self._json({'error': 'bad request'}, 400)
        p = urlparse(self.path).path.rstrip('/')
        if p == '/api/check':
            params, err = validate(body.get('params'))
            return self._json({'params': params, 'errors': err,
                               'mismatch': mismatch(params, body.get('opponent')),
                               'att_mismatch': mismatch(params, body.get('attacker'))})
        if p == '/api/new':
            params, err = validate(body.get('params'))
            if err:
                return self._json({'error': err[0], 'errors': err}, 400)
            gid, st = new_game(body.get('seed', 0), body.get('attacker'),
                               params, body.get('human_pick', True))
            return self._json({'gid': gid, 'board': board(st),
                               'params': params,
                               'promoted': st['promoted'], 'alabel': st['alabel']})
        st = _GAMES.get(body.get('gid'))
        if st is None:
            return self._json({'error': 'that game has expired'}, 404)
        if p == '/api/select':
            if not st['human_pick']:
                return self._json({'error': 'the pick station is not yours to '
                                            'choose from in this game'}, 400)
            if not select(st, body.get('i', 0)):
                return self._json({'error': 'that box is out of reach'}, 400)
            return self._json({'board': board(st), 'promoted': st['promoted']})
        if p == '/api/place':
            try:
                r, x, y = (int(body[k]) for k in 'rxy')
            except (KeyError, TypeError, ValueError):
                return self._json({'error': 'a placement is r, x and y'}, 400)
            if not place(st['env'], r, x, y):
                return self._json({'error': 'the box will not rest there'}, 400)
            advance(st)
            return self._json({'board': board(st), 'promoted': st['promoted']})
        if p == '/api/agent':
            # a policy that fails to load must not take the result screen down
            # with it - the page needs a body it can show as an error
            try:
                return self._json(opponent(st, body['spec']))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
        if p == '/api/recalc':
            params, err = recalc_params(st, body.get('params'))
            if err:
                return self._json({'error': err[0], 'errors': err}, 400)
            try:
                return self._json(recalc(st, body.get('spec') or 'random',
                                         params))
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
    # These only seed the setup form: it is then filled from whichever run you
    # pick as the opponent, and editable either way.
    ap.add_argument('--bin', default=None,
                    help='bin extent: an int for a cube, or WxLxH')
    ap.add_argument('--size_hi', default=None,
                    help='item side cap: an int, or per-axis WxLxH')
    ap.add_argument('--max_l', type=int, default=None,
                    help='leaf cap; match the --max_l the opponent was trained with')
    ap.add_argument('--nb', type=int, default=None, help='observable window N_B')
    ap.add_argument('--n_pick', type=int, default=None,
                    help='how many of the N_B are within reach')
    a = ap.parse_args(argv)
    DEVICE = a.device

    def extent(v):
        try:
            q = [int(t) for t in str(v).lower().split('x')]
            if len(q) not in (1, 3):
                raise ValueError
            return [q[0]] * 3 if len(q) == 1 else q
        except ValueError:
            ap.error(f'expected an int or WxLxH, got {v!r}')

    if a.bin is not None:
        CLI['bin'] = extent(a.bin)
    if a.size_hi is not None:
        CLI['size_hi'] = extent(a.size_hi)
    for k in ('max_l', 'nb', 'n_pick'):
        if getattr(a, k) is not None:
            CLI[k] = getattr(a, k)

    srv = QuietServer((a.host, a.port), Handler)
    runs = [r for r in A.list_runs() if not r.startswith(('att_', 'h'))]
    d = defaults()
    print(f"game: http://{a.host}:{a.port}/   bin {'x'.join(map(str, d['bin']))}"
          f"   (opponents: {', '.join(runs[:6]) or 'heuristics only'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
