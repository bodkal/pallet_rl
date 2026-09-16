"""Human vs. agent packing duel.

    python3 -m bpp.viz.game --port 8090      # -> http://127.0.0.1:8090

You pick the stream that generates the boxes (CUT-1 / CUT-2 / RS) and who you
want to play: the trained MCTS agent of Sec. V-C, the bare policy p_theta, or a
heuristic.  You then pack by hand under the paper's own rules - pick any item
out of the b-item buffer, optionally turn it 90 degrees when k=1, and drop it by
its front-left-bottom corner - until nothing fits any more.  The *same* sequence
is then packed by the agent and the two bins are scored side by side.

Both players face an identical stream: it is generated once from the game's seed
and the agent is handed that exact list.  They also face the same action space -
the mask the page draws is `env.compute_masks`, the one the agent searches over.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..datasets import gen_cut, gen_rs
from ..env import Cfg, combo_dims, compute_masks, new_state, step_v
from . import agents as A

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = A.ROOT
LOG = os.path.join(ROOT, 'runs', 'duel_log.jsonl')

DATASETS = ['cut1', 'cut2', 'rs']
DS_LABEL = {'cut1': 'CUT-1', 'cut2': 'CUT-2', 'rs': 'RS'}
_LOCK = threading.Lock()
GAMES: dict[str, 'Game'] = {}
DEFAULT_CFG = None


def default_cfg():
    return DEFAULT_CFG if DEFAULT_CFG is not None else Cfg()


def gen_one(dataset, seed, cfg):
    """One sequence of the chosen stream (Sec. IV), exactly as the trainer makes them."""
    if dataset == 'rs':
        s, ln = gen_rs(1, seed)
    else:
        s, ln = gen_cut(1, 1 if dataset == 'cut1' else 2, seed, cfg.W, cfg.L, cfg.H)
    return s[0].astype(np.int8), int(ln[0])


# ---------------------------------------------------------------------------
# one duel
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, dataset, opponent, seed, device, c_puct,
                 b=None, k=None, lookahead=0):
        self.id = secrets.token_hex(6)
        self.dataset = dataset
        self.opponent = opponent
        meta = A.opponent_meta(opponent, default_cfg())
        self.opponent_label = meta['label'] if meta else opponent
        self.opponent_note = meta['note'] if meta else ''
        self.run = A.opponent_run(opponent)
        # A trained net is locked to the action space it was trained on; the
        # heuristics read the mask, so they can pack any buffer or orientation.
        if self.run is None:
            d = default_cfg()
            self.cfg = Cfg(d.W, d.L, d.H,
                           b=int(b) if b else d.b, k=int(k) if k is not None else d.k)
            self.fixed = False
        else:
            self.cfg = A.config_for(opponent, default_cfg())
            self.fixed = True
        # Lookahead is pure information: you see these boxes, you cannot place
        # them.  Distinct from the buffer, which is what you may pick from.
        self.lookahead = max(0, int(lookahead))
        # MCTS rolls the true remaining stream out at every leaf, so it does know
        # what is coming - further than any lookahead we would grant the human.
        self.searches = A.searches(opponent)
        self.seed = int(seed)
        self.device = device
        self.c_puct = c_puct

        c = self.cfg
        self.seq, self.len = gen_one(dataset, self.seed, c)
        self.seqs = self.seq[None]
        self.lens = np.array([self.len], np.int32)
        self.hm, self.buf, self.nxt = new_state(self.seqs, self.lens, c)
        self.trace: list[dict] = []
        self.G = 0.0
        self.done = False
        self.reason = None
        self.t0 = time.time()
        self.elapsed = 0.0
        self.agent = None
        self._refresh()

    # -- state ----------------------------------------------------------
    def _refresh(self):
        c = self.cfg
        self.dims = combo_dims(self.buf, c.k)
        mask, self.zmap = compute_masks(self.hm, self.dims, c.H)
        self.mf = mask.reshape(1, -1)
        if not self.mf.any():
            self._finish('sequence_end' if len(self.trace) >= self.len
                         else 'no_feasible')

    def _finish(self, reason):
        if not self.done:
            self.done = True
            self.reason = reason
            self.elapsed = time.time() - self.t0

    def resign(self):
        self._finish('resigned')

    def place(self, slot, orient, x, y):
        c = self.cfg
        if self.done:
            return 'the game is already over'
        if not (0 <= slot < c.b and 0 <= orient <= c.k
                and 0 <= x < c.W and 0 <= y < c.L):
            return 'that is not a move in this action space'
        m = slot * (c.k + 1) + orient
        a = m * c.WL + x * c.L + y
        if not self.mf[0, a]:
            return 'that position is not legal for this box'
        A._record(self.trace, a, self.dims, self.zmap, self.buf, c)
        self.G += float(step_v(self.hm, self.buf, self.nxt, self.seqs, self.lens,
                               self.dims, self.zmap, np.array([a], np.int64), c)[0])
        self._refresh()
        return None

    @property
    def util(self):
        return self.G / 10.0

    def placed(self):
        return [[t['x'], t['y'], t['z'], t['w'], t['l'], t['h']] for t in self.trace]

    def result(self):
        return dict(utilization=self.util, n_items=len(self.trace),
                    reason=self.reason, placed=self.placed(),
                    seconds=self.elapsed or (time.time() - self.t0))

    def state(self):
        c = self.cfg
        nx = int(self.nxt[0])
        return dict(
            ok=True, game=self.id, dataset=self.dataset,
            dataset_label=DS_LABEL[self.dataset],
            opponent=self.opponent, opponent_label=self.opponent_label,
            opponent_note=self.opponent_note, searches=self.searches,
            seed=self.seed, W=c.W, L=c.L, H=c.H, b=c.b, k=c.k,
            hmap=self.hm[0].T.tolist(),                 # [y][x] for the canvas
            buf=[[int(v) for v in it] for it in self.buf[0]],
            mask=[int(v) for v in self.mf[0]],
            placed=self.placed(),
            fixed=self.fixed, lookahead=self.lookahead,
            ahead=[[int(v) for v in it]
                   for it in self.seq[nx:nx + self.lookahead]],
            seq_len=self.len, n_items=len(self.trace),
            remaining=self.len - len(self.trace), to_come=self.len - nx,
            utilization=self.util, done=self.done, reason=self.reason,
            seconds=self.elapsed or (time.time() - self.t0),
            agent=self.agent)

    # -- the opponent's turn ---------------------------------------------
    def run_agent(self):
        out = A.play_opponent(self.opponent, self.cfg, self.seq, self.len,
                              device=self.device, seed=self.seed, c_puct=self.c_puct)
        self.agent = dict(label=self.opponent_label, **out)
        return self.agent


def log_duel(g):
    h = g.result()
    row = dict(t=time.strftime('%Y-%m-%d %H:%M:%S'), dataset=g.dataset,
               opponent=g.opponent, opponent_label=g.opponent_label,
               seed=g.seed, seq_len=g.len, b=g.cfg.b, k=g.cfg.k,
               lookahead=g.lookahead,
               W=g.cfg.W, L=g.cfg.L, H=g.cfg.H,
               human_util=h['utilization'], human_items=h['n_items'],
               human_reason=h['reason'], human_seconds=round(h['seconds'], 1),
               agent_util=g.agent['utilization'], agent_items=g.agent['n_items'],
               agent_reason=g.agent['reason'],
               agent_seconds=round(g.agent['seconds'], 2))
    row['winner'] = ('human' if row['human_util'] > row['agent_util'] + 1e-9 else
                     'agent' if row['agent_util'] > row['human_util'] + 1e-9 else 'draw')
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a') as f:
        f.write(json.dumps(row) + '\n')
    return row


def history(n=50):
    if not os.path.exists(LOG):
        return []
    rows = []
    with open(LOG) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows[-n:][::-1]


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
class QuietServer(ThreadingHTTPServer):
    """A browser refresh aborts the in-flight response; that is not an error."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    device = 'cuda'
    c_puct = 4.0

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

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}')

    def _game(self, gid):
        g = GAMES.get(gid or '')
        if g is None:
            self._json(dict(ok=False, error='no such game - start a new one'), 404)
        return g

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path.rstrip('/') or '/'
        if p in ('/', '/index.html'):
            with open(os.path.join(HERE, 'game.html'), 'rb') as f:
                return self._send(f.read(), 'text/html; charset=utf-8')
        if p == '/api/setup':
            return self._json(dict(ok=True, datasets=DATASETS, labels=DS_LABEL,
                                   opponents=A.opponents(default_cfg()),
                                   history=history(12)))
        if p == '/api/state':
            g = self._game(q.get('game', [''])[0])
            return None if g is None else self._json(g.state())
        if p == '/api/history':
            return self._json(dict(ok=True, history=history(50)))
        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path.rstrip('/')
        try:
            b = self._body()
        except Exception as e:
            return self._json(dict(ok=False, error=f'bad request: {e}'), 400)

        if p == '/api/new':
            ds = b.get('dataset', 'cut2')
            opp = b.get('opponent', 'lowest')
            if ds not in DATASETS:
                return self._json(dict(ok=False, error=f'unknown stream {ds}'), 400)
            if A.opponent_meta(opp, default_cfg()) is None:
                return self._json(dict(ok=False, error=f'unknown opponent {opp}'), 400)
            seed = b.get('seed')
            seed = int(secrets.randbelow(10 ** 6)) if seed in (None, '') else int(seed)
            try:
                g = Game(ds, opp, seed, self.device, self.c_puct,
                         b=b.get('b'), k=b.get('k'),
                         lookahead=b.get('lookahead', 0))
            except Exception as e:
                return self._json(dict(ok=False, error=f'{type(e).__name__}: {e}'), 500)
            with _LOCK:
                GAMES[g.id] = g
                for k in list(GAMES)[:-40]:      # keep the last 40 games in memory
                    GAMES.pop(k, None)
            return self._json(g.state())

        if p == '/api/place':
            g = self._game(b.get('game'))
            if g is None:
                return
            err = g.place(int(b.get('slot', 0)), int(b.get('o', 0)),
                          int(b.get('x', -1)), int(b.get('y', -1)))
            s = g.state()
            s['ok'] = err is None
            if err:
                s['error'] = err
            return self._json(s)

        if p == '/api/resign':
            g = self._game(b.get('game'))
            if g is None:
                return
            g.resign()
            return self._json(g.state())

        if p == '/api/agent':
            g = self._game(b.get('game'))
            if g is None:
                return
            if not g.done:
                return self._json(dict(ok=False, error='finish your packing first'), 400)
            try:
                if g.agent is None:
                    g.run_agent()
                    log_duel(g)
            except Exception as e:
                return self._json(dict(ok=False, error=f'{type(e).__name__}: {e}'), 500)
            s = g.state()
            s['human'] = g.result()
            return self._json(s)

        self.send_error(404)

    def log_message(self, *a):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description='human vs. agent 3D bin-packing duel')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--c_puct', type=float, default=4.0)
    ap.add_argument('--bin', default=None, metavar='N|WxLxH',
                    help='bin for the HEURISTIC opponents, e.g. 12 (default 10). '
                         'A trained opponent always uses the geometry in its own '
                         'checkpoint, so this does not affect those.')
    ap.add_argument('--b', type=int, default=1, help='buffer size for the heuristics')
    ap.add_argument('--k', type=int, default=0, help='extra orientations for the heuristics')
    a = ap.parse_args(argv)
    parts = [10, 10, 10]
    if a.bin:
        try:
            parts = [int(v) for v in a.bin.lower().split('x')]
        except ValueError:
            ap.error(f'--bin: expected N or WxLxH, got {a.bin!r}')
        if len(parts) == 1:
            parts *= 3
        if len(parts) != 3 or min(parts) < 2:
            ap.error(f'--bin: expected N or WxLxH with every dim >= 2, got {a.bin!r}')
    globals()['DEFAULT_CFG'] = Cfg(*parts, b=a.b, k=a.k)
    Handler.device, Handler.c_puct = a.device, a.c_puct
    opps = A.opponents(default_cfg())
    srv = QuietServer((a.host, a.port), Handler)
    print(f'duel: http://{a.host}:{a.port}/')
    print(f'      {len(opps)} opponents '
          f"({sum(o['run'] is not None for o in opps)} trained, "
          f"{sum(o['run'] is None for o in opps)} heuristic)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
