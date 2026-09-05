"""Human vs. agent packing duel.

    python -m src.viz.game --port 8090      # -> http://127.0.0.1:8090

You pick the benchmark that generates the box stream (RS / CUT-1 / CUT-2) and
which agent you want to play against (the trained BPP-1 network, BPP-k MCTS, or
one of the heuristics).  You then pack the boxes by hand, one at a time, in the
order they arrive, until nothing fits any more.  The *same* sequence is then
replayed by the agent and the two packings are scored side by side.

Both players face an identical stream: the sequence is generated once from the
game's seed and stored, the agent's environment is handed that exact list.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import sys
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOG = os.path.join(ROOT, "runs", "duel_log.jsonl")

DATASETS = ["RS", "CUT-1", "CUT-2"]
_LOCK = threading.Lock()
GAMES: dict[str, "Game"] = {}
_NETS: dict[tuple, tuple] = {}


# ---------------------------------------------------------------------------
# what you can play against
# ---------------------------------------------------------------------------
def last_step(run_dir):
    """Steps trained so far, from the tail of metrics.jsonl.

    The file is append-only and reaches ~55 MB over a long run, and this is
    called for every run on every /api/setup, so read the last few KB rather
    than parsing all of it.
    """
    p = os.path.join(run_dir, "metrics.jsonl")
    try:
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return 0
    for line in reversed(tail.splitlines()):   # last COMPLETE line wins
        line = line.strip()
        if not line:
            continue
        try:
            return int(json.loads(line)["step"])
        except Exception:
            continue
    return 0


def runs_with_checkpoint():
    d = os.path.join(ROOT, "runs")
    out = []
    if not os.path.isdir(d):
        return out
    for r in sorted(os.listdir(d)):
        rd = os.path.join(d, r)
        if not os.path.exists(os.path.join(rd, "config.json")):
            continue
        if not (os.path.exists(os.path.join(rd, "best.pt")) or
                os.path.exists(os.path.join(rd, "latest.pt"))):
            continue
        try:
            cfg = json.load(open(os.path.join(rd, "config.json")))
        except Exception:
            continue
        out.append(dict(run=r, L=cfg["L"], W=cfg["W"], H=cfg["H"],
                        dataset=cfg["dataset"], orientations=cfg["orientations"],
                        step=last_step(rd), total_steps=cfg.get("total_steps", 0)))
    return out


def opponents():
    """Every playable opponent id, most interesting first."""
    out = []
    for r in runs_with_checkpoint():
        geom = {k: r[k] for k in ("L", "W", "H", "step", "total_steps")}
        # `dataset` is the stream this network was trained on - the setup screen
        # only offers it for that stream unless you ask for all of them
        out.append(dict(id=f"bpp1:{r['run']}", label=f"BPP-1 agent ({r['run']})",
                        note="the trained network, greedy, one box of lookahead",
                        run=r["run"], dataset=r["dataset"], lookahead=1,
                        orientations=r["orientations"], can_rotate=False, **geom))
        for k in (3, 5):
            out.append(dict(id=f"bppk:{r['run']}:{k}",
                            label=f"BPP-{k} MCTS ({r['run']})",
                            note=f"same network + permutation tree search over {k} boxes",
                            run=r["run"], dataset=r["dataset"], lookahead=k,
                            orientations=r["orientations"], can_rotate=False, **geom))
    d = default_cfg()
    hgeom = dict(L=d.L, W=d.W, H=d.H)
    out.append(dict(lookahead=1, orientations=1, can_rotate=True, dataset=None, id="boundary", label="boundary rule",
                    note="the paper's spare-cuboid heuristic (slow, strong)", run=None, **hgeom))
    out.append(dict(lookahead=1, orientations=1, can_rotate=True, dataset=None, id="dbl", label="deepest-bottom-left",
                    note="classic online placement rule", run=None, **hgeom))
    out.append(dict(lookahead=1, orientations=1, can_rotate=True, dataset=None, id="random", label="random feasible",
                    note="uniform over legal positions - the floor", run=None, **hgeom))
    return out


def opponent_run(opp: str):
    p = opp.split(":")
    return p[1] if p[0] in ("bpp1", "bppk") else None


def opponent_lookahead(opp: str) -> int:
    """How many boxes the opponent gets to see at once.  BPP-k sees k, everything
    else sees only the box in front of it.  The human is given exactly this."""
    p = opp.split(":")
    return int(p[2]) if p[0] == "bppk" else 1


def opponent_label(opp: str):
    for o in opponents():
        if o["id"] == opp:
            return o["label"]
    return opp


# Geometry for the heuristic opponents, which have no checkpoint to read one
# from.  Set by main() via --bin; the paper's 10^3 unless asked otherwise.
DEFAULT_CFG = None


def default_cfg():
    from ..config import build
    return DEFAULT_CFG if DEFAULT_CFG is not None else build("paper")


def config_for(run):
    if run:
        from ..config import Config
        return Config.from_json(os.path.join(ROOT, "runs", run, "config.json"))
    return default_cfg()


def build_policy(cfg, opp: str, device: str, sims: int):
    """An evaluate.py-style policy: policy(bin, obs) -> action index or None."""
    from ..baselines import BASELINES
    kind, *rest = opp.split(":")
    if kind in BASELINES:
        return BASELINES[kind](cfg)
    from ..evaluate import MCTSPolicy, NetPolicy, load_run
    run = rest[0]
    key = (run, device)
    with _LOCK:
        if key not in _NETS:
            _NETS[key] = load_run(run, "best.pt", device)[:3]
        rcfg, net, dev = _NETS[key]
    if kind == "bpp1":
        return NetPolicy(cfg, net, dev)
    if kind == "bppk":
        return MCTSPolicy(cfg, net, dev, int(rest[1]), sims)
    raise KeyError(f"unknown opponent {opp!r}")


# ---------------------------------------------------------------------------
# one duel
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, dataset, opponent, seed, device, sims, rotate=False):
        from ..bin3d import Bin3D
        from ..items import gen_sequence

        self.id = secrets.token_hex(6)
        self.dataset = dataset
        self.opponent = opponent
        self.opponent_label = opponent_label(opponent)
        self.run = opponent_run(opponent)
        self.cfg = config_for(self.run)
        # a trained net is locked to the action space it was trained on; the
        # heuristics read the mask, so they can pack either way
        if self.run is None and rotate:
            self.cfg.orientations = 2
        self.seed = int(seed)
        # you see exactly what the opponent sees: its lookahead minus the box
        # you are placing right now
        self.lookahead = opponent_lookahead(opponent)
        self.preview = self.lookahead - 1
        # BPP-k also reads the box just past its window and feeds it to the critic
        # at the end of each rollout (env.peek_next_after_lookahead).  It can
        # neither place nor reorder that box - it only values the bin knowing what
        # is coming.  BPP-1 short-circuits before this, and 'mean' opts out.
        self.peeks = self.lookahead > 1 and self.cfg.mcts_last_item == "observed"
        self.device = device
        self.sims = sims

        c = self.cfg
        rng = np.random.default_rng(self.seed)
        self.seq = [tuple(int(v) for v in d) for d in
                    gen_sequence(rng, c.L, c.W, c.H, c.item_min, c.item_max, dataset)]
        self.bin = Bin3D(c.L, c.W, c.H)
        self.idx = 0
        self.moves: list[dict] = []
        self.done = False
        self.reason = None
        self.t0 = time.time()
        self.elapsed = 0.0
        self.agent = None
        self._settle()

    # -- state ----------------------------------------------------------
    @property
    def item(self):
        return self.seq[self.idx] if self.idx < len(self.seq) else None

    def mask(self):
        it = self.item
        if it is None:
            return np.zeros(self.cfg.action_dim, dtype=np.float32)
        return self.bin.feasibility_mask(it, self.cfg.orientations)

    def _settle(self):
        """End the game when the stream runs dry or nothing fits any more."""
        if self.idx >= len(self.seq):
            self._finish("sequence_end")
        elif self.mask().sum() == 0:
            self._finish("no_feasible")

    def _finish(self, reason):
        if not self.done:
            self.done = True
            self.reason = reason
            self.elapsed = time.time() - self.t0

    def resign(self):
        self._finish("resigned")

    def place(self, x, y, o):
        c = self.cfg
        if self.done:
            return "the game is already over"
        it = self.item
        if not (0 <= x < c.L and 0 <= y < c.W and 0 <= o < c.orientations):
            return "off the pallet"
        a = o * c.L * c.W + x + c.L * y
        if not self.bin.is_feasible(a, it, c.orientations):
            return "that position is not legal for this box"
        z, l, w, h = self.bin.place(a, it, c.orientations)
        self.moves.append(dict(item=list(it), pos=[x, y, int(z)], dims=[l, w, h]))
        self.idx += 1
        self._settle()
        return None

    def result(self):
        return dict(utilization=self.bin.utilization, n_items=len(self.bin.placed),
                    reason=self.reason,
                    placed=[list(map(int, p)) for p in self.bin.placed],
                    seconds=self.elapsed or (time.time() - self.t0))

    def peek_item(self):
        """The box the opponent peeks at but may not place, if there is one."""
        j = self.idx + self.lookahead
        return list(self.seq[j]) if self.peeks and j < len(self.seq) else None

    def state(self):
        c = self.cfg
        m = self.mask()
        q = self.seq[self.idx + 1: self.idx + 1 + self.preview]
        return dict(
            ok=True, game=self.id, dataset=self.dataset,
            opponent=self.opponent, opponent_label=self.opponent_label,
            seed=self.seed, preview=self.preview, lookahead=self.lookahead,
            L=c.L, W=c.W, H=c.H, orientations=c.orientations,
            idx=self.idx, seq_len=len(self.seq), remaining=len(self.seq) - self.idx,
            hmap=self.bin.hmap.T.tolist(),               # [y][x], row-major for the UI
            placed=[list(map(int, p)) for p in self.bin.placed],
            item=list(self.item) if self.item else None,
            queue=[list(v) for v in q], peek=self.peek_item(), peeks=self.peeks,
            mask=[int(v) for v in m],
            utilization=self.bin.utilization, n_items=len(self.bin.placed),
            done=self.done, reason=self.reason,
            seconds=self.elapsed or (time.time() - self.t0),
            agent=self.agent,
        )

    # -- the opponent's turn ---------------------------------------------
    def run_agent(self):
        from ..env import PackingEnv
        from ..evaluate import rollout

        cfg = copy.deepcopy(self.cfg)
        kind, *rest = self.opponent.split(":")
        cfg.lookahead_k = int(rest[1]) if kind == "bppk" else 1
        policy = build_policy(cfg, self.opponent, self.device, self.sims)
        env = PackingEnv(cfg, dataset=self.dataset, sequences=[self.seq], seed=0)
        t0 = time.time()
        rec = rollout(env, policy, record=True)
        self.agent = dict(
            label=self.opponent_label,
            utilization=rec["utilization"], n_items=rec["n_items"],
            reason=rec["reason"], seconds=time.time() - t0,
            placed=[list(map(int, p)) for p in rec["placed"]],
            moves=[dict(item=t["item"], pos=t["pos"], dims=t["dims"])
                   for t in rec["trace"]],
        )
        return self.agent


def log_duel(g):
    human = g.result()
    row = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), dataset=g.dataset,
               opponent=g.opponent, opponent_label=g.opponent_label,
               seed=g.seed, lookahead=g.lookahead, peeks=g.peeks,
               seq_len=len(g.seq), orientations=g.cfg.orientations,
               L=g.cfg.L, W=g.cfg.W, H=g.cfg.H,
               human_util=human["utilization"], human_items=human["n_items"],
               human_reason=human["reason"], human_seconds=round(human["seconds"], 1),
               agent_util=g.agent["utilization"], agent_items=g.agent["n_items"],
               agent_reason=g.agent["reason"])
    row["winner"] = ("human" if row["human_util"] > row["agent_util"] + 1e-9 else
                     "agent" if row["agent_util"] > row["human_util"] + 1e-9 else "draw")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def history(n=50):
    if not os.path.exists(LOG):
        return []
    rows = []
    with open(LOG) as f:
        for line in f:
            line = line.strip()
            if line:
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
    device = "cuda"
    sims = 100

    def _send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass          # the page navigated away mid-response; nothing to do

    def _json(self, obj, code=200):
        self._send(json.dumps(obj), code=code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _game(self, gid):
        g = GAMES.get(gid or "")
        if g is None:
            self._json(dict(ok=False, error="no such game - start a new one"), 404)
        return g

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path.rstrip("/") or "/"
        if p in ("/", "/index.html"):
            with open(os.path.join(HERE, "game.html"), "rb") as f:
                return self._send(f.read(), "text/html; charset=utf-8")
        if p == "/api/setup":
            return self._json(dict(ok=True, datasets=DATASETS,
                                   opponents=opponents(), history=history(12)))
        if p == "/api/state":
            g = self._game(q.get("game", [""])[0])
            return None if g is None else self._json(g.state())
        if p == "/api/history":
            return self._json(dict(ok=True, history=history(50)))
        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path.rstrip("/")
        try:
            b = self._body()
        except Exception as e:
            return self._json(dict(ok=False, error=f"bad request: {e}"), 400)

        if p == "/api/new":
            ds = b.get("dataset", "CUT-2")
            opp = b.get("opponent", "dbl")
            if ds not in DATASETS:
                return self._json(dict(ok=False, error=f"unknown dataset {ds}"), 400)
            if opp not in {o["id"] for o in opponents()}:
                return self._json(dict(ok=False, error=f"unknown opponent {opp}"), 400)
            seed = b.get("seed")
            seed = int(secrets.randbelow(10 ** 6)) if seed in (None, "") else int(seed)
            try:
                g = Game(ds, opp, seed, self.device, self.sims,
                         rotate=bool(b.get("rotate")))
            except Exception as e:
                return self._json(dict(ok=False, error=f"{type(e).__name__}: {e}"), 500)
            with _LOCK:
                GAMES[g.id] = g
                for k in list(GAMES)[:-40]:      # keep the last 40 games in memory
                    GAMES.pop(k, None)
            return self._json(g.state())

        if p == "/api/place":
            g = self._game(b.get("game"))
            if g is None:
                return
            err = g.place(int(b.get("x", -1)), int(b.get("y", -1)), int(b.get("o", 0)))
            s = g.state()
            s["ok"] = err is None
            if err:
                s["error"] = err
            return self._json(s)

        if p == "/api/resign":
            g = self._game(b.get("game"))
            if g is None:
                return
            g.resign()
            return self._json(g.state())

        if p == "/api/agent":
            g = self._game(b.get("game"))
            if g is None:
                return
            if not g.done:
                return self._json(dict(ok=False,
                                       error="finish your packing first"), 400)
            try:
                if g.agent is None:
                    g.run_agent()
                    log_duel(g)
            except Exception as e:
                return self._json(dict(ok=False, error=f"{type(e).__name__}: {e}"), 500)
            s = g.state()
            s["human"] = g.result()
            return self._json(s)

        self.send_error(404)

    def log_message(self, *a):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="human vs. agent 3D bin-packing duel")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mcts-sims", type=int, default=100)
    ap.add_argument("--bin", default=None, metavar="N|LxWxH",
                    help="bin for the HEURISTIC opponents, e.g. 15 (default 10). "
                         "A trained opponent always uses the geometry in its own "
                         "config.json, so this does not affect those.")
    a = ap.parse_args(argv)
    if a.bin:
        from ..config import build as _build
        try:
            parts = [int(v) for v in a.bin.lower().split("x")]
        except ValueError:
            ap.error(f"--bin: expected N or LxWxH, got {a.bin!r}")
        if len(parts) == 1:
            parts *= 3
        if len(parts) != 3 or min(parts) < 2:
            ap.error(f"--bin: expected N or LxWxH with every dim >= 2, got {a.bin!r}")
        globals()["DEFAULT_CFG"] = _build("paper", L=parts[0], W=parts[1], H=parts[2])
    Handler.device, Handler.sims = a.device, a.mcts_sims
    opps = opponents()
    srv = QuietServer((a.host, a.port), Handler)
    print(f"duel: http://{a.host}:{a.port}/")
    print(f"      {len(opps)} opponents "
          f"({sum(o['run'] is not None for o in opps)} trained, "
          f"{sum(o['run'] is None for o in opps)} heuristic)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
