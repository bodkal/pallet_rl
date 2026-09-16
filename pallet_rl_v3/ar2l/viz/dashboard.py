"""Live training dashboard.

    python3 -m ar2l.viz.dashboard --port 8095

Serves a self-contained page that polls runs/<name>/log.jsonl, so you can watch
the held-out nominal utilisation, the utilisation of whatever dynamics each
algorithm is actually training on, and the losses converge while the grid runs.
Overlay runs by clicking their pills; click a card for a big plot that reads
every series at whichever iteration you hover.

The paper's number for that (method, N_B) cell is drawn as a dashed target
line, so "are we there yet" is answerable at a glance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

# AR2L Table 2, discrete setting: nominal (beta=0) and fully attacked (beta=100)
PAPER = {
    ('pct', 5): (76.2, 64.9), ('cppo', 5): (75.5, 65.2),
    ('rarl', 5): (74.6, 66.7), ('rfmdp', 5): (75.5, 63.9),
    ('ex05', 5): (76.5, 65.3), ('ex10', 5): (77.4, 66.5),
    ('ap05', 5): (76.1, 64.7), ('ap10', 5): (76.5, 65.7),
    ('pct', 10): (76.4, 55.7), ('cppo', 10): (75.6, 57.4),
    ('rarl', 10): (74.3, 63.3), ('rfmdp', 10): (74.4, 55.9),
    ('ex05', 10): (77.6, 59.7), ('ex10', 10): (76.0, 63.8),
    ('ap05', 10): (76.2, 56.1), ('ap10', 10): (73.6, 57.1),
    ('pct', 15): (76.8, 48.6), ('cppo', 15): (75.2, 52.3),
    ('rarl', 15): (73.2, 58.7), ('rfmdp', 15): (73.6, 54.1),
    ('ex05', 15): (77.7, 55.0), ('ex10', 15): (76.6, 58.5),
    ('ap05', 15): (75.2, 55.1), ('ap10', 15): (73.5, 55.8),
    ('pct', 20): (77.0, 41.9), ('cppo', 20): (74.1, 45.8),
    ('rarl', 20): (72.0, 58.7), ('rfmdp', 20): (73.8, 54.4),
    ('ex05', 20): (76.8, 54.4), ('ex10', 20): (76.1, 58.5),
    ('ap05', 20): (75.0, 53.1), ('ap10', 20): (73.4, 57.6),
}

_CACHE: dict = {}                 # path -> incremental parse state
_CACHE_LOCK = threading.Lock()    # ThreadingHTTPServer: polls can overlap


def list_runs():
    d = os.path.join(ROOT, 'runs')
    if not os.path.isdir(d):
        return []
    out = [r for r in sorted(os.listdir(d))
           if os.path.exists(os.path.join(d, r, 'log.jsonl'))]
    out.sort(key=lambda r: -os.path.getmtime(os.path.join(d, r, 'log.jsonl')))
    return out


def read_log(run):
    """Rows of runs/<run>/log.jsonl, parsed incrementally.

    One row per logging interval, so the file stays small - but it is
    append-only and polled every few seconds by a page that may have a dozen
    runs selected, while six trainers want the CPU more than the dashboard
    does.  Parse only the bytes appended since the last call.
    """
    p = os.path.join(ROOT, 'runs', run, 'log.jsonl')
    try:
        size = os.path.getsize(p)
    except OSError:
        return []
    with _CACHE_LOCK:
        c = _CACHE.get(p)
        if c is None or size < c['off']:    # unseen, or truncated by a rerun
            c = {'off': 0, 'rows': [], 'tail': ''}
            _CACHE[p] = c
        if size > c['off']:
            with open(p, 'rb') as f:
                f.seek(c['off'])
                chunk = f.read()
                c['off'] = f.tell()
            parts = (c['tail'] + chunk.decode('utf-8', 'replace')).split('\n')
            c['tail'] = parts.pop()         # trailing '' or a half-written line
            for line in parts:
                if not line.strip():
                    continue
                try:
                    c['rows'].append(json.loads(line))
                except json.JSONDecodeError:
                    continue                # partially written, retry next poll
        return list(c['rows'])


def method_of(run):
    """runs/ex10_nb20 -> 'ex10'; runs/att_pct_nb10 -> None (it is an attacker)."""
    base = run[4:] if run.startswith('att_') else run
    return base.rsplit('_nb', 1)[0] if '_nb' in base else None


def run_meta(run):
    d = os.path.join(ROOT, 'runs', run)
    try:
        args = json.load(open(os.path.join(d, 'args.json')))
    except Exception:
        args = {}
    rows = read_log(run)
    last = rows[-1] if rows else {}
    nb = args.get('nb')
    bench = PAPER.get((method_of(run), nb)) if args.get('algo') != 'attack' else None
    return dict(run=run, algo=args.get('algo', '?'), nb=nb,
                alpha=args.get('alpha'), rho=args.get('rho'),
                iters=int(args.get('iters', 0)), it=int(last.get('it', 0)),
                nom_util=last.get('nom_util', 0.0), nom_items=last.get('nom_items', 0.0),
                target=bench[0] if bench else None,
                has_ckpt=os.path.exists(os.path.join(d, 'best.pt')))


def summary():
    """results/table2.json, annotated with the paper's numbers for the page."""
    f = os.path.join(ROOT, 'results', 'table2.json')
    if not os.path.exists(f):
        return {}
    try:
        res = json.load(open(f))
    except json.JSONDecodeError:
        return {}
    for k, v in res.items():
        p = PAPER.get((v.get('method'), v.get('nb')))
        if p:
            v['paper'] = {'b0': p[0], 'b100': p[1]}
    return res


class QuietServer(ThreadingHTTPServer):
    """A browser refresh aborts the in-flight response; that is not an error."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(b)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass          # the page navigated away mid-response; nothing to do

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path.rstrip('/') or '/'
        if p in ('/', '/index.html'):
            with open(os.path.join(HERE, 'dashboard.html'), 'rb') as f:
                return self._send(f.read(), 'text/html; charset=utf-8')
        if p == '/runs':
            return self._send(json.dumps([run_meta(r) for r in list_runs()]))
        if p == '/log':
            return self._send(json.dumps(read_log(q.get('run', [''])[0])))
        if p == '/summary':
            return self._send(json.dumps(summary()))
        self.send_error(404)

    def log_message(self, *a):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8095)
    ap.add_argument('--host', default='127.0.0.1')
    a = ap.parse_args(argv)
    srv = QuietServer((a.host, a.port), Handler)
    print(f"dashboard: http://{a.host}:{a.port}/   "
          f"(runs: {', '.join(list_runs()) or 'none yet'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
