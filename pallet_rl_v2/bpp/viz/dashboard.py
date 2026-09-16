"""Live training dashboard.

    python3 -m bpp.viz.dashboard --port 8080

Serves a self-contained page that polls runs/<name>/log.jsonl, so you can watch
self-play utilisation, the held-out MCTS evaluation and the two losses converge
while training runs.  Overlay several runs by clicking their pills in the header.

The paper's own number for each (dataset, b, k) is drawn as a target line on the
utilisation charts, so "are we there yet" is answerable at a glance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..evaluate import BENCH

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

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

    One row per training iteration, so the file stays small - but it is
    append-only and polled every few seconds by a page that may have several
    runs selected, and the trainer wants the CPU more than the dashboard does.
    Parse only the bytes appended since the last call.
    """
    p = os.path.join(ROOT, 'runs', run, 'log.jsonl')
    try:
        size = os.path.getsize(p)
    except OSError:
        return []
    with _CACHE_LOCK:
        c = _CACHE.get(p)
        if c is None or size < c['off']:    # unseen, or truncated by a fresh run
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


def run_meta(run):
    d = os.path.join(ROOT, 'runs', run)
    try:
        args = json.load(open(os.path.join(d, 'args.json')))
    except Exception:
        args = {}
    key = (args.get('data'), int(args.get('b', 1)), int(args.get('k', 0)))
    bench = BENCH.get(key)
    return dict(run=run, data=args.get('data', '?'), b=key[1], k=key[2],
                iters=int(args.get('iters', 0)), games=int(args.get('games', 0)),
                sims=int(args.get('sims', 0)), c_puct=args.get('c_puct'),
                lr=args.get('lr'), batch=args.get('batch'),
                bench_items=bench[0] if bench else None,
                bench_util=bench[1] if bench else None,
                has_ckpt=os.path.exists(os.path.join(d, 'last.pt')))


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
            f = os.path.join(ROOT, 'results', 'summary.json')
            if os.path.exists(f):
                with open(f) as fh:
                    return self._send(fh.read())
            return self._send('{}')
        self.send_error(404)

    def log_message(self, *a):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
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
