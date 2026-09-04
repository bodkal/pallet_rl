"""Live training dashboard.

    python -m src.viz.dashboard --port 8080

Serves a self-contained page that polls runs/<name>/metrics.jsonl, so you can
watch space utilisation, the mask predictor and the invalid-action rate converge
while training runs.  Overlay several runs (e.g. the MP/MC/FE ablation) by
clicking their pills in the header.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def list_runs():
    d = os.path.join(ROOT, "runs")
    if not os.path.isdir(d):
        return []
    out = [r for r in sorted(os.listdir(d))
           if os.path.exists(os.path.join(d, r, "metrics.jsonl"))]
    out.sort(key=lambda r: -os.path.getmtime(os.path.join(d, r, "metrics.jsonl")))
    return out


_CACHE: dict = {}                 # path -> incremental parse state
_CACHE_LOCK = threading.Lock()    # ThreadingHTTPServer: polls can overlap
_KEEP = 4800                      # decimation bound: ~9 MB/run, not ~140 MB


def read_metrics(run, max_points=1200):
    """Rows from runs/<run>/metrics.jsonl, parsed incrementally.

    The file is append-only and reaches ~55 MB / 78k rows over a 100M-step run.
    Re-parsing all of it on every 4 s poll cost 645 ms of CPU per run, and
    pipeline_queue() does it for all 8 scripted stages whatever you have
    selected -- together about a core.  That matters because each training
    process only gets one core (D.1), so the dashboard was competing with the
    thing it is watching.  Parse only the bytes appended since the last call,
    and keep a decimated history so memory stays bounded however long the run.
    """
    p = os.path.join(ROOT, "runs", run, "metrics.jsonl")
    try:
        size = os.path.getsize(p)
    except OSError:
        return []
    with _CACHE_LOCK:
        c = _CACHE.get(p)
        if c is None or size < c["off"]:   # unseen, or truncated by a fresh run
            c = {"off": 0, "rows": [], "tail": "", "stride": 1, "seen": 0, "last": None}
            _CACHE[p] = c
        if size > c["off"]:
            with open(p, "rb") as f:
                f.seek(c["off"])
                chunk = f.read()
                c["off"] = f.tell()
            parts = (c["tail"] + chunk.decode("utf-8", "replace")).split("\n")
            c["tail"] = parts.pop()        # trailing "" or a half-written line
            for line in parts:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue               # partially written line, retry next poll
                c["last"] = r
                if c["seen"] % c["stride"] == 0:
                    c["rows"].append(r)
                c["seen"] += 1
                if len(c["rows"]) > _KEEP:  # halve, and thin future appends too
                    c["rows"] = c["rows"][::2]
                    c["stride"] *= 2
        rows, last = list(c["rows"]), c["last"]
    if last is not None and (not rows or rows[-1] is not last):
        rows.append(last)                  # the newest point is always shown
    if len(rows) > max_points:
        step = len(rows) / max_points
        rows = [rows[int(i * step)] for i in range(max_points - 1)] + [rows[-1]]
    return rows


PIPELINE_STAGES = [
    ("bpp1_cut2", "BPP-1 CUT-2"), ("bpp1_cut1", "BPP-1 CUT-1"),
    ("bpp1_rs", "BPP-1 RS"), ("bpp1_orient_rs", "re-orientation RS"),
    ("abl_none", "ablation --none"), ("abl_mp_fe", "ablation MP+FE"),
    ("abl_mp_mc", "ablation MP+MC"), ("abl_mc_fe", "ablation MC+FE"),
]


def pipeline_queue():
    """Which scripted stages are done / running / still queued."""
    out = []
    for run, label in PIPELINE_STAGES:
        d = os.path.join(ROOT, "runs", run)
        done = os.path.exists(os.path.join(d, "latest.pt"))
        started = os.path.exists(os.path.join(d, "metrics.jsonl"))
        rows = read_metrics(run, 4) if started else []
        cfg = {}
        f = os.path.join(d, "config.json")
        if os.path.exists(f):
            try:
                cfg = json.load(open(f))
            except Exception:
                pass
        total = cfg.get("total_steps", 0)
        step = rows[-1]["step"] if rows else 0
        finished = done and total and step >= total * 0.995
        out.append(dict(run=run, label=label, step=step, total=total,
                        state=("done" if finished else
                               "running" if started else "queued")))
    return out


class QuietServer(ThreadingHTTPServer):
    """A browser refresh aborts the in-flight response; that is not an error."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass          # the page navigated away mid-response; nothing to do

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                return self._send(f.read(), "text/html; charset=utf-8")
        if u.path.lstrip("/") == "runs":
            return self._send(json.dumps(list_runs()))
        if u.path.lstrip("/") == "config":
            run = q.get("run", [""])[0]
            f = os.path.join(ROOT, "runs", run, "config.json")
            if os.path.exists(f):
                with open(f) as fh:
                    return self._send(fh.read())
            return self._send("{}")
        if u.path.lstrip("/") == "queue":
            return self._send(json.dumps(pipeline_queue()))
        if u.path.lstrip("/") == "metrics":
            run = q.get("run", [""])[0]
            return self._send(json.dumps(read_metrics(run)))
        if u.path.lstrip("/") == "report":
            p = os.path.join(ROOT, "runs", q.get("run", [""])[0], "report.html")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return self._send(f.read(), "text/html; charset=utf-8")
        self.send_error(404)

    def log_message(self, *a):
        pass


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args(argv)
    srv = QuietServer((a.host, a.port), Handler)
    print(f"dashboard: http://{a.host}:{a.port}/   (runs: {', '.join(list_runs()) or 'none yet'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
