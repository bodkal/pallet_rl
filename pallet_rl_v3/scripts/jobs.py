#!/usr/bin/env python3
"""Job queue for the training grid: N commands at a time, in phase order.

    python3 scripts/jobs.py policies --par 4 --iters 4000
    python3 scripts/jobs.py attackers --par 4 --iters 1500

`policies` trains one packing policy per (method, N_B) cell of Table 2.
`attackers` then trains a dedicated permutation-based attacker against each of
those frozen policies -- the paper evaluates every policy under its own
attacker, so the mixture datasets need one per cell.  Attacker strength is the
binding constraint on every attacked column, so they are trained under the
conditions they are tested in: the frozen packer plays greedily, the entropy
bonus is annealed away, and the checkpoint is chosen by held-out attacked
utilisation rather than by the training average.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METHODS = [
    ("pct",   "--algo pct"),
    ("cppo",  "--algo cppo --cvar_q 0.5"),
    ("rarl",  "--algo rarl"),
    ("rfmdp", "--algo rfmdp --rho 0.1"),
    ("ex05",  "--algo exact  --alpha 0.5"),
    ("ex10",  "--algo exact  --alpha 1.0"),
    ("ap05",  "--algo approx --alpha 0.5 --rho 0.1"),
    ("ap10",  "--algo approx --alpha 1.0 --rho 0.1"),
]


def policy_jobs(nbs, methods, iters, extra):
    for nb in nbs:
        for name, flags in METHODS:
            if methods and name not in methods:
                continue
            run = f"{name}_nb{nb}"
            yield run, (f"python3 -m ar2l.train --name {run} {flags} --nb {nb} "
                        f"--iters {iters} --log_every 50 --eval_every 250 "
                        f"--save_every 250 --resume {extra}")


def attacker_jobs(nbs, methods, iters, extra):
    for nb in nbs:
        for name, _ in METHODS:
            if methods and name not in methods:
                continue
            src = f"{name}_nb{nb}"
            ck = os.path.join(ROOT, "runs", src, "best.pt")
            if not os.path.exists(ck):
                print(f"skip {src}: no checkpoint yet")
                continue
            run = f"att_{src}"
            yield run, (f"python3 -m ar2l.train --name {run} --algo attack "
                        f"--nb {nb} --init {ck} --freeze_pack --iters {iters} "
                        f"--ent_final 0.001 "
                        f"--log_every 50 --eval_every 250 --save_every 250 "
                        f"--resume {extra}")


def worker(q, done):
    while True:
        try:
            run, cmd = q.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        with open(os.path.join(ROOT, "runs", run + ".out"), "a") as f:
            f.write(f"\n### {cmd}\n"); f.flush()
            r = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=f)
        done.append((run, r.returncode, time.time() - t0))
        print(f"[{len(done)}] {run} rc={r.returncode} "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        q.task_done()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["policies", "attackers"])
    p.add_argument("--nbs", type=int, nargs="+", default=[10, 20])
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--par", type=int, default=4)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--extra", default="")
    a = p.parse_args()

    gen = policy_jobs if a.phase == "policies" else attacker_jobs
    jobs = list(gen(a.nbs, a.methods, a.iters, a.extra))
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    print(f"{len(jobs)} jobs, {a.par} at a time", flush=True)
    done = []
    ts = [threading.Thread(target=worker, args=(q, done), daemon=True)
          for _ in range(a.par)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"all done in {(time.time()-t0)/3600:.2f} h; "
          f"failures: {[r for r, c, _ in done if c]}")


if __name__ == "__main__":
    main()
