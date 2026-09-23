#!/usr/bin/env python3
"""Train a permutation-based attacker against each heuristic, for Table 1.

    python3 scripts/attack_heur.py --nbs 5 10 15 20 --par 3 --iters 1500

The packing policy is the heuristic itself and observes one item; the attacker
observes N_B.  Runs land in runs/h<heuristic>_nb<N_B>, which is where
scripts/eval_all.py table1 looks for them.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ar2l.heuristics import NAMES                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbs", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--heuristics", nargs="*", default=list(NAMES))
    p.add_argument("--par", type=int, default=3)
    p.add_argument("--iters", type=int, default=1500)
    a = p.parse_args()

    q = queue.Queue()
    for nb in a.nbs:
        for h in a.heuristics:
            run = f"h{h}_nb{nb}"
            q.put((run, f"python3 -m ar2l.train --name {run} --algo attack "
                        f"--nb {nb} --heur_pack {h} --iters {a.iters} "
                        f"--ent_final 0.001 --eval_every 250 "
                        f"--log_every 100 --save_every 250 --resume"))
    n = q.qsize()
    print(f"{n} attacker jobs, {a.par} at a time", flush=True)
    done = []

    def worker():
        while True:
            try:
                run, cmd = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            with open(os.path.join(ROOT, "runs", run + ".out"), "a") as f:
                r = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=f)
            done.append(run)
            print(f"[{len(done)}/{n}] {run} rc={r.returncode} "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.par)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("done")


if __name__ == "__main__":
    main()
