#!/usr/bin/env python3
"""Where does a training iteration spend its time?

Runs the real `ar2l.train.train` for a few iterations with timers wrapped
around the functions that matter -- the rollout, the env's feasibility / EMS /
arm-collision sweeps, network inference, GAE, the PPO update and its
forward / backward / optimiser steps, the held-out eval and the checkpoint
save -- and prints

  * the setup cost, the steady ms/it and a projected full-run time
  * a call tree of mean ms per iteration, with calls and ms per call
  * a timeline of one typical iteration and of one rollout step
  * the heaviest functions by self time, and a few pointers

Every argument after the profiler's own is handed to `ar2l.train`, so a run
is profiled under exactly the flags it trains with:

    python3 scripts/profile_train.py                        # config.yaml defaults
    python3 scripts/profile_train.py --prof_iters 40 -- --algo select --n_pick 5
    python3 scripts/profile_train.py --cprofile 25 -- --n_env 128

GPU work is asynchronous, so by default every timed call synchronises CUDA
on entry and exit; the time a kernel takes then lands on the call that
launched it.  That costs a little throughput -- `--sync 0` measures without
it, but GPU time then smears onto whichever call happens to wait.
The run is written to runs/_profile_<time> and removed afterwards
(`--keep` leaves it).
"""
from __future__ import annotations

import argparse
import cProfile
import functools
import io
import json
import os
import pstats
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                  # train() writes runs/<name> and reads data/

from ar2l import env as E                                         # noqa: E402
from ar2l import model as M                                       # noqa: E402
from ar2l import pack_collision as PC                             # noqa: E402
from ar2l import ppo as P                                         # noqa: E402
from ar2l import train as T                                       # noqa: E402
from ar2l.config import CFG, load as load_config                  # noqa: E402

#: calls that happen every `eval_every` / `save_every` iterations rather than
#: every iteration; they are costed separately and amortised
PERIODIC = ("nominal_score", "attack_score", "torch.save")


class Recorder:
    """Nested wall-clock intervals, tagged with the iteration they fell in."""

    def __init__(self, sync):
        self.sync = sync
        self.events = []        # (path, t0, t1, child_time, bucket, extra)
        self.stack = []         # [path, child_time]
        self.bucket = -1        # -1 = setup, k = the (k+1)-th iteration
        self.steps = []         # time of each Progress.step, the iteration ends
        self.t_loop = None      # when the training loop started

    def call(self, label, fn, a, k, extra=None):
        if self.sync:
            torch.cuda.synchronize()
        path = (self.stack[-1][0] + (label,)) if self.stack else (label,)
        frame = [path, 0.0]
        self.stack.append(frame)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if self.sync:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += t1 - t0
            self.events.append((path, t0, t1, frame[1], self.bucket,
                                extra(a, k) if extra else None))


REC: Recorder = None


def wrap(owner, attr, label=None, extra=None):
    """Replace `owner.attr` with a timed version of itself."""
    fn = getattr(owner, attr)
    label = label or attr

    @functools.wraps(fn)
    def timed(*a, **k):
        return REC.call(label, fn, a, k, extra)
    setattr(owner, attr, timed)


def instrument():
    B, A = E.BPPBatch, PC.ArmPackChecker
    # the environment
    for name in ("reset_done", "obs", "obs_cb", "step", "permute", "window",
                 "_positions", "_placeable", "_head_placeable", "_feas_one",
                 "_arm_any", "_arm_clear",
                 "_ems_corners", "_corner_mask", "_type_ok", "_type_under",
                 "_sweeps", "_support_ratio", "_trim_c", "utilization"):
        if hasattr(B, name):
            wrap(B, name, f"env.{name}")
    # the arm check, with the number of placements it was asked about
    wrap(A, "collides_batch", "arm.collides_batch",
         extra=lambda a, k: len(a[1]))
    for name in ("lazy_ik_batch", "fk_batch", "_capsules_b",
                 "capsules_hit_packs_b"):
        wrap(PC, name, f"arm.{name}")
    # the networks: rollout inference and the update share these, the call
    # path tells them apart
    for cls in (M.PackNet, M.PermNet):
        for name in ("forward", "value", "logits"):
            wrap(cls, name, f"{cls.__name__}.{name}")
    # the training loop, as ar2l.train sees these names
    wrap(T.Runner, "collect", "rollout")
    for name in ("to_torch", "sample", "gae", "flat_obs", "move_to_front",
                 "sup_tv_dual", "inf_tv_dual", "heur_act"):
        if hasattr(T, name):
            wrap(T, name)
    for name in ("nominal_score", "attack_score"):
        wrap(T, name)
        timed = getattr(T, name)

        # n_inst is nominal_score's third argument, attack_score's fourth
        def announce(*a, _f=timed, _i=2 if name == "nominal_score" else 3, **k):
            # the bar stops moving while the held-out eval runs
            print(f"\n[profile] held-out eval ({a[_i]} instances) ...",
                  file=sys.stderr, flush=True)
            return _f(*a, **k)
        setattr(T, name, announce)
    wrap(torch, "save", "torch.save")
    # inside the PPO update
    wrap(P.PPO, "update", "ppo.update", extra=lambda a, k: len(a[2]))
    wrap(torch.Tensor, "backward", "backward")
    wrap(torch.nn.utils, "clip_grad_norm_", "clip_grad_norm")
    wrap(torch.optim.Adam, "step", "adam.step")

    # iteration boundaries: the loop calls Progress.step once per iteration
    init, step = T.Progress.__init__, T.Progress.step

    def p_init(self, *a, **k):
        init(self, *a, **k)
        if REC.sync:
            torch.cuda.synchronize()
        REC.t_loop = time.perf_counter()
        REC.bucket = 0

    def p_step(self, **fields):
        if REC.sync:
            torch.cuda.synchronize()
        REC.steps.append(time.perf_counter())
        REC.bucket += 1
        return step(self, **fields)
    T.Progress.__init__, T.Progress.step = p_init, p_step


# ------------------------------------------------------------------ report
def ms(x):
    return f"{x * 1e3:9.1f}"


def fmt_s(s):
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.1f} s"


def is_periodic(path):
    return any(p in PERIODIC for p in path)


def bar(t0, dur, span, width):
    a = int(round(t0 / span * width))
    b = max(int(round((t0 + dur) / span * width)), a + 1)
    return " " * a + "█" * (b - a) + " " * max(width - b, 0)


def timeline(evs, origin, span, title, width=50, min_frac=0.005):
    """Gantt of `evs` (path, t0, t1, ...), merging back-to-back repeats."""
    rows = []
    for path, t0, t1, *_ in sorted(evs, key=lambda e: e[1]):
        lab = path[-1]
        if rows and rows[-1][0] == lab:
            rows[-1][2] = t1; rows[-1][3] += 1; rows[-1][4] += t1 - t0
        else:
            rows.append([lab, t0, t1, 1, t1 - t0])
    print(f"\n{title}  ({span * 1e3:.1f} ms)")
    print(f"  {'start ms':>9} {'busy ms':>9}  {'':<{width}}  stage")
    for lab, t0, t1, n, busy in rows:
        if busy < min_frac * span:
            continue
        rep = f" x{n}" if n > 1 else ""
        print(f"  {ms(t0 - origin)} {ms(busy)}  |{bar(t0 - origin, t1 - t0, span, width)}| "
              f"{lab}{rep}")


def report(args, targs, rec, wall_setup, prof):
    ev = rec.events
    warm = args.warmup
    n_it = len(rec.steps)
    ends = np.array(rec.steps)
    starts = np.concatenate([[rec.t_loop], ends[:-1]])
    meas = list(range(warm, n_it))
    if not meas:
        sys.exit("no iterations left after the warm-up; raise --prof_iters")
    by_bucket = defaultdict(list)
    for e in ev:
        by_bucket[e[4]].append(e)
    # an eval / save that lands inside an iteration's interval is not that
    # iteration's own cost
    per_it = []
    for b in meas:
        per = sum(e[2] - e[1] for e in by_bucket[b]
                  if len(e[0]) == 1 and is_periodic(e[0]))
        per_it.append(ends[b] - starts[b] - per)
    per_it = np.array(per_it)
    it_mean = per_it.mean()

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() and \
        targs.device.startswith("cuda") else "cpu"
    print("\n" + "=" * 78)
    print(f"TRAINING TIMING PROFILE   algo {targs.algo}  n_env {targs.n_env}  "
          f"T {targs.T}  nb {targs.nb}  n_pick {targs.n_pick}")
    print(f"  bin {targs.bin}  ems {targs.ems}  rot {targs.rot}  "
          f"arm_collision {CFG['env'].get('arm_collision', 0)}  "
          f"epochs {targs.epochs} x minibatches {targs.minibatches}  "
          f"width {targs.width}/{targs.heads}h/{targs.layers}L")
    print(f"  device {gpu}   cuda sync {'on' if args.sync else 'OFF'}"
          f"{'   cProfile ON (times inflated)' if prof else ''}")
    la, nc = os.getloadavg()[0], os.cpu_count()
    print(f"  cpu load {la:.1f} on {nc} cores"
          + ("   ** other jobs are running: times are inflated **"
             if la > 0.5 * nc else ""))
    print("=" * 78)

    # ---- summary ---------------------------------------------------------
    periodic = defaultdict(list)
    for e in ev:
        if len(e[0]) == 1 and e[0][0] in PERIODIC:
            periodic[e[0][0]].append(e[2] - e[1])
    every = {"nominal_score": targs.eval_every, "attack_score": targs.eval_every,
             "torch.save": targs.save_every}
    amort = {k: float(np.mean(v)) / every[k] for k, v in periodic.items()}
    full_it = it_mean + sum(amort.values())
    print(f"\nSetup (data, env, nets, before iteration 1): {fmt_s(wall_setup)}")
    print(f"Iterations: {n_it} run, first {warm} dropped as warm-up, "
          f"{len(meas)} measured")
    print(f"  per iteration  mean {it_mean * 1e3:.1f} ms   median "
          f"{np.median(per_it) * 1e3:.1f}   min {per_it.min() * 1e3:.1f}   "
          f"max {per_it.max() * 1e3:.1f}   ({1 / it_mean:.2f} it/s)")
    for k, v in periodic.items():
        what = "held-out eval" if k != "torch.save" else "checkpoint save"
        print(f"  {what:<16} {np.mean(v):7.2f} s per call, every {every[k]} it"
              f"  -> +{amort[k] * 1e3:.1f} ms/it amortised")
    print(f"  => {full_it * 1e3:.1f} ms/it all-in; the configured "
          f"{targs.iters} iterations ~ {fmt_s(full_it * targs.iters)}"
          f"  (+{fmt_s(wall_setup)} setup)")
    if torch.cuda.is_available() and gpu != "cpu":
        print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

    # ---- call tree -------------------------------------------------------
    tot = defaultdict(float); self_t = defaultdict(float); calls = defaultdict(int)
    xtra = defaultdict(int)
    for b in meas:
        for path, t0, t1, child, _, ex in by_bucket[b]:
            if is_periodic(path):
                continue
            tot[path] += t1 - t0; self_t[path] += t1 - t0 - child
            calls[path] += 1
            if ex is not None:
                xtra[path] += ex
    n = len(meas)
    top = sum(v for p, v in tot.items() if len(p) == 1)
    print(f"\nPER-ITERATION BREAKDOWN  (mean over {n} iterations; indented = "
          f"called from the line above)")
    print(f"  {'stage':<46} {'ms/it':>9} {'%it':>6} {'calls/it':>9} {'ms/call':>9}")

    def children(path):
        return sorted((p for p in tot if len(p) == len(path) + 1
                       and p[:-1] == path), key=lambda p: -tot[p])

    def show(path, depth):
        t = tot[path] / n
        if t < args.min_ms * 1e-3 and depth > 0:
            return
        c = calls[path] / n
        lab = "  " * depth + path[-1]
        if path in xtra:
            per = xtra[path] / calls[path]
            lab += (f" [{per:.0f} placements]" if path[-1] == "arm.collides_batch"
                    else f" [{per:.0f} samples]")
        print(f"  {lab:<46} {ms(t)} {t / it_mean * 100:5.1f}% {c:9.1f} "
              f"{tot[path] / calls[path] * 1e3:9.2f}")
        kids = children(path)
        for k in kids:
            show(k, depth + 1)
        if kids and self_t[path] / n > args.min_ms * 1e-3:
            s = self_t[path] / n
            print(f"  {'  ' * (depth + 1) + '(own code, untimed calls)':<46} "
                  f"{ms(s)} {s / it_mean * 100:5.1f}%")

    for p in children(()):
        show(p, 0)
    rest = it_mean - top / n
    print(f"  {'(train loop outside timed calls)':<46} {ms(rest)} "
          f"{rest / it_mean * 100:5.1f}%")

    # ---- timelines -------------------------------------------------------
    med = meas[int(np.argsort(per_it)[len(per_it) // 2])]
    evs = [e for e in by_bucket[med] if not is_periodic(e[0])]
    timeline([e for e in evs if len(e[0]) == 1], starts[med],
             ends[med] - starts[med], f"TIMELINE of iteration {med + 1} "
             f"(the median one), top-level stages")
    ro = [e for e in evs if e[0] == ("rollout",)]
    if ro:
        _, r0, r1, *_ = ro[0]
        kids = sorted((e for e in evs if len(e[0]) == 2 and e[0][0] == "rollout"
                       and r0 <= e[1] <= r1), key=lambda e: e[1])
        # a rollout step starts at every call of its first child (reset_done)
        first = kids[0][0][-1] if kids else None
        cuts = [e[1] for e in kids if e[0][-1] == first]
        if len(cuts) > 2:
            durs = np.diff(cuts)
            j = int(np.argsort(durs)[len(durs) // 2])
            s0, s1 = cuts[j], cuts[j + 1]
            timeline([e for e in kids if s0 <= e[1] < s1], s0, s1 - s0,
                     f"TIMELINE of one rollout step (the median of "
                     f"{len(cuts)}), inside `rollout`")
            sub = [e for e in evs if len(e[0]) >= 3 and s0 <= e[1] < s1
                   and len(e[0]) == 3]
            if sub:
                timeline(sub, s0, s1 - s0, "  ... and one level deeper")

    # ---- heaviest by self time ------------------------------------------
    flat_self = defaultdict(float); flat_calls = defaultdict(int)
    for p, v in self_t.items():
        flat_self[p[-1]] += v; flat_calls[p[-1]] += calls[p]
    print(f"\nHEAVIEST FUNCTIONS by self time (time not spent in a timed callee;"
          f" all call sites summed)")
    print(f"  {'function':<34} {'ms/it':>9} {'%it':>6} {'calls/it':>9} {'us/call':>9}")
    for lab, v in sorted(flat_self.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"  {lab:<34} {ms(v / n)} {v / n / it_mean * 100:5.1f}% "
              f"{flat_calls[lab] / n:9.1f} {v / flat_calls[lab] * 1e6:9.1f}")

    # ---- cProfile ----------------------------------------------------------
    if prof is not None:
        s = io.StringIO()
        st = pstats.Stats(prof, stream=s)
        st.sort_stats("tottime").print_stats(args.cprofile)
        print(f"\nPYTHON HOTSPOTS (cProfile, whole run incl. setup/eval, "
              f"sorted by own time)")
        lines = s.getvalue().splitlines()
        i = next((k for k, l in enumerate(lines) if "ncalls" in l), 0)
        print("\n".join("  " + l.replace(ROOT + "/", "") for l in lines[i:]
                        if l.strip()))

    # ---- pointers ----------------------------------------------------------
    hints = []
    share = lambda lab: sum(v for p, v in tot.items() if p[-1] == lab
                            and lab not in p[:-1]) / n / it_mean
    ro_s, up_s = share("rollout"), share("ppo.update")
    arm = share("arm.collides_batch")
    nets = sum(v for p, v in tot.items() if p[0] == "rollout"
               and p[-1].endswith((".forward", ".value", ".logits"))
               and not any(q.endswith((".forward", ".value", ".logits"))
                           for q in p[:-1])) / n / it_mean
    hints.append(f"rollout {ro_s * 100:.0f}% / ppo.update {up_s * 100:.0f}% of "
                 f"an iteration; rollout network inference is "
                 f"{nets * 100:.0f}%, so ~{(ro_s - nets) * 100:.0f}% is the "
                 f"numpy env on the CPU (GPU idle meanwhile).")
    if arm > 0.10:
        pl = sum(xtra[p] for p in xtra if p[-1] == "arm.collides_batch") / n
        hints.append(f"arm-collision check is {arm * 100:.0f}% "
                     f"({pl:.0f} uncached placements/it); `arm_collision: 0` "
                     f"or fewer candidates (ems, min_support) cut it.")
    fz = share("env._feas_one")
    if fz > 0.15:
        hints.append(f"feasibility sweeps (_feas_one) are {fz * 100:.0f}%; "
                     f"rot 2 doubles them, and n_pick k runs k extra sweeps "
                     f"per step for b_pick (env._placeable).")
    pk = share("env._placeable")
    if pk > 0.15:
        hints.append(f"env._placeable (the b_pick mask: one sweep per "
                     f"reachable box, n_pick {targs.n_pick}) alone is "
                     f"{pk * 100:.0f}%.")
    if up_s > 0.4:
        hints.append(f"the PPO update dominates: its cost is linear in "
                     f"epochs x minibatches ({targs.epochs} x "
                     f"{targs.minibatches}); larger minibatches (fewer of "
                     f"them) use the GPU better.")
    ev_am = sum(v for k, v in amort.items() if k != "torch.save")
    if ev_am > 0.1 * it_mean:
        hints.append(f"held-out evals add {ev_am / full_it * 100:.0f}% to the "
                     f"run; raise eval_every or lower eval_inst "
                     f"({targs.eval_inst}).")
    print("\nPOINTERS")
    for h in hints:
        print("  - " + h)

    if args.json:
        rows = [{"path": list(p), "ms_per_it": tot[p] / n * 1e3,
                 "self_ms_per_it": self_t[p] / n * 1e3,
                 "calls_per_it": calls[p] / n} for p in tot]
        json.dump({"ms_per_it": it_mean * 1e3, "ms_per_it_all_in": full_it * 1e3,
                   "setup_s": wall_setup, "iter_ms": (per_it * 1e3).tolist(),
                   "periodic_s": {k: v for k, v in periodic.items()},
                   "tree": rows, "train_args": {k: str(v) for k, v in
                                                vars(targs).items()}},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prof_iters", type=int, default=25,
                   help="iterations to run (default 25)")
    p.add_argument("--warmup", type=int, default=3,
                   help="first iterations left out of the averages (CUDA "
                        "init, allocator, caches); default 3")
    p.add_argument("--sync", type=int, default=1,
                   help="1 = cuda.synchronize around every timed call so GPU "
                        "time is attributed correctly (default)")
    p.add_argument("--eval", type=int, default=1,
                   help="1 = run one held-out eval at the end to cost it "
                        "(default); 0 skips it")
    p.add_argument("--cprofile", type=int, default=0, metavar="N",
                   help="also run cProfile and print its N top functions")
    p.add_argument("--min_ms", type=float, default=0.2,
                   help="hide tree rows under this many ms/it")
    p.add_argument("--top", type=int, default=15,
                   help="rows in the heaviest-functions table")
    p.add_argument("--json", default=None, help="also write the numbers here")
    p.add_argument("--progress", choices=("auto", "on", "off"), default="on",
                   help="the training progress bar on stderr (default on)")
    p.add_argument("--keep", action="store_true",
                   help="keep the runs/_profile_* directory")
    args, rest = p.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]

    # the same argument handling as ar2l.train.main
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(rest)
    if known.config:
        load_config(known.config)
    if "--name" in rest:
        i = rest.index("--name"); del rest[i:i + 2]
        print("[profile] --name ignored; profiling into a scratch run")
    name = f"_profile_{time.strftime('%Y%m%d_%H%M%S')}"
    targs = T.get_parser().parse_args(rest + ["--name", name])
    if targs.min_support is None:
        targs.min_support = float(CFG["env"]["min_support"])
    targs.data = targs.data or None
    targs.data_sha1 = None
    full_iters = targs.iters
    targs.iters = args.prof_iters
    targs.progress = args.progress
    targs.log_every = 10 ** 9           # only the last iteration logs
    if not args.eval:
        targs.eval_every = 10 ** 9
    # `train` evals and saves at its last iteration regardless; keep the real
    # cadences in the args so the report amortises them correctly
    eval_every, save_every = targs.eval_every, targs.save_every

    global REC
    REC = Recorder(bool(args.sync) and targs.device.startswith("cuda")
                   and torch.cuda.is_available())
    instrument()
    if not args.eval:
        T.nominal_score = lambda *a, **k: (0.0, 0.0)
        T.attack_score = lambda *a, **k: (0.0, 0.0)

    print(f"[profile] {args.prof_iters} iterations of `ar2l.train "
          f"{' '.join(rest)}` into runs/{name}", flush=True)
    prof = cProfile.Profile() if args.cprofile else None
    t_start = time.perf_counter()
    try:
        if prof:
            prof.enable()
        T.train(targs)
    finally:
        if prof:
            prof.disable()
        if not args.keep:
            shutil.rmtree(os.path.join("runs", name), ignore_errors=True)
    wall_setup = (REC.t_loop or t_start) - t_start
    targs.iters, targs.eval_every, targs.save_every = (full_iters, eval_every,
                                                       save_every)
    report(args, targs, REC, wall_setup, prof)


if __name__ == "__main__":
    main()
