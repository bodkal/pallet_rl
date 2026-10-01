"""The big experiment: every pallet file of a folder, packed onto as many
pallets as it takes.

One file is one day's boxes at one cell, in the order they arrived:

    TRAN_DATE,TRAN_HOUR,CONTAINER_ID,PALLET_ID,FROM_LOC_ID,V_BOX_DEPTH,V_BOX_WIDTH,V_BOX_HEIGHT,TYPE

read as `length_cm = V_BOX_DEPTH`, `width_cm = V_BOX_WIDTH`,
`height_cm = V_BOX_HEIGHT` and `type = TYPE - 1` (the file counts types from
1: blue, white, brown/master), then gridded exactly as `ar2l.orders` grids an
orders file -- `cell_cm`, `box_scale`, `box_round` -- onto the `pallet_cm`
pallet.  The packer takes the boxes in file order; when no box within reach
has anywhere left to go the pallet is closed and a fresh one is opened, and
packing carries on from the first box that did not go in, until the whole
file is on pallets.

Each file writes one result, under the same name as its source file, in the
format the cell's own experiment reports use:

    <file>,current date:,<now>,boxes:,<n>,palet: ,<PALLET_IDs in the file>,<env params>,<data params>,
    pallet_id,max_height,uneven_z_val,boxes_volume_per_pallet [%],time_to_pack [ms],total_number_of_boxes,number_of_blue_boxes,number_of_white_boxes,number_of_master_boxes,
    0,152,0,58,6529,32,6,6,20,

`max_height` is in real cm (cells x cell_cm x box_scale), `time_to_pack` is
the running total since the file started, and `uneven_z_val` is written as 0.
A run given a name adds `name:,<name>,` to the end of the first line, and
`ar2l.viz.compare` shows that name in its legend.  A report that already exists is appended
to: each run adds its own block, first line and header included.

The packer and the picker always take their most likely choice (`argmax`),
so the only randomness is the boxes' arrival order: `order_random` (0..1)
shuffles each file's box order before packing, as `ar2l.orders.randomize_order`
does for training -- 0 is the file's order, 1 a uniformly random one, 0.1
moves a box ~5 places in a 50-box pallet -- drawn from `seed`, so the same
seed shuffles the same way, and the report's first line records it.
"""
from __future__ import annotations

import csv
import glob
import os
import threading
import time
import uuid
from datetime import datetime

import numpy as np

from ..env import BPPBatch
from ..orders import (_cells, box_divisor, box_rounding, orders_bin,
                      randomize_order)
from . import agents as A

#: the file's columns -> what `ar2l.orders` calls them
COLS = {"length_cm": "V_BOX_DEPTH", "width_cm": "V_BOX_WIDTH",
        "height_cm": "V_BOX_HEIGHT"}
HEADER = ("pallet_id,max_height,uneven_z_val,boxes_volume_per_pallet [%],"
          "time_to_pack [ms],total_number_of_boxes,number_of_blue_boxes,"
          "number_of_white_boxes,number_of_master_boxes,")

_JOBS: dict = {}
_LOCK = threading.Lock()


def resolve(path):
    """A folder the page named, resolved against the project root."""
    return os.path.normpath(os.path.join(A.ROOT, os.path.expanduser(str(path))))


def read_file(path, S, p):
    """-> (seq (n, 4) int16 in cells, n_pallet_ids, n_boxes, skipped).

    A box that fits the pallet in no allowed orientation could never be
    placed; it is counted in `skipped` and left out rather than stalling the
    file on an empty pallet forever.
    """
    scale = box_divisor(p["box_scale"])
    rounding = box_rounding(p["box_round"])
    Lx, Ly, Lz = S
    rows, ids, skipped = [], set(), 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = [c.strip() for c in (rd.fieldnames or [])]
        missing = [c for c in list(COLS.values()) + ["TYPE"] if c not in cols]
        if missing:
            raise ValueError(f"{os.path.basename(path)}: missing column(s) "
                             f"{missing}")
        for n, raw in enumerate(rd, start=2):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            if not any(row.values()):
                continue
            where = f"{os.path.basename(path)}:{n}"
            sx, sy, sz = (_cells(row[COLS[c]], p["cell_cm"], f"{where} {c}",
                                 scale, rounding)
                          for c in ("length_cm", "width_cm", "height_cm"))
            t = int(float(row["TYPE"])) - 1
            if not 0 <= t < p["n_types"]:
                raise ValueError(f"{where}: TYPE {t + 1} is outside 1.."
                                 f"{p['n_types']}")
            if row.get("PALLET_ID"):
                ids.add(row["PALLET_ID"])
            flat = ((sx <= Lx and sy <= Ly)
                    or (p["rot"] >= 2 and sy <= Lx and sx <= Ly))
            if not (flat and sz <= Lz):
                skipped += 1
                continue
            rows.append((sx, sy, sz, t))
    seq = np.asarray(rows, np.int16).reshape(-1, 4)
    return seq, len(ids), len(seq) + skipped, skipped


def packer(spec, n_pick, device):
    """-> (policy, picker, label): who places, and who picks from the reach.

    A `select` run gets its own selector back, as the game hands it one.
    """
    policy, label, _ = A.load_policy(spec, device)
    picker = None
    if spec.startswith("run:") and n_pick > 1:
        run = spec.split(":")[1]
        info = A.run_info(run)
        if info and info["args"].get("algo") == "select":
            picker = A.load_attacker(f"mix:{run}", device)[0]
    return policy, picker, label


def pack_pallet(seq, policy, picker, S, p, cell_m, tick=None, stop=None):
    """Pack one pallet from the front of `seq`.  -> (env, rest of seq).

    The pallet is closed when no box within reach fits, not when the front
    one does not: the env ends a bin on its front box, so a stuck front with a
    placeable box behind it is permuted rather than taken as the end.
    """
    env = BPPBatch(1, S=S, nb=p["nb"], n_items=len(seq), max_l=p["max_l"],
                   rot=p["rot"], ems=p["ems"], stability=p["stability"],
                   min_support=p["min_support"], n_pick=p["n_pick"],
                   n_types=p["n_types"], types=False,
                   type_constraint=bool(p["type_constraint"]),
                   arm_collision=bool(p["arm_collision"]), arm_cell_m=cell_m,
                   arm_moves=p["base_x_moves"],
                   size_hi=seq[:, :3].max(0))
    env.reset(seq[None])
    while True:
        if stop is not None and stop.is_set():
            break
        if env.done[0]:
            if env.head[0] >= env.length[0] or env.n_pick < 2:
                break
            env.done[0] = False                # ask the rest of the reach
            env._invalidate(hmap=False)
            ok = np.flatnonzero(env._placeable()[0])
            if ok.size == 0:
                env.done[0] = True
                break
            env.permute(np.array([int(ok[0])]))
        if picker is not None:
            env.permute(picker(env)[0])
        if not env.obs()["l_mask"][0].any():
            env.done[0] = True
            continue
        env.step(policy(env)[0])
        if tick is not None:
            tick(env)
    rest = env.seq[0, int(env.head[0]):int(env.length[0])].copy()
    return env, rest


def params_text(spec, p, S):
    """The two parameter fields of the report's first line, comma-free."""
    j = lambda v: "x".join(f"{t:g}" for t in v)
    env = (f"packer={spec} bin={j(S)} nb={p['nb']} n_pick={p['n_pick']} "
           f"max_l={p['max_l']} rot={p['rot']} ems={p['ems']} "
           f"stability={p['stability']} min_support={p['min_support']:g} "
           f"n_types={p['n_types']} type_constraint={p['type_constraint']} "
           f"arm_collision={p['arm_collision']} "
           f"base_x_moves={'/'.join(f'{t:g}' for t in p['base_x_moves'])}")
    data = (f"cell_cm={p['cell_cm']:g} box_scale={p['box_scale']:g} "
            f"box_round={p['box_round']} "
            f"order_random={p.get('order_random', 0):g} "
            f"pallet_cm={j(p['pallet_cm']) if p['pallet_cm'] else 'none'}")
    return env, data


def clean_name(name):
    """An experiment name as the report can hold it: one line, comma-free."""
    return " ".join(str(name or "").replace(",", " ").split())


def write_report(path, name, n_boxes, n_ids, fields, rows, exp_name=""):
    """Write one run's block; an existing report is added to, not replaced.

    Every block starts with its own first line and column header, so a file
    that several runs have written still reads block by block.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"{name},current date:,{now},boxes:,{n_boxes},palet: ,{n_ids},"
             f"{fields[0]} ,{fields[1]},"
             + (f"name:,{exp_name}," if exp_name else ""), HEADER]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in (
            "pallet_id", "max_height", "uneven_z_val", "util_pct", "ms",
            "boxes", "blue", "white", "master")) + ",")
    with open(path, "a", newline="") as f:
        f.write("\n".join(lines) + "\n")


def _show(path):
    """A path as the page shows it: from the project root when inside it."""
    rel = os.path.relpath(path, A.ROOT)
    return path if rel.startswith("..") else rel


def start(folder, out, spec, p, cell_m, device, name="", seed=0):
    """Queue every .csv of `folder` and run them in a thread.  -> job id."""
    src = resolve(folder)
    if not os.path.isdir(src):
        raise ValueError(f"no such folder: {folder}")
    files = sorted(glob.glob(os.path.join(src, "*.csv")))
    if not files:
        raise ValueError(f"{folder} holds no .csv files")
    dst = resolve(out)
    if os.path.abspath(dst) == os.path.abspath(src):
        raise ValueError("write the results to another folder than the data")
    S = tuple(orders_bin(p["pallet_cm"], p["cell_cm"]) if p["pallet_cm"]
              else p["bin"])
    job = {"id": uuid.uuid4().hex[:10], "state": "running", "spec": spec,
           "name": clean_name(name), "seed": int(seed),
           "src": folder, "out": _show(dst), "bin": list(S),
           "t0": time.time(), "t1": None, "error": None,
           "files": [{"name": os.path.basename(f), "path": f, "state": "queued",
                      "total": 0, "done": 0, "pallets": [], "cur": None,
                      "ms": 0, "skipped": 0, "ids": 0, "out": None,
                      "error": None} for f in files],
           "stop": threading.Event()}
    with _LOCK:
        running = [j for j in _JOBS.values() if j["state"] == "running"]
        if running:
            raise ValueError("an experiment is already running; stop it first")
        _JOBS[job["id"]] = job
        _JOBS["latest"] = job
    threading.Thread(target=_run, args=(job, dst, spec, p, S, cell_m, device),
                     daemon=True).start()
    return job["id"]


def _run(job, dst, spec, p, S, cell_m, device):
    try:
        os.makedirs(dst, exist_ok=True)
        policy, picker, label = packer(spec, p["n_pick"], device)
        job["label"] = label
        env_f, data_f = params_text(spec, p, S)
        if p.get("order_random"):
            data_f += f" seed={job['seed']}"
        fields = (env_f, data_f)
        # a grid cell stands for cell_cm x box_scale of the real box
        cm = p["cell_cm"] * box_divisor(p["box_scale"])
        for f in job["files"]:
            if job["stop"].is_set():
                break
            _run_file(job, f, dst, policy, picker, p, S, cell_m, cm, fields)
        job["state"] = "stopped" if job["stop"].is_set() else "done"
    except Exception as e:                    # the page shows it, and stops
        job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"
    job["t1"] = time.time()


def _run_file(job, f, dst, policy, picker, p, S, cell_m, cm, fields):
    f["state"] = "running"
    t0 = time.time()
    try:
        seq, f["ids"], n_boxes, f["skipped"] = read_file(f["path"], S, p)
        if p.get("order_random") and len(seq):
            seq = randomize_order(seq[None], p["order_random"], job["seed"])[0]
        f["total"] = len(seq)
        rows, rest = [], seq
        while len(rest):
            base = f["done"]

            def tick(env, _b=base):
                f["done"] = _b + int(env.n_packed[0])
                f["cur"] = {"util": round(float(env.utilization()[0]) * 100, 1),
                            "boxes": int(env.n_packed[0])}
                f["ms"] = int((time.time() - t0) * 1000)

            env, left = pack_pallet(rest, policy, picker, S, p, cell_m, tick,
                                    job["stop"])
            if job["stop"].is_set():
                f["state"] = "stopped"
                return
            n = int(env.n_packed[0])
            if n == 0:
                # nothing went onto an empty pallet: that box cannot be placed
                # by this packer at all (the arm filter, say); set it aside
                f["skipped"] += 1
                rest = rest[1:]
                continue
            types = env.ptype[0, :n]
            ms = int((time.time() - t0) * 1000)
            row = {"pallet_id": len(rows),
                   "max_height": int(round(int(env.hmap[0].max()) * cm)),
                   "uneven_z_val": 0,
                   "util_pct": int(round(float(env.utilization()[0]) * 100)),
                   "ms": ms, "boxes": n,
                   "blue": int((types == 0).sum()),
                   "white": int((types == 1).sum()),
                   "master": int((types == 2).sum())}
            rows.append(row)
            f["pallets"].append(row)
            f["done"], f["cur"], f["ms"], rest = base + n, None, ms, left
        name = f["name"]
        write_report(os.path.join(dst, name), f["name"], n_boxes, f["ids"],
                     fields, rows, job["name"])
        f["out"], f["state"] = name, "done"
    except Exception as e:
        f["state"], f["error"] = "error", f"{type(e).__name__}: {e}"
    f["ms"] = int((time.time() - t0) * 1000)


def status(jid):
    """What the page draws: the job and every file's progress, or None."""
    job = _JOBS.get(jid or "latest")
    if job is None:
        return None
    now = job["t1"] or time.time()
    return {k: v for k, v in job.items() if k not in ("stop", "t0", "t1")} | {
        "elapsed": round(now - job["t0"], 1),
        "files": [{k: v for k, v in f.items() if k != "path"}
                  for f in job["files"]]}


def writing(folder):
    """Whether a running experiment writes its reports into `folder`."""
    job = _JOBS.get("latest")
    return bool(job and job["state"] == "running"
                and os.path.abspath(resolve(job["out"]))
                == os.path.abspath(resolve(folder)))


def stop(jid):
    job = _JOBS.get(jid or "latest")
    if job is not None:
        job["stop"].set()
    return job is not None
