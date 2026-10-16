"""Log match: the cell's own conveyor log, packed by an agent a pick at a time.

A log is what the cell's camera saw each time the robot picked (a `step`):
one row per box on the conveyor,

    step,time,conveyor,run_detected,box_no,type,status,size_L_cm,size_W_cm,
    size_H_cm,row_L_cm,row_W_cm,row_H_cm,rms,mid_x_m,mid_y_m,mid_z_m,picked

read as `type` 1/2/3 = blue/white/brown (the env's 0/1/2), the box's sides
`size_L_cm` x `size_W_cm` x `size_H_cm` -- the `row_*` measurement reads a
crate's rim, 2 cm for an 18 cm crate, so it is not used -- its place along
the belt `mid_y_m` (larger is nearer the robot), and `picked` = `yes` on the
box the robot took at that step.  The sides are gridded as the big
experiment grids a file: `box_pad_m`, `cell_cm`, `box_scale`, `box_round`.

A box stays on the belt, and in the log, step after step until it is picked,
so the rows are told apart first (`track`).  A box of a step is one already
on the belt when it has the same type and the same three sides, in any
order, within `tol_cm`, and has not slid back along the belt; the rows are
matched front to back, in belt order.  The box the robot just picked is gone
-- unless it turns up again in front of a box that stayed, where no new box
can come on (they come on at the back): then the pick failed.  A box the
camera missed is still looked for `KEEP` steps on.  Anything left over is a
new box.

The agent then packs one box per step, as the robot does, from a station of
every box seen so far that it has not packed: in the order they came, the
first `nb` in view and the first `n_pick` within reach, and never a box of a
later step.  A `select` run picks with its own selector; any other packer
takes the first box within reach that has somewhere to go.  When none has,
the pallet is closed and a new one opened, the station carried over; once
the log has run out the station is packed on, a box per step, until it is
empty.  A box that fits no empty pallet is set aside, as the big experiment
does.
"""
from __future__ import annotations

import csv
import os
import threading
import time
import uuid

import numpy as np

from ..orders import box_cells, box_divisor, box_pad_cm, box_rounding, orders_bin
from . import experiment as X
from . import game as G

#: what a log has to say about every box
COLS = ("step", "type", "size_L_cm", "size_W_cm", "size_H_cm", "mid_y_m",
        "picked")
SIDES = ("size_L_cm", "size_W_cm", "size_H_cm")
#: where the robot put a box it packed, in the newer logs: its corner on the
#: pallet in cm, and its turn -- 0 or 180 lays L along the pallet's y, +-90
#: along its x (the reading under which no two boxes of a pallet overlap)
PACK = ("box_x_cm", "box_y_cm", "box_z_cm", "orientation_deg")
#: two real boxes overlapping by more than this (cm: x/y, z) cannot share a
#: pallet -- when the later one starts in the lower half of the other: a
#: crate nests a few cm into the one under it, so one starting higher is
#: stacked on it, however deep it reaches in
CLASH_CM = (1.0, 4.0)
TOL_CM = 1.5        # two sightings of one box differ by up to this, per side
BACK_M = 0.05       # how far back along the belt a box may seem to slide
KEEP = 3            # steps a box the camera missed is still looked for
JOBS_KEPT = 4       # finished runs kept, for their pallets' replays

_JOBS: dict = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------- the log
def read_log(path, n_types=3):
    """-> [{'step', 'time', 'rows': [{'no', 'type', 'cm', 'y', 'picked'}]}],
    the steps in order and each one's boxes front (nearest the robot) first."""
    steps = {}
    name = os.path.basename(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = [c.strip() for c in (rd.fieldnames or [])]
        missing = [c for c in COLS if c not in cols]
        if missing:
            raise ValueError(f"{name}: missing column(s) {missing}")
        for n, raw in enumerate(rd, start=2):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            if not any(row.values()):
                continue
            where = f"{name}:{n}"
            try:
                s = int(float(row["step"]))
                t = int(float(row["type"])) - 1
                cm = tuple(float(row[c]) for c in SIDES)
                y = float(row["mid_y_m"])
                no = int(float(row.get("box_no") or 0))
            except ValueError:
                raise ValueError(f"{where}: step, box_no, type, the sides and "
                                 f"mid_y_m are numbers") from None
            if not 0 <= t < n_types:
                raise ValueError(f"{where}: type {t + 1} is outside 1..{n_types}")
            if min(cm) <= 0:
                raise ValueError(f"{where}: a side must be positive, got {cm}")
            st = steps.setdefault(s, {"step": s, "time": row.get("time", ""),
                                      "rows": []})
            r = {"no": no, "type": t, "cm": cm, "y": y,
                 "picked": row["picked"].lower() in ("yes", "y", "1", "true"),
                 # the newer logs: why a pick did not go onto the pallet, and
                 # where on it the box went when it did
                 "failed": row.get("not_packed_reason", ""), "pack": None}
            if r["picked"] and not r["failed"] and all(
                    row.get(c) for c in PACK):
                try:
                    r["pack"] = tuple(float(row[c]) for c in PACK)
                except ValueError:
                    raise ValueError(f"{where}: the pack position "
                                     f"{', '.join(PACK)} are numbers") from None
                r["at"] = row.get("pack_location", "")
                # the pallet it went onto, in the logs that say: a new id is
                # a new pallet, the one before it finished
                r["pallet_id"] = row.get("pallet_id", "")
            st["rows"].append(r)
    out = [steps[s] for s in sorted(steps)]
    # a log may start at any step; the match counts its own from 1, in order,
    # and keeps the log's number (`log`) to refer back to it
    for i, st in enumerate(out):
        st["log"], st["step"] = st["step"], i + 1
        st["rows"].sort(key=lambda r: (-r["y"], r["no"]))
    return out


def same(a, b, tol):
    """One box seen twice: its type, and its sides in any order within `tol`."""
    return a["type"] == b["type"] and all(
        abs(u - v) <= tol for u, v in zip(sorted(a["cm"]), sorted(b["cm"])))


def track(steps, tol_cm=TOL_CM):
    """Tell the log's boxes apart.  -> one {'id', 'type', 'cm', 'at'} per box
    the belt carried, `at` the index of the step it was first seen at.

    Each step gains `ids`, the box of each of its rows; `new`, the boxes first
    seen there, front first; and `real`, the box the robot picked, or None.
    """
    boxes, belt = [], []          # belt: what is on it, front first
    for k, st in enumerate(steps):
        belt.sort(key=lambda b: -b["y"])
        stay = [b["y"] for b in belt if not b["gone"]]
        seen, ids, new, came, last = set(), [], [], [], -1
        for r in st["rows"]:
            fit = [j for j in range(last + 1, len(belt))
                   if j not in seen and same(belt[j], r, tol_cm)
                   and r["y"] >= belt[j]["y"] - BACK_M]
            hit = next((j for j in fit if not belt[j]["gone"]), None)
            # only the box the robot just took: its pick failed, when the box
            # is in front of one that stayed -- a new box comes on at the back
            if hit is None and fit and any(r["y"] > y + BACK_M for y in stay):
                hit = fit[0]
            if hit is None:
                b = {"id": len(boxes), "type": r["type"], "cm": r["cm"], "at": k}
                boxes.append(b)
                came.append({"id": b["id"], "type": r["type"], "cm": r["cm"],
                             "y": r["y"], "gone": False, "miss": 0})
                new.append(b["id"])
                ids.append(b["id"])
            else:
                last = hit
                seen.add(hit)
                belt[hit].update(y=r["y"], gone=False, miss=0)
                ids.append(belt[hit]["id"])
        keep = []
        for j, b in enumerate(belt):
            if j in seen:
                keep.append(b)
            elif not b["gone"] and b["miss"] < KEEP:
                b["miss"] += 1
                keep.append(b)
        belt = keep + came
        # the robot's pick leaves the belt, unless it turns up again -- or the
        # log says the pick failed, and so it stayed
        real = [i for i, r in zip(ids, st["rows"]) if r["picked"]]
        took = [i for i, r in zip(ids, st["rows"]) if r["picked"]
                and not r["failed"]]
        for b in belt:
            if b["id"] in took:
                b["gone"] = True
        st.update(ids=ids, new=new, real=real[0] if real else None)
    return boxes


def real_pack(steps, cm_cell, p=None):
    """The robot's own packing, step by step, from a newer log; None for a
    log that does not say where the boxes went.

    One entry per step: its station as the page draws it (every box of that
    step, front first -- the robot sees nothing else), the box picked
    (`pick`, -1 for none), why the pick failed (`failed`), and the box put on
    the pallet, `[x, y, z, l, w, h]` in cells of `cm_cell` cm (`box`, None
    when nothing went on).  A box that would overlap one already on the real
    pallet by more than `CLASH_CM`, from its lower half, is on a new pallet
    (`pallet`) -- or, when the log has `pallet_id`, a box with another id
    than the one before it (`id`)."""
    if not any(r["pack"] for s in steps for r in s["rows"]):
        return None
    by_id = any(r.get("pallet_id") for s in steps for r in s["rows"])
    out, on, k, pid = [], [], 0, None
    for s in steps:
        rows = s["rows"]
        pick = next((j for j, r in enumerate(rows) if r["picked"]), -1)
        e = {"step": s["step"], "log": s["log"], "time": s["time"], "pick": pick,
             "w": [list(cells(r["cm"], p)) if p else [] for r in rows],
             "t": [r["type"] for r in rows],
             "failed": rows[pick]["failed"] or None if pick >= 0 else None,
             "box": None, "at": None, "pallet": k}
        r = rows[pick] if pick >= 0 else None
        if r is not None and r["pack"]:
            x, y, z, ori = r["pack"]
            L, W, H = r["cm"]
            lx, ly = (L, W) if int(round(ori)) % 180 else (W, L)
            b = (x, y, z, lx, ly, H)
            if by_id:
                new = pid is not None and r["pallet_id"] != pid
                pid = r["pallet_id"]
            else:
                new = any(b[2] < a[2] + a[5] / 2 and all(
                    min(a[i] + a[i + 3], b[i] + b[i + 3]) - max(a[i], b[i])
                    > CLASH_CM[i == 2] for i in range(3)) for a in on)
            if new:
                k, on = k + 1, []
            on.append(b)
            e.update(box=[round(v / cm_cell, 3) for v in b], at=r.get("at"),
                     pallet=k, id=r.get("pallet_id") or None, type=r["type"],
                     # the box as the agent's grid has it, padding and all
                     cells=list(cells(r["cm"], p)) if p else None)
        out.append(e)
    return out


def real_pallets(real, S, cm_cell):
    """One summary per real pallet, as `Pack.close` sums up the agent's: its
    boxes and their types, its fill -- each box at the size the agent's grid
    gives it, so the two fills measure the same boxes the same way -- its
    height in cm, the log steps and times it spans, and its failed picks."""
    if real is None:
        return None
    out = []
    for k in range(real[-1]["pallet"] + 1):
        on = [e for e in real if e["pallet"] == k and e["box"]]
        vol = sum(int(np.prod(e["cells"])) for e in on if e["cells"])
        out.append({
            "i": k, "id": on[0]["id"] if on else None, "boxes": len(on),
            "types": [sum(e["type"] == t for e in on) for t in range(3)],
            "util": vol / float(np.prod(S)),
            "max_height": int(round(max((e["box"][2] + e["box"][5] for e in on),
                                        default=0) * cm_cell)),
            "steps": [on[0]["step"], on[-1]["step"]] if on else None,
            "logs": [on[0]["log"], on[-1]["log"]] if on else None,
            "times": [on[0]["time"], on[-1]["time"]] if on else None,
            "failed": sum(bool(e["failed"]) for e in real if e["pallet"] == k)})
    return out


def summary(path, n_types=3, tol_cm=TOL_CM):
    """What the page says about a log before it is run."""
    steps = read_log(X.resolve(path), n_types)
    boxes = track(steps, tol_cm)
    real = real_pack(steps, 1.0)
    return {"steps": len(steps), "rows": sum(len(s["rows"]) for s in steps),
            "real": None if real is None else {
                "packed": sum(e["box"] is not None for e in real),
                "failed": sum(bool(e["failed"]) for e in real),
                "pallets": real[-1]["pallet"] + 1},
            "boxes": len(boxes),
            "picks": sum(s["real"] is not None for s in steps),
            "types": np.bincount([b["type"] for b in boxes],
                                 minlength=n_types).tolist(),
            "logs": [steps[0]["log"], steps[-1]["log"]] if steps else None,
            "first": steps[0]["time"] if steps else "",
            "last": steps[-1]["time"] if steps else ""}


def cells(cm, p):
    """A box's sides in cells, read as the big experiment reads a file."""
    return box_cells(dict(zip("lwh", cm)), p["cell_cm"], "log box",
                     box_divisor(p["box_scale"]), box_rounding(p["box_round"]),
                     box_pad_cm(p.get("box_pad_m", 0.0)), "lwh")


# ---------------------------------------------------------------- the run
def load(env, items):
    """Make `items` the conveyor from the env's head on: the station as it
    stands, with nothing behind it."""
    h = int(env.head[0])
    env.seq[0, h:h + len(items)] = items
    env.length[0] = h + len(items)
    env.done[0] = False
    env._invalidate(hmap=False)


def start(path, spec, p, cell_m, device, tol_cm=TOL_CM):
    """Read, track and pack a log in a thread.  -> job id."""
    q = X.resolve(path)
    if not os.path.isfile(q):
        raise ValueError(f"no such log: {path}")
    steps = read_log(q, p["n_types"])
    if not steps:
        raise ValueError(f"{path} holds no boxes")
    boxes = track(steps, tol_cm)
    S = tuple(orders_bin(p["pallet_cm"], p["cell_cm"]) if p["pallet_cm"]
              else p["bin"])
    for b in boxes:
        b["cells"] = cells(b["cm"], p)
        b["fits"] = G.fits(b["cells"], S, p["rot"])
    if not any(b["fits"] for b in boxes):
        raise ValueError(f"not one box of {path} fits a "
                         f"{'x'.join(map(str, S))} pallet")
    job = {"id": uuid.uuid4().hex[:10], "state": "running", "error": None,
           "spec": spec, "label": spec, "picker": None, "log": X._show(q),
           "bin": list(S), "p": p, "tol_cm": tol_cm,
           "n_steps": len(steps), "logs": [steps[0]["log"], steps[-1]["log"]], "at": 0, "drain": 0, "rows": sum(
               len(s["rows"]) for s in steps), "boxes": len(boxes),
           "picks": sum(s["real"] is not None for s in steps),
           "packed": 0, "skipped": [], "station": 0, "cur": None,
           "pallets": [], "match": [0, 0], "t0": time.time(), "t1": None,
           "stop": threading.Event(), "lock": threading.Lock(), "full": [],
           # the robot's own packing, on the agent's grid, when the log has it
           "real": real_pack(steps, p["cell_cm"] * box_divisor(p["box_scale"]), p)}
    job["real_pallets"] = real_pallets(job["real"], S,
                                       p["cell_cm"] * box_divisor(p["box_scale"]))
    with _LOCK:
        if any(j["state"] == "running" for j in _JOBS.values()):
            raise ValueError("a log match is already running; stop it first")
        _JOBS[job["id"]] = job
        _JOBS["latest"] = job
        for old in [k for k in _JOBS if k != "latest"][:-JOBS_KEPT]:
            _JOBS.pop(old)
    threading.Thread(target=_run, args=(job, steps, boxes, cell_m, device),
                     daemon=True).start()
    return job["id"]


def _run(job, steps, boxes, cell_m, device):
    p, S = job["p"], tuple(job["bin"])
    try:
        policy, picker, label = X.packer(job["spec"], p["n_pick"], device)
        job["label"], job["policy"] = label, policy
        job["picker"] = ("its own selector" if picker is not None
                         else "the first in reach that fits"
                         if p["n_pick"] > 1 else None)
        Pack(job, steps, boxes, policy, picker, S, cell_m).run()
        job["state"] = "stopped" if job["stop"].is_set() else "done"
    except Exception as e:                    # the page shows it, and stops
        job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"
    job["t1"] = time.time()


class Pack:
    """One log, packed: the station, the pallet being packed, and the
    pallets closed so far (on the job)."""

    def __init__(self, job, steps, boxes, policy, picker, S, cell_m):
        self.job, self.steps, self.boxes = job, steps, boxes
        self.policy, self.picker, self.S, self.cell_m = policy, picker, S, cell_m
        self.p = job["p"]
        # every box that fits: the env's conveyor is sized, and its sweeps
        # cover, all of them, whichever are at the station
        self.items = {b["id"]: np.array(b["cells"] + (b["type"],), np.int16)
                      for b in boxes if b["fits"]}
        self.seq = np.stack(list(self.items.values()))
        self.station = []                     # box ids, in the order they came
        self.packed = set()
        self.env = self.pal = None
        self.t0 = time.time()

    def run(self):
        job = self.job
        self.open()
        for k, st in enumerate(self.steps):
            if job["stop"].is_set():
                return
            job["at"] = k + 1
            for i in st["new"]:
                if i in self.items:
                    self.station.append(i)
                else:
                    job["skipped"].append(i)
            self.decide(st)
        # the log is over: what is left is packed on, a box per step
        while self.station and not job["stop"].is_set():
            job["drain"] += 1
            self.decide(None)
        if self.env.n_packed[0]:
            self.close("the log ran out")

    def open(self):
        self.env = X.pallet_env(self.seq, self.S, self.p, self.cell_m)
        self.pal = {"i": len(self.job["pallets"]), "windows": [], "snaps": [],
                    "cands": [], "taken": [], "cand_probs": [], "steps": [],
                    "match": [0, 0], "ms": 0}

    def view(self, st, pick=-1, probs=None):
        """The station as the page draws it: the first `nb` boxes, with
        which came at this step and which the robot took there."""
        vis = self.station[:self.p["nb"]]
        real = None if st is None else st["real"]
        new = set() if st is None else set(st["new"])
        where = (None if real is None else "view" if real in vis
                 else "waiting" if real in self.station
                 else "packed" if real in self.packed else "skipped")
        return {"w": [self.items[i][:3].tolist() for i in vis],
                "t": [int(self.items[i][3]) for i in vis],
                "ids": vis, "pick": pick, "probs": probs,
                "real": vis.index(real) if where == "view" else -1,
                "real_at": where,
                "new": [j for j, i in enumerate(vis) if i in new],
                "step": None if st is None else st["step"],
                "log": None if st is None else st["log"],
                "time": None if st is None else st["time"],
                "waiting": len(self.station)}

    def decide(self, st):
        """Pack one box of the station, opening pallets as they fill."""
        env, job, bad = self.env, self.job, set()
        while self.station:
            load(env, [self.items[i] for i in self.station])
            k = min(self.p["n_pick"], self.p["nb"], len(self.station))
            ok = env._placeable()[0, :k].copy()
            ok[list(bad)] = False
            if not ok.any():
                if env.n_packed[0] == 0:
                    # nothing in reach fits an empty pallet: that box cannot be
                    # placed by this packer at all (the arm filter, say)
                    job["skipped"].append(self.station.pop(0))
                    continue
                self.close("no box in reach has a place", st)
                self.open()
                env, bad = self.env, set()
                continue
            snap = G.env_state(env)
            first = int(np.flatnonzero(ok)[0])
            i, probs = first, None
            if self.picker is not None:
                idx, pr = self.picker(env)
                probs = G.rounded(pr[0][:min(len(self.station), self.p["nb"])])
                j = int(idx[0])
                # a box it may not take, or one with nowhere to go
                i = j if j < k and ok[j] else first
            env.permute(np.array([i]))
            pts = G.cand_points(env)
            if not pts:               # the rules let it, the candidates do not
                bad.add(i)
                continue
            act, prob = self.policy(env)
            win = self.view(st, i, probs)
            env.step(act)
            box = self.station.pop(i)
            self.packed.add(box)
            pal = self.pal
            pal["snaps"].append(snap)
            pal["windows"].append(win)
            pal["cands"].append(pts)
            pal["taken"].append(int(act[0]))
            pal["cand_probs"].append(G.rounded(prob[0][:len(pts)]))
            if 0 <= win["real"] < k:      # the robot's box was within reach
                hit = int(win["real"] == i)
                for m in (pal["match"], job["match"]):
                    m[0] += hit
                    m[1] += 1
            job["packed"] += 1
            job["station"] = len(self.station)
            job["cur"] = {"util": round(float(env.utilization()[0]) * 100, 1),
                          "boxes": int(env.n_packed[0])}
            return

    def close(self, reason, st=None):
        """Close the pallet being packed: its summary on the job, and all
        a replay needs kept beside it."""
        env, pal, job = self.env, self.pal, self.job
        n = int(env.n_packed[0])
        if self.station:          # what it was left with, and why it stopped
            load(env, [self.items[i] for i in self.station])
            pal["snaps"].append(G.env_state(env))
            pal["windows"].append(self.view(st))
        types = env.ptype[0, :n]
        cm = self.p["cell_cm"] * box_divisor(self.p["box_scale"])
        steps = [w["step"] for w in pal["windows"][:n] if w["step"] is not None]
        logs = [w["log"] for w in pal["windows"][:n] if w["log"] is not None]
        times = [w["time"] for w in pal["windows"][:n] if w["time"]]
        row = {"i": pal["i"], "boxes": n, "reason": reason,
               "util": float(env.utilization()[0]),
               "max_height": int(round(int(env.hmap[0].max()) * cm)),
               "types": [int((types == t).sum()) for t in range(3)],
               "steps": [steps[0], steps[-1]] if steps else None,
               "logs": [logs[0], logs[-1]] if logs else None,
               "times": [times[0], times[-1]] if times else None,
               "after_log": sum(w["step"] is None for w in pal["windows"][:n]),
               "match": pal["match"],
               "ms": int((time.time() - self.t0) * 1000)}
        self.t0 = time.time()
        pal.update(row, env=env,
                   placed=(env.packed[0, :n] * env.scale).round().astype(int).tolist(),
                   placed_types=env.ptype[0, :n].tolist())
        job["full"].append(pal)
        job["pallets"].append(row)
        job["cur"] = None


# ----------------------------------------------------------- the page's view
def get(jid):
    job = _JOBS.get(jid or "latest")
    if job is None:
        raise ValueError("that log match is no longer kept; run it again")
    return job


def status(jid):
    """What the page draws while it runs: the job, and every pallet's
    summary.  None when nothing has run."""
    job = _JOBS.get(jid or "latest")
    if job is None:
        return None
    now = job["t1"] or time.time()
    out = {k: v for k, v in job.items() if k in (
        "id", "state", "error", "spec", "label", "picker", "log", "bin",
        "tol_cm", "n_steps", "logs", "at", "drain", "rows", "boxes", "picks",
        "packed", "station", "cur", "pallets", "match")}
    real = job["real"]
    out.update(p=job["p"], skipped=len(job["skipped"]),
               real=None if real is None else {
                   "packed": sum(e["box"] is not None for e in real),
                   "failed": sum(bool(e["failed"]) for e in real),
                   "pallets": real[-1]["pallet"] + 1},
               real_pallets=job["real_pallets"],
               elapsed=round(now - job["t0"], 1))
    return out


def pallet(jid, k):
    """One closed pallet, everything its result view draws."""
    job = get(jid)
    try:
        pal = job["full"][int(k)]
    except (IndexError, TypeError, ValueError):
        raise ValueError(f"no pallet {k} in that run") from None
    return {k2: v for k2, v in pal.items() if k2 not in ("env", "snaps")} | {
        "label": job["label"], "picker": job["picker"], "bin": job["bin"],
        "params": job["p"], "real": job["real"],
        "real_pallets": job["real_pallets"]}


def points(jid, k, n, i):
    """Where box `i` of the station could go at step `n` of pallet `k`, as
    `game.points` answers it for a game: the packer's candidates, its odds
    over them, and the positions its action space offers that the rules
    turn down.  The step past the last box is the station it was closed on."""
    job = get(jid)
    try:
        pal = job["full"][int(k)]
    except (IndexError, TypeError, ValueError):
        raise ValueError(f"no pallet {k} in that run") from None
    with job["lock"]:
        snaps, env = pal["snaps"], pal["env"]
        n = max(0, min(int(n), len(snaps) - 1))
        G.set_state(env, snaps[n])
        i = int(i)
        reach = min(env.n_pick, env.nb, int(env.length[0] - env.head[0]))
        if not 0 <= i < reach:
            raise ValueError(f"box {i + 1} is not within reach at that step")
        env.done[:] = False
        env.permute(np.array([i]))
        cands = G.cand_points(env)
        feas, z, odims, _ = env._positions()
        bad = G.turned_down(env, feas, z, odims)
        probs = (G.rounded(job["policy"](env)[1][0][:len(cands)])
                 if cands else None)
        capped = int(feas.sum()) - len(cands)
    return {"n": n, "i": i, "cands": cands, "probs": probs, "bad": bad,
            "capped": capped}


def arm(jid, placed, S, moves=None):
    """The arm placing the last of `placed`, as `game.arm_replay` does for a
    game: the job stands in for the session, its rules and its arm."""
    job = get(jid)
    with job["lock"]:
        return G.arm_replay(job, placed, S, moves)


def stop(jid):
    job = _JOBS.get(jid or "latest")
    if job is not None:
        job["stop"].set()
    return job is not None
