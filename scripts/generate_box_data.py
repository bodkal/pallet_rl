#!/usr/bin/env python3
"""Synthetic pallet days that look like data/box_pallet_data/use_data.

    python3 scripts/generate_box_data.py                 # -> generated_data/1.csv .. 20.csv
    python3 scripts/generate_box_data.py --n 20 --seed 2024

Each generated day takes the *shape* of one real day -- how many pallets,
roughly how many boxes of each type each pallet holds, how many pallets are
built at once, which hours the line ran -- and jitters every part of it, so
no generated day is a copy of a real one.  The values themselves are new:

  - type 1 and type 2 boxes keep the one depth x width their type has and one
    of its two heights, as the real data does (the size *is* the type);
  - type 3 boxes come from a fresh catalog of SKUs, each a sibling of a real
    SKU moved a few percent per side.  No generated triple equals a real one,
    and every side and every volume stays inside the real type-3 range;
  - container and pallet ids are new, in each type's own id format, and the
    type-3 labels keep counting up from where the real ones stop.

TRAN_HOUR is written the way the source system writes it: hour, *month*,
second (07:06:02 on a June day), and every file is sorted on that string, so
boxes inside an hour come out in second order exactly like the real files.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import itertools
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "box_pallet_data")
COLS = ["TRAN_DATE", "TRAN_HOUR", "CONTAINER_ID", "PALLET_ID", "FROM_LOC_ID",
        "V_BOX_DEPTH", "V_BOX_WIDTH", "V_BOX_HEIGHT", "TYPE"]
DIMS = ["V_BOX_DEPTH", "V_BOX_WIDTH", "V_BOX_HEIGHT"]
TYPES = (1, 2, 3)

# How far a generated day may drift from the real day it is modelled on.
JITTER_N = 0.12          # log-sd of each pallet's box count
MIX_CONC = 40.0          # Dirichlet concentration of a pallet's type mix
HOUR_CONC = 60.0         # Dirichlet concentration of the hour profile
TALL_LOGIT_SD = 0.15     # per-day tall/short split of types 1 and 2
# Type-3 catalog: a sibling SKU moves this much per side (log-sd), and its
# popularity this much (log-sd).  NOVEL_FRAC more SKUs are drawn wider, with
# no real popularity behind them -- the long tail of one-off cartons.
SKU_SD = 0.06
SKU_POP_SD = 0.35
NOVEL_FRAC = 0.15
NOVEL_SD = 0.12
# Day-level SKU choice: a day's weights are Gamma(DAY_CONC * popularity), so a
# day leans on a handful of SKUs; REPEAT_P is the chance a type-3 box is the
# same SKU as the type-3 box before it on its pallet.  Both are calibrated so
# distinct-SKUs-per-day and same-SKU-in-a-row match the real files.
DAY_CONC = 0.2
REPEAT_P = 0.31
STAY_P = 0.0             # type 1/2: keep the previous box's height


# ---------------------------------------------------------------- reading
def read_days(src):
    days = []
    for f in sorted(glob.glob(os.path.join(src, "*.csv"))):
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            days.append({"file": os.path.basename(f), "rows": rows,
                         "date": dt.date.fromisoformat(rows[0]["TRAN_DATE"])})
    return days


def dims(r):
    return tuple(float(r[c]) for c in DIMS)


# ---------------------------------------------------------------- fitting
def fit(days):
    """Everything the generator needs, measured from the real days."""
    rows = [r for d in days for r in d["rows"]]
    m = {"fixed": {}, "bounds": {}, "days": []}

    for t in (1, 2):
        sz = Counter(dims(r) for r in rows if int(r["TYPE"]) == t)
        dw = {s[:2] for s in sz}
        hs = sorted({s[2] for s in sz}, reverse=True)
        assert len(dw) == 1 and len(hs) == 2, f"type {t}: {sorted(sz)}"
        (d, w), = dw
        m["fixed"][t] = {"d": d, "w": w, "tall": hs[0], "short": hs[1]}

    t3 = np.array([dims(r) for r in rows if int(r["TYPE"]) == 3])
    vol = t3.prod(1)
    m["bounds"] = {"lo": t3.min(0), "hi": t3.max(0),
                   "vlo": vol.min(), "vhi": vol.max()}
    m["skus"] = Counter((r["V_BOX_DEPTH"], r["V_BOX_WIDTH"], r["V_BOX_HEIGHT"])
                        for r in rows if int(r["TYPE"]) == 3)

    # where in its pallet a box of each type tends to sit (method of moments)
    pos = defaultdict(list)
    for d in days:
        by = defaultdict(list)
        for r in d["rows"]:
            by[r["PALLET_ID"]].append(int(r["TYPE"]))
        for seq in by.values():
            if len(seq) >= 5:
                for i, t in enumerate(seq):
                    pos[t].append((i + 0.5) / len(seq))
    m["beta"] = {}
    for t in TYPES:
        p = np.array(pos[t])
        k = p.mean() * (1 - p.mean()) / p.var() - 1
        m["beta"][t] = (p.mean() * k, (1 - p.mean()) * k)

    for d in days:
        seq = [r["PALLET_ID"] for r in d["rows"]]
        order = list(dict.fromkeys(seq))
        comp = [[sum(1 for r in d["rows"] if r["PALLET_ID"] == p and int(r["TYPE"]) == t)
                 for t in TYPES] for p in order]
        first = {p: seq.index(p) for p in order}
        last = {p: len(seq) - 1 - seq[::-1].index(p) for p in order}
        open_at = [sum(first[p] <= i <= last[p] for p in order) for i in range(len(seq))]
        hours = np.bincount([int(r["TRAN_HOUR"][:2]) for r in d["rows"]], minlength=24)
        tall = {t: np.mean([dims(r)[2] == m["fixed"][t]["tall"] for r in d["rows"]
                            if int(r["TYPE"]) == t] or [np.nan]) for t in (1, 2)}
        m["days"].append({"comp": comp, "hours": hours, "tall": tall, "n": len(seq),
                          # boxes placed while another pallet was also open
                          "overlap": sum(c >= 2 for c in open_at) / len(seq)})
    m["n_lo"] = min(d["n"] for d in m["days"])
    m["n_hi"] = max(d["n"] for d in m["days"])

    # overlap switching: inside a stretch where 2+ pallets are open, how often
    # the next box goes to a different pallet than the last one
    sw = tot = 0
    for d in days:
        seq = [r["PALLET_ID"] for r in d["rows"]]
        first = {p: seq.index(p) for p in set(seq)}
        last = {p: len(seq) - 1 - seq[::-1].index(p) for p in set(seq)}
        for i in range(1, len(seq)):
            if sum(first[p] <= i <= last[p] for p in first) >= 2:
                tot += 1
                sw += seq[i] != seq[i - 1]
    m["switch_p"] = sw / max(tot, 1)

    ids = {t: np.array([int(r["CONTAINER_ID"]) for r in rows if int(r["TYPE"]) == t])
           for t in TYPES}
    m["ids"] = ids
    m["real_ids"] = {r["CONTAINER_ID"] for r in rows}
    # type-3 labels are printed in sequence: ids per day, and the spacing of
    # consecutive labels inside a day
    dates = np.array([(d["date"] - days[0]["date"]).days for d in days for r in d["rows"]
                      if int(r["TYPE"]) == 3])
    m["t3_rate"] = np.polyfit(dates, ids[3], 1)[0]
    gaps = []
    for d in days:
        v = np.sort([int(r["CONTAINER_ID"]) for r in d["rows"] if int(r["TYPE"]) == 3])
        gaps += list(np.diff(v))
    m["t3_gap"] = np.array([g for g in gaps if g > 0])
    m["t3_max"] = int(ids[3].max())
    m["pallet_max"] = max(int(r["PALLET_ID"]) for r in rows)
    m["real_pallets"] = {r["PALLET_ID"] for r in rows}
    m["last_date"] = days[-1]["date"]
    m["date_gaps"] = np.diff([d["date"].toordinal() for d in days])
    m["carry_p"] = np.mean([bool({r["PALLET_ID"] for r in a["rows"]} &
                                 {r["PALLET_ID"] for r in b["rows"]})
                            for a, b in zip(days, days[1:])])
    m["loc"] = rows[0]["FROM_LOC_ID"]
    return m


# ---------------------------------------------------------------- catalog
def _round_like(v, ref):
    """Round to 0.5 cm where the real side is a whole or half cm, else 0.1."""
    step = 0.5 if (round(float(ref) * 10) % 5 == 0) else 0.1
    return round(round(v / step) * step, 1)


def _in_bounds(s, b):
    s = np.asarray(s)
    return bool((s >= b["lo"]).all() and (s <= b["hi"]).all()
                and b["vlo"] <= s.prod() <= b["vhi"])


def make_catalog(m, rng):
    """New type-3 SKUs: a sibling of every real SKU, plus a novel tail."""
    real = {tuple(float(x) for x in k) for k in m["skus"]}
    keys = list(m["skus"])
    cnt = np.array([m["skus"][k] for k in keys], float)
    out, seen = [], set()

    def draw(parent, sd):
        p = np.array([float(x) for x in parent])
        for _ in range(200):
            s = p * np.exp(rng.normal(0, sd, 3))
            s = np.array([_round_like(v, r) for v, r in zip(s, parent)])
            if (p[0] >= p[1]) != (s[0] >= s[1]):          # keep its footprint's way round
                s[[0, 1]] = s[[1, 0]]
            t = tuple(s)
            if _in_bounds(s, m["bounds"]) and t not in real and t not in seen:
                seen.add(t)
                return t
        return None

    for k, c in zip(keys, cnt):
        s = draw(k, SKU_SD)
        if s:
            out.append((s, c * np.exp(rng.normal(0, SKU_POP_SD))))
    for _ in range(int(NOVEL_FRAC * len(keys))):
        s = draw(keys[rng.integers(len(keys))], NOVEL_SD)
        if s:
            out.append((s, 1.0))
    skus = [s for s, _ in out]
    pop = np.array([w for _, w in out])
    return skus, pop / pop.sum()


# ---------------------------------------------------------------- one day
def _jitter_share(p, conc, rng, floor=0.0):
    p = np.asarray(p, float)
    on = p > 0
    out = np.zeros_like(p)
    out[on] = rng.dirichlet(conc * p[on] / p[on].sum() + floor + 1e-3)
    return out


def make_day(m, tmpl, skus, pop, rng):
    """One day's boxes in true time order: (pallet index, type, dims, t-seconds)."""
    # pallets: each real pallet's size and type mix, jittered; a pure pallet
    # stays pure
    comp = []
    for c in tmpl["comp"]:
        n = max(1, int(round(sum(c) * np.exp(rng.normal(0, JITTER_N)))))
        share = _jitter_share(c, MIX_CONC, rng)
        comp.append(rng.multinomial(n, share))
    total = sum(int(c.sum()) for c in comp)
    if not m["n_lo"] <= total <= m["n_hi"]:                 # keep the day in range
        f = np.clip(total, m["n_lo"], m["n_hi"]) / total
        comp = [np.maximum(np.round(c * f).astype(int), 0) for c in comp]
        comp = [c if c.sum() else np.array([0, 0, 1]) for c in comp]

    # each pallet's order: types sit where they sit on real pallets
    tall = {}
    for t in (1, 2):
        p = tmpl["tall"][t]
        p = np.nanmean([d["tall"][t] for d in m["days"]]) if np.isnan(p) else p
        lg = np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
        tall[t] = 1 / (1 + np.exp(-(lg + rng.normal(0, TALL_LOGIT_SD))))
    w = rng.gamma(DAY_CONC * pop * len(pop) + 1e-6)
    day_q = w / w.sum()

    pallets = []
    for c in comp:
        types = np.repeat(TYPES, c)
        key = np.array([rng.beta(*m["beta"][t]) for t in types])
        types = types[np.argsort(key, kind="stable")]
        seq, prev = [], {}
        for t in types:
            if t in (1, 2):
                f = m["fixed"][t]
                if t in prev and rng.random() < STAY_P:
                    h = prev[t]
                else:
                    h = f["tall"] if rng.random() < tall[t] else f["short"]
                prev[t] = h
                seq.append((t, (f["d"], f["w"], h)))
            else:
                if 3 in prev and rng.random() < REPEAT_P:
                    s = prev[3]
                else:
                    s = skus[rng.choice(len(skus), p=day_q)]
                prev[3] = s
                seq.append((3, s))
        pallets.append(seq)

    # hand-over: pallets follow each other, and around each change the end of
    # one and the start of the next share the line for a while -- as long,
    # summed over the day, as on the real day
    seqs = [[(i, t, s) for t, s in p] for i, p in enumerate(pallets)]
    n = len(seqs)
    budget = int(round(tmpl["overlap"] * sum(map(len, seqs)) * np.exp(rng.normal(0, 0.2))))
    o = rng.multinomial(budget, [1 / (n - 1)] * (n - 1)) if n > 1 else []
    tail, head = [0] * n, [0] * n
    for i in range(n - 1):
        tail[i] = min((o[i] + 1) // 2, len(seqs[i]) - head[i])
        head[i + 1] = min(o[i] - tail[i], len(seqs[i + 1]))
    out = []
    for i, sq in enumerate(seqs):
        out += sq[head[i]: len(sq) - tail[i]]
        if i < n - 1:
            a, b = sq[len(sq) - tail[i]:], seqs[i + 1][:head[i + 1]]
            cur = 0
            while a or b:
                if not (a, b)[cur] or ((a and b) and rng.random() < m["switch_p"]):
                    cur = 1 - cur
                out.append((a, b)[cur].pop(0))

    # clock: the real day's hour profile, jittered, every hour the real day
    # worked keeping at least one box.  Inside an hour the real seconds rise
    # with the box order (the minute is lost to the month), so sorting on the
    # stamp keeps runs of one SKU together -- do the same.
    on = tmpl["hours"] > 0
    share = np.zeros(24)
    share[on] = rng.dirichlet(HOUR_CONC * tmpl["hours"][on] / tmpl["hours"].sum() + 0.3)
    base = on.astype(int) if len(out) >= on.sum() else np.zeros(24, int)
    per_hour = base + rng.multinomial(len(out) - base.sum(), share)
    secs = np.concatenate([h * 3600 + np.sort(rng.integers(0, 60, k))
                           for h, k in enumerate(per_hour)])
    return [(p, t, s, int(x)) for (p, t, s), x in zip(out, secs)]


# ---------------------------------------------------------------- ids
class Ids:
    def __init__(self, m, rng):
        self.m, self.rng = m, rng
        self.used = set(m["real_ids"])
        self.t3 = m["t3_max"]
        self.pallet = m["pallet_max"] + int(rng.integers(40, 160))
        self.used_pallets = set(m["real_pallets"])

    def container(self, t, date):
        if t == 3:                               # printed in sequence, never reused
            due = self.m["t3_max"] + int(self.m["t3_rate"] * (date - self.m["last_date"]).days)
            self.t3 = max(self.t3, due - int(self.rng.integers(400, 1600)))
            self.t3 += int(self.rng.choice(self.m["t3_gap"]))
            self.used.add(str(self.t3))
            return self.t3
        ref = self.m["ids"][t]
        lo, hi = int(ref.min()), int(ref.max())
        spread = 0.02 * (hi - lo)
        while True:                              # near a real id, never on one
            v = int(np.clip(ref[self.rng.integers(len(ref))]
                            + self.rng.normal(0, spread), lo, hi))
            if str(v) not in self.used:
                self.used.add(str(v))
                return v

    def new_pallet(self):
        while True:
            self.pallet += int(self.rng.integers(1, 9))
            if str(self.pallet) not in self.used_pallets:
                self.used_pallets.add(str(self.pallet))
                return self.pallet


def next_date(m, d, rng):
    d = d + dt.timedelta(days=int(rng.choice(m["date_gaps"])))
    while d.weekday() in (4, 5):                 # Fri, Sat: no real day falls there
        d += dt.timedelta(days=1)
    return d


# ---------------------------------------------------------------- main
def fmt(v):
    return f"{v:.1f}"


def generate(src, dst, n, seed):
    rng = np.random.default_rng(seed)
    days = read_days(src)
    m = fit(days)
    skus, pop = make_catalog(m, rng)
    ids = Ids(m, rng)
    tmpls = rng.choice(len(m["days"]), size=n, replace=n > len(m["days"]))
    os.makedirs(dst, exist_ok=True)
    date, last_pallet = m["last_date"], None
    for k, ti in enumerate(tmpls, start=1):
        date = next_date(m, date, rng)
        boxes = make_day(m, m["days"][ti], skus, pop, rng)
        n_pal = 1 + max(p for p, *_ in boxes)
        pid = [ids.new_pallet() for _ in range(n_pal)]
        if last_pallet and n_pal > 1 and rng.random() < m["carry_p"]:
            pid[0] = last_pallet                 # yesterday's pallet, finished today
        last_pallet = pid[boxes[-1][0]]
        rows = []
        for p, t, s, x in boxes:
            h, sec = x // 3600, x % 3600
            rows.append({"TRAN_DATE": date.isoformat(),
                         # the source system's format: hour, MONTH, second
                         "TRAN_HOUR": f"{h:02d}:{date.month:02d}:{sec:02d}",
                         "CONTAINER_ID": ids.container(t, date),
                         "PALLET_ID": pid[p], "FROM_LOC_ID": m["loc"],
                         "V_BOX_DEPTH": fmt(s[0]), "V_BOX_WIDTH": fmt(s[1]),
                         "V_BOX_HEIGHT": fmt(s[2]), "TYPE": t})
        rows.sort(key=lambda r: r["TRAN_HOUR"])     # stable, like the real export
        with open(os.path.join(dst, f"{k}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, COLS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    check(m, dst, n)
    return m


def check(m, dst, n):
    """The rules, asserted on what was written."""
    real_skus = {tuple(float(x) for x in k) for k in m["skus"]}
    b = m["bounds"]
    for k in range(1, n + 1):
        with open(os.path.join(dst, f"{k}.csv"), newline="") as fh:
            for r in csv.DictReader(fh):
                t, s = int(r["TYPE"]), dims(r)
                if t in (1, 2):
                    f = m["fixed"][t]
                    assert s[:2] == (f["d"], f["w"]) and s[2] in (f["tall"], f["short"]), (k, r)
                else:
                    assert _in_bounds(s, b), (k, r)
                    assert s not in real_skus, (k, r)
                assert r["CONTAINER_ID"] not in m["real_ids"], (k, r)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", default=os.path.join(DATA, "use_data"))
    p.add_argument("--dst", default=os.path.join(DATA, "generated_data"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=2024)
    a = p.parse_args()
    generate(a.src, a.dst, a.n, a.seed)
    print(f"wrote {a.n} files to {os.path.relpath(a.dst, ROOT)}")


if __name__ == "__main__":
    main()
