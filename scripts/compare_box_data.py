#!/usr/bin/env python3
"""Compare generated pallet days with the real ones and write an HTML report.

    python3 scripts/compare_box_data.py          # -> generated_data/report.html

Reads data/box_pallet_data/use_data (real) and generated_data (synthetic),
measures both the same way, and fills scripts/box_report_template.html with
the numbers.  `--json` also writes the raw numbers next to the report.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
import datetime as dt

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "box_pallet_data")
TEMPLATE = os.path.join(os.path.dirname(__file__), "box_report_template.html")
DIMS = ["V_BOX_DEPTH", "V_BOX_WIDTH", "V_BOX_HEIGHT"]


def _num_key(path):
    s = os.path.basename(path)[:-4]
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def read(folder):
    days = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv")), key=_num_key):
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["t"] = int(r["TYPE"])
            r["d"] = tuple(float(r[c]) for c in DIMS)
        days.append({"file": os.path.basename(f), "rows": rows,
                     "date": rows[0]["TRAN_DATE"]})
    return days


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov D: the widest gap between the CDFs."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a)
                               - np.searchsorted(b, x, "right") / len(b))))


def ks_crit(n, m):
    return 1.36 * np.sqrt((n + m) / (n * m))


def summary(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "q": [float(x) for x in np.percentile(v, [10, 25, 50, 75, 90])]}


def hist(a, b, bins):
    ha = np.histogram(a, bins)[0] / len(a)
    hb = np.histogram(b, bins)[0] / len(b)
    return {"edges": [float(x) for x in bins], "real": ha.tolist(), "gen": hb.tolist()}


def measure(days):
    rows = [r for d in days for r in d["rows"]]
    out = {"files": len(days), "boxes": len(rows)}
    out["pallets"] = len({r["PALLET_ID"] for r in rows})
    out["types"] = {t: sum(r["t"] == t for r in rows) for t in (1, 2, 3)}
    t3 = np.array([r["d"] for r in rows if r["t"] == 3])
    out["t3"] = t3
    out["t3_skus"] = Counter(map(tuple, t3))
    out["tall"] = {}
    for t in (1, 2):
        hs = [r["d"][2] for r in rows if r["t"] == t]
        out["tall"][t] = {"tall": max(hs), "short": min(hs),
                          "share": float(np.mean(np.array(hs) == max(hs)))}
    per_day, pallets = [], []
    for d in days:
        rs = d["rows"]
        sk = [r["d"] for r in rs if r["t"] == 3]
        pal = defaultdict(list)
        for r in rs:
            pal[r["PALLET_ID"]].append(r)
        hours = sorted({int(r["TRAN_HOUR"][:2]) for r in rs})
        per_day.append({"file": d["file"], "date": d["date"], "boxes": len(rs),
                        "pallets": len(pal),
                        "types": [sum(r["t"] == t for r in rs) for t in (1, 2, 3)],
                        "skus": len(set(sk)), "t3": len(sk),
                        "first": hours[0], "last": hours[-1], "hours": len(hours),
                        "vol": sum(np.prod(r["d"]) for r in rs) / 1e6})
        for pid, v in pal.items():
            pallets.append({"file": d["file"], "id": pid, "n": len(v),
                            "vol": sum(np.prod(r["d"]) for r in v) / 1e6,
                            "boxes": [[*r["d"], r["t"]] for r in v]})
    out["days"], out["pallet_list"] = per_day, pallets
    out["hours"] = (np.bincount([int(r["TRAN_HOUR"][:2]) for r in rows], minlength=24)
                    / len(rows)).tolist()
    # where each type sits in its pallet, 0 = first box, 1 = last
    pos = defaultdict(list)
    for p in pallets:
        n = p["n"]
        if n >= 5:
            for i, b in enumerate(p["boxes"]):
                pos[b[3]].append((i + 0.5) / n)
    out["pos"] = {t: pos[t] for t in (1, 2, 3)}
    # same size twice in a row, per type
    rep = {}
    for t in (1, 2, 3):
        s = n = 0
        for d in days:
            for a, b in zip(d["rows"], d["rows"][1:]):
                if a["t"] == b["t"] == t:
                    n += 1
                    s += a["d"] == b["d"]
        rep[t] = s / max(n, 1)
    out["repeat"] = rep
    # how pallets share the line: share of boxes landing while 2+ pallets are open
    ov = tot = 0
    for d in days:
        seq = [r["PALLET_ID"] for r in d["rows"]]
        first = {p: seq.index(p) for p in set(seq)}
        last = {p: len(seq) - 1 - seq[::-1].index(p) for p in set(seq)}
        for i in range(len(seq)):
            tot += 1
            ov += sum(first[p] <= i <= last[p] for p in first) >= 2
    out["overlap"] = ov / tot
    out["ids"] = {r["CONTAINER_ID"] for r in rows}
    out["pallet_ids"] = {r["PALLET_ID"] for r in rows}
    out["dates"] = [d["date"] for d in days]
    out["id_len"] = {t: Counter(len(r["CONTAINER_ID"]) for r in rows if r["t"] == t)
                     for t in (1, 2, 3)}
    out["stamp_ok"] = float(np.mean([r["TRAN_HOUR"].split(":")[1] == r["TRAN_DATE"][5:7]
                                     for r in rows]))
    out["sorted"] = all([r["TRAN_HOUR"] for r in d["rows"]] ==
                        sorted(r["TRAN_HOUR"] for r in d["rows"]) for d in days)
    return out


def rel(a, b):
    return abs(b - a) / abs(a) if a else abs(b - a)


def scores(R, G):
    """How far G sits from R on each property, one row per property.

    `score` is a gap that is 0 for a perfect match: relative error for a
    single number, the KS D for a distribution, the total-variation distance
    for the hour profile, and the move in mean position for box order.
    """
    out = []

    def add(group, name, r, g, unit, s, how):
        out.append({"group": group, "name": name, "real": float(r), "gen": float(g),
                    "unit": unit, "score": float(s), "how": how})

    def mean(M, k):
        return float(np.mean([d[k] for d in M["days"]]))

    def ratio(M):
        return float(np.mean([d["skus"] / d["t3"] for d in M["days"] if d["t3"]]))

    def mpos(M, t):
        return float(np.mean(M["pos"][t]))

    tr, tg = sum(R["types"].values()), sum(G["types"].values())
    for t, nm in ((1, "Blue boxes"), (2, "White boxes"), (3, "Brown cartons")):
        r, g = R["types"][t] / tr, G["types"][t] / tg
        add("Type mix", f"{nm}, share of all boxes", r, g, "%", rel(r, g), "relative")
    for t, nm in ((1, "Blue boxes that are 28 cm tall"), (2, "White boxes that are 25 cm tall")):
        r, g = R["tall"][t]["share"], G["tall"][t]["share"]
        add("Heights", nm, r, g, "%", rel(r, g), "relative")
    for i, nm in enumerate(("Depth", "Width", "Height")):
        a, b = R["t3"][:, i], G["t3"][:, i]
        add("Brown carton sizes", nm, a.mean(), b.mean(), "cm", ks(a, b), "ks")
    a, b = R["t3"].prod(1) / 1000, G["t3"].prod(1) / 1000
    add("Brown carton sizes", "Volume", a.mean(), b.mean(), "L", ks(a, b), "ks")
    for k, nm, u in (("boxes", "Boxes per day", ""), ("pallets", "Pallets per day", ""),
                     ("hours", "Working hours per day", "h")):
        add("Days", nm, mean(R, k), mean(G, k), u, rel(mean(R, k), mean(G, k)), "relative")
    hr, hg = np.array(R["hours"]), np.array(G["hours"])
    add("Days", "Hour-of-day profile", (hr * np.arange(24)).sum(),
        (hg * np.arange(24)).sum(), "h", 0.5 * np.abs(hr - hg).sum(), "tv")
    pr, pg = [p["n"] for p in R["pallet_list"]], [p["n"] for p in G["pallet_list"]]
    add("Pallets", "Boxes per pallet", np.mean(pr), np.mean(pg), "", ks(pr, pg), "ks")
    vr, vg = [p["vol"] for p in R["pallet_list"]], [p["vol"] for p in G["pallet_list"]]
    add("Pallets", "Box volume per pallet", np.mean(vr), np.mean(vg), "m³", ks(vr, vg), "ks")
    add("Pallets", "Boxes placed while two pallets share the line", R["overlap"],
        G["overlap"], "%", rel(R["overlap"], G["overlap"]), "relative")
    for t, nm in ((1, "Blue"), (2, "White"), (3, "Brown")):
        add("Order", f"{nm}: where it sits in its pallet", mpos(R, t), mpos(G, t), "pos",
            abs(mpos(G, t) - mpos(R, t)), "shift")
    for t, nm in ((1, "Blue"), (2, "White"), (3, "Brown")):
        add("Order", f"{nm}: same size twice in a row", R["repeat"][t], G["repeat"][t], "%",
            rel(R["repeat"][t], G["repeat"][t]), "relative")
    add("Order", "Carton sizes in use per brown box, per day", ratio(R), ratio(G), "%",
        rel(ratio(R), ratio(G)), "relative")
    return out


def baseline(days, n_gen, k=40, seed=0):
    """The same gaps between two random halves of the real days -- how far
    real days drift from each other, the yardstick for the generated ones."""
    rng = np.random.default_rng(seed)
    n_a = min(n_gen, len(days) // 2)
    runs = []
    for _ in range(k):
        idx = rng.permutation(len(days))
        a = measure([days[i] for i in idx[n_a:]])
        b = measure([days[i] for i in idx[:n_a]])
        runs.append([s["score"] for s in scores(a, b)])
    return np.array(runs)


def compare(real_dir, gen_dir):
    real_days, gen_days = read(real_dir), read(gen_dir)
    R, G = measure(real_days), measure(gen_days)
    rep = {"generated_on": dt.date.today().isoformat()}

    # ---- the rules
    b_lo, b_hi = R["t3"].min(0), R["t3"].max(0)
    v_r, v_g = R["t3"].prod(1), G["t3"].prod(1)
    real_sk = set(R["t3_skus"])
    real_rows = [r for d in real_days for r in d["rows"]]
    gen_rows = [r for d in gen_days for r in d["rows"]]
    fixed, ref = {}, {}
    for t in (1, 2):
        ref[t] = {r["d"] for r in real_rows if r["t"] == t}
        fixed[t] = float(np.mean([r["d"] in ref[t] for r in gen_rows if r["t"] == t]))
    inb = (G["t3"] >= b_lo).all(1) & (G["t3"] <= b_hi).all(1)
    inv = (v_g >= v_r.min()) & (v_g <= v_r.max())
    rep["rules"] = {
        "t1_exact": fixed[1], "t2_exact": fixed[2],
        "t3_in_bounds": float(inb.mean()), "t3_vol_in_bounds": float(inv.mean()),
        "t3_copied": sum(1 for s in G["t3_skus"] if s in real_sk),
        "t3_gen_skus": len(G["t3_skus"]), "t3_real_skus": len(real_sk),
        "ids_copied": len(G["ids"] & R["ids"]),
        "pallets_copied": len(G["pallet_ids"] & R["pallet_ids"]),
        "stamp_ok": G["stamp_ok"], "sorted": G["sorted"],
        "bounds": {"lo": b_lo.tolist(), "hi": b_hi.tolist(),
                   "vlo": float(v_r.min()) / 1000, "vhi": float(v_r.max()) / 1000},
        "gen_bounds": {"lo": G["t3"].min(0).tolist(), "hi": G["t3"].max(0).tolist(),
                       "vlo": float(v_g.min()) / 1000, "vhi": float(v_g.max()) / 1000},
        "fixed": {t: {"d": next(iter(ref[t]))[0], "w": next(iter(ref[t]))[1],
                      "tall": R["tall"][t]["tall"], "short": R["tall"][t]["short"]}
                  for t in (1, 2)},
    }

    # ---- overview
    rep["overview"] = {k: {"real": R[k], "gen": G[k]} for k in ("files", "boxes", "pallets")}
    rep["overview"]["skus"] = {"real": len(real_sk), "gen": len(G["t3_skus"])}
    rep["overview"]["dates"] = {"real": [R["dates"][0], R["dates"][-1]],
                                "gen": [G["dates"][0], G["dates"][-1]]}
    rep["types"] = {"real": R["types"], "gen": G["types"]}
    rep["tall"] = {"real": R["tall"], "gen": G["tall"]}

    # ---- type-3 sizes
    sizes = {}
    names = ["depth", "width", "height"]
    for i, nm in enumerate(names):
        a, b = R["t3"][:, i], G["t3"][:, i]
        lo, hi = np.floor(b_lo[i] / 2.5) * 2.5, np.ceil(b_hi[i] / 2.5) * 2.5
        sizes[nm] = {"hist": hist(a, b, np.arange(lo, hi + 2.5, 2.5)),
                     "real": summary(a), "gen": summary(b), "ks": ks(a, b),
                     "crit": float(ks_crit(len(a), len(b)))}
    a, b = v_r / 1000, v_g / 1000                     # litres
    sizes["volume"] = {"hist": hist(a, b, np.arange(0, np.ceil(a.max() / 5) * 5 + 5, 5)),
                       "real": summary(a), "gen": summary(b), "ks": ks(a, b),
                       "crit": float(ks_crit(len(a), len(b)))}
    rep["sizes"] = sizes
    # footprint scatter: one dot per distinct SKU, sized by how often it ships
    rep["skus"] = {
        "real": [[*k, c] for k, c in R["t3_skus"].most_common()],
        "gen": [[*k, c] for k, c in G["t3_skus"].most_common()],
    }

    def rank_share(cnt, k=40):
        v = np.array(sorted(cnt.values(), reverse=True), float)
        return (np.cumsum(v) / v.sum())[:k].tolist()
    rep["rank"] = {"real": rank_share(R["t3_skus"]), "gen": rank_share(G["t3_skus"])}

    # ---- days and pallets
    rep["days"] = {"real": R["days"], "gen": G["days"]}

    def pals(M):
        return [{"file": p["file"], "id": p["id"], "n": p["n"], "vol": p["vol"]}
                for p in M["pallet_list"]]
    rep["pallets"] = {"real": pals(R), "gen": pals(G)}
    rep["hours"] = {"real": R["hours"], "gen": G["hours"]}
    rep["pos"] = {t: {"real": summary(R["pos"][t])["q"], "gen": summary(G["pos"][t])["q"],
                      "rmean": float(np.mean(R["pos"][t])), "gmean": float(np.mean(G["pos"][t]))}
                  for t in (1, 2, 3)}
    rep["repeat"] = {"real": R["repeat"], "gen": G["repeat"]}

    # ---- one score table, next to the same gaps between halves of the real data
    score = scores(R, G)
    base = baseline(real_days, len(gen_days))
    for row, col in zip(score, base.T):
        row["base_mean"], row["base_p90"] = float(col.mean()), float(np.percentile(col, 90))
        row["verdict"] = ("within" if row["score"] <= row["base_p90"] else
                          "near" if row["score"] <= 1.5 * row["base_p90"] else "differs")
    rep["score"] = score
    rep["baseline_splits"] = int(base.shape[0])

    # ---- hero: real and generated pallets, in their real box order
    def pick(M):
        ok = [p for p in M["pallet_list"] if 45 <= p["n"] <= 80 and 0.9 <= p["vol"] <= 1.7
              and len({b[3] for b in p["boxes"]}) == 3]
        return [{"file": p["file"], "id": p["id"], "n": p["n"], "vol": p["vol"],
                 "boxes": p["boxes"]} for p in ok]
    rep["hero"] = {"real": pick(R), "gen": pick(G)}

    rep["id_len"] = {"real": {t: dict(R["id_len"][t]) for t in (1, 2, 3)},
                     "gen": {t: dict(G["id_len"][t]) for t in (1, 2, 3)}}
    rep["rules"]["rows"] = {t: sum(r["t"] == t for r in gen_rows) for t in (1, 2, 3)}
    rep["rules"]["id_range"] = {t: [min(int(r["CONTAINER_ID"]) for r in gen_rows if r["t"] == t),
                                    max(int(r["CONTAINER_ID"]) for r in gen_rows if r["t"] == t)]
                                for t in (1, 2, 3)}
    rep["rules"]["pallet_range"] = [min(map(int, G["pallet_ids"])), max(map(int, G["pallet_ids"]))]
    rep["rules"]["real_id_max"] = {t: max(int(r["CONTAINER_ID"]) for r in real_rows if r["t"] == t)
                                   for t in (1, 2, 3)}
    rep["rules"]["real_pallet_max"] = max(map(int, R["pallet_ids"]))

    # a few raw lines of each, as they sit in the files
    def head(folder, name, n=9):
        with open(os.path.join(folder, name)) as fh:
            return {"file": name, "lines": [next(fh).rstrip("\n") for _ in range(n)]}
    rep["sample"] = {"real": head(real_dir, real_days[1]["file"]),
                     "gen": head(gen_dir, gen_days[0]["file"])}
    try:                                   # the knobs the generator ran with
        import generate_box_data as gbd
        rep["params"] = {k: getattr(gbd, k) for k in
                         ("JITTER_N", "MIX_CONC", "HOUR_CONC", "TALL_LOGIT_SD", "SKU_SD",
                          "SKU_POP_SD", "NOVEL_FRAC", "NOVEL_SD", "DAY_CONC", "REPEAT_P")}
    except ImportError:
        rep["params"] = {}
    return rep


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return round(float(o), 5)
    return o


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--real", default=os.path.join(DATA, "use_data"))
    p.add_argument("--gen", default=os.path.join(DATA, "generated_data"))
    p.add_argument("--out", default=None, help="default: <gen>/report.html")
    p.add_argument("--body", default=None,
                   help="also write the page without its <html>/<head> shell here")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    rep = to_jsonable(compare(a.real, a.gen))
    blob = json.dumps(rep, separators=(",", ":"))
    out = a.out or os.path.join(a.gen, "report.html")
    if a.json:
        with open(os.path.splitext(out)[0] + ".json", "w") as fh:
            json.dump(rep, fh, indent=1)
    with open(TEMPLATE) as fh:
        page = fh.read().replace("/*__DATA__*/null", blob.replace("</", "<\\/"))
    head, _, body = page.partition("<!--/head-->")
    with open(out, "w") as fh:
        fh.write("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                 "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, "
                 "viewport-fit=cover\">\n" + head + "</head>\n<body>\n" + body
                 + "</body>\n</html>\n")
    if a.body:                    # the same page without its shell, for hosts that add one
        with open(a.body, "w") as fh:
            fh.write(page)
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
