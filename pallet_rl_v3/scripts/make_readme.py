#!/usr/bin/env python3
"""Fill the RESULTS block of README.md from results/*.json.

    python3 scripts/make_readme.py

Idempotent: rewrites whatever sits between the RESULTS markers, so it can be
re-run after every evaluation.
"""
from __future__ import annotations

import json
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
START, END = "<!-- RESULTS:start -->", "<!-- RESULTS:end -->"


def headline():
    """PCT and ExactAR2L(1.0) at the two ends of the beta sweep."""
    f = os.path.join(ROOT, "results/table2.json")
    if not os.path.exists(f):
        return ""
    res = json.load(open(f))
    from ar2l.viz.report import LABEL
    out = ["", "### Headline", "",
           "Space utilisation on 3000 held-out instances of 150 items; "
           "&beta; is the share of them reordered by that policy&rsquo;s own "
           "permutation-based attacker. *drop* is what the attacker costs.", ""]
    for nb in sorted({v["nb"] for v in res.values()}):
        out += [f"**N_B = {nb}**", "",
                "| method | &beta;=0 | &beta;=100 | drop | items at &beta;=100 |"
                " Std. at &beta;=100 |", "|---|---|---|---|---|---|"]
        for m in LABEL:
            k = f"{m}_nb{nb}"
            if k not in res:
                continue
            v = res[k]
            d = v["0"]["uti"] - v["100"]["uti"]
            out.append(f"| {LABEL[m]} | {v['0']['uti']:.1f} | {v['100']['uti']:.1f}"
                       f" | &minus;{d:.1f} | {v['100']['num']:.1f} |"
                       f" {v['100']['std']:.1f} |")
        out.append("")
    return "\n".join(out)


def main():
    body = [headline()]
    for which in ("table2", "table1"):
        if not os.path.exists(os.path.join(ROOT, "results", which + ".json")):
            continue
        r = subprocess.run(["python3", os.path.join(ROOT, "scripts/summarize.py"),
                            "--which", which], capture_output=True, text=True)
        body.append(r.stdout)
    p = os.path.join(ROOT, "README.md")
    s = open(p).read()
    a, b = s.index(START), s.index(END)
    s = s[:a + len(START)] + "\n" + "\n".join(body) + "\n" + s[b:]
    open(p, "w").write(s)
    print("README RESULTS block updated")


if __name__ == "__main__":
    main()
