# Archived experiments

Everything here predates the action-space rewrite of 2026-09-15 (rotation plus
the exact empty-maximal-space enumeration — see `../TODO.md` section A.4).

**The checkpoints under `runs_norot/` cannot be re-run.** They were trained
against `env._ems_axis`, the per-axis extreme-point candidate generator, which
no longer exists in `ar2l/env.py`. Loading one would silently evaluate a policy
in an environment it never saw. They are kept only as provenance for the numbers
below.

| | what it is | why it is here |
|---|---|---|
| `runs_norot/` | 16 policies + 16 attackers + 12 heuristic attackers, one orientation, old EMS. `PIPE/GRID/GRID2/STRONG.log` are the driver logs | superseded; environment gone |
| `results/table2_att800.json` | Table 2 with attackers trained to only 800 iterations | superseded twice — by the 3000-iteration attackers, then by the rewrite |
| `results/table1_att800.json` | same, Table 1 | as above |
| `results/table1_stale_norot.json` | byte-identical duplicate of `../results/table1_norot.json` | duplicate |
| `results/calibration_norot.json` | byte-identical duplicate of `../results/calibration.json` | duplicate |
| `figures/` | report, attacker-behaviour plot, and the replay/heatmap frames rendered before rotation | the packer in them has no rotated boxes, so they misrepresent the current agent |

## What was deliberately *not* archived

Kept in `../results/` because the next experiment needs them:

- **`table2_norot.json`** — the pre-rotation Table 2. This is the direct
  before/after for the whole rewrite (mean \|gap\| to the paper 6.5 -> 3.5) and
  is the single most useful comparison on disk.
- **`table1_norot.json`** — the only Table 1 baseline until the new heuristic
  attackers finish.
- **`calibration.json`** — the stability-rule sweep. Still the evidence for the
  centre-of-mass rule, and still cited in the README.
- **`ablate_env.json`** — the rotation / EMS ablation on the six heuristics.
  This is the measurement that attributes +6.7 points to rotation.

Kept in `../runs/`: the whole current grid, plus `attrot{1,2}_dbl_nb10`, the
pair of attackers trained against DBL under one and two orientations that
measure how much attacker damage rotation absorbs (-23.2 vs -18.1).
