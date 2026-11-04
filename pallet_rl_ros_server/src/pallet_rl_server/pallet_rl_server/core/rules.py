"""The packing rules, resolved once from the settings file.

`Rules` holds every number the feasibility check reads -- in grid cells --
so a request only has to build the pallet state, never re-derive the rules.

Box types (the env's numbering): 0 blue, 1 white, 2 brown.

`type_rule` says when a box may be stacked on a box of another type:
  touch          only the cells it rests on must be its own type
  column         nothing anywhere under its footprint may be another type
  mixed_touch    the cell's blue/white/brown table (below), "on" as touch
  mixed_column   the same table, "on" as column

The mixed table -- (a) top within `mixed_top_gap_cm` of the lid, (c) brown
footprint >= `mixed_area_frac` of the small blue, (d) no small blue box can
still go on that blue:

                    normally          own fallback   soft_mix last resort
  blue  on white    (a)               yes            yes
  blue  on brown    (a)               yes            yes
  white on blue     (a) or (d)        yes            yes
  white on brown    (a)               yes            yes
  brown on blue     (a), (c) or (d)   yes            yes, (c) dropped
  brown on white    never             never          never

A box's own fallback opens when it has no normal place at all; with
`soft_mix` that is replaced by a last resort that opens only when no box
within reach has a place.  (b) blue on blue rests on blue with all of its
base; (w) white on white rests on at least the small white's base -- both
hold under every type rule and are lifted by the fallbacks.
"""
from __future__ import annotations

import math

import numpy as np

# a column no box has reached; the floor takes every type
TYPE_FLOOR = -1
TYPE_RULES = ("touch", "column", "mixed_touch", "mixed_column")
COLUMN_RULES = ("column", "mixed_column")
MIXED_RULES = ("mixed_touch", "mixed_column")
BLUE, WHITE, BROWN = 0, 1, 2

# what a mixed rule says of a type on another: no, yes, if the pair's own
# condition holds, or if its top ends near the lid
NO, YES, IF, TOP = 0, 1, 2, 3
TOP_PAIRS = ((BLUE, WHITE), (BLUE, BROWN), (WHITE, BROWN))
SOFT_PAIRS = ((BLUE, WHITE), (BLUE, BROWN), (WHITE, BLUE), (WHITE, BROWN),
              (BROWN, BLUE))

# `ems`: which positions are candidates
EMS_OFF, EMS_SPACES, EMS_CORNER, EMS_BOTH = 0, 1, 2, 3
# pallets at least this many cells square enumerate EMS by the pruned path
EMS_PRUNE_S = 25

# a column no small box has capped
NO_CAP = int(np.iinfo(np.int16).max)

# float hairs off exact boundaries
CM_EPS = 1e-6
SUPPORT_EPS = 1e-6

# `stability: cdrl` (Zhao et al. 2021): (area ratio, corners) pairs
SUPPORT_RULES = ((0.60, 4), (0.80, 3), (0.95, 0))

ROUNDING = {
    "up": lambda x: math.ceil(x - CM_EPS),
    "down": lambda x: math.floor(x + CM_EPS),
    "nearest": lambda x: math.ceil(x - 0.5 - CM_EPS),   # 1.5 -> 1, 1.51 -> 2
}


def cm_to_cells(cm, cell_cm, rounding, pad_cm=0.0):
    """One box side in cells: (cm + pad) / cell, rounded, never below 1."""
    return max(1, ROUNDING[rounding]((float(cm) + pad_cm) / cell_cm))


def stack_allowance(n, side_cm, allow_cm, cell_cm):
    """(n + 1,) cells a later box may still start above a small box whose
    shorter footprint side is k cells: linear from allow_cm[0] at
    side_cm[0] to allow_cm[1] at side_cm[1], no limit from side_cm[1] up."""
    s0, s1 = (float(v) for v in side_cm)
    a0, a1 = (float(v) for v in allow_cm)
    if not 0 < s0 < s1 or min(a0, a1) < 0:
        raise ValueError(f"stack_cap_side_cm {side_cm} / stack_cap_allow_cm "
                         f"{allow_cm} are not valid")
    side = np.arange(n + 1) * float(cell_cm)
    allow = np.floor(np.interp(side, (s0, s1), (a0, a1)) / cell_cm + CM_EPS)
    return np.where(side < s1 - CM_EPS, allow, NO_CAP).astype(np.int64)


def mixed_tables(n_types, soft):
    """`(rule, fallback)`: rule[k, t, u] is what type t may do on type u under
    the normal rules (k=0), its own fallback (k=1) or the last resort (k=2);
    fallback[t] is whether the fallback gives type t anything more."""
    T = int(n_types)
    rule = np.stack([np.where(np.eye(T, dtype=bool), YES, NO)] * 3)
    rule = rule.astype(np.int8)

    def put(ks, pairs, v):
        for t, u in pairs:
            if max(t, u) < T:
                rule[ks, t, u] = v

    put([0], ((WHITE, BLUE), (BROWN, BLUE)), IF)
    put([0], TOP_PAIRS, TOP)
    rule[1] = rule[2] = rule[0]
    if soft:
        put([2], SOFT_PAIRS, YES)
    else:
        put([1, 2], SOFT_PAIRS, YES)
    return rule, (rule[0] != rule[1]).any(1)


class Rules:
    """Every rule parameter, in grid cells.  Built once per server.

    `r` is the `rules:` section of the settings file, `cell_cm` the real cm
    per grid cell, `pad_cm` / `rounding` how box sides become cells.
    """

    def __init__(self, r, bin_cells, cell_cm, pad_cm, rounding):
        self.Lx, self.Ly, self.Lz = (int(v) for v in bin_cells)
        self.nb = int(r["nb"])
        self.n_pick = int(r["n_pick"] if r["n_pick"] is not None else self.nb)
        if not 1 <= self.n_pick <= self.nb:
            raise ValueError(f"n_pick must be 1..nb, got {self.n_pick}")
        self.rot = int(r["rot"])
        self.ems = int(r["ems"])
        if not EMS_OFF <= self.ems <= EMS_BOTH:
            raise ValueError(f"ems is 0-3, got {self.ems}")
        self.max_l = int(r["max_l"])
        self.stability = str(r["stability"])
        if self.stability not in ("com", "cdrl"):
            raise ValueError(f"stability is com or cdrl, got {self.stability}")
        self.min_support = float(r["min_support"])
        self.min_total_support = float(r["min_total_support"])
        for k in ("min_support", "min_total_support"):
            if not 0.0 <= getattr(self, k) <= 1.0:
                raise ValueError(f"{k} is a fraction, got {getattr(self, k)}")

        # ---- box types
        self.n_types = int(r["n_types"])
        if not 1 <= self.n_types <= 31:
            raise ValueError(f"n_types is 1..31, got {self.n_types}")
        self.type_constraint = bool(r["type_constraint"])
        self.type_rule = str(r["type_rule"])
        if self.type_rule not in TYPE_RULES:
            raise ValueError(f"type_rule is one of {TYPE_RULES}")
        self.mixed = self.type_rule in MIXED_RULES
        self.column = self.type_rule in COLUMN_RULES
        self.typed = self.type_constraint and self.n_types > 1
        self.soft_mix = bool(r["soft_mix"])
        self.tables = mixed_tables(self.n_types, self.soft_mix)
        soft = mixed_tables(self.n_types, True)[0]
        self.soft_types = (soft[2] != soft[1]).any(1)
        self.soft_on = bool(self.soft_mix and self.mixed and self.typed
                            and self.soft_types.any())
        self.outline_resort = bool(r["outline_resort"])
        self.outline_when = str(r["outline_when"])
        self.outline_area = str(r["outline_area"])
        if self.outline_when not in ("station", "box"):
            raise ValueError("outline_when is station or box")
        if self.outline_area not in ("outline", "all"):
            raise ValueError("outline_area is outline or all")
        self.outline_on = bool(self.outline_resort and self.mixed
                               and self.typed and self.ems != EMS_OFF)
        pick = r.get("pick_feasible")
        self.pick_feasible = bool(self.typed and self.n_pick > 1
                                  if pick is None else pick)

        # ---- the mixed rules' thresholds, in cells
        self.top_gap = int(np.floor(float(r["mixed_top_gap_cm"]) / cell_cm
                                    + CM_EPS))
        self.area_frac = float(r["mixed_area_frac"])
        sb = r["mixed_small_blue_cm"]
        self.small_blue = tuple(cm_to_cells(v, cell_cm, rounding,
                                            pad_cm if i < 2 else 0.0)
                                for i, v in enumerate(sb))
        ws = r.get("white_support_cm")
        self.small_white = (tuple(cm_to_cells(v, cell_cm, rounding, pad_cm)
                                  for v in ws) if ws else ())
        self.white_area = (int(np.prod(self.small_white))
                           if self.small_white else 0)
        self.white_on = self.white_area > 0 and self.n_types > WHITE
        self.blue_on = bool(r["blue_support"]) and self.n_types > 1

        # ---- stack height over small boxes
        self.stack_cap = bool(r["stack_cap"])
        self.stack_types = np.asarray(r["stack_cap_types"] or [],
                                      np.int64).reshape(-1)
        self.stack_allow = stack_allowance(
            min(self.Lx, self.Ly), r["stack_cap_side_cm"],
            r["stack_cap_allow_cm"], cell_cm)

        self.arm_collision = bool(r["arm_collision"])
        # the contact count costs a second sweep; only when a rule reads it
        self.needs_count = (self.stability != "com" or self.min_support > 0
                            or self.mixed or self.white_area > 0
                            or self.blue_on)
        self.scale = float(max(self.Lx, self.Ly, self.Lz))
