"""The stateless planner: the whole pallet in, one placement out.

`Planner.plan(placed, station)` rebuilds the pallet from the boxes on it,
lets the selector pick a station box, lets the packer place it, and returns
the result.  It keeps nothing about any pallet between calls; the only thing
it holds is what does not change -- the rules, the networks and the arm's
geometry cache.  No ROS here: `node.py` wraps it.

Units at this boundary are real cm in the pallet frame (origin at the grid's
corner, x along the pallet length, y along its width, z up).  Inside, a box
is padded by `box_pad_cm` on length and width and rounded to grid cells,
exactly as the training data was.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import settings as S
from .core.arm import ArmChecker
from .core.nets import greedy, load_nets
from .core.pallet import Pallet
from .core.rules import Rules, cm_to_cells

OK, PALLET_DONE, BAD_REQUEST = 0, 1, 2


@dataclass
class Box:
    """A carton in real cm.  Position (centre x, y; bottom z) is read only
    for boxes already on the pallet."""
    length_cm: float
    width_cm: float
    height_cm: float
    type: int
    x_cm: float = 0.0
    y_cm: float = 0.0
    z_cm: float = 0.0


@dataclass
class Plan:
    status: int
    message: str = ""
    station_index: int = -1          # which request station box to pick
    placement: Box | None = None     # where it goes, sides as it stands
    rotated: bool = False            # turned 90 deg: its length is along y
    pick_probability: float = 0.0
    place_probability: float = 0.0
    n_candidates: int = 0
    warnings: list = field(default_factory=list)
    ms: float = 0.0


class Planner:
    def __init__(self, cfg, log=print):
        """`cfg`: a settings dict (`settings.load`)."""
        self.cfg = cfg
        srv, grid = cfg["server"], cfg["grid"]
        self.cell_cm = float(grid["cell_cm"])
        self.pad_cm = float(grid["box_pad_cm"])
        self.rounding = grid["box_round"]
        self.z_tol = float(srv["z_tolerance_cm"])
        self.device = srv["device"]
        self.rules = Rules(cfg["rules"], grid["pallet_cells"], self.cell_cm,
                           self.pad_cm, self.rounding)
        self.arm = (ArmChecker(cfg["robot"], self.cell_cm / 100.0,
                               self.rules.Ly)
                    if self.rules.arm_collision else None)
        self.pack, self.select, args = load_nets(
            srv["checkpoint"], srv["selector"] or None, self.device)

        bad = S.checkpoint_mismatches(cfg, args)
        mode = srv["check_checkpoint"]
        if bad and mode != "off":
            text = "; ".join(f"{k}: settings {a} vs checkpoint {b}"
                             for k, a, b in bad)
            if mode == "error":
                raise ValueError(f"settings differ from the checkpoint: {text}")
            log(f"WARNING settings differ from the checkpoint: {text}")

    # ------------------------------------------------------------- units
    def _cells(self, b, pad):
        return cm_to_cells(b, self.cell_cm, self.rounding,
                           self.pad_cm if pad else 0.0)

    def _station_cells(self, box):
        return [self._cells(box.length_cm, True), self._cells(box.width_cm, True),
                self._cells(box.height_cm, False), int(box.type)]

    def _placed_cells(self, box):
        """Grid row (x, y, z, sx, sy, sz, type): the padded footprint centred
        where the box's centre is, snapped to the nearest cell."""
        sx, sy, sz, t = self._station_cells(box)
        c = self.cell_cm
        row = []
        for centre, s, L, axis in ((box.x_cm, sx, self.rules.Lx, "x"),
                                   (box.y_cm, sy, self.rules.Ly, "y")):
            v = math.floor(centre / c - s / 2.0 + 0.5)
            snapped = min(max(v, 0), L - s)
            if abs(snapped - v) > 1:
                raise ValueError(f"a placed box (centre {axis} = {centre} cm, "
                                 f"{s} cells) is off the pallet")
            row.append(snapped)
        row.append(math.floor(box.z_cm / c + 0.5))
        return row + [sx, sy, sz, t]

    def _check(self, boxes, what):
        for i, b in enumerate(boxes):
            if min(b.length_cm, b.width_cm, b.height_cm) <= 0:
                raise ValueError(f"{what}[{i}] has a side <= 0")
            if not 0 <= int(b.type) < self.rules.n_types:
                raise ValueError(f"{what}[{i}] type {b.type} is not in "
                                 f"0..{self.rules.n_types - 1}")

    # -------------------------------------------------------------- plan
    @torch.no_grad()
    def plan(self, placed, station):
        """`placed`: every box on the pallet, as placed, in placement order
        if known.  `station`: the conveyor, front first -- the first `n_pick`
        can be picked, the rest up to `nb` are preview."""
        t0 = time.perf_counter()
        try:
            out = self._plan(list(placed), list(station))
        except ValueError as e:
            out = Plan(BAD_REQUEST, str(e))
        out.ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _plan(self, placed, station):
        R, warn = self.rules, []
        if not station:
            raise ValueError("no boxes at the station")
        self._check(placed, "placed")
        self._check(station, "station")
        if len(station) > R.nb:
            warn.append(f"{len(station)} station boxes sent, the policy sees "
                        f"the first {R.nb}")
        pal = Pallet(R, self.arm, [self._placed_cells(b) for b in placed],
                     [self._station_cells(b) for b in station[:R.nb]])
        for i, (b, z) in enumerate(zip(placed, pal.landed)):
            if abs(z * self.cell_cm - b.z_cm) > self.z_tol:
                warn.append(f"placed[{i}] reported at z {b.z_cm:g} cm but "
                            f"rests at {z * self.cell_cm:g} cm")

        # 1. the selector brings a station box to the front
        idx, p_pick = 0, 1.0
        if self.select is not None:
            o = pal.obs_cb()
            if R.pick_feasible and not o["b_pick"].any():
                return Plan(PALLET_DONE, "no box within reach has a legal "
                            "place", warnings=warn)
            idx, p_pick = greedy(self.select(self._torch(o)))
            pal.permute(idx)

        # 2. the packer places it
        o = pal.obs()
        n_cand = int(o["l_mask"].sum())
        if not n_cand:
            return Plan(PALLET_DONE, "the chosen box has no legal place",
                        station_index=idx, warnings=warn)
        a, p_place = greedy(self.pack(self._torch(o)))
        x, y, r, z, sx, sy, sz = (int(v) for v in pal.cand[a])

        box = station[idx]
        length, width = ((box.width_cm, box.length_cm) if r
                         else (box.length_cm, box.width_cm))
        c = self.cell_cm
        place = Box(length, width, box.height_cm, int(box.type),
                    x_cm=(x + sx / 2.0) * c, y_cm=(y + sy / 2.0) * c,
                    z_cm=z * c)
        return Plan(OK, "; ".join(warn), station_index=idx, placement=place,
                    rotated=bool(r), pick_probability=p_pick,
                    place_probability=p_place, n_candidates=n_cand,
                    warnings=warn)

    def _torch(self, o):
        return {k: torch.as_tensor(v).to(self.device) for k, v in o.items()}

    # ------------------------------------------------- grid-level access
    @torch.no_grad()
    def plan_cells(self, placed, station):
        """The same decision on grid rows -- for tests and replays.
        -> (station index, (x, y, orientation, z) or None)."""
        pal = Pallet(self.rules, self.arm, placed, station)
        idx = 0
        if self.select is not None:
            o = pal.obs_cb()
            if self.rules.pick_feasible and not o["b_pick"].any():
                return idx, None
            idx, _ = greedy(self.select(self._torch(o)))
            pal.permute(idx)
        o = pal.obs()
        if not o["l_mask"].any():
            return idx, None
        a, _ = greedy(self.pack(self._torch(o)))
        return idx, tuple(int(v) for v in pal.cand[a][:4])
