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
from .points import turned_down
from .core.arm import ArmChecker
from .core.nets import greedy, load_nets, probs
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
    # the step as the online view draws it (`Planner._view`); None on a bad
    # request
    view: dict | None = None


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
        # the snapshot's extras: the points the rules turned down (whenever a
        # snapshot is written), and every box in reach scored as if it were
        # picked (`enable_extra_3d_viw`, slower)
        self.views = bool(srv["snapshot_file"])
        self.extra = self.views and bool(srv["enable_extra_3d_viw"])
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
        cells = [self._placed_cells(b) for b in placed]
        pal = Pallet(R, self.arm, cells,
                     [self._station_cells(b) for b in station[:R.nb]])
        for i, (b, z) in enumerate(zip(placed, pal.landed)):
            if abs(z * self.cell_cm - b.z_cm) > self.z_tol:
                warn.append(f"placed[{i}] reported at z {b.z_cm:g} cm but "
                            f"rests at {z * self.cell_cm:g} cm")
        view = self._view(pal, cells, placed, station)

        # 1. the selector brings a station box to the front
        idx, p_pick = 0, 1.0
        if self.select is not None:
            o = pal.obs_cb()
            view["reach"] = o["b_pick"][0, : len(view["station"])].tolist()
            if R.pick_feasible and not o["b_pick"].any():
                self._every_box(pal, view, 0)
                return Plan(PALLET_DONE, "no box within reach has a legal "
                            "place", warnings=warn, view=view)
            logits = self.select(self._torch(o))
            idx, p_pick = greedy(logits)
            view["pick_probs"] = probs(logits)[: len(view["station"])]
            pal.permute(idx)
        view["pick"] = idx

        # 2. the packer places it
        o = pal.obs()
        n_cand = int(o["l_mask"].sum())
        if self.views:
            view["bad"] = turned_down(pal)
        if not n_cand:
            self._every_box(pal, view, idx)
            return Plan(PALLET_DONE, "the chosen box has no legal place",
                        station_index=idx, warnings=warn, view=view)
        logits = self.pack(self._torch(o))
        a, p_place = greedy(logits)
        x, y, r, z, sx, sy, sz = (int(v) for v in pal.cand[a])
        view["cands"] = self._cands(pal, n_cand)
        view["cand_probs"] = probs(logits)[:n_cand]
        view["taken"] = a
        view["place_cells"] = [x, y, z, sx, sy, sz]
        self._every_box(pal, view, idx)

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
                    warnings=warn, view=view)

    def _view(self, pal, cells, placed, station):
        """The step in grid cells, for the online view: the pallet as the
        planner rebuilt it (each box at the height it rests at), the
        conveyor, and -- filled in by `_plan` as it decides -- the picker's
        and the packer's probabilities and what they chose."""
        R = self.rules
        rows = [[int(v) for v in (x, y, int(z), sx, sy, sz, t)]
                for (x, y, _, sx, sy, sz, t), z in zip(cells, pal.landed)]
        cm = lambda b: [float(b.length_cm), float(b.width_cm),
                        float(b.height_cm)]
        return {
            "bin": [R.Lx, R.Ly, R.Lz], "cell_cm": self.cell_cm,
            "n_types": R.n_types, "nb": R.nb, "n_pick": R.n_pick,
            "type_rule": R.type_rule if R.type_constraint else "none",
            "placed": rows, "placed_cm": [cm(b) for b in placed],
            "station": [[int(v) for v in self._station_cells(b)]
                        for b in station[: R.nb]],
            "station_cm": [cm(b) for b in station[: R.nb]],
            "reach": None, "pick_probs": None, "pick": -1,
            "cands": [], "cand_probs": [], "taken": -1, "place_cells": None,
            # the points the rules turned down for the box picked, and with
            # `enable_extra_3d_viw` every box in reach's own points
            "bad": None, "boxes": None,
        }

    @staticmethod
    def _cands(pal, n):
        """The legal candidates (they come first in `cand`) as the view
        takes them: (x, y, z, orientation), as the game's 3D bin draws them."""
        return [[int(c[0]), int(c[1]), int(c[3]), int(c[2])]
                for c in pal.cand[:n]]

    def _every_box(self, pal, view, front):
        """With `enable_extra_3d_viw`: each box within reach in turn brought
        to the front and asked as if it were picked -- its candidates, the
        packer's probabilities over them, and the points the rules turned
        down -- as `view["boxes"][i]` (station order).  Runs after the
        decision; `pal` is left with some other box in front.  `front`: the
        station index in front now."""
        if not self.extra:
            return
        n = len(view["station"])
        order = [front] + [j for j in range(n) if j != front]
        boxes = [None] * n
        for j in range(min(self.rules.n_pick, n)):
            k = order.index(j)
            if k:
                pal.permute(k)
                order = [order[k]] + order[:k] + order[k + 1:]
            o = pal.obs()
            nc = int(o["l_mask"].sum())
            box = {"cands": self._cands(pal, nc), "cand_probs": [],
                   "taken": -1, "bad": turned_down(pal)}
            if nc:
                logits = self.pack(self._torch(o))
                box["cand_probs"] = probs(logits)[:nc]
                box["taken"] = greedy(logits)[0]
            boxes[j] = box
        view["boxes"] = boxes

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
