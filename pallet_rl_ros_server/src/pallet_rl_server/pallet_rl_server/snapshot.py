"""The last step, written to a JSON file for the game's "Online game" tab.

After every request the node hands the plan to `write`: the pallet as the
planner rebuilt it, the conveyor, the picker's and the packer's probabilities,
every legal candidate point and the one chosen (`Plan.view`, in grid cells),
plus the answer in cm.  The file is replaced atomically, so a reader never
sees half of it.  No ROS here.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict


def build(plan, pallet_id, seq, cfg):
    """-> the snapshot dict of one request."""
    srv = cfg["server"]
    return {
        "seq": int(seq),
        "time": time.time(),
        "pallet_id": str(pallet_id),
        "status": int(plan.status),
        "message": plan.message,
        "warnings": list(plan.warnings),
        "ms": round(float(plan.ms), 1),
        "station_index": int(plan.station_index),
        "placement_cm": asdict(plan.placement) if plan.placement else None,
        "rotated": bool(plan.rotated),
        "pick_probability": float(plan.pick_probability),
        "place_probability": float(plan.place_probability),
        "n_candidates": int(plan.n_candidates),
        "checkpoint": os.path.basename(srv["checkpoint"]),
        "selector": srv["selector"],
        # the extra 3D view data is on: every box in reach, and the game
        # lets the arm's base be moved by hand
        "extra_3d_viw": bool(srv["enable_extra_3d_viw"]),
        # what the game needs to draw the UR20 placing the box
        "robot": dict(cfg["robot"]),
        "view": plan.view,
    }


def write(path, snap):
    """Write `snap` to `path` atomically (temp file in the same folder, then
    rename)."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".last_step.", dir=folder)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(snap, f)
        os.chmod(tmp, 0o644)        # mkstemp makes it 0600
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
