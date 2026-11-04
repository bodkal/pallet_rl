"""Read the server's YAML settings and check them against the checkpoint.

Every value the server uses comes from one file (`config/server.yaml`):
`server` (service, checkpoint, device), `grid` (cm <-> cells), `rules` (the
packing rules) and `robot` (the arm check).  Relative paths are read from the
file's own folder.

The network itself was trained under some rules; if the file says otherwise
the policy is being asked about a world it never saw.  `check_checkpoint`
compares the two at start-up and warns (or refuses to start).
"""
from __future__ import annotations

import os

import yaml

REQUIRED = {
    "server": ("service", "checkpoint", "selector", "device",
               "check_checkpoint", "z_tolerance_cm"),
    "grid": ("cell_cm", "pallet_cells", "box_pad_cm", "box_round"),
    "rules": ("nb", "n_pick", "rot", "ems", "max_l", "stability",
              "min_support", "min_total_support", "n_types", "type_constraint",
              "type_rule", "soft_mix", "outline_resort", "outline_when",
              "outline_area", "mixed_top_gap_cm", "mixed_area_frac",
              "mixed_small_blue_cm", "white_support_cm", "blue_support",
              "stack_cap", "stack_cap_types", "stack_cap_side_cm",
              "stack_cap_allow_cm", "arm_collision", "pick_feasible"),
    "robot": ("base_from_box_tf", "base_x_moves", "tool_length_m",
              "ik_solution"),
}


def load(path):
    """-> settings dict, with `server.checkpoint` made absolute."""
    path = os.path.abspath(path)
    with open(path) as f:
        s = yaml.safe_load(f) or {}
    for sec, keys in REQUIRED.items():
        have = s.get(sec) or {}
        missing = [k for k in keys if k not in have]
        unknown = sorted(set(have) - set(keys))
        if missing or unknown:
            raise ValueError(f"{path} [{sec}]: missing {missing}, "
                             f"unknown {unknown}")
    extra = sorted(set(s) - set(REQUIRED))
    if extra:
        raise ValueError(f"{path}: unknown section(s) {extra}")
    ck = os.path.expanduser(s["server"]["checkpoint"])
    s["server"]["checkpoint"] = os.path.join(os.path.dirname(path), ck)
    if s["grid"]["box_round"] not in ("up", "down", "nearest"):
        raise ValueError("grid.box_round is up, down or nearest")
    return s


def checkpoint_mismatches(s, args):
    """[(name, settings value, checkpoint value)] that differ.  Only what
    the checkpoint recorded is compared."""
    r, g = s["rules"], s["grid"]
    ours = {k: r[k] for k in ("nb", "n_pick", "rot", "ems", "max_l",
                              "stability", "min_support", "min_total_support",
                              "n_types", "type_constraint", "type_rule",
                              "soft_mix", "outline_resort", "outline_when",
                              "outline_area", "blue_support", "stack_cap")}
    ours["white_support_cm"] = r["white_support_cm"]
    ours["pallet_cells"] = list(g["pallet_cells"])
    ours["cell_cm"] = g["cell_cm"]
    ours["box_pad_cm"] = g["box_pad_cm"]
    ours["box_round"] = g["box_round"]

    theirs = {k: args[k] for k in ours if k in args}
    if "white_support" in args:
        theirs["white_support_cm"] = (args.get("white_support_cm")
                                      if args["white_support"] else None)
    if "bin" in args:
        theirs["pallet_cells"] = list(args["bin"])
    if "cell_cm" in args and "box_scale" in args:
        theirs["cell_cm"] = args["cell_cm"] * max(1.0, args["box_scale"] or 1)
    if "box_pad_m" in args:
        theirs["box_pad_cm"] = 100.0 * (args["box_pad_m"] or 0.0)

    def same(a, b):
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return (a is not None and b is not None and len(a) == len(b)
                    and all(same(x, y) for x, y in zip(a, b)))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-9
        return a == b

    return [(k, ours[k], v) for k, v in theirs.items() if not same(ours[k], v)]
