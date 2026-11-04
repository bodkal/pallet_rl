"""The server decides exactly as the training code does.

Plays real pallets with the original `ar2l` environment and networks (from
the pallet_rl repo this workspace sits in, or `PALLET_RL_REPO`), and at
every step rebuilds the same state in the server from the boxes alone:

  * the observations (C, B, L) are bit-identical, before and after the pick
  * the logits of both networks are identical
  * the picked box and the placement are the same
  * the cm request path gives the same answer as the grid path

Skipped when the repo is not there.  Run with the workspace venv:

    .venv/bin/python -m pytest src/pallet_rl_server/test -q
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from pallet_rl_server import settings  # noqa: E402
from pallet_rl_server.core.pallet import Pallet  # noqa: E402
from pallet_rl_server.planner import OK, Box, Planner  # noqa: E402

# this workspace lives at <repo>/pallet_rl_ros_server
REPO = os.path.expanduser(os.environ.get(
    "PALLET_RL_REPO", os.path.join(HERE, "..", "..", "..", "..")))
CKPT = os.path.join(REPO, "runs/w_128_h_1_l_1_colom_center_top/best.pt")
DATA = os.path.join(REPO, "data/train_data/all_data.csv")
if not (os.path.isfile(CKPT) and os.path.isfile(DATA)):
    pytest.skip("pallet_rl repo not found", allow_module_level=True)
sys.path.insert(0, REPO)

from ar2l.env import BPPBatch  # noqa: E402
from ar2l.evaluate import load_nets  # noqa: E402
from ar2l.model import sample  # noqa: E402
from ar2l.orders import load_orders, randomize_order  # noqa: E402

CFG = settings.load(os.path.join(PKG, "config", "server.yaml"))
CFG["server"]["check_checkpoint"] = "off"

# rule sets to compare under: the shipped one, then every other branch
VARIANTS = {
    "shipped": {},
    "mixed_column_soft": dict(type_rule="mixed_column", soft_mix=1),
    "mixed_touch_outline": dict(type_rule="mixed_touch", soft_mix=1,
                                outline_resort=1, outline_when="box"),
    "mixed_column_outline_all": dict(type_rule="mixed_column", soft_mix=0,
                                     outline_resort=1, outline_area="all"),
    "touch_total_support": dict(type_rule="touch", min_total_support=0.8,
                                white_support_cm=None, blue_support=0),
    "cdrl_no_arm": dict(stability="cdrl", arm_collision=0, ems=1),
}
PALLETS = {"shipped": 8}           # the rest play 3


def ar2l_env(rules, seq):
    """The training env under `rules`, playing one pallet."""
    r = rules
    return BPPBatch(
        1, S=tuple(CFG["grid"]["pallet_cells"]), nb=r["nb"],
        n_items=seq.shape[0], size_hi=seq[:, :3].max(0),
        stability=r["stability"], ems=r["ems"], rot=r["rot"],
        max_l=r["max_l"], min_support=r["min_support"], n_pick=r["n_pick"],
        n_types=r["n_types"], types=False,
        type_constraint=bool(r["type_constraint"]), type_rule=r["type_rule"],
        soft_mix=bool(r["soft_mix"]), outline_resort=bool(r["outline_resort"]),
        outline_when=r["outline_when"], outline_area=r["outline_area"],
        mixed_white_support=bool(r["white_support_cm"]),
        mixed_white_support_cm=r["white_support_cm"],
        blue_support=bool(r["blue_support"]),
        min_total_support=r["min_total_support"],
        stack_cap=bool(r["stack_cap"]),
        arm_collision=bool(r["arm_collision"]))


def state_rows(env):
    """The ar2l env's pallet and window as request rows (cells)."""
    n = int(env.n_packed[0])
    c = np.rint(env.packed[0, :n] * env.scale).astype(np.int64)
    placed = np.concatenate([c, env.ptype[0, :n, None].astype(np.int64)], 1)
    win, valid = env.window()
    return placed, win[0][valid[0]].astype(np.int64)


def same_obs(a, b, keys):
    for k in keys:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def to_t(o):
    return {k: torch.as_tensor(v) for k, v in o.items()}


@pytest.fixture(scope="module")
def ref_nets():
    nets = load_nets(CKPT, "cpu", ("pack", "mixer"))
    return nets["pack"], nets["mixer"]


@pytest.fixture(scope="module")
def pallets():
    seqs, _ = load_orders(DATA, 1.0, tuple(CFG["grid"]["pallet_cells"]),
                          rot=2, box_scale=4, box_round="nearest",
                          box_pad_m=0.04)
    seqs = randomize_order(seqs, 0.1, 0)
    rng = np.random.default_rng(1)
    return [seqs[i] for i in rng.choice(len(seqs), 8, replace=False)]


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_parity(variant, ref_nets, pallets):
    cfg = copy.deepcopy(CFG)
    cfg["rules"].update(VARIANTS[variant])
    cfg["server"]["checkpoint"] = CKPT
    planner = Planner(cfg)
    R, cell = planner.rules, planner.cell_cm
    pack, mixer = ref_nets
    steps = 0
    for seq in pallets[: PALLETS.get(variant, 3)]:
        seq = seq[(seq[:, :3] > 0).all(1)]
        env = ar2l_env(cfg["rules"], seq)
        env.reset(seq[None])
        while not env.done.all():
            placed, station = state_rows(env)
            pal = Pallet(R, planner.arm, placed, station)
            assert np.array_equal(pal.landed, placed[:, 2])

            # the pick
            o_ref, o_new = env.obs_cb(), pal.obs_cb()
            same_obs(o_ref, o_new, o_ref)
            lg_ref, lg_new = mixer.logits(to_t(o_ref)), planner.select(to_t(o_new))
            assert torch.equal(lg_ref, lg_new)
            idx = int(sample(lg_ref, True)[0][0])
            env.permute(np.array([idx]))
            pal.permute(idx)

            # the placement
            o_ref, o_new = env.obs(), pal.obs()
            same_obs(o_ref, o_new, o_ref)
            if not o_ref["l_mask"].any():
                break
            lg_ref, lg_new = pack.logits(to_t(o_ref)), planner.pack(to_t(o_new))
            assert torch.equal(lg_ref, lg_new)
            a = int(sample(lg_ref, True)[0][0])
            ref = (int(env._lxy[0, a, 0]), int(env._lxy[0, a, 1]),
                   int(env._lxy[0, a, 2]), int(env._lz[0, a]))
            assert tuple(pal.cand[a][:4]) == ref

            # the planner's own entry points, grid and cm
            assert planner.plan_cells(placed, station) == (idx, ref)
            plan = planner.plan(
                [Box(*cm_box(row, cell, planner.pad_cm)) for row in placed],
                [Box(*cm_box(np.r_[0, 0, 0, row], cell, planner.pad_cm)[:4])
                 for row in station])
            assert plan.status == OK, plan.message
            x, y, r, z = ref
            sx, sy = (station[idx, :2][::-1] if r else station[idx, :2])
            assert plan.station_index == idx and plan.rotated == bool(r)
            assert plan.placement.x_cm == (x + sx / 2.0) * cell
            assert plan.placement.y_cm == (y + sy / 2.0) * cell
            assert plan.placement.z_cm == z * cell

            env.step(np.array([a]))
            steps += 1
    assert steps > 0


def cm_box(row, cell, pad):
    """Grid row (x, y, z, sx, sy, sz, type) -> Box args in cm whose sides
    grid back to sx, sy, sz: a side of k cells is k * cell - pad cm."""
    x, y, z, sx, sy, sz, t = (int(v) for v in row)
    return (sx * cell - pad, sy * cell - pad, sz * cell, t,
            (x + sx / 2.0) * cell, (y + sy / 2.0) * cell, z * cell)
