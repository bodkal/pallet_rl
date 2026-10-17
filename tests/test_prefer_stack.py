"""Stack over floor: the model's placement, sent up from the floor.

A blue or white box the policy would put at z = 0 goes to its likeliest
legal placement with z > 0 among those whose probability is at most `tol`
below the floor one's.  Brown boxes, and a choice already off the floor, are
left alone; the probabilities stay the policy's.
"""
import numpy as np

from ar2l.env import BPPBatch
from ar2l.viz import agents as A
from ar2l.viz import game as G

BLUE, WHITE, BROWN = [12, 7, 7, 0], [12, 7, 6, 1], [6, 5, 4, 2]


def env(seq):
    """A blue already at (0, 0), so the next box has spots on top of it."""
    e = BPPBatch(1, S=[31, 27, 40], nb=3, n_items=len(seq), n_types=3,
                 types=False, arm_collision=False)
    e.reset(np.asarray(seq, np.int16)[None])
    assert G.place(e, 0, 0, 0)
    return e


def scored(floor, up):
    """A policy that gives each floor spot `floor` and each z > 0 spot `up`,
    the first floor spot a little more, and takes the likeliest."""
    def fn(e):
        lm = e.obs()["l_mask"][0]
        z = e._lz[0]
        w = np.where(z > 0, up, floor) * lm
        w[np.flatnonzero((z == 0) & lm)[0]] += 1e-3
        return w.argmax()[None], w[None]
    return fn


def go(seq, floor, up, tol):
    e = env(seq)
    act, p = A.prefer_stack(scored(floor, up), tol)(e)
    e.obs()
    return int(e._lz[0, act[0]]), p


def test_floor_goes_up_within_the_threshold():
    for box in (BLUE, WHITE):
        assert go([BLUE, box, BROWN], 0.30, 0.27, 0.05)[0] > 0


def test_floor_stays_past_the_threshold():
    assert go([BLUE, BLUE, BROWN], 0.30, 0.20, 0.05)[0] == 0
    assert go([BLUE, BLUE, BROWN], 0.30, 0.27, 0.0)[0] == 0


def test_brown_stays_on_the_floor():
    assert go([BLUE, BROWN, BLUE], 0.30, 0.29, 0.30)[0] == 0


def test_the_likeliest_spot_up_wins():
    e = env([BLUE, BLUE, BROWN])

    def fn(e):
        lm = e.obs()["l_mask"][0]
        up = np.flatnonzero((e._lz[0] > 0) & lm)
        w = (e._lz[0] == 0) * lm * 0.30
        w[up] = 0.10
        w[up[-1]] = 0.25
        return w.argmax()[None], w[None]
    act, p = A.prefer_stack(fn, 0.30)(e)
    assert act[0] == np.flatnonzero((e._lz[0] > 0) & e.obs()["l_mask"][0])[-1]
    assert p[0].max() == np.float64(0.30)       # the policy's own scores


def test_label_and_off():
    pol = lambda e: (np.zeros(1, int), np.ones((1, 1)))
    assert A.stack_rule(pol, "x", {"prefer_stack": 0}) == (pol, "x")
    _, lab = A.stack_rule(pol, "x", {"prefer_stack": 1, "prefer_stack_pct": 30})
    assert lab == "x, stacks within 30 points"
