"""Box types: the stream, the stacking rule, and the tensors it produces.

The rule is that a box may rest only on boxes of its own type, and that the
bin floor takes any type.  What "rest on" means is the whole content of the
rule, so the reference implementation here is a voxel grid: a placement is
legal when every cell the item actually *touches* -- the cells of its base
that have something directly beneath them -- carries its type.  An overhang
bridging a foreign type touches nothing there and is legal; a contact patch on
one is not.
"""
import numpy as np
import pytest
import torch

import ar2l.env as E
from ar2l.env import BPPBatch, TYPE_FLOOR, sample_items, type_classes
from ar2l.model import PackNet, PermNet, sample
from ar2l.ppo import to_torch
from ar2l.train import move_to_front


CLASSES = [{"p": 1, "lo": [1, 1, 1], "hi": [3, 3, 3]},
           {"p": 1, "lo": [2, 1, 1], "hi": [4, 3, 2]},
           {"p": 1, "lo": [1, 2, 1], "hi": [3, 4, 3]}]


def typed(n_env=8, **kw):
    kw.setdefault("S", (8, 8, 10))
    kw.setdefault("nb", 1)
    kw.setdefault("n_items", 120)
    kw.setdefault("types", CLASSES)
    kw.setdefault("n_types", 3)
    return BPPBatch(n_env, **kw)


def random_action(env, rng):
    m = env.obs()["l_mask"]
    return np.array([rng.choice(np.nonzero(r)[0]) if r.any() else 0 for r in m])


def voxels(env, b):
    """The bin as a voxel grid of type ids, -1 where empty, and its items."""
    vox = np.full((env.Lx, env.Ly, env.Lz), TYPE_FLOOR, np.int8)
    items = (env.packed[b, : env.n_packed[b]] * env.scale).round().astype(int)
    out = []
    for i, (x, y, z, sx, sy, sz) in enumerate(items):
        t = int(env.ptype[b, i])
        under = vox[x:x + sx, y:y + sy, z - 1] if z else None
        out.append(((x, y, z, sx, sy, sz), t,
                    np.array([], np.int8) if under is None
                    else under[under >= 0]))
        vox[x:x + sx, y:y + sy, z:z + sz] = t
    return vox, out


def play(env, rng, steps=None):
    n = 0
    while not env.done.all() and (steps is None or n < steps):
        env.step(random_action(env, rng))
        n += 1
    return env


# ------------------------------------------------------------------- stream
def test_every_box_carries_a_type_from_its_own_size_class():
    """The type is the SKU: it decides which bounds the sides are drawn from."""
    probs, lo, hi = type_classes(CLASSES)
    assert probs.tolist() == [1 / 3] * 3
    it = sample_items(np.random.default_rng(0), (4000,), classes=(probs, lo, hi))
    assert it.shape == (4000, 4)
    seen = np.bincount(it[:, 3], minlength=3)
    assert (seen > 1000).all(), f"a type never came up: {seen}"
    for t in range(3):
        m = it[:, 3] == t
        assert (it[m, :3] >= lo[t]).all() and (it[m, :3] <= hi[t]).all()
    # and the classes really are different, or this test proves nothing
    assert not np.array_equal(lo[0], lo[1])


def test_type_shares_are_honoured():
    classes = type_classes([{"p": 8, "lo": 1, "hi": 2}, {"p": 2, "lo": 1, "hi": 2}])
    it = sample_items(np.random.default_rng(1), (20000,), classes=classes)
    share = np.bincount(it[:, 3], minlength=2) / it.shape[0]
    assert abs(share[0] - 0.8) < 0.02, share


def test_no_classes_is_the_untyped_stream_unchanged():
    """`types=None` must draw exactly what the pre-type `sample_items` did."""
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    got = sample_items(rng_a, (50, 9), 1, 5)
    want = rng_b.integers(1, 6, size=(50, 9, 3), dtype=np.int16)
    assert np.array_equal(got[..., :3], want), "the item stream was re-rolled"
    assert not got[..., 3].any()


@pytest.mark.parametrize("bad", [
    [{"p": 1, "lo": [3, 1, 1], "hi": [2, 1, 1]}],      # lo above hi
    [{"p": 0}, {"p": 0}],                              # no mass anywhere
    [{"p": -1}, {"p": 2}],                             # a negative share
    [],                                                # no classes at all
])
def test_broken_type_classes_are_rejected(bad):
    with pytest.raises(ValueError):
        type_classes(bad)


def test_n_types_must_match_the_classes():
    with pytest.raises(ValueError):
        type_classes(CLASSES, n_types=2)


@pytest.mark.parametrize("bad", [[[1, 1, 1, 3]], [[1, 1, 1, -1]]])
def test_a_sequence_with_an_unknown_type_is_rejected(bad):
    env = typed(1)
    with pytest.raises(ValueError, match="type_id"):
        env.reset(np.array([bad], np.int16))


def test_a_three_column_sequence_loads_as_one_type():
    """Pre-type datasets on disk stay readable, as the type they were drawn as."""
    env = typed(2, n_items=5)
    env.reset(np.full((2, 5, 3), 2, np.int16))
    assert env.seq.shape == (2, 5, 4)
    assert not env.seq[..., 3].any()


# --------------------------------------------------------------- the rule
def test_no_box_is_ever_stacked_on_another_type():
    """The checklist item: no episode log may contain a mixed-type contact."""
    rng = np.random.default_rng(0)
    contacts = 0
    for seed in range(8):
        env = play(typed(8, seed=seed, n_pick=1), rng)
        for b in range(env.n_env):
            for (place, t, touch) in voxels(env, b)[1]:
                if place[2] == 0:
                    assert touch.size == 0, "a floor placement rested on a box"
                    continue
                assert touch.size, f"{place} floats"
                assert (touch == t).all(), \
                    f"type {t} at {place} rests on {sorted(set(touch.tolist()))}"
                contacts += 1
    assert contacts > 200, f"nothing was ever stacked ({contacts}), so nothing was tested"


def test_the_rule_actually_removes_placements():
    """Without this the test above would pass on a simulator that never stacks."""
    rng = np.random.default_rng(3)
    on = typed(8, seed=5, type_constraint=True)
    off = typed(8, seed=5, type_constraint=False)
    blocked = 0
    for _ in range(30):
        for f in ("hmap", "tmap", "seq", "head", "packed", "ptype", "n_packed",
                  "done", "volume"):
            setattr(off, f, getattr(on, f).copy())
        off._invalidate()
        fa, fb = on._positions()[0], off._positions()[0]
        assert not (fa & ~fb).any(), "the rule invented a placement"
        blocked += int((fb & ~fa).sum())
        if on.done.all():
            break
        on.step(random_action(on, rng))
        on.reset_done()
    assert blocked > 50, f"the rule never bound ({blocked} placements removed)"


def test_an_overhang_over_a_foreign_type_is_legal():
    """Only what the box touches counts, not everything below its footprint.

    A 3x1x1 box of type 0 over a column of type 0 at height 2 and a column of
    type 1 at height 1: the first two cells of its base rest on its own type
    and the third hangs over the foreign one without touching it.
    """
    env = BPPBatch(1, S=(6, 6, 6), nb=1, rot=1, ems=False, seed=0,
                   n_types=3, types=CLASSES, min_support=0.6)
    def offer(t_far):
        env.reset(np.array([[[3, 1, 1, 0]] * 4], np.int16))
        env.hmap[0, 0:2, 0] = 2          # a type-0 shelf, two cells wide
        env.tmap[0, 0:2, 0] = 0
        env.hmap[0, 2, 0] = 1            # one cell lower, so it is never touched
        env.tmap[0, 2, 0] = t_far
        env._invalidate()
        return bool(env._positions()[0][0, 0, 0, 0])

    assert offer(0) is True, "the same type below is legal either way"
    assert offer(1) is True, "an untouched foreign column blocked the placement"
    # raise the foreign column into contact and the same placement must go
    env.reset(np.array([[[3, 1, 1, 0]] * 4], np.int16))
    env.hmap[0, 0:3, 0] = 2
    env.tmap[0, 0:2, 0] = 0
    env.tmap[0, 2, 0] = 1
    env._invalidate()
    assert not env._positions()[0][0, 0, 0, 0], \
        "a contact cell of a foreign type was allowed"


def test_the_floor_takes_every_type():
    env = typed(3, n_items=4)
    env.reset(np.array([[[2, 2, 1, t]] * 4 for t in range(3)], np.int16))
    feas = env._positions()[0]
    assert feas.reshape(3, -1).any(1).all(), "an empty bin refused a type"
    assert (env.obs()["l_type"] == env.n_types).all(), \
        "a candidate on the floor did not report the floor"


def test_type_under_is_what_the_box_lands_on():
    """`l_type` must name the type of the layer each candidate rests on."""
    rng = np.random.default_rng(11)
    env = typed(6, seed=2, n_pick=1)
    checked = 0
    for _ in range(25):
        o = env.obs()
        for b in range(env.n_env):
            for i in np.nonzero(o["l_mask"][b])[0]:
                x, y, z, sx, sy, _, tu = env.candidates(b)[i]
                if z == 0:
                    assert tu == TYPE_FLOOR
                else:
                    under = env.tmap[b, x:x + sx, y:y + sy]
                    touch = under[env.hmap[b, x:x + sx, y:y + sy] == z]
                    assert touch.size and (touch == tu).all(), (b, i, tu, touch)
                checked += 1
        env.step(random_action(env, rng))
        env.reset_done()
    assert checked > 500


def test_type_under_survives_the_rule_being_lifted():
    """With the rule off a candidate may straddle types; it still reports one.

    `_type_under` then names the largest type in the contact patch, which is
    the only thing a single index can say about a mixed one -- the point of
    the test is that the sweep stays in range and keeps naming a type that is
    genuinely down there.
    """
    rng = np.random.default_rng(12)
    env = typed(6, seed=4, type_constraint=False)
    for _ in range(20):
        o = env.obs()
        for b in range(env.n_env):
            for i in np.nonzero(o["l_mask"][b])[0]:
                x, y, z, sx, sy, _, tu = env.candidates(b)[i]
                under = env.tmap[b, x:x + sx, y:y + sy]
                touch = under[env.hmap[b, x:x + sx, y:y + sy] == z] if z else []
                assert tu == (TYPE_FLOOR if z == 0 else max(touch))
        env.step(random_action(env, rng))
        env.reset_done()


def test_the_single_type_shortcut_agrees_with_the_sweep():
    """`_type_under` answers `n_types = 1` from the landing height alone.

    It is the same claim the coded sweep makes -- a footprint above the floor
    rests on the only type there is -- so the two have to agree at every
    candidate, including the ones whose own corner column is empty and whose
    landing height comes from a taller cell elsewhere under the footprint.
    """
    rng = np.random.default_rng(0)
    for seed in range(4):
        env = BPPBatch(6, S=(8, 8, 10), nb=1, seed=seed, n_types=1, types=False)
        while not env.done.all():
            _, z, odims, tu = env._positions()
            for r in range(odims.shape[1]):
                env.n_types = 2               # force the coded path
                want = env._type_under(odims[:, r], z[:, r])
                env.n_types = 1
                assert np.array_equal(tu[:, r], want), (seed, r)
            env.step(random_action(env, rng))


def test_one_type_is_the_untyped_simulator():
    """`n_types=1` must leave the rule with nothing to say."""
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    a = BPPBatch(6, S=(8, 8, 10), nb=1, seed=3, n_types=1, types=False)
    b = BPPBatch(6, S=(8, 8, 10), nb=1, seed=3, n_types=1, types=False,
                 type_constraint=False)
    assert np.array_equal(a.seq, b.seq)
    while not (a.done.all() or b.done.all()):
        act = random_action(a, rng_a)
        assert np.array_equal(a._positions()[0], b._positions()[0])
        a.step(act); b.step(random_action(b, rng_b) * 0 + act)
    assert np.array_equal(a.utilization(), b.utilization())


# ------------------------------------------------------------- the fallback
def test_a_reach_of_one_ends_the_episode_on_a_blocked_box():
    """With nothing to choose from, a box that cannot go anywhere ends the bin.

    The bin is floored wall to wall with type 0 and the next box is a type 1
    that still fits geometrically, so what stops it is the rule and nothing
    else -- which is what `type_blocked` reports.
    """
    env = BPPBatch(1, S=(4, 4, 6), nb=1, n_pick=1, rot=1, ems=False, seed=0,
                   n_types=3, types=CLASSES, min_support=0.0)
    env.reset(np.array([[[2, 2, 1, 1]] * 6], np.int16))
    env.hmap[0] = 1
    env.tmap[0] = 0
    env._invalidate()
    assert env.n_feasible()[0] == 0, "the rule let a type 1 onto a type 0 floor"
    assert bool(env.type_blocked()[0]), "the box was blocked by geometry, not type"
    env.step(np.zeros(1, np.int64))
    assert bool(env.done[0])


def test_a_wider_reach_hides_the_blocked_boxes_from_the_permuter():
    """`b_pick` is what the permuter may hand over, so a dead box leaves it."""
    env = BPPBatch(1, S=(4, 4, 6), nb=4, n_pick=3, rot=1, ems=False, seed=0,
                   n_types=3, types=CLASSES, min_support=0.0)
    # slots 0 and 2 are foreign types on a type-0 floor; slot 1 is type 0
    env.reset(np.array([[[2, 2, 1, 1], [2, 2, 1, 0], [2, 2, 1, 2],
                         [2, 2, 1, 0]]], np.int16))
    env.hmap[0] = 1
    env.tmap[0] = 0
    env._invalidate()
    assert env.pick_feasible, "the station should be filtered under the rule"
    pick = env.obs_cb()["b_pick"][0]
    assert pick.tolist() == [False, True, False, False], pick.tolist()
    assert env.obs_cb()["b_mask"][0].all(), "a blocked box went invisible too"


def test_the_episode_ends_when_the_whole_station_is_blocked():
    env = BPPBatch(1, S=(4, 4, 6), nb=3, n_pick=3, rot=1, ems=False, seed=0,
                   n_types=3, types=CLASSES, min_support=0.0)
    env.reset(np.array([[[2, 2, 1, 1], [2, 2, 1, 2], [2, 2, 1, 1]]], np.int16))
    env.hmap[0] = 1
    env.tmap[0] = 0
    env._invalidate()
    assert not env.obs_cb()["b_pick"].any(), "a blocked box stayed pickable"
    env.step(np.zeros(1, np.int64))
    assert bool(env.done[0])


def test_the_front_slot_of_b_pick_is_exactly_what_step_will_accept():
    """The invariant behind "the station empties when the episode ends".

    `step` terminates on the *head* box having no placement, and `b_pick`'s
    first slot is the same question asked of the same box, so the two have to
    agree at every step or a bin could die with its station still full.
    """
    rng = np.random.default_rng(6)
    env = typed(8, nb=5, n_pick=4, seed=13)
    blocked = 0
    while not env.done.all():
        front = env.obs_cb()["b_pick"][:, 0]
        assert np.array_equal(front, (env.n_feasible() > 0) & ~env.done)
        env.step(random_action(env, rng))
        blocked += int(env.type_blocked().sum())
    assert blocked > 0, "the rule never ended a bin, so nothing was tested"


def test_the_station_filter_stays_off_when_it_cannot_matter():
    """It costs a sweep per reachable box, so an untyped run must not pay it."""
    assert not BPPBatch(2, nb=4, n_pick=2, n_types=1, types=False).pick_feasible
    assert not BPPBatch(2, nb=4, n_pick=1).pick_feasible
    assert not BPPBatch(2, nb=4, n_pick=2, type_constraint=False).pick_feasible
    assert BPPBatch(2, nb=4, n_pick=2).pick_feasible


def test_a_permuter_reaches_past_a_blocked_box():
    """End to end: with a reach the run keeps going where a FIFO would stop."""
    rng = np.random.default_rng(0)
    long = short = 0
    for seed in range(6):
        for n_pick, tally in ((1, "short"), (5, "long")):
            env = typed(8, seed=seed, nb=5, n_pick=n_pick)
            while not env.done.all():
                if n_pick > 1:
                    # stand in for a trained permuter: take any live box
                    pick = env.obs_cb()["b_pick"]
                    env.permute(np.where(pick.any(1), pick.argmax(1), 0))
                env.step(random_action(env, rng))
            if tally == "long":
                long += env.n_packed.sum()
            else:
                short += env.n_packed.sum()
    assert long > short, (long, short)


# ------------------------------------------------------------------ tensors
def net_kw():
    return dict(d=32, n_head=2, n_layer=1, n_types=3, type_embed=16)


def test_the_networks_take_the_typed_observation():
    """Shapes end to end, for both policies and both critics."""
    env = typed(5, nb=4, n_pick=2)
    pack, perm = PackNet(**net_kw()), PermNet(**net_kw())
    o, ocb = to_torch(env.obs(), "cpu"), to_torch(env.obs_cb(), "cpu")
    assert o["c_type"].shape == o["c_mask"].shape
    assert o["b_type"].shape == o["b_mask"].shape
    assert o["l_type"].shape == o["l_mask"].shape
    for t in (o["c_type"], o["b_type"], o["l_type"]):
        assert t.dtype == torch.long
        assert int(t.min()) >= 0 and int(t.max()) <= 3, "an index left the table"
    lg, v = pack(o)
    assert lg.shape == o["l_mask"].shape and v.shape == (5,)
    lg2, v2 = perm(ocb)
    assert lg2.shape == ocb["b_mask"].shape and v2.shape == (5,)


def test_the_projection_is_wider_by_the_embedding():
    pack = PackNet(**net_kw())
    assert [m.in_features for m in pack.enc.embed] == [6 + 16, 3 + 16, 6 + 16]
    assert [m.in_features for m in pack.critic.enc.embed] == [6 + 16, 3 + 16]
    assert pack.enc.type_embedding.num_embeddings == 4, "no row for the floor"


def test_the_type_embedding_changes_the_output_and_gets_a_gradient():
    """Otherwise the type would be carried around and quietly ignored."""
    env = typed(4, nb=3)
    pack = PackNet(**net_kw())
    o = to_torch(env.obs(), "cpu")
    base = pack.logits(o)
    flipped = dict(o); flipped["l_type"] = (o["l_type"] + 1) % 4
    assert not torch.allclose(base, pack.logits(flipped)), \
        "the candidate's under-type does not reach the pointer"
    pack.logits(o).sum().backward()
    g = pack.enc.type_embedding.weight.grad
    assert g is not None and float(g.abs().sum()) > 0


def test_the_attacker_ingests_types_through_a_ppo_update():
    """pi_perm, on a real rollout batch, with the shapes PPO actually hands it."""
    from ar2l.ppo import PPO
    env = typed(6, nb=4, n_pick=3)
    perm = PermNet(**net_kw())
    ppo = PPO(perm, minibatches=2, epochs=1)
    obs = []
    for _ in range(3):
        o = to_torch(env.obs_cb(), "cpu")
        obs.append(o)
        idx, _, _ = sample(perm.logits(o))
        env.permute(idx.numpy())
        env.step(env.obs()["l_mask"].argmax(1))
        env.reset_done()
    from ar2l.train import flat_obs
    flat = flat_obs(obs)
    n = flat["b"].shape[0]
    assert flat["b_type"].shape == flat["b_mask"].shape
    stats = ppo.update(flat, torch.zeros(n, dtype=torch.long),
                       torch.zeros(n), torch.randn(n), torch.randn(n))
    assert np.isfinite(stats["pg"]) and np.isfinite(stats["vf"])


def test_a_permutation_carries_the_types_with_the_sizes():
    """Eq. 18's bootstrap permutes the window by hand; it must move as a unit."""
    env = typed(4, nb=5, n_pick=5)
    o = to_torch(env.obs_cb(), "cpu")
    idx = torch.tensor([0, 2, 4, 1])
    out = move_to_front(o, idx)
    ar = torch.arange(4)
    assert torch.equal(out["b"][ar, 0], o["b"][ar, idx])
    assert torch.equal(out["b_type"][ar, 0], o["b_type"][ar, idx])
    assert torch.equal(out["b_mask"][ar, 0], o["b_mask"][ar, idx])
    # and the env's own permutation agrees with the tensor one
    env.permute(idx.numpy())
    assert torch.equal(to_torch(env.obs_cb(), "cpu")["b_type"], out["b_type"])


def test_ragged_rollouts_pad_the_type_axes_too():
    """`flat_obs` re-pads the node axes; an unpadded type array would misalign."""
    from ar2l.train import flat_obs
    rng = np.random.default_rng(0)
    env = typed(4, nb=3)
    obs = []
    for _ in range(6):
        obs.append(to_torch(env.obs(), "cpu"))
        env.step(random_action(env, rng))
        env.reset_done()
    assert len({o["l"].shape[1] for o in obs}) > 1, "nothing was ragged"
    flat = flat_obs(obs)
    for a, b in (("c", "c_type"), ("b", "b_type"), ("l", "l_type")):
        assert flat[a].shape[:2] == flat[b].shape
    pack = PackNet(**net_kw())
    assert pack.logits(flat).shape == flat["l_mask"].shape


def test_a_pre_type_checkpoint_loads_with_zeroed_type_columns():
    """The migration: a narrow `fc_in` is padded, and nothing else is forgiven."""
    import tempfile, os
    from ar2l.evaluate import _fit_state, load_nets
    kw = net_kw()
    net = PackNet(**kw)
    sd = {k: v for k, v in net.state_dict().items()
          if not k.endswith("type_embedding.weight")}
    for k in list(sd):
        if ".embed." in k and k.endswith(".weight") and sd[k].dim() == 2:
            sd[k] = sd[k][:, : sd[k].shape[1] - kw["type_embed"]].clone()
    got = _fit_state(PackNet(**kw), sd)
    for k, v in got.state_dict().items():
        if ".embed." in k and k.endswith(".weight") and v.dim() == 2:
            assert bool((v[:, -kw["type_embed"]:] == 0).all()), k
            assert torch.equal(v[:, : sd[k].shape[1]], sd[k]), k
    # it still runs, and on the typed observation
    env = typed(3, nb=3)
    assert got.logits(to_torch(env.obs(), "cpu")).shape[0] == 3

    for broken, why in ((dict(sd, nonsense=torch.zeros(1)), "a stray weight"),
                        ({k: v for k, v in sd.items() if k != "ptr.q.weight"},
                         "a missing head")):
        with pytest.raises(ValueError):
            _fit_state(PackNet(**kw), broken)


# -------------------------------------------------------------- integration
def test_the_heuristics_keep_the_rule():
    """They score the env's own candidate list, so the rule has to reach them."""
    from ar2l import heuristics as H
    for name in H.NAMES:
        env = typed(4, seed=17, n_items=80)
        for _ in range(30):
            m = env.obs()["l_mask"]
            if not m.any():
                break
            env.step(np.where(m.any(1), H.act(env, name), 0))
        for b in range(env.n_env):
            for place, t, touch in voxels(env, b)[1]:
                assert place[2] == 0 or (touch == t).all(), (name, place, t)
        assert env.utilization().mean() > 0, name


def test_a_network_policy_episode_log_holds_no_mixed_stack():
    """Checklist item 1, end to end through the policy rather than the mask.

    The packer and the permuter both act, greedily, on the observation the env
    hands them -- so this is the episode log a training rollout writes, and the
    claim is about that log and not about the candidate list behind it.
    """
    env = typed(8, nb=5, n_pick=3, seed=21)
    pack, perm = PackNet(**net_kw()), PermNet(**net_kw())
    steps = 0
    while not env.done.all():
        idx, _, _ = sample(perm.logits(to_torch(env.obs_cb(), "cpu")), greedy=True)
        env.permute(idx.numpy())
        o = to_torch(env.obs(), "cpu")
        a, _, _ = sample(pack.logits(o), greedy=True)
        env.step(a.numpy())
        steps += 1
    stacked = 0
    for b in range(env.n_env):
        for place, t, touch in voxels(env, b)[1]:
            if place[2] == 0:
                continue
            assert touch.size and (touch == t).all(), (b, place, t, touch)
            stacked += 1
    assert steps > 5 and stacked > 20, (steps, stacked)


def test_a_packed_bin_matches_its_type_map():
    """`tmap` is what the rule is tested against, so it must be the real top."""
    rng = np.random.default_rng(5)
    env = play(typed(6, seed=9), rng)
    for b in range(env.n_env):
        vox = voxels(env, b)[0]
        for x in range(env.Lx):
            for y in range(env.Ly):
                h = int(env.hmap[b, x, y])
                want = TYPE_FLOOR if h == 0 else int(vox[x, y, h - 1])
                assert int(env.tmap[b, x, y]) == want, (b, x, y, h)
