"""The game server: the parameter form, and the pick station.

Nothing here loads a checkpoint, so the whole file runs on CPU in a second.
What it pins is the two things the page cannot check for itself: that a
parameter set the form will accept is one `BPPBatch` will actually run, and
that choosing a box off the pick station permutes the conveyor exactly once,
against the order the strip is showing.
"""
import numpy as np
import pytest

from ar2l.viz import game as G

BASE = {"bin": [12, 12, 14], "size_lo": [2, 2, 2], "size_hi": [5, 5, 5],
        "n_items": 40, "nb": 6, "n_pick": 3, "max_l": 64, "rot": 2, "ems": 1,
        "stability": "com", "min_support": 0.3}


def start(**over):
    p, err = G.validate(dict(BASE, **over))
    assert not err, err
    return G.new_game(3, "none", p, human_pick=over.pop("human_pick", True))[1]


# ------------------------------------------------------------- the form
def test_the_form_is_filled_from_the_run_not_from_the_config():
    """Whatever `config.yaml` says, a run brings its own geometry."""
    p = G.run_params("run:sel_2cm_k5") if _has("sel_2cm_k5") else None
    if p is None:
        pytest.skip("no runs/ checkpoints to read a configuration from")
    import json
    import os
    a = json.load(open(os.path.join(G.A.run_dir("sel_2cm_k5"), "args.json")))
    assert p["bin"] == list(G.A.extent(a["bin"]))
    assert p["nb"] == a["nb"]
    # a run trained before --n_pick existed recorded null, meaning "all of them"
    assert p["n_pick"] == (a["n_pick"] or a["nb"])


def _has(run):
    return G.A.run_info(run) is not None


@pytest.mark.parametrize("bad, why", [
    ({"nb": 4, "n_pick": 7}, "reach k"),
    ({"size_lo": [9, 9, 9], "size_hi": [4, 4, 4]}, "smallest"),
    ({"bin": [3, 3, 3]}, "does not fit the bin"),
    ({"bin": "nonsense"}, "whole number"),
    ({"bin": [12, 12]}, "three numbers"),
    ({"stability": "zzz"}, "stability rule"),
    ({"min_support": 3.0}, "support floor"),
    ({"rot": 5}, "orientations"),
    ({"bin": [G.MAX_AXIS + 1, 4, 4]}, "between 1 and"),
])
def test_the_form_refuses_what_the_simulator_could_not_run(bad, why):
    _, err = G.validate(dict(BASE, **bad))
    assert err and any(why in e for e in err), (bad, err)


def test_anything_the_form_accepts_the_simulator_accepts():
    """The point of validating server-side: no accepted set may raise."""
    for over in ({}, {"rot": 1}, {"ems": 0}, {"stability": "cdrl"},
                 {"min_support": 0.0}, {"min_support": 1.0}, {"n_pick": 1},
                 {"nb": 1, "n_pick": 1}, {"bin": [4, 4, 4], "size_hi": [4, 4, 4]},
                 {"size_lo": [5, 5, 5], "size_hi": [5, 5, 5]}):
        st = start(**over)
        b = G.board(st)
        assert b["Lx"] and len(b["picks"]) == min(b["n_pick"], b["nb"])


def test_the_parameters_reach_the_simulator():
    st = start(bin=[9, 11, 13], rot=1, ems=0, stability="cdrl", min_support=0.0)
    env, b = st["env"], G.board(st)
    assert (b["Lx"], b["Ly"], b["Lz"]) == (9, 11, 13)
    assert b["rot"] == 1 and env.ems is False and env.stability == "cdrl"
    assert b["ncand"] == b["nfree"], "ems off means every free cell is a candidate"
    assert all(2 <= v <= 5 for it in b["window"] for v in it)


def test_drift_from_the_run_is_reported_field_by_field():
    if not _has("sel_2cm_k5"):
        pytest.skip("no runs/ checkpoints")
    p = G.run_params("run:sel_2cm_k5")
    assert G.mismatch(p, "run:sel_2cm_k5") == []
    off = {m["key"] for m in G.mismatch(dict(p, nb=99, bin=[1, 2, 3]),
                                        "run:sel_2cm_k5")}
    assert off == {"nb", "bin"}


# ----------------------------------------------------------- pick station
def test_picking_takes_the_box_you_pointed_at():
    st = start()
    w0 = [list(x) for x in G.board(st)["window"]]
    for i in range(3):
        assert G.select(st, i)
        assert G.board(st)["item"] == w0[i]


def test_the_strip_keeps_its_order_however_often_you_change_your_mind():
    """Each pick is one permutation of the *remembered* window, not two."""
    st = start()
    w0 = [list(x) for x in G.board(st)["window"]]
    for i in (2, 0, 1, 2, 1, 0, 2):
        G.select(st, i)
        b = G.board(st)
        assert [list(x) for x in b["window"]] == w0
        assert b["item"] == w0[i] and b["sel"] == i


def test_the_preview_tail_is_out_of_reach():
    st = start()
    assert not G.select(st, 3), "slot 3 is preview at k = 3"
    assert not G.select(st, 99)
    assert not G.select(st, -1)
    assert G.board(st)["sel"] == 0


def test_the_box_you_picked_is_the_box_that_goes_in():
    st = start()
    env = st["env"]
    w0 = [list(x) for x in G.board(st)["window"]]
    G.select(st, 2)
    r, x, y = (int(v) for v in np.argwhere(G.free_positions(env))[0])
    assert G.place(env, r, x, y)
    got = (env.packed[0, 0][3:6] * env.scale).round().astype(int).tolist()
    assert sorted(got) == sorted(w0[2])
    # the two you passed over are still at the front, in their old order
    G.advance(st)
    want = [w0[0], w0[1]] + w0[3:]
    assert [list(t) for t in G.board(st)["window"]][:len(want)] == want


def test_a_pick_leaves_the_conveyor_no_shorter_than_it_found_it():
    """A pick rewrites the sequence in place; it must not lose an item."""
    st = start()
    env = st["env"]
    before = env.seq.copy()
    for i in (2, 1, 0, 2):
        G.select(st, i)
        head = int(env.head[0])
        assert sorted(map(tuple, env.seq[0, head:head + env.nb].tolist())) == \
               sorted(map(tuple, before[0, head:head + env.nb].tolist()))
        assert (env.seq[0, head + env.nb:] == before[0, head + env.nb:]).all(), \
            "only the window may move"


def test_free_positions_restores_the_filter_it_borrowed():
    for ems in (0, 1):
        st = start(ems=ems)
        G.free_positions(st["env"])
        assert st["env"].ems is bool(ems)


def test_the_bin_is_finished_only_when_every_reachable_box_is_stuck():
    """With a reach you are stuck only if all k are; without one, if yours is."""
    st = start()
    b = G.board(st)
    assert len(b["picks"]) == 3 and b["done"] is False
    # a bin with no room at all is done under either reading
    st["env"].hmap[:] = st["env"].Lz
    st["env"]._invalidate()
    assert G.board(st)["done"] is True
    assert all(p == 0 for p in G.board(st)["picks"])


def test_withholding_the_pick_pins_you_to_the_front_box():
    st = start(human_pick=False)
    b = G.board(st)
    assert b["human_pick"] is False and b["sel"] == 0
    assert b["item"] == list(b["window"][0])


def test_k_of_one_is_the_fifo_conveyor_of_the_paper():
    st = start(n_pick=1)
    b = G.board(st)
    assert b["n_pick"] == 1 and len(b["picks"]) == 1
    assert b["human_pick"] is False, "there is nothing to choose from"
    assert not G.select(st, 1)


# ------------------------------------------------------- recalculation
# Heuristics only, so this section keeps the file's promise not to load a
# checkpoint: what is under test is which parameters a rerun is allowed to
# move, not which policy plays it.
HEUR = "heur:dbl"


def test_a_recalculation_replays_the_boxes_it_was_dealt():
    """The stream fields come from the session, whatever the request says."""
    st = start()
    p, err = G.recalc_params(st, {"n_items": 999, "size_lo": [1, 1, 1],
                                  "size_hi": [9, 9, 9], "bin": [16, 16, 14]})
    assert not err, err
    for k in G.STREAM_KEYS:
        assert p[k] == st["p"][k], f"{k} is what deals the boxes; it may not move"
    assert p["bin"] == [16, 16, 14], "everything else is free to"


def test_a_recalculation_at_the_games_own_values_reproduces_the_duel():
    st = start()
    duel = G.opponent(st, HEUR)
    row = G.recalc(st, HEUR, G.recalc_params(st, {})[0])
    assert (row["util"], row["items"]) == (duel["util"], duel["items"])
    assert row["changed"] == [], "nothing moved, so nothing to report"


def test_a_recalculation_reports_the_fields_it_moved():
    st = start()
    p, err = G.recalc_params(st, {"n_pick": 1, "ems": 0})
    assert not err, err
    row = G.recalc(st, HEUR, p)
    assert {c["key"]: (c["is"], c["was"]) for c in row["changed"]} == \
           {"n_pick": ("1", "3"), "ems": ("0", "1")}


def test_a_recalculation_refuses_what_the_simulator_could_not_run():
    st = start()
    _, err = G.recalc_params(st, {"n_pick": 99})
    assert err and any("reach k" in e for e in err)
    # and a bin the dealt boxes no longer fit is refused on their behalf, since
    # they are the ones the rerun is pinned to
    _, err = G.recalc_params(st, {"bin": [3, 3, 3]})
    assert err and any("does not fit the bin" in e for e in err)


def test_the_recalculation_form_covers_every_field_the_setup_form_has():
    """No parameter may fall between the two panels unnoticed."""
    assert set(G.STREAM_KEYS) | set(G.RECALC_KEYS) == set(G.PARAM_KEYS)
    assert not set(G.STREAM_KEYS) & set(G.RECALC_KEYS)


def test_a_recalculation_plays_the_model_it_was_handed():
    """The model is the rerun's other axis, and comes from the call, not the
    session: a row must be able to race someone the game never faced."""
    st = start()
    p = G.recalc_params(st, {})[0]
    labels = {G.recalc(st, "heur:" + h, p)["label"] for h in ("dbl", "lsah")}
    assert labels == {"DBL", "LSAH"}
    # and the session's own cached opponent is untouched by any of it
    assert st["opp_spec"] is None
