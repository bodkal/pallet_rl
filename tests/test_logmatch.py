"""The log match: a conveyor log told apart into boxes, then packed a pick
per step, with a heuristic so nothing loads a checkpoint."""
import time

from ar2l.viz import game as G
from ar2l.viz import logmatch as LM

HEAD = ("step,time,conveyor,run_detected,box_no,type,status,size_L_cm,"
        "size_W_cm,size_H_cm,row_L_cm,row_W_cm,row_H_cm,rms,mid_x_m,mid_y_m,"
        "mid_z_m,picked\n")


def write(path, steps):
    """`steps`: per step, its boxes as (type, (l, w, h), y, picked)."""
    lines = []
    for s, rows in enumerate(steps, start=1):
        for no, (t, (l, w, h), y, pk) in enumerate(rows, start=1):
            lines.append(f"{s},2026-10-13 21:{s:02d}:00,PL11PR,{len(rows)},{no},"
                         f"{t},passed,{l},{w},{h},{l},{w},2.0,0.01,0.0,{y},"
                         f"-0.3,{'yes' if pk else ''}\n")
    path.write_text(HEAD + "".join(lines))
    return str(path)


def tracked(tmp_path, steps):
    s = LM.read_log(write(tmp_path / "log.csv", steps))
    return LM.track(s), s


BROWN = (3, (30.2, 24.4, 19.0))


def box(kind, y, picked=False):
    return (kind[0], kind[1], y, picked)


def test_the_rows_are_read_front_first_and_types_count_from_one(tmp_path):
    s = LM.read_log(write(tmp_path / "a.csv", [[
        (3, (20, 14.8, 12), -1.4, False), (1, (50, 30, 18), -0.9, True)]]))
    assert [r["type"] for r in s[0]["rows"]] == [0, 2]
    assert [r["picked"] for r in s[0]["rows"]] == [True, False]


def test_a_box_still_on_the_belt_is_not_new(tmp_path):
    a, b, c = (3, (18, 18.6, 13.7)), (2, (45.5, 27.5, 18)), (1, (50, 30, 18))
    boxes, s = tracked(tmp_path, [[box(a, -0.9), box(b, -1.1, True)],
                                  [box(a, -0.9), box(c, -1.1)]])
    assert len(boxes) == 3
    assert s[0]["real"] == 1
    assert s[1]["ids"] == [0, 2] and s[1]["new"] == [2]


def test_twins_with_one_picked_bring_one_new_box_at_the_back(tmp_path):
    ys = (-0.9, -1.14, -1.38, -1.61)
    boxes, s = tracked(tmp_path, [
        [box(BROWN, y, i == 2) for i, y in enumerate(ys)],
        [box(BROWN, y) for y in ys]])
    assert len(boxes) == 5
    # the back one slid into the picked one's place, and a new one came on
    assert s[1]["ids"] == [0, 1, 3, 4] and s[1]["new"] == [4]


def test_a_twin_of_the_back_box_picked_is_a_new_box(tmp_path):
    a = (1, (50, 30, 18))
    boxes, s = tracked(tmp_path, [[box(a, -0.9), box(BROWN, -1.6, True)],
                                  [box(a, -0.9), box(BROWN, -1.6)]])
    assert s[1]["new"] == [2]


def test_a_pick_that_failed_keeps_its_box(tmp_path):
    a, b, c = (3, (18, 18.6, 13.8)), (3, (20, 14.8, 12.4)), (2, (45.5, 27.5, 25))
    # the robot "took" a, which is still there in front of b next step
    boxes, s = tracked(tmp_path, [[box(a, -1.09, True), box(b, -1.25)],
                                  [box(a, -1.09), box(b, -1.25), box(c, -1.45)]])
    assert s[1]["ids"] == [0, 1, 2] and s[1]["new"] == [2]


def test_a_box_the_camera_missed_is_found_again(tmp_path):
    a, b = (1, (50, 30, 18)), (3, (20, 14.8, 12.4))
    boxes, s = tracked(tmp_path, [[box(a, -0.9), box(b, -1.25)],
                                  [box(a, -0.9, True)],
                                  [box(b, -1.25)]])
    assert len(boxes) == 2 and s[2]["new"] == []


def test_sides_match_in_any_order_within_the_tolerance(tmp_path):
    boxes, s = tracked(tmp_path, [[(3, (16.3, 14.8, 17.3), -0.87, False)],
                                  [(3, (14.8, 16.8, 18.0), -0.86, False)],
                                  [(3, (18.7, 14.8, 16.0), -0.88, False)]])
    assert [x["new"] for x in s] == [[0], [], []]
    s = LM.read_log(str(tmp_path / "log.csv"))
    assert len(LM.track(s, tol_cm=0.5)) == 3


def test_a_box_that_slid_back_along_the_belt_is_another_box(tmp_path):
    boxes, s = tracked(tmp_path, [[box(BROWN, -0.9)], [box(BROWN, -1.5)]])
    assert s[1]["new"] == [1]


# ------------------------------------------------------------------ the run
def params(**over):
    p, err = G.exp_params(dict(G.defaults(), bin=[10, 10, 6], pallet_cm=None,
                               cell_cm=1.0, box_scale=1.0, box_pad_m=0.0,
                               nb=3, n_pick=2, max_l=64, arm_collision=0,
                               type_constraint=0) | over)
    assert not err, err
    return p


def run(tmp_path, steps, **over):
    path = write(tmp_path / "log.csv", steps)
    p = params(**over)
    j = LM.start(path, "heur:dbl", p, G.cell_m(p), "cpu")
    while LM.status(j)["state"] == "running":
        time.sleep(0.05)
    s = LM.status(j)
    assert s["state"] == "done", s
    return j, s


def test_one_box_per_step_and_never_one_of_a_later_step(tmp_path):
    # six boxes no two of which are alike, two coming on at each step
    small = [(1, (2, 2, 1)), (2, (5, 2, 1)), (3, (2, 2, 4)),
             (1, (8, 2, 1)), (2, (2, 9, 3)), (3, (4, 4, 3))]
    steps = [[box(small[2 * k], -1.3), box(small[2 * k + 1], -1.5)]
             for k in range(3)]
    j, s = run(tmp_path, steps)
    pal = LM.pallet(j, 0)
    assert [w["step"] for w in pal["windows"]] == [1, 2, 3, None, None, None]
    sl = LM.read_log(str(tmp_path / "log.csv"))
    seen = {b["id"]: sl[b["at"]]["step"] for b in LM.track(sl)}
    for w in pal["windows"]:
        if w["step"] is not None:
            assert all(seen[i] <= w["step"] for i in w["ids"])
    assert s["packed"] == s["boxes"] == 6 and s["drain"] == 3


def test_a_full_pallet_opens_the_next_with_the_same_station(tmp_path):
    # 10x10x6 holds six 10x10x1 slabs; one comes on per step
    slab = (1, (10, 10, 1))
    j, s = run(tmp_path, [[box(slab, -0.9, True)] for k in range(8)])
    assert [q["boxes"] for q in s["pallets"]] == [6, 2]
    first = LM.pallet(j, 0)
    assert first["reason"] == "no box in reach has a place"
    assert first["windows"][-1]["pick"] == -1
    assert first["windows"][-1]["step"] == 7


def test_a_box_no_pallet_holds_is_set_aside(tmp_path):
    j, s = run(tmp_path, [[box((1, (11, 10, 1)), -0.9)],
                          [box((1, (2, 2, 1)), -0.9)]])
    assert s["skipped"] == 1 and s["packed"] == 1


def test_the_points_of_a_closed_pallet_are_asked_again(tmp_path):
    slab = (1, (10, 10, 1))
    j, s = run(tmp_path, [[box(slab, -0.9, True)] for k in range(7)])
    r = LM.points(j, 0, 0, 0)
    assert r["cands"] and r["n"] == 0
    assert LM.points(j, 0, 6, 0)["cands"] == []      # the pallet it closed on


# ------------------------------------------------ the robot's own packing
PHEAD = HEAD.rstrip("\n") + (",box_id,sku,pack_location,not_packed_reason,"
                             "box_x_cm,box_y_cm,box_z_cm,orientation_deg,"
                             "pack_x_m,pack_y_m,pack_z_m\n")


def write_packed(path, steps):
    """`steps`: per step, (type, (l, w, h), y, pick) with `pick` None, a
    failure reason, or the robot's (x, y, z, orientation) in cm."""
    lines = []
    for s, rows in enumerate(steps, start=1):
        for no, (t, (l, w, h), y, pk) in enumerate(rows, start=1):
            pos = ("yes,1,X,A01,," + ",".join(str(v) for v in pk) + ",0,0,0"
                   if isinstance(pk, tuple) else
                   f"yes,-,-,,{pk},,,,,,," if pk else ",,,,,,,,,,,")
            lines.append(f"{s},2026-10-12 12:{s:02d}:00,PL01PR,{len(rows)},{no},"
                         f"{t},passed,{l},{w},{h},{l},{w},{h},0,0.0,{y},-0.4,"
                         f"{pos}\n")
    path.write_text(PHEAD + "".join(lines))
    return str(path)


def test_a_failed_pick_leaves_the_box_on_the_belt(tmp_path):
    a = (3, (20, 14, 16))
    s = LM.read_log(write_packed(tmp_path / "p.csv", [
        [(*a, -1.29, "grasp validation failed")], [(*a, -1.29, None)]]))
    boxes = LM.track(s)
    assert len(boxes) == 1 and s[1]["new"] == []
    assert s[0]["real"] == 0


def test_the_real_pallet_is_built_from_the_pack_positions(tmp_path):
    blue, brown = (1, (50, 30, 28)), (3, (31, 26, 24))
    s = LM.read_log(write_packed(tmp_path / "p.csv", [
        [(*blue, -1.5, (66, 0, 0, -90))],           # turned: L along x
        [(*brown, -1.6, (35, 0, 0, 0))],            # as is: L along y
        [(*brown, -1.6, "could not reach pick pose")],
        [(*brown, -1.6, (34, 0, 0, 0))]]))          # where a box stands: a new pallet
    real = LM.real_pack(s, 1.0)
    assert real[0]["box"] == [66, 0, 0, 50, 30, 28]
    assert real[1]["box"] == [35, 0, 0, 26, 31, 24]
    assert real[2]["box"] is None and real[2]["failed"]
    assert [e["pallet"] for e in real] == [0, 0, 0, 1]


def test_an_older_log_has_no_real_pallet(tmp_path):
    s = LM.read_log(write(tmp_path / "a.csv", [[box(BROWN, -0.9, True)]]))
    assert LM.real_pack(s, 1.0) is None


def test_each_real_pallet_is_summed_up_at_the_agents_box_sizes(tmp_path):
    blue, brown = (1, (50, 30, 28)), (3, (31, 26, 24))
    s = LM.read_log(write_packed(tmp_path / "p.csv", [
        [(*blue, -1.5, (66, 0, 0, -90))],
        [(*brown, -1.6, (35, 0, 0, 0))],
        [(*brown, -1.6, "could not reach pick pose")],
        [(*brown, -1.6, (34, 0, 0, 0))]]))
    p = params(box_pad_m=0.04)
    real = LM.real_pack(s, 1.0, p)
    R = LM.real_pallets(real, (100, 100, 100), 1.0)
    assert [q["boxes"] for q in R] == [2, 1]
    assert R[0]["types"] == [1, 0, 1] and R[0]["failed"] == 1
    assert R[0]["steps"] == [1, 2] and R[1]["steps"] == [4, 4]
    # padded as the agent's grid pads them: 54 x 34 x 28 and 35 x 30 x 24
    assert R[0]["util"] == (54 * 34 * 28 + 35 * 30 * 24) / 1e6
    assert R[0]["max_height"] == 28


def test_steps_count_from_one_whatever_the_log_starts_at(tmp_path):
    path = tmp_path / "late.csv"
    path.write_text(HEAD + "".join(
        f"{s},2026-10-12 18:{s % 60:02d}:00,PL01PR,1,1,3,passed,20,14,16,20,14,"
        f"16,0,0.0,-0.9,-0.4,yes\n" for s in (36, 37, 40)))
    s = LM.read_log(str(path))
    assert [st["step"] for st in s] == [1, 2, 3]
    assert [st["log"] for st in s] == [36, 37, 40]


def test_a_crate_nesting_into_the_one_under_it_is_the_same_pallet(tmp_path):
    white18, white25 = (2, (45.5, 27.5, 18)), (2, (45.5, 27.5, 25))
    s = LM.read_log(write_packed(tmp_path / "p.csv", [
        [(*white18, -1.2, (0, 54, 105, 0))],
        [(*white25, -1.2, (0, 54, 118, 0))]]))       # 5 cm into the one under it
    assert [e["pallet"] for e in LM.real_pack(s, 1.0)] == [0, 0]


def test_a_new_pallet_id_finishes_the_real_pallet(tmp_path):
    path = tmp_path / "ids.csv"
    head = PHEAD.replace("pack_location,", "pack_location,pallet_id,")
    rows = [(1, "P1", (0, 0, 0)), (2, "P1", (0, 0, 0)),       # overlap, same id
            (3, "P2", (60, 0, 0))]                           # no overlap, new id
    path.write_text(head + "".join(
        f"{s},2026-10-12 12:{s:02d}:00,PL02PR,1,1,1,passed,50,30,28,50,30,28,0,"
        f"0.0,-1.4,-0.3,yes,1,X,A01,{pid},,{x},{y},{z},0,0,0,0\n"
        for s, pid, (x, y, z) in rows))
    real = LM.real_pack(LM.read_log(str(path)), 1.0)
    assert [e["pallet"] for e in real] == [0, 0, 1]
    assert [q["id"] for q in LM.real_pallets(real, (100, 100, 100), 1.0)] == ["P1", "P2"]
