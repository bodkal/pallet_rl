"""The big experiment: a folder of pallet files, each packed onto as many
pallets as it takes, with a heuristic so nothing loads a checkpoint."""
import time

from ar2l.viz import experiment as X
from ar2l.viz import game as G

HEAD = ("TRAN_DATE,TRAN_HOUR,CONTAINER_ID,PALLET_ID,FROM_LOC_ID,"
        "V_BOX_DEPTH,V_BOX_WIDTH,V_BOX_HEIGHT,TYPE\n")


def params(**over):
    p, err = G.exp_params(dict(G.defaults(), bin=[10, 10, 6], pallet_cm=None,
                               cell_cm=1.0, box_scale=1.0, nb=3, n_pick=1,
                               max_l=64, arm_collision=0, order_random=0.0) | over)
    assert not err, err
    return p


def write(path, rows):
    path.write_text(HEAD + "".join(
        f"2024-05-30,13:05:05,{i},{pid},PL33PR,{l},{w},{h},{t}\n"
        for i, (pid, l, w, h, t) in enumerate(rows)))


def run(tmp_path, rows, spec="heur:dbl", run_kw=None, **over):
    src = tmp_path / "in"
    src.mkdir(parents=True)
    write(src / "day.csv", rows)
    p = params(**over)
    j = X.start(str(src), str(tmp_path / "out"), spec, p, G.cell_m(p), "cpu",
                **(run_kw or {}))
    while X.status(j)["state"] == "running":
        time.sleep(0.05)
    return X.status(j)


def test_the_columns_are_mapped_and_types_count_from_one(tmp_path):
    f = tmp_path / "a.csv"
    write(f, [("A", 4, 3, 2, 1), ("B", 5, 2, 1, 3)])
    seq, ids, n, skipped = X.read_file(str(f), (10, 10, 6), params())
    assert seq.tolist() == [[4, 3, 2, 0], [5, 2, 1, 2]]
    assert (ids, n, skipped) == (2, 2, 0)


def test_a_full_pallet_opens_the_next_and_every_box_goes_somewhere(tmp_path):
    # 10x10x6 holds six 10x10x1 slabs; fourteen of them take three pallets
    s = run(tmp_path, [("P1", 10, 10, 1, 1)] * 8 + [("P2", 10, 10, 1, 2)] * 6,
            type_constraint=0)
    assert s["state"] == "done", s
    f = s["files"][0]
    assert [q["boxes"] for q in f["pallets"]] == [6, 6, 2]
    assert f["done"] == f["total"] == 14


def test_the_report_is_in_the_cells_format(tmp_path):
    run(tmp_path, [("P1", 10, 10, 1, 1)] * 7 + [("P2", 10, 10, 2, 3)],
        type_constraint=0)
    out = (tmp_path / "out" / "day.csv").read_text().splitlines()
    first = out[0].split(",")
    assert first[:7] == ["day.csv", "current date:", first[2], "boxes:", "8",
                         "palet: ", "2"]
    assert out[1] == X.HEADER
    rows = [r.split(",") for r in out[2:]]
    assert [r[0] for r in rows] == ["0", "1"]
    assert rows[0][1] == "6" and rows[0][2] == "0" and rows[0][3] == "100"
    # blue, white, master: the 7th blue box and the master box share pallet 1
    assert rows[1][5:9] == ["2", "1", "0", "1"]
    assert all(r[-1] == "" for r in rows)            # the trailing comma


def test_a_box_that_fits_no_pallet_is_counted_not_packed(tmp_path):
    s = run(tmp_path, [("P1", 10, 10, 1, 1), ("P1", 30, 10, 1, 1)])
    f = s["files"][0]
    assert f["skipped"] == 1 and f["total"] == 1 and f["state"] == "done"


def test_the_experiment_form_refuses_what_the_simulator_could_not_run():
    _, err = G.exp_params(dict(G.defaults(), nb=2, n_pick=5))
    assert any("reach k" in e for e in err)


def test_the_experiment_form_reaches_the_packer(tmp_path):
    s = run(tmp_path, [("P1", 10, 10, 1, 1)] * 3, nb=4, n_pick=2)
    assert "nb=4 n_pick=2" in (tmp_path / "out" / "day.csv").read_text()
    assert s["state"] == "done"


def test_a_second_run_adds_to_the_report_rather_than_replacing_it(tmp_path):
    rows = [("P1", 10, 10, 1, 1)] * 3
    run(tmp_path, rows)
    out = tmp_path / "out" / "day.csv"
    once = out.read_text()
    (tmp_path / "in").rename(tmp_path / "in0")      # `run` makes a fresh folder
    run(tmp_path, rows)
    twice = out.read_text().splitlines()
    assert len(twice) == 2 * len(once.splitlines())
    assert twice[: len(once.splitlines())] == once.splitlines()
    assert twice[len(once.splitlines()) + 1] == X.HEADER


def test_file_order_has_no_seed_and_no_suffix(tmp_path):
    run(tmp_path, [("P1", 10, 10, 1, 1)] * 3, run_kw=dict(name="n"))
    text = (tmp_path / "out" / "day.csv").read_text()
    assert "seed=" not in text and "name:,n," in text


def test_a_seeded_order_replays_exactly(tmp_path):
    rows = [("P1", 1 + i % 9, 1 + i % 7, 1, 1) for i in range(30)]
    a = run(tmp_path / "a", rows, type_constraint=0, order_random=0.5,
            run_kw=dict(seed=3))
    b = run(tmp_path / "b", rows, type_constraint=0, order_random=0.5,
            run_kw=dict(seed=3))
    key = lambda s: [(q["boxes"], q["util_pct"]) for q in s["files"][0]["pallets"]]
    assert key(a) == key(b)


def test_order_random_shuffles_by_seed_and_zero_keeps_the_file(tmp_path):
    rows = [("P1", 1 + i % 9, 1 + i % 7, 1, 1) for i in range(30)]
    f = tmp_path / "a.csv"
    write(f, rows)
    seq = X.read_file(str(f), (10, 10, 6), params())[0]
    from ar2l.orders import randomize_order
    assert (randomize_order(seq[None], 0.0, 1)[0] == seq).all()
    a = randomize_order(seq[None], 0.5, 1)[0]
    assert (a == randomize_order(seq[None], 0.5, 1)[0]).all()
    assert not (a == seq).all()
    s = run(tmp_path, rows, type_constraint=0, order_random=0.5,
            run_kw=dict(name="o", seed=4))
    assert s["state"] == "done", s
    text = (tmp_path / "out" / "day.csv").read_text()
    assert "order_random=0.5 " in text and " seed=4," in text
    assert "name:,o," in text


def test_a_name_already_in_the_results_folder_is_refused(tmp_path):
    rows = [("P1", 10, 10, 1, 1)] * 3
    run(tmp_path, rows, type_constraint=0, box_pad_m=0.0,
        run_kw=dict(name="first"))
    out = str(tmp_path / "out")
    assert X.name_used(out, "first") == ["day.csv"]
    assert X.name_used(out, "First") == [] and X.name_used(out, "") == []
    p = params(box_pad_m=0.0)
    try:
        X.start(str(tmp_path / "in"), out, "heur:dbl", p, G.cell_m(p), "cpu",
                name=" first ")
    except ValueError as e:
        assert "already" in str(e)
    else:
        raise AssertionError("a used name was let through")
    (tmp_path / "in").rename(tmp_path / "in0")      # `run` makes a fresh folder
    s = run(tmp_path, rows, type_constraint=0, box_pad_m=0.0,
            run_kw=dict(name="second"))                # same folder, new name
    assert s["state"] == "done", s
    assert X.name_used(out, "second") == ["day.csv"]
