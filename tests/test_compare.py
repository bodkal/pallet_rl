"""The Compare tab: reading a folder of big-experiment reports, picking
experiments by name, and deleting them by name from every report."""
import pytest

from ar2l.viz import compare as C
from ar2l.viz import experiment as X
from ar2l.viz import game as G


def block(name, pallets, first=None):
    """One run's block as `experiment.write_report` writes it."""
    head = (f"day.csv,current date:,2026-10-01 08:00:00,boxes:,{sum(pallets)},palet: ,1,"
            "packer=heur:dbl ,cell_cm=1," + (f"name:,{name}," if name else ""))
    rows = [f"{i},150,0,{50 + i},100,{n},0,0,{n}," for i, n in enumerate(pallets)]
    return "\n".join([head, X.HEADER] + rows) + "\n"


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "a.csv").write_text(block("", [5, 3]) + block("base", [6, 2])
                                    + block("Base", [8]) + block("base", [7, 1]))
    (tmp_path / "b.csv").write_text(block("base", [4]) + block("wide", [4]))
    return tmp_path


def test_delete_named_drops_every_block_of_that_name_and_keeps_the_rest(folder):
    before = (folder / "a.csv").read_text()
    assert C.delete_named(str(folder / "a.csv"), ["base"]) == 2
    exps = C.read_report(str(folder / "a.csv"))
    assert [C.key(e) for e in exps] == ["Exp 1", "Base"]
    # what is kept is the original text, block for block
    keep = block("", [5, 3]) + block("Base", [8])
    assert (folder / "a.csv").read_text() == keep
    assert before != keep


def test_delete_named_leaves_a_report_without_that_name_alone(folder):
    text = (folder / "b.csv").read_text()
    assert C.delete_named(str(folder / "b.csv"), ["nope", ""]) == 0
    assert (folder / "b.csv").read_text() == text


def test_a_report_left_empty_is_removed(folder):
    assert G.cmp_delete(str(folder), ["base", "wide"]) == {"a.csv": 2, "b.csv": 2}
    assert not (folder / "b.csv").exists()
    assert [k["key"] for k in G.cmp_reports(str(folder))["keys"]] == ["Exp 1", "Base"]


def test_the_folder_lists_every_key_and_the_winners_follow_the_pick(folder):
    r = G.cmp_reports(str(folder))
    # by the first place each ran at; `base` ran twice in a.csv
    assert [(k["key"], k["named"], k["files"], k["blocks"]) for k in r["keys"]] == [
        ("Exp 1", False, 1, 1), ("base", True, 2, 3), ("wide", True, 1, 1),
        ("Base", True, 1, 1)]
    a = r["files"][0]
    assert a["file"] == "a.csv" and a["win"] == ["Base"] and a["best"] == [1, 8]
    assert a["exps"][1]["pallets"] == [{"id": 0, "boxes": 6, "volume": 50.0},
                                       {"id": 1, "boxes": 2, "volume": 51.0}]
    # leave the one-pallet run out and the two-pallet runs compete
    r = G.cmp_reports(str(folder), {"Exp 1", "base"})
    assert r["files"][0]["win"] == ["base"] and r["files"][0]["best"] == [2, 1]
    assert len(r["keys"]) == 4                 # every key is still offered
    assert G.cmp_reports(str(folder), set())["files"] == []


def test_no_delete_while_an_experiment_writes_there(folder, monkeypatch):
    monkeypatch.setitem(X._JOBS, "latest", {"state": "running", "out": str(folder)})
    with pytest.raises(ValueError, match="writing to this folder"):
        G.cmp_delete(str(folder), ["base"])
    assert len(C.read_report(str(folder / "a.csv"))) == 4


def test_a_missing_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no such folder"):
        G.cmp_reports(str(tmp_path / "nope"))
