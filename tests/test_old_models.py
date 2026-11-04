"""Old models: a run moved into runs/old drops out of the hall of fame and
the run lists, stays loadable as `old/<name>`, and comes back on a resume."""
import json

import pytest

from ar2l.viz import agents as A
from ar2l.viz import trainer as TR


def _run(root, name, sub=()):
    d = root.joinpath('runs', *sub, name)
    d.mkdir(parents=True)
    (d / 'args.json').write_text(json.dumps({'name': name, 'algo': 'nominal',
                                             'iters': 10, 'seed': 3}))
    (d / 'log.jsonl').write_text(json.dumps({'it': 5, 'nom_util': 0.5}) + '\n')
    return d


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, 'ROOT', str(tmp_path))
    monkeypatch.setattr(A, 'ROOT', str(tmp_path))
    return tmp_path


def test_moved_out_and_back(root):
    _run(root, 'a')
    assert TR.to_old('a') == 'old/a'
    assert (root / 'runs' / 'old' / 'a' / 'args.json').exists()
    assert not (root / 'runs' / 'a').exists()
    assert TR.from_old('a') == 'a'
    assert (root / 'runs' / 'a' / 'args.json').exists()


def test_old_is_off_the_lists_and_the_hall_of_fame(root):
    _run(root, 'a')
    _run(root, 'b')
    TR.to_old('b')
    assert A.list_runs() == ['a']
    assert A.list_runs(old=True) == ['old/b']
    assert {r['name']: r['old'] for r in TR.runs()} == {'a': False, 'old/b': True}
    assert [r['name'] for g in TR.fame() for r in g['runs']] == ['a']


def test_an_old_model_loads_its_settings(root):
    _run(root, 'b', sub=('old',))
    assert TR.run_values('old/b')['values']['name'] == 'b'
    assert A.run_dir('old/b') == str(root / 'runs' / 'old' / 'b')


def test_a_name_taken_either_side_is_refused(root):
    _run(root, 'a')
    _run(root, 'a', sub=('old',))
    with pytest.raises(ValueError):
        TR.to_old('a')
    with pytest.raises(ValueError):
        TR.from_old('a')


def test_a_new_run_may_not_take_an_old_models_name(root):
    _run(root, 'b', sub=('old',))
    _, errs = TR.check({'name': 'b'})
    assert any('old model' in e for e in errs)
    _, errs = TR.check({'name': 'old'})
    assert any('runs/old' in e for e in errs)


def test_a_running_run_stays(root, monkeypatch):
    _run(root, 'a')

    class Live:
        pid = -1

        def poll(self):
            return None
    monkeypatch.setitem(TR._JOBS, 'a', {'proc': Live()})
    with pytest.raises(ValueError, match='training'):
        TR.to_old('a')
    assert (root / 'runs' / 'a').exists()


def test_the_dashboard_lists_old_models_flagged(root, monkeypatch):
    from ar2l.viz import dashboard as D
    monkeypatch.setattr(D, 'ROOT', str(root))
    _run(root, 'a')
    _run(root, 'b', sub=('old',))
    assert D.all_runs() == ['a', 'old/b']
    assert {m['run']: m['old'] for m in map(D.run_meta, D.all_runs())} == {
        'a': False, 'old/b': True}
    assert D.read_log('old/b') == [{'it': 5, 'nom_util': 0.5}]
    assert D.delete_run('old/b') == 'runs/.trash/b'
