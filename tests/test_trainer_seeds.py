"""The New training page's `n_seeds`: one Start, one run per seed."""
import json

import pytest

from ar2l.viz import trainer as TR


def test_one_seed_is_the_run_as_filled_in():
    v = {'name': 'exp', 'seed': '5', 'n_seeds': '1'}
    assert TR.seed_runs(v) == [v]


def test_more_seeds_name_every_run_by_its_seed():
    v = {'name': 'exp', 'seed': '5', 'n_seeds': '3', 'lr': '0.0003'}
    got = TR.seed_runs(v)
    assert [(g['name'], g['seed']) for g in got] == [
        ('exp_seed_5', '5'), ('exp_seed_6', '6'), ('exp_seed_7', '7')]
    assert all(g['lr'] == '0.0003' and g['n_seeds'] == '1' for g in got)


def test_a_blank_seed_counts_from_the_config_default():
    got = TR.seed_runs({'name': 'exp', 'seed': '', 'n_seeds': '2'})
    d = TR.parser(TR.fresh_config()).get_default('seed')
    assert [g['name'] for g in got] == [f'exp_seed_{d}', f'exp_seed_{d + 1}']


def test_no_name_is_reported_once_not_renamed():
    _, errs = TR.check({'name': '', 'n_seeds': '3'})
    assert errs == ['a run needs a name']


@pytest.mark.parametrize('n', ['0', '17', '2.5', 'x'])
def test_a_bad_n_seeds_is_an_error(n):
    _, errs = TR.check({'name': 'exp', 'n_seeds': n})
    assert errs and errs[0].startswith('n_seeds')


def test_a_taken_seed_name_stops_the_whole_group(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, 'ROOT', str(tmp_path))
    taken = tmp_path / 'runs' / 'exp_seed_1'
    taken.mkdir(parents=True)
    (taken / 'log.jsonl').write_text('')
    _, errs = TR.check({'name': 'exp', 'seed': '0', 'n_seeds': '3'})
    assert any('exp_seed_1' in e for e in errs)
    with pytest.raises(ValueError):
        TR.start({'name': 'exp', 'seed': '0', 'n_seeds': '3'})
    assert not (tmp_path / 'runs' / 'exp_seed_0').exists()


def test_a_recorded_run_loads_as_one_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, 'ROOT', str(tmp_path))
    d = tmp_path / 'runs' / 'old'
    d.mkdir(parents=True)
    (d / 'args.json').write_text(json.dumps({'name': 'old', 'seed': 0}))
    assert TR.run_values('old')['values']['n_seeds'] == '1'


def test_the_form_has_n_seeds_right_after_seed():
    dests = [f['dest'] for f in TR.form()['fields']]
    assert dests[dests.index('seed') + 1] == 'n_seeds'
