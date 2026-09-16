"""Hyperparameter screening: short training runs, one factor changed at a time.

    python3 sweep.py --list                 # show the plan
    python3 sweep.py --run --concurrent 3   # execute it
    python3 sweep.py --report               # rank what has finished

Every config is trained on the same data budget and then scored on the *validation*
split with one fixed search setting, so the ranking reflects the network each config
produced and not the evaluator it happened to be trained with.  The test split is
never touched here - it stays clean for the headline numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'results', 'sweep')

# the configuration the reported runs used - every arm is this plus one change
BASE = dict(data='cut2', iters=30, games=256, sims=100, c_puct=4.0,
            dir_eps=0.05, dir_alpha=0.3, scale=1.0, temp=1.0, temp_moves=0,
            lr=1e-3, lr_min=1e-4, wd=1e-4, batch=256, updates=120,
            cap=200_000, per_alpha=0.6, augment=1, filter=1, value=1,
            eval_every=10, eval_n=100, eval_split='val', seed=0)

# name -> the single change, and a one-line statement of what it asks
ARMS = [
    # ordered so that each wave of four answers a coherent question, and the
    # controls land first - without the noise floor nothing after it is readable
    ('base',       {},                          'control'),
    ('base_seed1', dict(seed=1),                'control, second seed - the noise floor'),
    ('scale0.5',   dict(scale=0.5),             'sharper value target and MCTS leaf value'),
    ('scale2',     dict(scale=2.0),             'softer value target and MCTS leaf value'),

    ('upd60',      dict(updates=60),            'half the gradient steps per iteration'),
    ('upd240',     dict(updates=240),           'double the gradient steps per iteration'),
    ('lr5e-4',     dict(lr=5e-4, lr_min=5e-5),  'half the learning rate'),
    ('lr2e-3',     dict(lr=2e-3, lr_min=2e-4),  'double the learning rate'),

    ('sims50',     dict(sims=50),               'half the search per move during self-play'),
    ('sims200',    dict(sims=200),              'double the search per move during self-play'),
    ('cpuct2',     dict(c_puct=2.0),            'less exploration inside the tree'),
    ('cpuct6',     dict(c_puct=6.0),            'more exploration inside the tree'),

    ('dir0',       dict(dir_eps=0.0),           'no Dirichlet noise at the root'),
    ('dir0.15',    dict(dir_eps=0.15),          'three times the Dirichlet noise'),
    ('nofilter',   dict(filter=0),              'keep episodes that lost to the baseline'),
    ('noval',      dict(value=0),               'drop the value head and its loss term'),

    ('cap50k',     dict(cap=50_000),            'quarter the replay buffer'),
    ('per_unif',   dict(per_alpha=0.0),         'uniform replay instead of prioritised'),
    ('noaug',      dict(augment=0),             'no 8-fold symmetry augmentation'),
    ('batch512',   dict(batch=512),             'double the minibatch'),

    # follow-ups, added once wave 2 showed `updates` rising monotonically
    ('upd480',     dict(updates=480),           'four times the gradient steps - still rising?'),
    ('upd240_lr5e-4', dict(updates=240, lr=5e-4, lr_min=5e-5),
                                                'do the two best single changes compose?'),

    # follow-ups from wave 3: c_puct 2.0 beat 4.0 during *training* (my earlier
    # sweep tuned c_puct at inference on a frozen net - a different quantity),
    # and sims=50 was only half the wall clock, so it deserves an equal-compute
    # arm rather than the equal-episode one it lost on
    ('cpuct1',     dict(c_puct=1.0),            'is even less tree exploration better still?'),
    ('cpuct3',     dict(c_puct=3.0),            'bracket the new c_puct optimum'),
    ('sims50_2x',  dict(sims=50, iters=60),     'equal wall clock: twice the episodes, half the search'),
    ('best_combo', dict(updates=240, c_puct=2.0), 'the two winning changes together'),

    # Where does the buffer stop paying?  The paper stops at b=3 for a physical
    # reason (robot arm reach), not an empirical one, and its numbers are still
    # climbing there.  Same data budget and same 100 sims throughout, so this is
    # the realistic "just turn b up" curve; policy-only separates the network's
    # use of the extra choice from the search dilution a bigger action space causes.
    ('b2',         dict(b=2),                   'buffer of two'),
    ('b3',         dict(b=3),                   'buffer of three - the paper stops here'),
    ('b5',         dict(b=5),                   'buffer of five - past the paper'),
    ('b7',         dict(b=7),                   'buffer of seven - looking for the turn'),
]

EVAL = dict(sims=100, c_puct=4.0, reps=2, n=200, split='val', seed=0)


def cfg_for(name):
    for n, ch, why in ARMS:
        if n == name:
            return {**BASE, **ch}, ch, why
    raise KeyError(name)


def run_name(name):
    return 'sw_' + name.replace('.', 'p').replace('-', '_')


def train_cmd(name):
    cfg, _, _ = cfg_for(name)
    cmd = [sys.executable, '-m', 'bpp.train', '--name', run_name(name)]
    for k, v in cfg.items():
        cmd += [f'--{k}', str(v)]
    return cmd


def already_done(name):
    return os.path.exists(os.path.join(OUT, f'{name}.json'))


def evaluate(name):
    """Score the finished checkpoint on the validation split, one fixed setting."""
    from bpp.evaluate import run_eval
    ck = os.path.join(ROOT, 'runs', run_name(name), 'last.pt')
    if not os.path.exists(ck):
        return None
    cfg, change, why = cfg_for(name)
    r = run_eval(ck, cfg['data'], ROOT, n=EVAL['n'], sims=EVAL['sims'],
                 c_puct=EVAL['c_puct'], reps=EVAL['reps'], seed=EVAL['seed'],
                 split=EVAL['split'])
    log = [json.loads(l) for l in
           open(os.path.join(ROOT, 'runs', run_name(name), 'log.jsonl')) if l.strip()]
    r.update(arm=name, change=change, question=why,
             sp_util_last3=round(sum(x['sp_util'] for x in log[-3:]) / max(1, len(log[-3:])), 2),
             kept_last3=round(sum(x['kept'] for x in log[-3:]) / max(1, len(log[-3:])), 1),
             mins=log[-1]['mins'] if log else None, iters_done=len(log))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{name}.json'), 'w') as f:
        json.dump(r, f, indent=2)
    return r


def execute(names, concurrent):
    """Train in waves of `concurrent`, then evaluate each wave (one GPU, so the
    evaluations run after the training processes have released their memory)."""
    todo = [n for n in names if not already_done(n)]
    print(f'{len(todo)} arms to run, {concurrent} at a time\n', flush=True)
    waves = [todo[i:i + concurrent] for i in range(0, len(todo), concurrent)]
    for w, wave in enumerate(waves):
        print(f'--- wave {w+1}: {", ".join(wave)}', flush=True)
        t0 = time.time()
        procs = []
        for n in wave:
            lf = open(os.path.join(ROOT, 'runs', f'{run_name(n)}.out'), 'w')
            procs.append((n, subprocess.Popen(train_cmd(n), cwd=ROOT,
                                              stdout=lf, stderr=subprocess.STDOUT), lf))
        for n, p, lf in procs:
            rc = p.wait()
            lf.close()
            print(f'    {n}: train rc={rc}', flush=True)
        for n, _, _ in procs:
            r = evaluate(n)
            if r:
                print(f"    {n}: val {r['mcts_util']:.2f}% (policy {r['greedy_util']:.1f}%) "
                      f"sp {r['sp_util_last3']:.1f}%  {r['mins']:.0f} min", flush=True)
        print(f'    wave took {(time.time()-t0)/60:.0f} min\n', flush=True)


def report():
    rows = []
    for n, _, _ in ARMS:
        f = os.path.join(OUT, f'{n}.json')
        if os.path.exists(f):
            rows.append(json.load(open(f)))
    if not rows:
        print('nothing finished yet')
        return
    by = {r['arm']: r for r in rows}
    ref = by.get('base')
    sigma = None
    if ref and 'base_seed1' in by:
        sigma = abs(by['base_seed1']['mcts_util'] - ref['mcts_util'])
    rows.sort(key=lambda r: -r['mcts_util'])
    w = max(len(r['arm']) for r in rows)
    print(f"{'arm':<{w}}  {'val MCTS':>9}  {'vs base':>8}  {'policy':>7}  "
          f"{'self-play':>9}  {'kept':>5}  {'min':>4}   question")
    for r in rows:
        d = r['mcts_util'] - ref['mcts_util'] if ref else 0.0
        flag = ''
        if sigma is not None and r['arm'] not in ('base', 'base_seed1'):
            flag = '  *' if abs(d) > 2 * sigma else '   '
        print(f"{r['arm']:<{w}}  {r['mcts_util']:>8.2f}%  {d:>+7.2f}{flag}  "
              f"{r['greedy_util']:>6.1f}%  {r['sp_util_last3']:>8.1f}%  "
              f"{r['kept_last3']:>5.0f}  {r['mins']:>4.0f}   {r['question']}")
    if sigma is not None:
        print(f"\nseed-to-seed spread of the control: {sigma:.2f} points.  "
              f"'*' marks arms beyond 2x that - everything else is noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--concurrent', type=int, default=3)
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    names = a.only or [n for n, _, _ in ARMS]
    if a.list:
        print(f'base: {BASE}\n')
        for n, ch, why in ARMS:
            print(f'  {n:<12} {str(ch or "-"):<34} {why}')
        print(f'\neval: {EVAL}')
    if a.run:
        execute(names, a.concurrent)
        report()
    if a.report:
        report()


if __name__ == '__main__':
    main()
