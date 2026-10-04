"""The game's "New training" tab: start `ar2l.train` runs from the page.

The form is `ar2l.train`'s own argparser, read flag by flag, so every flag is
on it and a flag added there shows up here without touching this file.  The
defaults are read from `config.yaml` afresh each time the form loads (merged
under `--config` when one is given), so an edit to the file shows on reload.

A run is a child process of the game server, not of the page: closing the
browser leaves it training, and reopening the tab finds it again.  It goes
when the server goes, and it goes *saved*: Ctrl+C or closing the terminal
stops the server, which asks every run to stop, and `ar2l.train` answers a
stop by finishing its iteration, writing last.pt and exiting -- so `--resume`
carries on from there.  `PR_SET_PDEATHSIG` sends the same request if the
server is killed outright.  Its stdout goes to `runs/<name>/train.out`.

"Stop & save" is that same request; "Cancel" kills the run and moves its
folder to runs/.trash.  Runs this server did not start -- from the command
line, or from a server that has since gone -- are found by scanning /proc
and can be stopped and resumed the same way.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import importlib
import ctypes
import io
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

from .. import config as C
from .. import orders as O
from .. import train as T

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
#: never a field: argparse's own, and --config, which the page has as the
#: file the whole form is loaded from
SKIP = ('help', 'config')
#: the config section each flag's default comes from, where its name does not
#: say -- `add_cm_args` reads the eval section
SECTION = {'cell_cm': 'eval', 'pallet_cm': 'eval', 'box_scale': 'eval',
           'box_round': 'eval', 'seed': 'run'}
ORDER = ('run name', 'train', 'env', 'ppo', 'model', 'eval', 'run', 'other')

_CFG_LOCK = threading.RLock()
_JOBS: dict = {}                 # name -> job, in start order
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------- the form

def fresh_config(path=None):
    """config.yaml as it is on disk now, with AR2L_CONFIG and then `path`
    merged over it, the way `ar2l.train` itself would read them."""
    cfg = C._read(C.PATH)
    for over in (os.environ.get('AR2L_CONFIG'), path):
        if over:
            q = over if os.path.isabs(over) else os.path.join(ROOT, over)
            cfg = C._merge(cfg, C._read(q))
    return cfg


def reload_train():
    """Re-import `ar2l.train` (and the orders flags it borrows), so a flag
    or a help text edited there shows on the form without restarting."""
    global T
    with _CFG_LOCK:
        importlib.reload(O)
        T = importlib.reload(T)


def parser(cfg):
    """`ar2l.train`'s parser with `cfg` as its defaults.  It reads the shared
    CFG at build time, so that is swapped for the build and put straight back."""
    with _CFG_LOCK:
        saved = copy.deepcopy(C.CFG)
        C.CFG.clear(); C.CFG.update(cfg)
        try:
            return T.get_parser()
        finally:
            C.CFG.clear(); C.CFG.update(saved)


def _help(a, default):
    """The help text as `-h` prints it: %(default)s filled in, %% a percent."""
    try:
        return (a.help or '') % dict(vars(a), default=_text(a, default))
    except (KeyError, TypeError, ValueError):
        return a.help or ''


def _section(dest, cfg):
    if dest == 'name':
        return 'run name'
    if dest in SECTION:
        return SECTION[dest]
    return next((s for s in ORDER if dest in cfg.get(s, {})), 'other')


def checkpoints():
    """Every checkpoint under runs/, newest first: what --init can start from."""
    out = glob.glob(os.path.join(ROOT, 'runs', '*', '*.pt'))
    out.sort(key=lambda q: -os.path.getmtime(q))
    return [os.path.relpath(q, ROOT) for q in out]


def _text(a, v):
    """A default as the form's text box shows it."""
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return (' '.join if a.nargs else 'x'.join)(str(q) for q in v)
    return str(v)


def form(config_path=None):
    """Every flag of `ar2l.train`, with its default and how to edit it."""
    reload_train()
    cfg = fresh_config(config_path)
    fields = []
    for a in parser(cfg)._actions:
        if a.dest in SKIP:
            continue
        default = a.default
        # `main` resolves this one from the config when it is left unset
        if a.dest == 'min_support' and default is None:
            default = cfg['env']['min_support']
        if isinstance(a, argparse._StoreTrueAction):
            kind = 'flag'
        elif a.choices:
            kind = 'choice'
        elif a.type in (int, float):
            kind = a.type.__name__
        else:
            kind = 'text'
        fields.append({
            'dest': a.dest, 'flag': a.option_strings[-1], 'kind': kind,
            'section': _section(a.dest, cfg),
            'default': bool(default) if kind == 'flag' else _text(a, default),
            'choices': [str(c) for c in a.choices] if a.choices else None,
            'help': _help(a, default),
            'nargs': a.nargs, 'required': a.required,
            'suggest': checkpoints() if a.dest == 'init' else None})
    fields.sort(key=lambda f: ORDER.index(f['section']))
    return {'fields': fields, 'sections': list(ORDER),
            'config': config_path or os.environ.get('AR2L_CONFIG') or 'config.yaml'}


def runs():
    """Every run with a recorded configuration, newest first, and how far it
    got -- what the form can load to resume or copy."""
    out = []
    for a in glob.glob(os.path.join(ROOT, 'runs', '*', 'args.json')):
        d = os.path.dirname(a)
        try:
            args = json.load(open(a))
        except (OSError, ValueError):
            continue
        rows, _ = _rows(os.path.join(d, 'log.jsonl'), 0)
        with _JOBS_LOCK:
            j = _JOBS.get(os.path.basename(d))
        out.append({'name': os.path.basename(d), 'algo': args.get('algo'),
                    'it': rows[-1]['it'] if rows else 0, 'iters': args.get('iters'),
                    'last_pt': os.path.exists(os.path.join(d, 'last.pt')),
                    'running': bool(j and j['proc'].poll() is None),
                    'mtime': os.path.getmtime(a)})
    out.sort(key=lambda r: -r['mtime'])
    return out


def run_values(name, config_path=None):
    """A run's args.json as the form's values: every field the parser has,
    written the way the form writes it.  A flag added since the run was
    trained keeps the form's default (it is left out here)."""
    d = os.path.join(ROOT, 'runs', os.path.basename(str(name)))
    try:
        args = json.load(open(os.path.join(d, 'args.json')))
    except (OSError, ValueError) as e:
        raise ValueError(f'no recorded configuration for {name!r}: {e}')
    out = {}
    for a in parser(fresh_config(args.get('config') or config_path))._actions:
        if a.dest in SKIP or a.dest not in args:
            continue
        v = args[a.dest]
        out[a.dest] = (bool(v) if isinstance(a, argparse._StoreTrueAction)
                       else _text(a, v))
    return {'values': out, 'config': args.get('config') or ''}


def argv_of(values, config_path=None):
    """The command line for the form's values: every field that is set, so the
    run is exactly what the form showed even if config.yaml changes later."""
    p = parser(fresh_config(config_path))
    argv = ['--config', config_path] if config_path else []
    for a in p._actions:
        if a.dest in SKIP or a.dest not in values:
            continue
        v = values[a.dest]
        if isinstance(a, argparse._StoreTrueAction):
            if v:
                argv.append(a.option_strings[-1])
            continue
        v = str(v).strip()
        if v == '':
            continue
        argv.append(a.option_strings[-1])
        argv += v.replace(',', ' ').split() if a.nargs else [v]
    return argv


def check(values, config_path=None):
    """The argv and its errors, by parsing it with the real parser."""
    errs = []
    name = str(values.get('name') or '').strip()
    if not name:
        errs.append('a run needs a name')
    elif not re.fullmatch(r'[A-Za-z0-9_.\-]+', name) or name.startswith('.'):
        errs.append('run name: letters, digits, _ . - only')
    try:
        argv = argv_of(values, config_path)
        p = parser(fresh_config(config_path))
    except Exception as e:                        # a bad --config file
        return [], errs + [f'config: {e}']
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            p.parse_args(argv)
    except SystemExit:
        last = buf.getvalue().strip().splitlines()
        msg = last[-1].split('error: ', 1)[-1] if last else 'bad arguments'
        if not (msg.endswith('required: --name') and not name):   # said above
            errs.append(msg)
    if name and not errs:
        with _JOBS_LOCK:
            j = _JOBS.get(name)
            if j and j['proc'].poll() is None:
                errs.append(f'{name} is already training')
        if (os.path.exists(os.path.join(ROOT, 'runs', name, 'log.jsonl'))
                and not values.get('resume')):
            errs.append(f'runs/{name} already exists: tick --resume to carry '
                        f'it on, or pick another name')
    return argv, errs


# ---------------------------------------------------------------- the runs

def _die_with_parent():
    """In the child, before exec: SIGTERM when the parent thread exits."""
    try:
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


# PR_SET_PDEATHSIG fires when the *thread* that forked exits, and the server
# answers each request on a thread of its own that ends with the request.  So
# every run is forked from this one thread, which lives as long as the server.
_SPAWN: queue.Queue = queue.Queue()


def _spawner():
    while True:
        argv, out, box, done = _SPAWN.get()
        try:
            box['proc'] = subprocess.Popen(
                [sys.executable, '-u', '-m', 'ar2l.train', *argv], cwd=ROOT,
                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                # its own session: a Ctrl+C in the server's terminal reaches
                # the server only, which then asks each run to save -- a
                # second Ctrl+C can then not kill a run halfway through that
                start_new_session=True,
                preexec_fn=_die_with_parent if sys.platform.startswith('linux') else None)
        except Exception as e:
            box['error'] = e
        done.set()


threading.Thread(target=_spawner, name='train-spawner', daemon=True).start()


def start(values, config_path=None):
    argv, errs = check(values, config_path)
    if errs:
        raise ValueError('\n'.join(errs))
    name = values['name'].strip()
    # `auto` draws no bar into a file, and the bar is the only thing that
    # moves every iteration: the page reads the iteration and the speed off
    # it long before the first `--log_every` line
    if '--progress' not in argv:
        argv += ['--progress', 'on']
    elif argv[argv.index('--progress') + 1] == 'auto':
        argv[argv.index('--progress') + 1] = 'on'
    d = os.path.join(ROOT, 'runs', name)
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, 'train.out')
    out = open(log, 'ab')
    out.write(f'\n$ python -m ar2l.train {shlex.join(argv)}\n'.encode())
    out.flush()
    box, done = {}, threading.Event()
    _SPAWN.put((argv, out, box, done))
    done.wait()
    out.close()                                   # the child holds its own copy
    if 'error' in box:
        raise ValueError(f'could not start: {box["error"]}')
    a = parser(fresh_config(config_path)).parse_args(argv)
    jsonl = os.path.join(d, 'log.jsonl')
    with _JOBS_LOCK:
        _JOBS.pop(name, None)
        _JOBS[name] = {'name': name, 'proc': box['proc'], 'argv': argv,
                       'log': log, 'started': time.time(), 'ended': None,
                       'stop': None, 'adopted': False, 'iters': a.iters, 'algo': a.algo,
                       'device': a.device, 'eval_every': a.eval_every,
                       'save_every': a.save_every, 'log_every': a.log_every,
                       # rows before this offset were written by an earlier
                       # process (a --resume); the speed is this one's alone
                       'offset': os.path.getsize(jsonl) if os.path.exists(jsonl) else 0}
    return name


class Pid:
    """A run this server did not start, behind the bits of Popen used here."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        if self.returncode is None:
            try:
                with open(f'/proc/{self.pid}/stat') as f:
                    if f.read().rsplit(')', 1)[1].split()[0] == 'Z':
                        raise OSError
            except OSError:
                self.returncode = 0           # its exit status is not ours to read
        return self.returncode

    def send_signal(self, sig):
        try:
            os.kill(self.pid, sig)
        except ProcessLookupError:
            pass

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)

    def wait(self, timeout=None):
        t = time.time()
        while self.poll() is None:
            if timeout is not None and time.time() - t > timeout:
                raise subprocess.TimeoutExpired('ar2l.train', timeout)
            time.sleep(0.2)
        return self.returncode


_SCAN = {'t': 0.0}


def adopt():
    """Every `ar2l.train` running from this project that this server does not
    know of -- started on the command line, or by a server that has gone --
    taken into the list, at most every 5 s."""
    if time.time() - _SCAN['t'] < 5 or not os.path.isdir('/proc'):
        return
    _SCAN['t'] = time.time()
    with _JOBS_LOCK:
        known = {j['proc'].pid for j in _JOBS.values() if j['proc'].poll() is None}
    for pid in (int(q) for q in os.listdir('/proc') if q.isdigit()):
        if pid in known or pid == os.getpid():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().decode(errors='replace').split('\0')
            cwd = os.readlink(f'/proc/{pid}/cwd')
            started = os.stat(f'/proc/{pid}').st_mtime
        except OSError:
            continue
        if 'ar2l.train' not in cmd or cwd != ROOT:
            continue
        argv = [q for q in cmd[cmd.index('ar2l.train') + 1:] if q]
        try:
            cp = _get(argv, '--config')
            a = parser(fresh_config(cp)).parse_args(argv)
        except (SystemExit, Exception):
            continue
        d = os.path.join(ROOT, 'runs', a.name)
        jsonl = os.path.join(d, 'log.jsonl')
        out = os.path.join(d, 'train.out')
        with _JOBS_LOCK:
            j = _JOBS.get(a.name)
            if j and j['proc'].poll() is None:
                continue
            _JOBS.pop(a.name, None)
            _JOBS[a.name] = {
                'name': a.name, 'proc': Pid(pid), 'argv': argv,
                # a command-line run prints to its own terminal, not to a file
                'log': out, 'started': started, 'ended': None, 'stop': None,
                'adopted': True, 'iters': a.iters, 'algo': a.algo,
                'device': a.device, 'eval_every': a.eval_every,
                'save_every': a.save_every, 'log_every': a.log_every,
                # where this process began is unknown; the speed is then read
                # off the rows since the clock last restarted, as for a resume
                'offset': 0,
                # a process older than train.py's stop-and-save only dies
                'graceful': started > os.path.getmtime(T.__file__)}


def _get(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def stop(name, mode='save'):
    """`save`: ask the run to stop -- it finishes its iteration, writes
    last.pt and exits.  `cancel`: kill it, and move its folder to runs/.trash."""
    with _JOBS_LOCK:
        j = _JOBS.get(name)
    if not j:
        return False
    if mode == 'cancel':
        j['stop'] = 'cancel'
        if j['proc'].poll() is None:
            j['proc'].kill()
            try:
                j['proc'].wait(10)
            except subprocess.TimeoutExpired:
                pass
        with _JOBS_LOCK:
            _JOBS.pop(name, None)
        return {'moved_to': trash(name)}
    if j['proc'].poll() is not None:
        return False
    j['stop'] = 'save'
    j['proc'].terminate()
    return True


def trash(name):
    """runs/<name> moved into runs/.trash, as the dashboard deletes a run:
    `rm -rf runs/.trash` empties it for good."""
    name = os.path.basename(str(name))
    src = os.path.join(ROOT, 'runs', name)
    if not name or not os.path.isdir(src):
        return None
    dest = os.path.join(ROOT, 'runs', '.trash', name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        dest += time.strftime('_%Y%m%d-%H%M%S')
    shutil.move(src, dest)
    return os.path.relpath(dest, ROOT)


def dismiss(name):
    """Take a finished run's card off the page; the run itself stays."""
    with _JOBS_LOCK:
        j = _JOBS.get(name)
        if j and j['proc'].poll() is not None:
            _JOBS.pop(name)
            return True
    return False


def resume(name, config_path=None):
    """Start `name` again from its last.pt, with the settings it recorded."""
    rv = run_values(name, config_path)
    return start(dict(rv['values'], resume=True), rv['config'] or None)


def stop_all():
    """The server is going: ask every run of ours to save and wait for them.
    A second Ctrl+C stops the waiting, not the runs -- they are in their own
    sessions, and finish saving on their own."""
    with _JOBS_LOCK:
        jobs = [j for j in _JOBS.values()
                if j['proc'].poll() is None and not j['adopted']]
    if not jobs:
        return
    for j in jobs:
        j['stop'] = 'save'
        j['proc'].terminate()
    print(f"stopping {len(jobs)} training run(s): each saves last.pt after its "
          f"current iteration (Ctrl+C again to stop waiting; they still save)",
          flush=True)
    try:
        for j in jobs:
            j['proc'].wait()
            print(f"  {j['name']}: saved -- resume it from the New training tab",
                  flush=True)
    except KeyboardInterrupt:
        pass


def _tail(path, n=40, cap=64 * 1024):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - cap))
            text = f.read().decode(errors='replace')
    except OSError:
        return ''
    # tqdm's carriage returns: keep what each line ended up showing
    lines = [ln.rstrip('\r').rsplit('\r', 1)[-1] for ln in text.split('\n')]
    return '\n'.join(lines[-n:])


#: the held-out number each algorithm is judged on, as `train._held` reads it;
#: an attacker is best when the packer it faces does worst
HELD = {'attack': 'att_util', 'select': 'sel_util'}


def _rows(path, offset):
    """log.jsonl as (every row, the rows this process wrote)."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return [], []
    def parse(b):
        out = []
        for ln in b.decode(errors='replace').splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass                          # a line still being written
        return out
    return parse(data), parse(data[offset:])


def _thin(pts, n=400):
    """At most `n` points for the chart, the last one always kept."""
    if len(pts) <= n:
        return pts
    step = len(pts) / n
    return [pts[int(i * step)] for i in range(n)] + [pts[-1]]


def _ckpt(d, f):
    p = os.path.join(d, f)
    try:
        st = os.stat(p)
        return {'age': time.time() - st.st_mtime, 'mb': st.st_size / 2**20}
    except OSError:
        return None


_GPU = {'t': 0.0, 'mem': {}}


def _gpu_mem():
    """MiB of GPU memory per pid, from nvidia-smi, at most every 5 s."""
    if time.time() - _GPU['t'] > 5:
        _GPU['t'], _GPU['mem'] = time.time(), {}
        try:
            q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=3)
            for ln in q.stdout.splitlines():
                pid, mem = (int(v) for v in ln.split(','))
                _GPU['mem'][pid] = mem
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return _GPU['mem']


#: tqdm's bar: `  37/1000 [21:05<9:08:41, 34.20s/it, util=...]`
BAR = re.compile(r'(\d+)/(\d+) \[[\d:]+<[\d:?]+,\s*([\d.]+)(s/it|it/s)')


def _bar(path):
    """(iterations done, seconds per iteration) from the last progress bar
    drawn into the run's output, or None."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            text = f.read().decode(errors='replace')
    except OSError:
        return None
    # only the bar of the latest process: `start` heads each one's output
    text = text.rsplit('\n$ python -m ar2l.train', 1)[-1]
    m = None
    for m in BAR.finditer(text):
        pass
    if m is None:
        return None
    r = float(m.group(3))
    tail = text[m.end():].split('\n', 1)[0].split('\r', 1)[0]
    u = re.search(r'util=([\d.]+)%', tail)
    k = re.search(r'items=([\d.]+)', tail)
    return (int(m.group(1)), (r if m.group(4) == 's/it' else 1 / r if r else None),
            float(u.group(1)) / 100 if u else None, float(k.group(1)) if k else None)


def stats(j):
    """Everything the run's card shows, from its log and checkpoint files."""
    d = os.path.join(ROOT, 'runs', j['name'])
    jsonl = os.path.join(d, 'log.jsonl')
    if os.path.exists(jsonl) and os.path.getsize(jsonl) < j['offset']:
        # a resume cut the rows past last.pt; tell this process's rows apart
        # by the restart of `t` instead, as for an adopted run
        j['offset'] = 0
    rows, mine = _rows(jsonl, j['offset'])
    bar = _bar(j['log'])
    if bar and j['proc'].poll() is None:
        # the bar is this process's own; log rows past it are from a run that
        # died after its last save, and this one is redoing those iterations
        rows = [r for r in rows if r.get('it', 0) <= bar[0]]
        mine = [r for r in mine if r.get('it', 0) <= bar[0]]
    key = HELD.get(j['algo'], 'nom_util')
    sign = -1 if j['algo'] == 'attack' else 1
    last = rows[-1] if rows else {}
    it = last.get('it', 0)

    # evals: the held-out number only changes when one runs
    evals, prev = [], None
    for r in rows:
        v = r.get(key)
        if v is not None and v != prev:
            evals.append([r['it'], v])
        prev = v
    best = max(evals, key=lambda e: sign * e[1]) if evals else None

    # best.pt is written at a save point whose score beats every earlier one
    best_ck, hi, k = None, -1e9, 0
    saves = list(range(j['save_every'], it + 1, j['save_every']))
    if (it == j['iters'] or last.get('stopped')) and it not in saves:
        saves.append(it)            # the last save, or a stop's
    for sv in saves:
        while k < len(evals) and evals[k][0] <= sv:
            k += 1
        sc = sign * evals[k - 1][1] if k else (-1.0 if sign < 0 else 0.0)
        if sc > hi:
            hi, best_ck = sc, sv

    # speed: this process's rows only, overall and over the last few.  `t`
    # restarts with every process, so a drop marks where a later one began
    cut = max((i for i in range(1, len(mine)) if mine[i]['t'] < mine[i - 1]['t']),
              default=0)
    mine = mine[cut:]
    # rows from before this process mean it carried on from an earlier one
    resumed_at = mine[0]['it'] if mine and len(rows) > len(mine) else None
    rate = recent = None
    if len(mine) >= 2:
        rate = (mine[-1]['t'] - mine[0]['t']) / max(1, mine[-1]['it'] - mine[0]['it'])
        w = mine[-6:]
        recent = (w[-1]['t'] - w[0]['t']) / max(1, w[-1]['it'] - w[0]['it'])
    elif mine:
        before = rows[-len(mine) - 1]['it'] if len(rows) > len(mine) else 0
        rate = recent = mine[-1]['t'] / max(1, mine[-1]['it'] - before)
    # the bar is ahead of the log by up to --log_every iterations
    if bar and bar[0] > it:
        it = bar[0]
        recent = bar[1] or recent
        rate = rate or bar[1]
    left = j['iters'] - it
    eta = rate * left if rate and j['proc'].poll() is None else None
    utils = [r['util'] for r in rows[-5:] if 'util' in r]
    util, items = last.get('util'), last.get('items')
    if bar and bar[0] >= it and bar[2] is not None:
        util, items = bar[2], bar[3]
    nxt = lambda every: min(j['iters'], (it // every + 1) * every) if it < j['iters'] else None
    return {
        'it': it, 'iters': j['iters'], 'resumed_at': resumed_at,
        'log_every': j['log_every'], 'live': bool(bar),
        'util': util, 'items': items,
        'util_avg': sum(utils) / len(utils) if utils else None,
        'held_key': key, 'held': evals[-1][1] if evals else None,
        'best': best, 'n_evals': len(evals), 'lower_better': sign < 0,
        'next_eval': nxt(j['eval_every']), 'next_save': nxt(j['save_every']),
        'rate': rate, 'recent': recent, 'eta': eta,
        'since_log': time.time() - os.path.getmtime(os.path.join(d, 'log.jsonl'))
                     if rows else None,
        'best_pt': dict(_ckpt(d, 'best.pt') or {}, it=best_ck) if _ckpt(d, 'best.pt') else None,
        'last_pt': dict(_ckpt(d, 'last.pt'), it=saves[-1] if saves else None)
                   if _ckpt(d, 'last.pt') else None,
        'curve': _thin([[r['it'], r['util']] for r in rows if 'util' in r]),
        'evals': evals,
        'gpu_mb': _gpu_mem().get(j['proc'].pid) if j['proc'].poll() is None else None}


def _state(j, rc, st):
    if rc is None:
        return 'stopping' if j['stop'] == 'save' else 'running'
    if j['stop'] == 'save':
        # a run older than stop-and-save dies on the request instead
        return 'saved' if rc == 0 and j.get('graceful', True) else 'stopped'
    if st['it'] >= st['iters']:
        return 'done'
    if j['adopted']:
        return 'stopped'               # it ended elsewhere; its exit is not ours
    return 'stopped' if rc < 0 else 'failed'


def status():
    adopt()
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    out = []
    for j in reversed(jobs):                      # newest first
        rc = j['proc'].poll()
        if rc is not None and j['ended'] is None:
            j['ended'] = time.time()
        st = stats(j)
        state = _state(j, rc, st)
        last_pt = os.path.exists(os.path.join(ROOT, 'runs', j['name'], 'last.pt'))
        out.append(dict(st,
            name=j['name'], pid=j['proc'].pid, algo=j['algo'], device=j['device'],
            state=state, adopted=j['adopted'], graceful=j.get('graceful', True),
            resumable=rc is not None and last_pt and st['it'] < st['iters'],
            rc=rc, cmd='python -m ar2l.train ' + shlex.join(j['argv']),
            started=j['started'], ended=j['ended'],
            elapsed=(j['ended'] or time.time()) - j['started'],
            out=_tail(j['log'])))
    return out
