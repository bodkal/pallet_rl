"""The "Online game" tab: watch the ROS pallet RL server decide, step by step.

The server (`pallet_rl_ros_server`, node `pallet_rl_server`) writes every
request it answers to one JSON file (`server.snapshot_file` in its
`server.yaml`, `/tmp/pallet_rl_server/last_step.json` by default): the pallet
as it rebuilt it, in grid cells, the conveyor, the picker's probability for
each box in reach, every legal candidate point with the packer's probability,
and what it chose.  See `pallet_rl_server/snapshot.py`.

Here a thread watches that file and keeps every step it has seen, so the page
can follow the live one or go back through the ones before.  It starts the
first time the page asks, and reads nothing but that file -- the server stays
stateless and knows nothing of this page.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

import numpy as np

from .. import pack_collision as PC

DEFAULT_FILE = '/tmp/pallet_rl_server/last_step.json'
#: steps kept in memory; the oldest drop off
KEEP = 2000
#: how often the file is checked, seconds
EVERY = 0.2

_LOCK = threading.Lock()
#: the arm checker holds one height map at a time
_ARM_LOCK = threading.Lock()
_S = {'file': DEFAULT_FILE, 'steps': deque(maxlen=KEEP), 'next': 0,
      'sig': None, 'error': None, 'thread': None, 'arm': {}}


def configure(path):
    """Set the snapshot file to watch (the game's `--online_file`)."""
    _S['file'] = os.path.abspath(os.path.expanduser(path))


def _read():
    """Read the file if it changed since the last look; keep it as a new
    step when it is a new request."""
    path = _S['file']
    try:
        st = os.stat(path)
    except FileNotFoundError:
        _S['error'] = None
        return
    except OSError as e:
        _S['error'] = str(e)
        return
    sig = (st.st_mtime_ns, st.st_size, st.st_ino)
    if sig == _S['sig']:
        return
    try:
        with open(path) as f:
            snap = json.load(f)
    except (OSError, ValueError) as e:
        # caught between the server's write and rename is impossible (it
        # renames), but a hand-copied file may be half there: try again
        _S['error'] = f'{type(e).__name__}: {e}'
        return
    _S['sig'], _S['error'] = sig, None
    with _LOCK:
        steps = _S['steps']
        last = steps[-1] if steps else None
        if last and (last['seq'], last['time']) == (snap.get('seq'), snap.get('time')):
            return
        snap['id'] = _S['next']
        _S['next'] += 1
        steps.append(snap)


def _watch():
    while True:
        try:
            _read()
        except Exception as e:          # never let the watcher die
            _S['error'] = f'{type(e).__name__}: {e}'
        time.sleep(EVERY)


def start():
    """Start the watcher, once."""
    with _LOCK:
        if _S['thread'] is None:
            t = threading.Thread(target=_watch, name='online-watch', daemon=True)
            _S['thread'] = t
            t.start()
    _read()


def _meta(s):
    v = s.get('view') or {}
    return {'id': s['id'], 'seq': s.get('seq'), 'time': s.get('time'),
            'pallet_id': s.get('pallet_id', ''), 'status': s.get('status'),
            'n_placed': len(v.get('placed') or [])}


def status(have=None):
    """The steps kept (one line each) and, unless the page already has it
    (`have`, a step id), the latest step in full."""
    start()
    with _LOCK:
        steps = list(_S['steps'])
    last = steps[-1] if steps else None
    return {'file': _S['file'], 'exists': os.path.exists(_S['file']),
            'error': _S['error'], 'steps': [_meta(s) for s in steps],
            'last': last if last and last['id'] != have else None}


def step(i):
    """One kept step, in full."""
    with _LOCK:
        for s in _S['steps']:
            if s['id'] == i:
                return s
    raise ValueError(f'step {i} is no longer kept')


def clear():
    """Forget every kept step (the file itself stays)."""
    with _LOCK:
        _S['steps'].clear()
    return {'ok': True}


def _session(robot, cell_cm):
    """A session dict as `game.arm_pose` takes it, for the server's robot on
    its grid; the arm checker is built once per robot and cell size."""
    key = json.dumps([robot, cell_cm], sort_keys=True)
    st = _S['arm'].get(key)
    if st is None:
        chk, base = PC.checker_from_config(robot, float(cell_cm) / 100.0)
        st = {'p': {'base_x_moves': robot.get('base_x_moves') or [0.0]},
              'robot': robot, 'arm': chk, 'base_from_box': base}
        _S['arm'][key] = st
    return st


def arm(i, moves=None, at=None):
    """The UR20 placing step `i`'s chosen box, against the pallet as the
    server saw it -- the same pose and capsules the Play tab draws.  `at`
    ([x, y, z, l, w, h] cells): place that box there instead (a point picked
    in the view, or another box in reach where the packer would put it).
    `moves`: the base x moves to try, in order (the robot's own by default)."""
    from . import game as G
    s = step(i)
    v = s.get('view') or {}
    pc = [int(c) for c in at] if at else v.get('place_cells')
    if not pc or len(pc) != 6:
        return {'error': 'nothing was placed at this step'}
    if not s.get('robot'):
        return {'error': 'the snapshot has no robot settings'}
    S = v['bin']
    hm = np.zeros((int(S[0]), int(S[1])), int)
    for x, y, z, l, w, h, _ in v['placed']:
        blk = hm[x:x + l, y:y + w]
        np.maximum(blk, z + h, out=blk)
    x, y, z, l, w, h = pc
    with _ARM_LOCK:
        st = _session(s['robot'], v['cell_cm'])
        if moves is not None:
            moves = [float(m) for m in moves] or None
        return G.arm_pose(st, hm, [l, w, h], [x, y, z], moves)
