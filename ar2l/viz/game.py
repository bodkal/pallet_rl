"""Play the packing problem yourself, then hand the same instance to an agent.

    python3 -m ar2l.viz.game --port 8096

The instance is built from the *opponent's own* training configuration: pick a
run and the setup form fills itself with the bin, the item bounds, the window
`nb` and the reach `n_pick` that run was trained under, so the duel is fought
on the geometry the agent actually knows.  Every field stays editable - the
server re-validates the whole set and refuses a combination the simulator
could not run - and anything you change away from the run's own value is
flagged, because an agent replayed off its training distribution is being
asked a question it was never trained on.

You pack that bin under the simulator's own stability rule: every cell the
page lets you click is one the box would actually rest in.  That is a superset
of `env.obs()['l_mask']`, the EMS-corner candidate list the policy scores - a
human is not bound to the leaf nodes of a packing tree, so the page marks
which of your placements the packer would also have considered rather than
hiding the rest.

With `n_pick > 1` the conveyor is a pick station rather than a queue: the
first `n_pick` boxes are within the cell's reach and you choose which of them
to pack, exactly as a `select`-trained agent's permuter does.  The rest of the
window is preview - seen, reasoned about, not reachable.

Pick an attacker and the same learned adversary reorders the conveyor for both
of you - which is the whole point of AR2L: a policy is only as good as its
worst ordering.  The instance is then handed to the opponent (any trained run,
or a heuristic) and the two bins are scored side by side.

The boxes come from the random generator (the `types:` size classes), or from
an orders file -- the real pallets of `ar2l.orders`, read with the same
`pallet_cm` / `cell_cm` / `box_pad_m` / `box_scale` / `box_round` as training and
evaluation, one pallet per game (a chosen one, or one dealt by the seed), its
box order randomised by `order_random`.  A run trained on a data file presets
the form to that file and those settings.

Pointing at a cell also shows the UR20 placing the box there, in the 3D bin:
its IK pose, the link capsules of `ar2l.pack_collision` (red where one hits the
pack) and the columns it hits, crossed out on the top view -- the same check
the real cell runs.  Where the robot stands, the tool length and how big a
cell is in metres live under `robot:` in `config.yaml`.  With the robot-arm
filter on (`env.arm_collision`, or the form's switch) a placement the arm
cannot make is not a legal one at all -- for you, the packer and the opponent.

The result screen can then recalculate: the same boxes in the same order, the
agent replayed under parameters you edit, one row per setting.  The three
fields that feed the draw itself - `n_items`, `size_lo`, `size_hi` - are held
at the game's own values there, because changing one deals a different
sequence and the row would no longer be measuring the field you moved.
The box padding is the one source field a recalculation may move: the same
pallet in the same order, its boxes read from the file again at the new
padding.

The "Big experiment" tab packs every pallet file of a folder, under the setup
form's rules, onto as many pallets as each file takes, and writes one report
per file (`ar2l.viz.experiment`), with a live progress bar per file.

The "New training" tab starts `ar2l.train` runs with every flag editable,
preset from `config.yaml` (`ar2l.viz.trainer`).  A run keeps going when the
browser is closed and stops only with this server -- Ctrl+C or closing its
terminal.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import pack_collision as PC
from ..config import CFG
from ..env import (BPPBatch, TYPE_FLOOR, TYPE_RULES, sample_items,
                   stack_allowance, type_classes)
from ..orders import (ROUNDING, box_divisor, load_orders, orders_bin,
                      randomize_order)
from . import agents as A
from . import compare as C
from . import dashboard as D
from . import experiment as X
from . import trainer as TR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = A.ROOT
_GAMES: dict = {}
_LOCK = threading.Lock()
DEVICE = 'cuda'

#: what the command line pinned, which then overrides config.yaml in the form
CLI: dict = {}

#: the editable geometry and rule fields, and the three that are per-axis
#: triples; `PARAM_KEYS` adds where the boxes come from, below
GAME_KEYS = ('bin', 'n_items', 'size_lo', 'size_hi', 'nb', 'n_pick',
             'max_l', 'rot', 'ems', 'stability', 'min_support',
             'n_types', 'type_constraint', 'type_rule', 'arm_collision',
             'stack_cap',
             'stack_cap_types', 'stack_cap_side_cm', 'stack_cap_allow_cm',
             'base_x_moves')
#: what a run does not record, so there is nothing to compare it on
UNRECORDED = ('base_x_moves', 'stack_cap_types', 'stack_cap_side_cm',
              'stack_cap_allow_cm')
TRIPLES = ('bin', 'size_lo', 'size_hi')

#: where the boxes come from: the generator, or a file of real pallets and how
#: it is read.  In `orders` mode the bin (when `pallet_cm` is set), the item
#: bounds and `n_items` are derived from the data rather than typed in.
SOURCE_KEYS = ('source', 'data', 'pallet', 'cell_cm', 'pallet_cm',
               'box_pad_m', 'box_scale', 'box_round', 'order_random')
#: what a run's own configuration fixes about its data -- the pallet picked
#: and the order shuffle are the game's choice, not the run's
RUN_SOURCE_KEYS = ('source', 'data', 'cell_cm', 'pallet_cm', 'box_pad_m',
                   'box_scale', 'box_round')
#: the three that feed `sample_items`, and so fix the box stream a session was
#: dealt, plus everything that picks and reads a data file.  A recalculation
#: holds them at the values the game was played under -- move one and the same
#: seed deals a *different* sequence, so the rerun would no longer be about the
#: parameter that was changed.
#: The box padding is the exception: a rerun re-reads the same pallet, in the
#: order it was dealt, at the new padding (`session_seq`).
STREAM_KEYS = (('n_items', 'size_lo', 'size_hi', 'n_types')
               + tuple(k for k in SOURCE_KEYS if k != 'box_pad_m'))
#: every field the form carries
PARAM_KEYS = GAME_KEYS + SOURCE_KEYS
#: everything else: geometry and rules, which the stream is indifferent to
RECALC_KEYS = tuple(k for k in PARAM_KEYS if k not in STREAM_KEYS)

# Caps that exist only to keep the *page* alive: the top view paints one
# rectangle per floor cell and the browser has to redraw all of them on every
# hover, so a 500-wide bin is not a configuration, it is a hang.
MAX_AXIS = 200
MAX_NB = 60
MAX_ITEMS = 5000

#: loaded data files, keyed by everything that changes what is read from one
_DATA: dict = {}


def data_path(path):
    """A data file the page named, resolved against the project root."""
    q = os.path.normpath(os.path.join(ROOT, os.path.expanduser(str(path))))
    if not q.lower().endswith(('.csv', '.npy')):
        raise ValueError(f'a data file is an orders .csv or an instance .npy, '
                         f'got {path!r}')
    if not os.path.isfile(q):
        raise ValueError(f'no such data file: {path}')
    return q


def load_data(p):
    """-> (table, ids, bin) for an `orders` parameter set; cached.

    The bin is `pallet_cm` in cells when that is set, the form's own bin
    otherwise; a `.npy` is already in cells and ignores the CSV settings.
    """
    q = data_path(p['data'])
    csv = q.lower().endswith('.csv')
    S = (tuple(orders_bin(p['pallet_cm'], p['cell_cm']))
         if csv and p['pallet_cm'] else tuple(p['bin']))
    key = (q, os.path.getmtime(q), S, p['rot'],
           (p['cell_cm'], p['box_pad_m'], p['box_scale'], p['box_round'])
           if csv else None)
    if key not in _DATA:
        if csv:
            tbl, ids = load_orders(q, p['cell_cm'], S, rot=p['rot'],
                                   box_scale=p['box_scale'],
                                   box_round=p['box_round'],
                                   box_pad_m=p['box_pad_m'])
        else:
            tbl = np.load(q)
            ids = [str(i) for i in range(len(tbl))]
        while len(_DATA) > 8:
            _DATA.pop(next(iter(_DATA)))
        _DATA[key] = (np.asarray(tbl), ids, S)
    return _DATA[key]


def fits(box, S, rot):
    """Does a (sx, sy, sz) box fit bin S in some allowed orientation?"""
    sx, sy, sz = (int(v) for v in box[:3])
    flat = (sx <= S[0] and sy <= S[1]) or (rot >= 2 and sy <= S[0] and sx <= S[1])
    return flat and sz <= S[2]


# --------------------------------------------------------------- parameters
def defaults():
    """config.yaml's env, plus anything the command line pinned over it."""
    e, t = CFG['env'], CFG['train']
    p = {'bin': list(A.extent(e['bin'])),
         'n_items': int(e['n_items']),
         'size_lo': list(A.extent(e['size_lo'])),
         'size_hi': list(A.extent(e['size_hi'])),
         'nb': int(t['nb']),
         'n_pick': int(t['n_pick'] or t['nb']),
         'max_l': int(e['max_l']),
         'rot': int(e['rot']),
         'ems': int(e['ems']),
         'stability': str(e['stability']),
         'min_support': float(e['min_support']),
         'n_types': int(e['n_types']),
         'type_constraint': int(e['type_constraint']),
         'type_rule': str(e.get('type_rule', 'touch')),
         'arm_collision': int(e.get('arm_collision', 0)),
         'stack_cap': int(e['stack_cap']),
         'stack_cap_types': [int(t) for t in e['stack_cap_types'] or []],
         'stack_cap_side_cm': [float(v) for v in e['stack_cap_side_cm']],
         'stack_cap_allow_cm': [float(v) for v in e['stack_cap_allow_cm']],
         'base_x_moves': PC.base_x_moves(CFG['robot'])}
    ev = CFG['eval']
    p.update({'source': 'orders' if t.get('data') else 'random',
              'data': str(t.get('data') or ''),
              'pallet': -1,                    # -1: the seed deals one
              'cell_cm': float(ev['cell_cm']),
              'pallet_cm': (None if ev['pallet_cm'] is None
                            else [float(v) for v in ev['pallet_cm']]),
              'box_pad_m': float(ev.get('box_pad_m') or 0.0),
              'box_scale': float(ev['box_scale']),
              'box_round': str(ev['box_round']),
              'order_random': float(ev['order_random'])})
    p.update(CLI)
    return p


def stream_classes(p):
    """The size classes this session deals from, clipped to its own envelope.

    The classes live in `config.yaml`, but the page lets the item-side bounds
    be moved; a class that then sticks out of the envelope would deal boxes the
    form says cannot exist.  Clipping keeps the two agreeing, and a class the
    envelope swallows whole falls back to the envelope itself rather than to an
    empty range.
    """
    lo, hi = list(p['size_lo']), list(p['size_hi'])
    spec = CFG['env']['types']
    if spec is None or len(spec) != int(p['n_types']):
        return type_classes(None, int(p['n_types']), lo, hi)
    probs, clo, chi = type_classes(spec, int(p['n_types']), lo, hi)
    clo = np.clip(clo, lo, hi)
    chi = np.clip(chi, lo, hi)
    bad = (clo > chi).any(1)
    clo[bad], chi[bad] = lo, hi
    return probs, clo, chi


def run_params(spec):
    """The env `spec` was trained in, as the form's own fields.

    A heuristic or `random` was not trained in anything, so it falls back to
    the shipped defaults; those opponents run in whatever bin you give them.
    """
    p = defaults()
    if not spec or spec in ('none', 'random') or spec.startswith('heur:'):
        return p
    info = A.run_info(spec.split(':')[1])
    if info is None:
        return p
    a = info['args']
    for k in ('n_items', 'max_l', 'rot', 'ems', 'stability', 'min_support',
              'nb', 'n_types', 'type_constraint', 'type_rule', 'arm_collision',
              'stack_cap'):
        if a.get(k) is not None:
            p[k] = a[k]
    # a run from before --type_rule existed was trained under `touch`
    if a.get('type_rule') is None:
        p['type_rule'] = 'touch'
    for k in TRIPLES:
        if a.get(k) is not None:
            p[k] = list(A.extent(a[k]))
    # a run trained before --n_pick existed recorded `null`, i.e. the whole
    # window was within reach
    p['n_pick'] = int(a['n_pick'] or p['nb'])
    # the data it was trained on, read the way it was read.  A run trained on
    # the generator keeps whatever source the form had: which boxes to play is
    # the game's question, and resetting it to the generator on every
    # opponent change would throw away the file you chose.  The drift check
    # still says it was trained on random boxes.
    if a.get('data'):
        p['source'] = 'orders'
        p['data'] = a['data']
        for k in ('cell_cm', 'box_scale', 'box_round'):
            if a.get(k) is not None:
                p[k] = a[k]
        # a run from before the padding existed read its boxes unpadded
        p['box_pad_m'] = float(a.get('box_pad_m') or 0.0)
        p['pallet_cm'] = a.get('pallet_cm')
    # A flag on the command line outranks the run, as it outranks config.yaml
    # everywhere else here; without this the preset would silently undo it the
    # moment an opponent was chosen, which is every time.  The field is then
    # marked as drift, which is exactly what it is.
    p.update(CLI)
    p['algo'] = a.get('algo')
    return p


def validate(raw):
    """-> (params, errors).  Every field the page sent, checked in one place.

    The page validates as you type for the immediate feedback, but nothing is
    trusted from it: a bin the sweeps cannot run, or an item bound no draw can
    satisfy, has to fail here rather than raise out of `BPPBatch.__init__`
    and take the request down with a 500.
    """
    p, err = defaults(), []
    raw = raw or {}

    def whole(k, lo, hi, what):
        v = raw.get(k, p[k])
        try:
            n = int(v)
        except (TypeError, ValueError):
            err.append(f'{what}: expected a whole number, got {v!r}')
            return p[k]
        if not lo <= n <= hi:
            err.append(f'{what}: must be between {lo} and {hi}, got {n}')
            return p[k]
        return n

    def triple(k, lo, hi, what):
        v = raw.get(k, p[k])
        v = list(v) if isinstance(v, (list, tuple)) else [v] * 3
        if len(v) != 3:
            err.append(f'{what}: expected three numbers, got {len(v)}')
            return p[k]
        out = []
        for ax, t in zip('xyz', v):
            try:
                n = int(t)
            except (TypeError, ValueError):
                err.append(f'{what} {ax}: expected a whole number, got {t!r}')
                return p[k]
            if not lo <= n <= hi:
                err.append(f'{what} {ax}: must be between {lo} and {hi}, got {n}')
                return p[k]
            out.append(n)
        return out

    p['bin'] = triple('bin', 1, MAX_AXIS, 'bin')
    p['size_lo'] = triple('size_lo', 1, MAX_AXIS, 'smallest item side')
    p['size_hi'] = triple('size_hi', 1, MAX_AXIS, 'largest item side')
    p['n_items'] = whole('n_items', 1, MAX_ITEMS, 'items in the sequence')
    p['nb'] = whole('nb', 1, MAX_NB, 'window N_B')
    p['n_pick'] = whole('n_pick', 1, MAX_NB, 'reach k')
    p['max_l'] = whole('max_l', 1, 1 << 20, 'candidate cap')
    p['rot'] = whole('rot', 1, 2, 'orientations')
    p['ems'] = whole('ems', 0, 3, 'EMS filter')
    p['n_types'] = whole('n_types', 1, 16, 'box types')
    p['type_constraint'] = whole('type_constraint', 0, 1, 'stacking rule')
    p['arm_collision'] = whole('arm_collision', 0, 1, 'robot-arm filter')
    p['stack_cap'] = whole('stack_cap', 0, 1, 'stack-height limit')

    def numbers(k, what, cast=float):
        """A list typed as "12, 30" (or sent as a list), as `cast`s."""
        v = raw.get(k, p[k])
        if isinstance(v, str):
            v = v.replace(',', ' ').split()
        try:
            return [cast(t) for t in v]
        except (TypeError, ValueError):
            err.append(f'{what}: expected numbers, got {v!r}')
            return p[k]

    types = numbers('stack_cap_types', 'limited box types', int)
    # a type the boxes never carry simply never sets a limit, as in the env
    if any(t < 0 for t in types):
        err.append(f'limited box types: a type is 0 or more, got {types}')
    else:
        p['stack_cap_types'] = types
    side = numbers('stack_cap_side_cm', 'stack-limit sides (cm)')
    allow = numbers('stack_cap_allow_cm', 'stack-limit allowances (cm)')
    if len(side) != 2 or len(allow) != 2:
        err.append('stack limit: give two sides and two allowances in cm')
    else:
        try:
            stack_allowance(1, side, allow, 1.0)
            p['stack_cap_side_cm'], p['stack_cap_allow_cm'] = side, allow
        except ValueError as ex:
            err.append(f'stack limit: {ex}')
    mv = raw.get('base_x_moves', p['base_x_moves'])
    try:
        if isinstance(mv, str):
            mv = [t for t in mv.replace(',', ' ').split()]
        if not 1 <= len(mv) <= 32:
            raise ValueError
        mv = PC.base_x_moves({'base_x_moves': list(mv)})
        p['base_x_moves'] = mv
    except (TypeError, ValueError):
        err.append('arm base x moves: 1 to 32 numbers between -1 and 1, '
                   'tried in order')

    # ---- where the boxes come from ----------------------------------------
    src = str(raw.get('source', p['source']))
    if src not in ('random', 'orders'):
        err.append(f"box source: expected 'random' or 'orders', got {src!r}")
    else:
        p['source'] = src
    p['data'] = str(raw.get('data', p['data']) or '').strip()
    p['pallet'] = whole('pallet', -1, 1 << 30, 'pallet')
    for k, lo, hi, what in (('cell_cm', 1e-3, 1e3, 'cell size (cm)'),
                            ('box_pad_m', 0.0, 1.0, 'box padding (m)'),
                            ('order_random', 0.0, 1.0, 'order randomness')):
        try:
            v = float(raw.get(k, p[k]))
            if not lo <= v <= hi:
                raise ValueError
            p[k] = v
        except (TypeError, ValueError):
            err.append(f'{what}: expected a number between {lo:g} and {hi:g}')
    try:
        p['box_scale'] = float(raw.get('box_scale', p['box_scale']))
        box_divisor(p['box_scale'])
    except (TypeError, ValueError):
        err.append('box scale: 0 (sizes as in the file) or a number >= 1 '
                   'to divide every side by')
    r = str(raw.get('box_round', p['box_round']))
    if r not in ROUNDING:
        err.append(f'box rounding: one of {", ".join(sorted(ROUNDING))}, got {r!r}')
    else:
        p['box_round'] = r
    pc = raw.get('pallet_cm', p['pallet_cm'])
    if pc in (None, '', [], [None, None, None]):
        p['pallet_cm'] = None
    else:
        try:
            pc = [float(v) for v in pc]
            if len(pc) != 3 or min(pc) <= 0:
                raise ValueError
            p['pallet_cm'] = pc
        except (TypeError, ValueError):
            err.append('pallet (cm): three positive numbers, or empty to use '
                       'the bin in cells')

    s = str(raw.get('stability', p['stability']))
    if s not in ('com', 'cdrl'):
        err.append(f"stability rule: expected 'com' or 'cdrl', got {s!r}")
    else:
        p['stability'] = s
    s = str(raw.get('type_rule', p['type_rule']))
    if s not in TYPE_RULES:
        err.append(f"type rule: expected one of {', '.join(TYPE_RULES)}, "
                   f"got {s!r}")
    else:
        p['type_rule'] = s
    try:
        ms = float(raw.get('min_support', p['min_support']))
        if not 0.0 <= ms <= 1.0:
            raise ValueError
        p['min_support'] = ms
    except (TypeError, ValueError):
        err.append('support floor: expected a fraction between 0 and 1')

    # ---- the combinations, once every field is individually sane -----------
    orders = p['source'] == 'orders'
    if orders and not err:
        err += derive_orders(p)
    if p['n_pick'] > p['nb']:
        err.append(f"reach k = {p['n_pick']} is larger than the window "
                   f"N_B = {p['nb']}: you cannot reach a box you cannot see")
    for ax, (lo, hi, L) in enumerate(zip(p['size_lo'], p['size_hi'], p['bin'])):
        n = 'xyz'[ax]
        if orders:
            break          # real boxes are checked one by one, turned if need be
        if lo > hi:
            err.append(f'item side {n}: the smallest ({lo}) is above the '
                       f'largest ({hi})')
        elif hi > L:
            # an item wider than the axis it is drawn for can never be placed
            # in its own pose, so the episode would die on the first such draw
            err.append(f'largest item side {n} = {hi} does not fit the bin '
                       f'({n} = {L})')
    cells = p['bin'][0] * p['bin'][1]
    if cells > 40000:
        err.append(f'a {p["bin"][0]}x{p["bin"][1]} floor is {cells} cells; '
                   f'the top view redraws all of them on every hover')
    return p, err


def derive_orders(p, table=True):
    """Fill the fields an orders file fixes, in place.  -> errors.

    The bin (when `pallet_cm` is set), the item-side envelope and `n_items`
    come from the data, so the form cannot claim a stream the file does not
    hold.  `table=False` is a recalculation: the session's pallet is already
    dealt, so only the bin can move and nothing is re-read.
    """
    if not p['data']:
        return ['box source is an orders file, but no data file is given']
    try:
        tbl, ids, S = load_data(p)
    except (OSError, ValueError) as e:
        return [str(e)]
    p['bin'] = list(S)
    if not -1 <= p['pallet'] < len(tbl):
        return [f"pallet: the file holds {len(tbl)} pallets (0 to "
                f"{len(tbl) - 1}, or -1 for one dealt by the seed), got "
                f"{p['pallet']}"]
    real = tbl[..., 0] > 0
    boxes = tbl[real]
    p['size_lo'] = boxes[:, :3].min(0).astype(int).tolist()
    p['size_hi'] = boxes[:, :3].max(0).astype(int).tolist()
    p['n_items'] = int(real.sum(1).max())
    if p['pallet'] >= 0:
        p['n_items'] = int(real[p['pallet']].sum())
    t = int(boxes[:, 3].max())
    if t >= p['n_types']:
        return [f"the data has box type {t}, but box types is "
                f"{p['n_types']}; raise it to at least {t + 1}"]
    return []


def pallet_list(p):
    """The pallets of an orders parameter set, for the page's picker."""
    try:
        tbl, ids, _ = load_data(p)
    except (OSError, ValueError):
        return []
    real = tbl[..., 0] > 0
    n = real.sum(1)
    # boxes of each type on each pallet, indexed by the 0-based type
    nt = max(int(p['n_types']), int(tbl[..., 3][real].max(initial=-1)) + 1)
    types = [np.bincount(tbl[i, real[i], 3].astype(int), minlength=nt).tolist()
             for i in range(len(ids))]
    return [{'i': i, 'id': ids[i], 'boxes': int(n[i]), 'types': types[i]}
            for i in range(len(ids))]


def compared_keys(p):
    """What a run's configuration is compared on for this parameter set.

    With an orders file the item bounds and `n_items` are the file's, not
    settings, and the file itself (and how it is read) is compared instead.
    """
    keys = tuple(k for k in GAME_KEYS if k not in UNRECORDED)
    if p['source'] == 'orders':
        return tuple(k for k in keys
                     if k not in ('size_lo', 'size_hi', 'n_items')) + RUN_SOURCE_KEYS
    return keys + ('source',)


def diff_params(p, want, keys=PARAM_KEYS):
    """Which of `keys` differ between two parameter sets, formatted for the page."""
    out = []
    for k in keys:
        a, b = p[k], want[k]
        if isinstance(a, list) and isinstance(b, (list, tuple)):
            a, b = list(a), list(b)
        if k == 'data' and a and b:
            try:                        # the same file, however it is spelled
                a, b = data_path(a), data_path(b)
            except ValueError:
                pass
        if a != b:
            if k == 'base_x_moves':
                out.append({'key': k, 'is': ', '.join(f'{t:g}' for t in a),
                            'was': ', '.join(f'{t:g}' for t in b)})
                continue
            fmt = (lambda v: 'x'.join(f'{t:g}' if isinstance(t, float) else str(t)
                                      for t in v)
                   if isinstance(v, (list, tuple)) else
                   ('none' if v is None else f'{v:g}' if isinstance(v, float)
                    else str(v)))
            out.append({'key': k, 'is': fmt(p[k]), 'was': fmt(want[k])})
    return out


def mismatch(p, spec):
    """Which fields `p` moved away from the run behind `spec`."""
    if not spec or spec in ('none', 'random') or spec.startswith('heur:'):
        return []
    want = run_params(spec)
    info = A.run_info(spec.split(':')[1])
    if info is not None and not info['args'].get('data'):
        want['source'] = 'random'      # what it was trained on, for the drift
    return diff_params(p, want, compared_keys(p))


# -------------------------------------------------------------------- board
def free_positions(env, arm=True):
    """(R, S, S) every placement the box would rest in, EMS filter off.

    `arm=False` also lifts the robot-arm filter, so the page can tell a cell
    the arm rules out from one the box would not rest in.

    The policy's action space is the corners of the empty maximal spaces, but
    a human packing by hand is not bound to it: anything that lands stably and
    stays inside the bin is a placement they can reach for.  Run the same
    sweep `_positions` runs, with `ems` off, so the square-footprint and
    terminal-state filters still apply.
    """
    ems, armc = env.ems, env.arm_collision   # restore what it *was*: the game
    env.ems = 0                              # can itself be run with --ems 0
    env.arm_collision = armc and arm
    env._invalidate(hmap=False)          # the height map is unchanged; only the filter
    try:
        return env._positions()[0][0].copy()
    finally:
        env.ems, env.arm_collision = ems, armc
        env._invalidate(hmap=False)


def grids(env):
    """Per-orientation legality and landing height for one position.

    The page is a grid you click, so it needs a value at *every* cell, not the
    sparse candidate list the network scores.  Three layers come back:

        cand    the cell is in the candidate list -- what the packer scores
        free    the box would rest there stably -- what *you* may do, a
                superset of `cand`
        z       where the box would land, legal or not, so hovering an illegal
                cell can still say why
    """
    k = int(env.obs()['l_mask'][0].sum())   # refreshes the candidate table first
    cand, z, odims, tunder = env._positions()
    return cand[0], free_positions(env), z[0], odims[0], k, tunder[0]


def type_blocked(env, z, odims):
    """(R, S, S) the placements the box-type rule rejects, under `type_rule`.

    The page says why a cell is illegal; under `column` the foreign box may be
    deep under the footprint, so the top-type map alone cannot tell it.
    """
    if not (env.type_constraint and env.n_types > 1):
        return np.zeros(z.shape, bool)
    t = env.head_item()[1]
    return np.stack([~env._type_ok(np.asarray(d)[None], zr[None], t)[0]
                     for d, zr in zip(odims, z)])


def arm_blocked(env, free):
    """(R, S, S) the placements the robot-arm filter alone rules out."""
    if not env.arm_collision:
        return np.zeros_like(free)
    return free_positions(env, arm=False) & ~free


def cand_points(env):
    """[[x, y, z, r]] the packer's candidate list for the item in front.

    What the result screen draws as the "places this box could have gone":
    the corner the box's footprint starts at, the height it would land at,
    and the orientation it would be turned to.
    """
    k = int(env.obs()['l_mask'][0].sum())
    lxy, lz = env._lxy[0, :k], env._lz[0, :k]
    return np.stack([lxy[:, 0], lxy[:, 1], lz, lxy[:, 2]], -1).astype(int).tolist()


def trace_points(trace):
    """`cand_points` for every step of an `A.play` trace."""
    return [[[int(c[0]), int(c[1]), int(c[2]), int(r)]
             for c, r in zip(t['cands'], t['cand_rot'])] for t in trace]


def trace_windows(trace):
    """The conveyor at every step of an `A.play` trace, as the result draws it.

    `w`/`t` are the window as it stood before anyone reordered it, and `pick`
    is the slot of the box that went in -- moved to the front by the attacker
    or the run's own selector, or the front one when nobody chose.
    `probs` is the picker's probability for each box in view, or None when
    nobody chose.
    """
    return [{'w': t['window'], 't': t['window_types'],
             'pick': t['perm_idx'] or 0,
             'probs': None if t['perm_probs'] is None
                      else rounded(t['perm_probs'][:len(t['window'])])}
            for t in trace]


def final_window(ep):
    """[the conveyor as the episode left it], or [] when it ran empty.

    `pick` is -1: nothing was packed from it.
    """
    if not ep.get('final_window'):
        return []
    return [{'w': ep['final_window'], 't': ep['final_window_types'],
             'pick': -1, 'probs': None}]


def rounded(p):
    """Probabilities as the page shows them: four places is plenty."""
    return [round(float(v), 4) for v in p]


def human_final(st):
    """`final_window` for your side: the strip as it stood when you stopped."""
    if not st['wmask0'].any():
        return []
    return [{'w': st['win0'][st['wmask0'], :3].tolist(),
             't': st['win0'][st['wmask0'], 3].tolist(), 'pick': -1,
             'probs': (None if st['pprobs'] is None else
                       rounded(st['pprobs'][:int(st['wmask0'].sum())]))}]


def pack_probs(st):
    """The opponent packer's probability for every point of `cand_points`.

    None when there is no opponent to ask or nowhere left to go.  The policy
    reads `env.obs()` itself, so the list it scores is the one `cand_points`
    returns for the same position.
    """
    pol, env = st.get('pol'), st['env']
    if pol is None or env.done[0] or not env.n_feasible()[0]:
        return None
    _, prob = pol(env)
    k = int(env.obs()['l_mask'][0].sum())
    return rounded(prob[0][:k])


def place(env, r, x, y):
    """Drop the leading item at (orientation r, x, y); False if it cannot rest.

    The cell may be outside the candidate list, so there is no action index to
    step.  Rather than duplicate `step`'s bookkeeping, rebuild the placement
    table with the EMS filter off and nothing truncated away, find the cell in
    it, and step that index normally.
    """
    ems, max_l = env.ems, env.max_l
    env.ems = 0
    env.max_l = env.rot * env.Lx * env.Ly   # every (orientation, x, y), none dropped
    env._invalidate(hmap=False)
    try:
        o = env.obs()
        lxy = env._lxy[0]
        hit = np.flatnonzero((lxy[:, 2] == r) & (lxy[:, 0] == x) & (lxy[:, 1] == y)
                             & (o['l_mask'][0] > 0))
        if hit.size == 0:
            return False
        # `step` re-reads n_feasible() to decide the bin is finished, and does
        # it with ems still off - which is right: the game is over when *you*
        # have nowhere left, not when the packer would have run out.
        env.step(np.array([int(hit[0])]))
    finally:
        env.ems, env.max_l = ems, max_l
        env._invalidate(hmap=False)
    return True


# ---------------------------------------------------------------- robot arm
quat_tf = PC.quat_tf


def cell_m(p):
    """Metres one grid cell stands for (see `pack_collision.robot_cell_m`)."""
    return PC.robot_cell_m(CFG['robot'], p['cell_cm'], p['box_scale'])


def arm_checker(st):
    """The session's `ArmPackChecker`, built once for its grid."""
    if st.get('arm') is None:
        st['arm'], st['base_from_box'] = PC.checker_from_config(
            CFG['robot'], cell_m(st['p']))
    return st['arm']


def arm_view(st, r, x, y):
    """The arm placing the held box at (orientation r, x, y), for the 3D view."""
    env = st['env']
    _, z, odims, _ = env._positions()
    if not (0 <= r < odims.shape[1] and 0 <= x < env.Lx and 0 <= y < env.Ly):
        return {'error': 'no such placement'}
    size = [int(v) for v in odims[0, r]]
    pos = [int(x), int(y), int(z[0, r, x, y])]
    return arm_pose(st, env.hmap[0].astype(int), size, pos)


def replay_hmap(placed, S):
    """The height map a list of placed [x, y, z, l, w, h] boxes leaves in bin S."""
    hm = np.zeros((int(S[0]), int(S[1])), int)
    for x, y, z, l, w, h in placed:
        blk = hm[x:x + l, y:y + w]
        np.maximum(blk, z + h, out=blk)
    return hm


def stack_kw(p):
    """The type rule and stack-height limit of a parameter set, as `BPPBatch`
    (and `agents.play`) take them."""
    return dict(type_rule=p['type_rule'], stack_cap=bool(p['stack_cap']), stack_types=p['stack_cap_types'],
                stack_side_cm=p['stack_cap_side_cm'],
                stack_allow_cm=p['stack_cap_allow_cm'])


def arm_replay(st, placed, S, moves=None):
    """The arm placing the last of `placed`, against the pack the others left.

    A replay step is "box n has just gone in", so the arm is checked against
    the bin as it stood a moment before -- boxes 0..n-1 -- exactly as the live
    game checks the box you are holding against the bin as it is.
    """
    try:
        placed = [[int(v) for v in b] for b in placed]
        S = [int(v) for v in S]
        if not placed or any(len(b) != 6 for b in placed) or len(S) != 3:
            raise ValueError
    except (TypeError, ValueError):
        return {'error': 'a replay step is a list of [x, y, z, l, w, h] and a bin'}
    *before, (x, y, z, l, w, h) = placed
    return arm_pose(st, replay_hmap(before, S), [l, w, h], [x, y, z], moves)


def arm_pose(st, hm, size, pos, moves=None):
    """The arm placing a `size` box at `pos` over height map `hm`.

    Everything comes back in the bin's cell coordinates, through the same
    `CellFrame` the collision check uses, so what is drawn is what was tested:
    the link capsules (each flagged if it hits the pack), the kinematic chain
    from the robot base to the tool tip, and the columns the arm collides with.
    The base stands at the first of `moves` (the session's `base_x_moves`
    when None) that clears the pack, as the env's filter does, or at the
    first one when none does.
    """
    chk = arm_checker(st)
    chk.set_heightmap(hm)
    moves = PC.base_x_moves({'base_x_moves': st['p']['base_x_moves']
                             if moves is None else moves})
    bases = PC.moved_bases(st['base_from_box'], moves,
                           hm.shape[1] / chk.cell_frame.scale_y)
    k = next((i for i, Bi in enumerate(bases)
              if not chk.is_arm_collid_with_pack(size, pos, Bi)[0]), 0)
    B = bases[k]
    joints = chk.get_arm_joints(size, pos, B)
    out = {'size': size, 'pos': pos, 'base_move': moves[k]}
    if not joints:
        out['reachable'] = False
        return out
    joint_tf = PC.get_forward_kinematics(joints, (1, 2, 3, 4, 5, 6))

    box_from_base = np.linalg.inv(B)
    cf = chk.cell_frame
    to_cell = lambda T: cf((box_from_base @ T)[:3, 3]).tolist()
    tool = joint_tf[5] @ PC.transform(t=[0.0, 0.0, float(CFG['robot']['tool_length_m'])])
    chain = [to_cell(np.eye(4))] + [to_cell(T) for T in joint_tf] + [to_cell(tool)]

    caps = PC.build_arm_capsules(joint_tf, box_from_base, chk.upper_arm_off_a,
                                 chk.upper_arm_off_b, chk.arm_params, cf)
    hs = cf.height_scale()
    mask = PC.collision_mask(caps, hm, chk.pad, hs)
    out.update({
        'reachable': True,
        'hit': bool(PC.any_capsule_hits_pyramid(caps, chk.pyramid, chk.pad)),
        'caps': [{'name': n, 'a': c.a.tolist(), 'b': c.b.tolist(),
                  'ra': c.ra, 'rb': c.rb,
                  'hit': bool(chk.pyramid.hits(c, chk.pad))}
                 for n, c in zip(('upper arm', 'forearm', 'wrist'), caps)],
        'chain': chain,
        # the pedestal, drawn from the base down to the pallet floor: joint 0's
        # 0.45 m diameter from UrKin::get_joints_diameter
        'base_r': 0.225 * cf.scale_x,
        # the columns it hits, with their heights, so a replay (which has no
        # height map of its own on the page) can still tint their tops
        'cells': [[int(x), int(y), int(hm[x, y])] for x, y in np.argwhere(mask > 0)],
        'depth': PC.max_penetration_depth(caps, hm, chk.pad, hs),
        'joints_deg': np.degrees(joints).round(1).tolist(),
    })
    return out


# -------------------------------------------------------------- pick station
def snapshot(st):
    """Remember the conveyor as the page is about to draw it.

    Every pick is applied to *this* order rather than to whatever the last
    pick left behind, so choosing slot 3 and then changing your mind to slot 1
    is one permutation of the window and not two composed ones -- which is
    what the trained permuter does, and the only reading under which the strip
    on screen can keep a stable order to click.
    """
    win, wmask = st['env'].window()
    st['win0'], st['wmask0'] = win[0].copy(), wmask[0].copy()
    st['sel'] = 0


def restore(st):
    """Put the remembered window back into the sequence."""
    env = st['env']
    off = env.head[0] + np.arange(env.nb)
    ok = off < env.n_items           # past the end the offsets clamp and would
    env.seq[0, off[ok]] = st['win0'][ok]     # otherwise write duplicates back
    env._invalidate(hmap=False)


def select(st, i):
    """Bring reachable item `i` of the remembered window to the front."""
    env = st['env']
    try:
        i = int(i)
    except (TypeError, ValueError):
        return False
    if not 0 <= i < min(env.n_pick, env.nb) or not st['wmask0'][i]:
        return False
    restore(st)
    if i:
        env.permute(np.array([i]))
    st['sel'] = i
    return True


def reach(st):
    """How many placements each box within reach still has.

    The bin is finished when *no* reachable box can be placed, not when the
    one at the front cannot: with a reach of five, four dead boxes and a live
    one is a live game.  One sweep per reachable box, at human pace.
    """
    env = st['env']
    keep, out = st['sel'], []
    for i in range(min(env.n_pick, env.nb)):
        if not st['wmask0'][i] or env.done[0]:
            out.append(0)
            continue
        select(st, i)
        out.append(int(free_positions(env).sum()))
    select(st, keep)
    return out


def board(st):
    """Everything the page draws for one position."""
    env = st['env']
    cand, free, z, odims, k, tunder = grids(env)
    item = env.seq[0, min(int(env.head[0]), env.n_items - 1)]
    picks = reach(st)
    nfree = int(free.sum())
    return {
        'Lx': int(env.Lx), 'Ly': int(env.Ly), 'Lz': int(env.Lz),
        'rot': int(odims.shape[0]),
        'dims': odims.tolist(),                      # the footprint per orientation
        'hmap': env.hmap[0].tolist(),
        # the type on top of each column, and what each cell of each
        # orientation would come to rest on: between them the page can say
        # *why* a cell it will not take is not takeable
        'tmap': env.tmap[0].tolist(),
        'tunder': tunder.astype(int).tolist(),
        'n_types': int(env.n_types),
        'type_constraint': bool(env.type_constraint),
        'type_rule': env.type_rule,
        'arm_collision': bool(env.arm_collision),
        'floor_type': int(TYPE_FLOOR),
        'item': item[:3].tolist(),
        'item_type': int(item[3]),
        'mask': cand.astype(np.uint8).tolist(),      # the packer's candidate list
        'free': free.astype(np.uint8).tolist(),      # everything you may click
        'armblock': arm_blocked(env, free).astype(np.uint8).tolist(),
        'typeblock': type_blocked(env, z, odims).astype(np.uint8).tolist(),
        'zmap': z.astype(int).tolist(),
        'ncand': k,
        'nfree': nfree,
        'window': st['win0'][st['wmask0'], :3].tolist(),
        'window_types': st['win0'][st['wmask0'], 3].tolist(),
        'nb': int(env.nb),
        'n_pick': int(env.n_pick),
        'sel': int(st['sel']),
        'pick_probs': (None if st['pprobs'] is None
                       else rounded(st['pprobs'][:int(st['wmask0'].sum())])),
        'pick_by': st['plabel'],
        # the opponent's packer over the packer's candidate list for the box
        # being packed: [x, y, z, r] and its probability, index for index
        'cands': cand_points(env),
        'cand_probs': pack_probs(st),
        'picks': picks,                # placements open to each reachable box
        'human_pick': bool(st['human_pick']),
        'placed': (env.packed[0, : env.n_packed[0]] * env.scale)
                  .round().astype(int).tolist(),
        'placed_types': env.ptype[0, : env.n_packed[0]].tolist(),
        'util': float(env.utilization()[0]),
        'items': int(env.n_packed[0]),
        # stuck means stuck with what you can actually take: with the pick
        # yours that is every reachable box, and without it only the one you
        # were handed
        'done': bool(env.done[0]) or not (any(picks) if st['human_pick']
                                          else picks[st['sel']]),
    }


def advance(st):
    """Let the attacker reorder the conveyor, then freeze it for the page."""
    env = st['env']
    st['pprobs'] = None
    if st['att'] is not None and not env.done[0]:
        idx, pp = st['att'](env)
        i = st['promoted'] = int(idx[0])
        env.permute(idx)
        # the attacker scored the window before it moved box i to the front;
        # reorder its scores the same way so they line up with the strip
        pp = pp[0]
        st['pprobs'] = np.concatenate([pp[i:i + 1], np.delete(pp, i)])
    else:
        st['promoted'] = None
    snapshot(st)
    if st['pprobs'] is None and st.get('picker') is not None and not env.done[0]:
        st['pprobs'] = st['picker'](env)[1][0]


def deal_pallet(p, seed):
    """-> (seq, pallet index, pallet id, order): one pallet of the file,
    reordered; `order` is the file rows `seq` was taken from, in turn.

    `pallet = -1` lets the seed pick it, so "replay this exact instance"
    deals the same pallet in the same order again.
    """
    tbl, ids, _ = load_data(p)
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(tbl))) if p['pallet'] < 0 else int(p['pallet'])
    L = int((tbl[i, :, 0] > 0).sum())
    # shuffle the row numbers rather than the rows, so a recalculation can
    # deal the same order again from the file read at another padding
    idx = tbl[i:i + 1].copy()
    idx[0, :, 3] = np.arange(idx.shape[1])
    order = randomize_order(idx, p['order_random'], rng)[0, :L, 3].astype(int)
    return tbl[i][order], i, ids[i], order


def session_seq(st, p):
    """The session's boxes as a rerun under `p` plays them: the pallet that
    was dealt, in the order it was dealt, read at `p`'s box padding."""
    if (st['p']['source'] != 'orders'
            or p['box_pad_m'] == st['p']['box_pad_m']):
        return st['seq']
    tbl, _, _ = load_data(dict(st['p'], box_pad_m=p['box_pad_m']))
    return tbl[st['p']['pallet']][st['order']]


def new_game(seed, attacker_spec, params, human_pick=True, opp_spec=None):
    att, alabel, _ = A.load_attacker(attacker_spec, DEVICE)
    p = dict(params)
    pid = order = None
    if p['source'] == 'orders':
        seq, p['pallet'], pid, order = deal_pallet(p, seed)
        # the game is this pallet: its length, and the bounds of its own boxes
        p['n_items'] = len(seq)
        p['size_lo'] = seq[:, :3].min(0).astype(int).tolist()
        p['size_hi'] = seq[:, :3].max(0).astype(int).tolist()
    else:
        seq = sample_items(np.random.default_rng(seed), (p['n_items'],),
                           p['size_lo'], p['size_hi'], stream_classes(p))
    env = BPPBatch(1, S=p['bin'], nb=p['nb'], n_items=p['n_items'],
                   size_lo=p['size_lo'], size_hi=p['size_hi'],
                   max_l=p['max_l'], rot=p['rot'], ems=p['ems'],
                   stability=p['stability'], min_support=p['min_support'],
                   n_pick=p['n_pick'], n_types=p['n_types'], types=False,
                   type_constraint=bool(p['type_constraint']),
                   arm_collision=bool(p['arm_collision']), arm_cell_m=cell_m(p),
                   arm_moves=p['base_x_moves'], **stack_kw(p))
    env.reset(seq[None])
    st = {'env': env, 'seq': seq, 'p': p, 'att': att, 'alabel': alabel,
          'pallet_id': pid, 'order': order,
          'human_pick': bool(human_pick) and p['n_pick'] > 1,
          'promoted': None, 'opp_spec': None, 'opp': None,
          'hcands': [], 'htaken': [], 'hwin': [], 'hprobs': [],
          'pol': None, 'picker': None, 'plabel': None}
    # the opponent, asked at every position what it would do there: its
    # packer scores your candidate points, and with no attacker on a select
    # run's own picker scores the boxes in reach
    if opp_spec:
        try:
            st['pol'] = A.load_policy(opp_spec, DEVICE)[0]
            if att is None:
                st['picker'], st['plabel'] = opp_permuter(st, opp_spec)
        except Exception:
            st['pol'] = st['picker'] = None
    if att is not None:
        st['plabel'] = alabel
    gid = uuid.uuid4().hex[:12]
    with _LOCK:
        _GAMES[gid] = st
        for old in list(_GAMES)[:-40]:      # keep the last few dozen sessions
            _GAMES.pop(old, None)
    advance(st)
    return gid, st


def opp_permuter(st, spec, n_pick=None):
    """Who chooses the agent's box: the attacker, or its own selector.

    A `select` run is trained *with* a permuter that hands it one of the `k`
    boxes within reach; replayed without one it would silently take slot 0
    every step, which is not the policy that was trained.  So when no attacker
    is on, a select run gets its own mixer back.

    `n_pick` is the reach of the replay being set up rather than the session's,
    because a recalculation may have changed it: a select run recalculated down
    to k = 1 has nothing left to select, and must not be handed its mixer.
    """
    if st['att'] is not None:
        return st['att'], st['alabel']
    k = st['p']['n_pick'] if n_pick is None else n_pick
    if spec.startswith('run:'):
        run = spec.split(':')[1]
        info = A.run_info(run)
        if info and info['args'].get('algo') == 'select' and k > 1:
            fn, lab, _ = A.load_attacker(f'mix:{run}', DEVICE)
            return fn, lab
    return None, None


def hand_over(st, spec, p, seq=None):
    """Play this session's own item stream with `spec`, under parameters `p`.

    `st['seq']` is what makes the comparison an instance rather than a sample:
    the boxes, and the order they arrive in, are the ones that were dealt once
    at `new_game`.  Everything the parameters touch -- the bin, the window, the
    reach, the action space, the stability rule -- is free to differ from the
    game that was played, which is what a recalculation varies.  `seq`
    replaces the stream when a recalculation re-read it at another padding.
    """
    seq = st['seq'] if seq is None else seq
    policy, label, _ = A.load_policy(spec, DEVICE)
    perm, plabel = opp_permuter(st, spec, p['n_pick'])
    t0 = time.time()
    ep = A.play(seq, policy, perm, nb=p['nb'], S=p['bin'],
                size_hi=p['size_hi'], max_l=p['max_l'], n_pick=p['n_pick'],
                rot=p['rot'], ems=p['ems'], stability=p['stability'],
                min_support=p['min_support'], n_types=p['n_types'],
                type_constraint=bool(p['type_constraint']),
                arm_collision=bool(p['arm_collision']), arm_cell_m=cell_m(p),
                arm_moves=p['base_x_moves'], **stack_kw(p))
    return {'label': label, 'util': ep['util'], 'items': ep['items'],
            'placed': ep['placed'], 'placed_types': ep['placed_types'],
            # before box n went in, the placements the packer had for it
            'cands': trace_points(ep['trace']),
            'taken': [int(t['choice']) for t in ep['trace']],
            'cand_probs': [rounded(t['probs']) for t in ep['trace']],
            'windows': trace_windows(ep['trace']) + final_window(ep),
            'seconds': time.time() - t0,
            'reason': 'ran out of room',
            # who picked the agent's box, so the result screen can say whether
            # the two sides had the same freedom
            'picker': plabel or ('first in reach' if p['n_pick'] > 1 else None)}


def opponent(st, spec):
    """Replay the identical instance with a chosen policy.

    Cached on the session: the page asks for the comparison once when the bin
    is full and again on every redraw of the result screen, and a trained
    policy is far too slow to replay on each of those.
    """
    if st.get('opp_spec') == spec:
        return st['opp']
    out = hand_over(st, spec, st['p'])
    st['opp_spec'], st['opp'] = spec, out
    return out


def recalc_params(st, raw):
    """-> (params, errors) for a recalculation of this session.

    The stream fields are taken from the session and not from the request, so
    "the same boxes" is a guarantee the server makes rather than a promise the
    page is trusted to keep.  Everything else is validated exactly as the setup
    form is, because it ends up in the same `BPPBatch`.
    """
    raw = raw or {}
    p = dict(st['p'])
    p.update({k: raw[k] for k in RECALC_KEYS if k in raw})
    p.update({k: st['p'][k] for k in STREAM_KEYS})
    if p['source'] != 'orders':
        return validate(p)
    # the pallet is dealt and fixed; the rerun may move the bin but must not
    # re-derive it from the file, so it is checked as the generator's is,
    # against the boxes that will actually be played
    p = dict(p, source='random')
    q, err = validate(p)
    q.update({k: st['p'][k] for k in STREAM_KEYS})
    err = [e for e in err if not e.startswith('largest item side')]
    try:
        seq = session_seq(st, q)
    except (OSError, ValueError) as e:
        return q, err + [str(e)]
    q['size_lo'] = seq[:, :3].min(0).astype(int).tolist()
    q['size_hi'] = seq[:, :3].max(0).astype(int).tolist()
    bad = [b for b in seq if not fits(b, q['bin'], q['rot'])]
    if bad:
        err.append(f"{len(bad)} of this pallet's boxes do not fit a "
                   f"{'x'.join(map(str, q['bin']))} bin, the first "
                   f"{'x'.join(map(str, bad[0][:3].tolist()))}")
    return q, err


def recalc(st, spec, p):
    """One row of the recalculation table: the same instance, other rules."""
    # `placed` stays in: the page draws the rerun's bin in 3D beside the game's
    out = hand_over(st, spec, p, session_seq(st, p))
    out['params'] = p
    # what moved from the game that was played, and what that leaves the agent
    # standing on: a rerun can walk a policy off its training configuration
    # just as the setup form can, and for the same reason it has to say so
    out['changed'] = diff_params(p, st['p'], RECALC_KEYS)
    out['drift'] = mismatch(p, spec)
    return out


def exp_params(raw):
    """-> (params, errors) for the big experiment: the setup form's rules.

    The boxes come from the experiment's own folder, so the form's source and
    item bounds say nothing here; everything else is checked as for a game.
    """
    p, err = validate(dict(raw or {}, source='random'))
    err = [e for e in err if not e.startswith(('item side', 'largest item side'))]
    if p['n_types'] < 3:
        err.append('box types: the files carry types 1 to 3, so at least 3')
    return p, err


#: what the experiment's form carries, and so what drift is judged on there
EXP_KEYS = ('nb', 'n_pick', 'max_l', 'rot', 'ems', 'stability', 'min_support',
            'type_constraint', 'type_rule', 'arm_collision', 'stack_cap',
            'pallet_cm',
            'box_pad_m', 'box_scale', 'box_round')


def exp_mismatch(p, spec):
    """Which of the experiment's fields `p` moved away from the run `spec`.

    A run trained on the generator never read a pallet in cm, so the fields
    that read the files are only compared for a run trained on data.
    """
    if not spec or spec in ('none', 'random') or spec.startswith('heur:'):
        return []
    info = A.run_info(spec.split(':')[1])
    keys = (EXP_KEYS if info and info['args'].get('data')
            else EXP_KEYS[:EXP_KEYS.index('pallet_cm')])
    return diff_params(p, run_params(spec), keys)


def cmp_reports(folder, picks=None):
    """What the Compare tab draws: every report of `folder`, its experiments
    cut down to `picks` (their `compare.key`; None keeps all), and who won.

    `keys` lists every experiment of the folder whatever is picked, so the
    page can offer them all: by first position run at, with how many files
    hold it and how many blocks (a name run twice in a file is two).
    """
    src = X.resolve(folder)
    if not os.path.isdir(src):
        raise ValueError(f'no such folder: {folder}')
    keys, files = {}, []
    for f in sorted(glob.glob(os.path.join(src, '*.csv'))):
        exps = C.read_report(f)
        for e in exps:
            k = keys.setdefault(C.key(e), {'key': C.key(e), 'named': bool(e['name']),
                                           'first': e['num'], 'files': 0,
                                           'blocks': 0, '_last': None})
            k['first'] = min(k['first'], e['num'])
            k['blocks'] += 1
            if k['_last'] != f:
                k['files'], k['_last'] = k['files'] + 1, f
        exps = [e for e in exps if picks is None or C.key(e) in picks]
        if not exps:
            continue
        win, best = C.winners(exps)
        files.append({'file': os.path.basename(f),
                      'win': [C.key(exps[j]) for j in win], 'best': list(best),
                      'exps': [{'key': C.key(e), 'num': e['num'], 'time': e['time'],
                                'boxes': e['boxes'], 'params': e['params'],
                                'pallets': [{'id': int(r['pallet_id']),
                                             'boxes': int(r[C.BOXES]),
                                             'volume': float(r[C.VOLUME])}
                                            for r in e['rows']]}
                               for e in exps]})
    return {'dir': X._show(src),
            'keys': [{n: v for n, v in k.items() if n != '_last'} for k in
                     sorted(keys.values(), key=lambda k: (k['first'], k['key']))],
            'files': files}


def cmp_delete(folder, names):
    """Delete the experiments keyed `names` (a name, or `Exp <num>` for an
    unnamed one) from every report of `folder`."""
    if X.writing(folder):
        raise ValueError('an experiment is writing to this folder; '
                         'wait for it or stop it first')
    src = X.resolve(folder)
    out = {}
    for f in sorted(glob.glob(os.path.join(src, '*.csv'))):
        n = C.delete_named(f, names)
        if n:
            out[os.path.basename(f)] = n
    return out


class QuietServer(ThreadingHTTPServer):
    """A browser refresh aborts the in-flight response; that is not an error."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)



def program_version():
    """'v1.<commits> (<short hash>) : <last commit date>' from git."""
    def git(*a):
        return subprocess.run(('git', '-C', HERE) + a, capture_output=True,
                               text=True, timeout=5).stdout.strip()
    try:
        n, h = git('rev-list', '--count', 'HEAD'), git('rev-parse', '--short', 'HEAD')
        d = git('log', '-1', '--format=%cd', '--date=format:%Y-%m-%d')
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return f'v1.{n} ({h}) : {d}' if n and h else 'unknown'

class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype='application/json', code=200):
        b = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(b)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass          # the page navigated away mid-response; nothing to do

    def _json(self, obj, code=200):
        self._send(json.dumps(obj), code=code)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path.rstrip('/') or '/'
        if p in ('/', '/index.html'):
            with open(os.path.join(HERE, 'game.html'), 'rb') as f:
                html = f.read().replace(b'{{VERSION}}', program_version().encode())
                return self._send(html, 'text/html; charset=utf-8')
        # the training dashboard, as its own server serves it at /
        if p == '/dashboard':
            if not u.path.endswith('/'):     # its requests are relative to it
                self.send_response(301)
                self.send_header('Location', '/dashboard/')
                return self.end_headers()
            with open(os.path.join(HERE, 'dashboard.html'), 'rb') as f:
                return self._send(f.read(), 'text/html; charset=utf-8')
        if p == '/dashboard/runs':
            return self._json([D.run_meta(r) for r in D.list_runs()])
        if p == '/dashboard/log':
            return self._json(D.read_log((parse_qs(u.query).get('run') or [''])[0]))
        if p == '/dashboard/summary':
            return self._json(D.summary())
        if p == '/api/setup':
            runs = [r for r in A.list_runs() if A.run_info(r)]
            policies = [r for r in runs if not r.startswith(('att_', 'h'))]
            attackers = [r for r in runs
                         if r.startswith(('att_', 'h')) or r in policies]
            return self._json({'policies': policies, 'attackers': attackers,
                               'heuristics': list(A.HEURISTICS),
                               'defaults': defaults(),
                               'limits': {'axis': MAX_AXIS, 'nb': MAX_NB,
                                          'items': MAX_ITEMS}})
        if p == '/api/params':
            q = parse_qs(u.query)
            spec = (q.get('spec') or [''])[0]
            try:
                return self._json({'params': run_params(spec)})
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
        if p == '/api/cmp':
            q = parse_qs(u.query, keep_blank_values=True)
            # `pick` is a JSON list of keys; left out, every experiment is kept
            picks = (set(json.loads(q['pick'][0] or '[]')) if 'pick' in q
                     else None)
            try:
                return self._json(cmp_reports((q.get('dir') or [''])[0], picks))
            except ValueError as e:
                return self._json({'error': str(e)}, 400)
        if p == '/api/train/form':
            try:
                return self._json(TR.form((parse_qs(u.query).get('config') or [''])[0]
                                          or None))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 400)
        if p == '/api/train/runs':
            return self._json(TR.runs())
        if p == '/api/train/run':
            try:
                return self._json(TR.run_values((parse_qs(u.query).get('name') or [''])[0]))
            except ValueError as e:
                return self._json({'error': str(e)}, 404)
        if p == '/api/train/status':
            return self._json(TR.status())
        if p == '/api/sys':
            return self._json(TR.system())
        if p == '/api/exp/status':
            q = parse_qs(u.query)
            s = X.status((q.get('job') or [''])[0])
            return self._json(s or {'error': 'no experiment has run yet'},
                              200 if s else 404)
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            return self._json({'error': 'bad request'}, 400)
        p = urlparse(self.path).path.rstrip('/')
        if p == '/dashboard/delete':
            try:
                return self._json({'moved_to': D.delete_run(body.get('run', ''))})
            except (ValueError, OSError) as e:
                return self._json({'error': str(e)})
        if p == '/api/check':
            params, err = validate(body.get('params'))
            return self._json({'params': params, 'errors': err,
                               'pallets': (pallet_list(params)
                                           if params['source'] == 'orders'
                                           and not err else []),
                               'mismatch': mismatch(params, body.get('opponent')),
                               'att_mismatch': mismatch(params, body.get('attacker'))})
        if p == '/api/new':
            params, err = validate(body.get('params'))
            if err:
                return self._json({'error': err[0], 'errors': err}, 400)
            gid, st = new_game(body.get('seed', 0), body.get('attacker'),
                               params, body.get('human_pick', True),
                               body.get('opponent'))
            return self._json({'gid': gid, 'board': board(st),
                               'params': st['p'], 'pallet_id': st['pallet_id'],
                               'promoted': st['promoted'], 'alabel': st['alabel']})
        if p == '/api/exp/check':
            params, err = exp_params(body.get('params'))
            return self._json({'params': params, 'errors': err,
                               'mismatch': [] if err else
                               exp_mismatch(params, body.get('spec'))})
        if p == '/api/exp/start':
            params, err = exp_params(body.get('params'))
            if err:
                return self._json({'error': err[0], 'errors': err}, 400)
            try:
                seed = int(body.get('seed') or 0)
            except (TypeError, ValueError):
                return self._json({'error': 'order seed: a whole number'}, 400)
            try:
                jid = X.start(body.get('dir') or '', body.get('out') or '',
                              body.get('spec') or 'random', params,
                              cell_m(params), DEVICE, body.get('name') or '',
                              seed=seed)
            except ValueError as e:
                return self._json({'error': str(e)}, 400)
            return self._json({'job': jid})
        if p == '/api/cmp/delete':
            names = [n for n in body.get('names') or [] if str(n).strip()]
            if not names:
                return self._json({'error': 'no experiment named to delete'}, 400)
            try:
                return self._json({'removed': cmp_delete(body.get('dir') or '', names)})
            except (ValueError, OSError) as e:
                return self._json({'error': str(e)}, 400)
        if p == '/api/train/check':
            argv, err = TR.check(body.get('values') or {}, body.get('config') or None)
            return self._json({'cmd': 'python -m ar2l.train ' + ' '.join(argv),
                               'errors': err})
        if p == '/api/train/start':
            try:
                return self._json({'name': TR.start(body.get('values') or {},
                                                    body.get('config') or None)})
            except ValueError as e:
                return self._json({'error': str(e)}, 400)
        if p == '/api/train/stop':
            try:
                return self._json({'ok': TR.stop(body.get('name'),
                                                 body.get('mode') or 'save')})
            except OSError as e:
                return self._json({'error': f'could not move the run away: {e}'}, 500)
        if p == '/api/train/resume':
            try:
                return self._json({'name': TR.resume(body.get('name') or '')})
            except ValueError as e:
                return self._json({'error': str(e)}, 400)
        if p == '/api/train/dismiss':
            return self._json({'ok': TR.dismiss(body.get('name'))})
        if p == '/api/exp/stop':
            return self._json({'ok': X.stop(body.get('job'))})
        st = _GAMES.get(body.get('gid'))
        if st is None:
            return self._json({'error': 'that game has expired'}, 404)
        if p == '/api/select':
            if not st['human_pick']:
                return self._json({'error': 'the pick station is not yours to '
                                            'choose from in this game'}, 400)
            if not select(st, body.get('i', 0)):
                return self._json({'error': 'that box is out of reach'}, 400)
            return self._json({'board': board(st), 'promoted': st['promoted']})
        if p == '/api/place':
            try:
                r, x, y = (int(body[k]) for k in 'rxy')
            except (KeyError, TypeError, ValueError):
                return self._json({'error': 'a placement is r, x and y'}, 400)
            pts = cand_points(st['env'])
            win = {'w': st['win0'][st['wmask0'], :3].tolist(),
                   't': st['win0'][st['wmask0'], 3].tolist(), 'pick': st['sel'],
                   'probs': (None if st['pprobs'] is None else
                             rounded(st['pprobs'][:int(st['wmask0'].sum())]))}
            probs = pack_probs(st)
            if not place(st['env'], r, x, y):
                return self._json({'error': 'the box will not rest there'}, 400)
            st['hcands'].append(pts)
            st['hwin'].append(win)
            st['hprobs'].append(probs)
            st['htaken'].append(next((i for i, c in enumerate(pts)
                                      if (c[3], c[0], c[1]) == (r, x, y)), -1))
            advance(st)
            return self._json({'board': board(st), 'promoted': st['promoted']})
        if p == '/api/arm':
            try:
                r, x, y = (int(body[k]) for k in 'rxy')
            except (KeyError, TypeError, ValueError):
                return self._json({'error': 'an arm pose is r, x and y'}, 400)
            try:
                return self._json(arm_view(st, r, x, y))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
        if p == '/api/arm_replay':
            try:
                return self._json(arm_replay(st, body.get('placed'), body.get('bin'),
                                             body.get('moves')))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
        if p == '/api/agent':
            # a policy that fails to load must not take the result screen down
            # with it - the page needs a body it can show as an error
            try:
                return self._json(dict(opponent(st, body['spec']),
                                       human_cands=st['hcands'],
                                       human_taken=st['htaken'],
                                       human_windows=st['hwin'] + human_final(st),
                                       human_cand_probs=st['hprobs']))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
        if p == '/api/recalc':
            params, err = recalc_params(st, body.get('params'))
            if err:
                return self._json({'error': err[0], 'errors': err}, 400)
            try:
                return self._json(recalc(st, body.get('spec') or 'random',
                                         params))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
        self.send_error(404)

    def log_message(self, *a):
        pass


def main(argv=None):
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8096)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--device', default='cuda')
    # These only seed the setup form: it is then filled from whichever run you
    # pick as the opponent, and editable either way.
    ap.add_argument('--bin', default=None,
                    help='bin extent: an int for a cube, or WxLxH')
    ap.add_argument('--size_hi', default=None,
                    help='item side cap: an int, or per-axis WxLxH')
    ap.add_argument('--max_l', type=int, default=None,
                    help='leaf cap; match the --max_l the opponent was trained with')
    ap.add_argument('--nb', type=int, default=None, help='observable window N_B')
    ap.add_argument('--n_pick', type=int, default=None,
                    help='how many of the N_B are within reach')
    ap.add_argument('--ems', type=int, choices=(0, 1, 2, 3), default=None,
                    help="the packer's candidates: 0 every position, 1 EMS "
                         "corners, 2 corner cells, 3 both")
    a = ap.parse_args(argv)
    DEVICE = a.device

    def extent(v):
        try:
            q = [int(t) for t in str(v).lower().split('x')]
            if len(q) not in (1, 3):
                raise ValueError
            return [q[0]] * 3 if len(q) == 1 else q
        except ValueError:
            ap.error(f'expected an int or WxLxH, got {v!r}')

    if a.bin is not None:
        CLI['bin'] = extent(a.bin)
    if a.size_hi is not None:
        CLI['size_hi'] = extent(a.size_hi)
    for k in ('max_l', 'nb', 'n_pick', 'ems'):
        if getattr(a, k) is not None:
            CLI[k] = getattr(a, k)

    srv = QuietServer((a.host, a.port), Handler)
    runs = [r for r in A.list_runs() if not r.startswith(('att_', 'h'))]
    d = defaults()
    print(f"game: http://{a.host}:{a.port}/   bin {'x'.join(map(str, d['bin']))}"
          f"   (opponents: {', '.join(runs[:6]) or 'heuristics only'})")
    # Training runs started from the page live as long as this server: closing
    # the terminal (SIGHUP) or a kill (SIGTERM) shuts it down, and them with it
    def bye(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, bye)
    signal.signal(signal.SIGTERM, bye)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        TR.stop_all()


if __name__ == '__main__':
    main()
