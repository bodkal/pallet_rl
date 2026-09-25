"""The defaults, read from `config.yaml`.

One file holds every default: the argparsers of `ar2l.train` and
`ar2l.evaluate` take theirs from it, and so do the values that have no flag --
the PPO clip, the pointer temperature, the packed-item capacity.  A flag on
the command line still wins over the file.

    from .config import CFG
    CFG["ppo"]["lr"]        # 3e-4
    cfg("env", "min_support")

`AR2L_CONFIG=/path/to/other.yaml` (or `load(path)`, which `--config` calls)
switches files; anything the replacement leaves out keeps the shipped value,
so a partial file is a patch rather than a full restatement.
"""
from __future__ import annotations

import copy
import os

import yaml

# `config.yaml` beside the package, i.e. at the root of the reproduction
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "config.yaml")

SECTIONS = ("env", "train", "ppo", "model", "run", "eval", "robot")


def _read(path):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a mapping of sections, got "
                         f"{type(d).__name__}")
    unknown = set(d) - set(SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {sorted(unknown)}; "
                         f"expected any of {list(SECTIONS)}")
    return d


def _merge(base, over):
    out = copy.deepcopy(base)
    for sec, vals in over.items():
        unknown = set(vals or {}) - set(out.get(sec, {}))
        if unknown:
            raise ValueError(f"unknown key(s) {sorted(unknown)} in section "
                             f"[{sec}]")
        out.setdefault(sec, {}).update(vals or {})
    return out


#: the shipped defaults, kept so a partial override file is a patch
BASE = _read(PATH)
CFG = copy.deepcopy(BASE)

_env_path = os.environ.get("AR2L_CONFIG")


def load(path=None):
    """Merge `path` over the shipped defaults and return the result.

    Mutates the module-level `CFG` in place, so anything that read `CFG`
    earlier -- the argparsers do, at call time -- sees the new values.
    """
    path = path or _env_path
    if path:
        CFG.clear()
        CFG.update(_merge(BASE, _read(path)))
    return CFG


def cfg(section, key):
    return CFG[section][key]


load()          # honour AR2L_CONFIG on import
