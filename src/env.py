"""Online 3D bin-packing environment (BPP-1 / BPP-k)."""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os

import numpy as np

from .bin3d import Bin3D
from .items import gen_sequence

# episode termination reasons
DONE_SEQ_END = "sequence_end"       # ran out of items (best case)
DONE_NO_FEASIBLE = "no_feasible"    # no legal LP for the arriving item
DONE_INVALID = "invalid_action"     # agent chose an infeasible LP


class PackingEnv:
    """One bin, one arriving stream of items.

    Observation dict:
        hmap      (L, W) int32       current height map
        item      (3,)   int         dimensions of the item to pack now
        mask      (A,)   float32     ground-truth feasibility mask
        lookahead (k, 3) int         current + next k-1 items (BPP-k / MCTS)
    """

    def __init__(self, cfg, dataset=None, sequences=None, seed=0):
        self.cfg = cfg
        self.dataset = dataset or cfg.dataset
        self.rng = np.random.default_rng(seed)
        self.sequences = sequences        # fixed benchmark, or None to sample
        self._seq_ptr = 0
        self.bin = Bin3D(cfg.L, cfg.W, cfg.H)
        self.seq: list = []
        self.idx = 0
        self.done_reason = None

    # -- helpers ----------------------------------------------------------
    def _new_sequence(self):
        if self.sequences is not None:
            s = self.sequences[self._seq_ptr % len(self.sequences)]
            self._seq_ptr += 1
            return list(s)
        return gen_sequence(self.rng, self.cfg.L, self.cfg.W, self.cfg.H,
                            self.cfg.item_min, self.cfg.item_max, self.dataset)

    def _obs(self):
        c = self.cfg
        item = self.seq[self.idx]
        k = max(1, c.lookahead_k)
        look = self.seq[self.idx: self.idx + k]
        while len(look) < k:                     # pad past the end of the stream
            look = look + [look[-1]]
        return {
            "hmap": self.bin.hmap.copy(),
            "item": np.asarray(item, dtype=np.int32),
            "mask": self.bin.feasibility_mask(item, c.orientations),
            "lookahead": np.asarray(look, dtype=np.int32),
        }

    # -- API ---------------------------------------------------------------
    def reset(self):
        self.bin = Bin3D(self.cfg.L, self.cfg.W, self.cfg.H)
        self.seq = self._new_sequence()
        self.idx = 0
        self.done_reason = None
        return self._obs()

    def peek_next_after_lookahead(self):
        """Dimensions of the first item beyond the lookahead window ('Last')."""
        j = self.idx + max(1, self.cfg.lookahead_k)
        if j < len(self.seq):
            return np.asarray(self.seq[j], dtype=np.int32)
        return None

    def step(self, action: int):
        c = self.cfg
        item = self.seq[self.idx]
        info = {}

        if not self.bin.is_feasible(action, item, c.orientations):
            info["invalid"] = True
            if c.invalid_action_mode == "terminate":
                self.done_reason = DONE_INVALID
                return self._terminal_obs(), 0.0, True, self._info(info)
            # 'resample' mode: fall back to a random feasible LP if one exists
            m = self.bin.feasibility_mask(item, c.orientations)
            f = np.flatnonzero(m)
            if len(f) == 0:
                self.done_reason = DONE_NO_FEASIBLE
                return self._terminal_obs(), 0.0, True, self._info(info)
            action = int(self.rng.choice(f))
        else:
            info["invalid"] = False

        self.bin.place(action, item, c.orientations)
        l, w, h = item
        reward = 10.0 * (l * w * h) / c.bin_volume
        self.idx += 1

        if self.idx >= len(self.seq):
            self.done_reason = DONE_SEQ_END
            return self._terminal_obs(), reward, True, self._info(info)

        obs = self._obs()
        if obs["mask"].sum() == 0:
            self.done_reason = DONE_NO_FEASIBLE
            return obs, reward, True, self._info(info)
        return obs, reward, False, self._info(info)

    def _terminal_obs(self):
        c = self.cfg
        return {"hmap": self.bin.hmap.copy(),
                "item": np.zeros(3, dtype=np.int32),
                "mask": np.zeros(c.action_dim, dtype=np.float32),
                "lookahead": np.zeros((max(1, c.lookahead_k), 3), dtype=np.int32)}

    def _info(self, info):
        info.update(utilization=self.bin.utilization,
                    n_items=len(self.bin.placed),
                    reason=self.done_reason)
        return info


class VecPackingEnv:
    """Synchronous vector of PackingEnv, with auto-reset on episode end."""

    def __init__(self, cfg, n, dataset=None, seed=0, sequences=None):
        self.cfg = cfg
        self.envs = [PackingEnv(cfg, dataset, sequences, seed=seed * 1000 + i)
                     for i in range(n)]
        self.n = n
        if sequences is not None:
            # Each PackingEnv walks `sequences` from its own pointer, so without
            # an offset all n envs would pack the *same* sequence at the same
            # time and the batch would carry n copies of one episode. Spread the
            # starting pointers evenly over the pool instead.
            for i, e in enumerate(self.envs):
                e._seq_ptr = i * len(sequences) // n

    def reset(self):
        return _stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, i = e.step(int(a))
            if d:
                i["final_utilization"] = e.bin.utilization
                i["final_items"] = len(e.bin.placed)
                o = e.reset()
            obs.append(o); rews.append(r); dones.append(d); infos.append(i)
        return (_stack(obs), np.asarray(rews, np.float32),
                np.asarray(dones, np.bool_), infos)


def _stack(obs_list):
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


# ---------------------------------------------------------------------------
# the same vector env, split across processes
# ---------------------------------------------------------------------------
def _make_envs(cfg, dataset, sequences, idx, n_total, seed):
    """Build the envs with GLOBAL index `idx`, exactly as VecPackingEnv would.

    Both the per-env seed and the sequence-pool offset are functions of the
    global index, so a worker holding envs 8..15 must build them with those
    numbers -- not 0..7 -- or the batch stops matching the serial version.
    """
    envs = [PackingEnv(cfg, dataset, sequences, seed=seed * 1000 + i) for i in idx]
    if sequences is not None:
        for e, i in zip(envs, idx):
            e._seq_ptr = i * len(sequences) // n_total
    return envs


def _step_envs(envs, actions):
    obs, rews, dones, infos = [], [], [], []
    for e, a in zip(envs, actions):
        o, r, d, i = e.step(int(a))
        if d:
            i["final_utilization"] = e.bin.utilization
            i["final_items"] = len(e.bin.placed)
            o = e.reset()
        obs.append(o); rews.append(r); dones.append(d); infos.append(i)
    return obs, rews, dones, infos


def _env_worker(remote, parent_remote, cfg, dataset, sequences, idx, n_total, seed):
    parent_remote.close()
    envs = _make_envs(cfg, dataset, sequences, idx, n_total, seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(_step_envs(envs, data))
            elif cmd == "reset":
                remote.send([e.reset() for e in envs])
            else:                                    # "close"
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        remote.close()


class SubprocVecPackingEnv:
    """VecPackingEnv with the env loop spread over worker processes.

    Same envs, same seeds, same batch -- so the gradients are identical to the
    serial version, bit for bit.  It only moves the numpy feasibility-mask work
    off the single core that VecPackingEnv.step pins it to.

    Start method is `fork` (overridable with PALLET_VECENV_START): the workers
    are pure numpy and never touch CUDA, and fork lets a large --seq-pool be
    inherited copy-on-write instead of pickled to every worker.  Build this
    BEFORE moving the network to the GPU.
    """

    def __init__(self, cfg, n, dataset=None, seed=0, sequences=None, workers=4):
        self.cfg, self.n = cfg, n
        workers = max(1, min(int(workers), n))
        ctx = mp.get_context(os.environ.get("PALLET_VECENV_START", "fork"))
        # contiguous chunks, so concatenating the replies rebuilds the batch in
        # env order and row k is still env k
        bounds = [round(k * n / workers) for k in range(workers + 1)]
        self.slices = [slice(bounds[k], bounds[k + 1]) for k in range(workers)]
        self.remotes, self.procs = [], []
        for sl in self.slices:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_env_worker,
                            args=(child, parent, cfg, dataset, sequences,
                                  list(range(sl.start, sl.stop)), n, seed),
                            daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.closed = False
        atexit.register(self.close)

    def _gather(self, what):
        try:
            return [r.recv() for r in self.remotes]
        except EOFError:
            self.close()
            raise RuntimeError(f"an env worker died during {what}") from None

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        out = []
        for part in self._gather("reset"):
            out += part
        return _stack(out)

    def step(self, actions):
        for r, sl in zip(self.remotes, self.slices):
            r.send(("step", actions[sl]))
        obs, rews, dones, infos = [], [], [], []
        for o, rw, d, i in self._gather("step"):
            obs += o; rews += rw; dones += d; infos += i
        return (_stack(obs), np.asarray(rews, np.float32),
                np.asarray(dones, np.bool_), infos)

    def close(self):
        if getattr(self, "closed", True):
            return
        self.closed = True
        for r in self.remotes:
            try:
                r.send(("close", None)); r.close()
            except (OSError, BrokenPipeError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __del__(self):
        self.close()


def make_vec_env(cfg, n, dataset=None, seed=0, sequences=None, workers=0):
    """Serial by default; `workers` > 1 spreads the env loop over processes."""
    if workers and workers > 1:
        return SubprocVecPackingEnv(cfg, n, dataset, seed, sequences, workers)
    return VecPackingEnv(cfg, n, dataset, seed, sequences)


def build_obs_tensor(hmap, item, cfg):
    """(B, 4, L, W) network input: normalised height map + stretched item dims.

    Paper Sec. 3.2 "State input": d_n is stretched into a three-channel L x W x 3
    tensor, concatenated with the height map -> L x W x 4.
    """
    B = hmap.shape[0]
    x = np.zeros((B, 4, cfg.L, cfg.W), dtype=np.float32)
    x[:, 0] = hmap.astype(np.float32) / cfg.H
    x[:, 1] = (item[:, 0].astype(np.float32) / cfg.L)[:, None, None]
    x[:, 2] = (item[:, 1].astype(np.float32) / cfg.W)[:, None, None]
    x[:, 3] = (item[:, 2].astype(np.float32) / cfg.H)[:, None, None]
    return x
