"""Online 3D bin-packing environment (BPP-1 / BPP-k)."""
from __future__ import annotations

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
