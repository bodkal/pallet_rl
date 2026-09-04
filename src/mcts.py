"""BPP-k: Monte-Carlo permutation tree search (paper Sec. 3.3, Algorithm 1).

The trained BPP-1 network is used unchanged -- no extra training.  We search over
*permutations* of the k lookahead items: a path from the root to a leaf is one
virtual placing order.  Each node hallucinates the placement of one item by
updating the height map, conditioning the current item's decision on the future.

Order dependence: if a later-arriving item q is virtually placed before an
earlier item p, then p may never occupy any LP overlapping q.  This is enforced
by raising the height map to H over q's footprint before p is placed
(paper: "H_p(x, y) <- H for all x in [x_q, x_q + l_q], y in [y_q, y_q + w_q]").

Backup uses the MAX return, not the mean (paper supplemental B).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .bin3d import Bin3D
from .env import build_obs_tensor


class _Node:
    __slots__ = ("item_idx", "action", "bin", "footprints", "reward",
                 "parent", "children", "untried", "N", "Q")

    def __init__(self, item_idx, action, bin_, footprints, reward, parent, untried):
        self.item_idx = item_idx      # index into the lookahead window
        self.action = action
        self.bin = bin_
        self.footprints = footprints  # (x, y, l, w) of items placed so far
        self.reward = reward
        self.parent = parent
        self.children = []
        self.untried = list(untried) if untried is not None else []
        self.N = 0
        self.Q = -1e9                 # max return seen through this node


class PermutationMCTS:
    def __init__(self, cfg, net, device):
        self.cfg, self.net, self.device = cfg, net, device
        self.mean_item = np.array(
            [(cfg.item_min + cfg.item_max) / 2.0] * 3, dtype=np.int32)

    # -- network helpers ---------------------------------------------------
    @torch.no_grad()
    def _policy_action(self, bin_, item, hmap):
        """Greedy feasible LP for `item` on the (possibly blocked) height map."""
        cfg = self.cfg
        mask = bin_.feasibility_mask(item, cfg.orientations, hmap=hmap)
        if mask.sum() == 0:
            return None, mask
        x = build_obs_tensor(hmap[None], np.asarray(item, np.int32)[None], cfg)
        logits, _, mask_logits = self.net(torch.as_tensor(x, device=self.device))
        used = (torch.sigmoid(mask_logits) > 0.5).float()
        # never return an action the *true* mask rejects: the simulator must stay
        # legal even when the predictor is wrong
        tm = torch.as_tensor(mask, device=self.device)[None]
        pl = self.net.projected_logits(logits, used) + torch.log(
            torch.where(tm > 0.5, torch.ones_like(tm), torch.zeros_like(tm)))
        return int(pl.argmax(-1).item()), mask

    @torch.no_grad()
    def _value(self, hmap, item):
        x = build_obs_tensor(hmap[None], np.asarray(item, np.int32)[None], self.cfg)
        _, v, _ = self.net(torch.as_tensor(x, device=self.device))
        return float(v.item())

    def _reward(self, item):
        l, w, h = item
        return 10.0 * (l * w * h) / self.cfg.bin_volume

    # -- tree operations ---------------------------------------------------
    def _place(self, node, idx, items):
        """EXPAND: virtually place lookahead item `idx` in `node`'s state."""
        item = items[idx]
        # block footprints of already-placed items that arrive AFTER this one
        blocked = [f for (j, f) in node.footprints if j > idx]
        hm = node.bin.blocked_hmap(blocked) if blocked else node.bin.hmap
        a, _ = self._policy_action(node.bin, item, hm)
        if a is None:
            return None
        b = node.bin.copy()
        x, y, l, w, h = b.decode(a, item, self.cfg.orientations)
        b.place(a, item, self.cfg.orientations)
        return _Node(idx, a, b, node.footprints + [(idx, (x, y, l, w))],
                     self._reward(item), node, None)

    def _default_policy(self, node, items, remaining, last_item):
        """Roll out the remaining items in arrival order; add V at the leaf."""
        b = node.bin.copy()
        foot = list(node.footprints)
        total = 0.0
        for idx in sorted(remaining):
            blocked = [f for (j, f) in foot if j > idx]
            hm = b.blocked_hmap(blocked) if blocked else b.hmap
            a, _ = self._policy_action(b, items[idx], hm)
            if a is None:
                return total                     # terminal: cannot place
            x, y, l, w, h = b.decode(a, items[idx], self.cfg.orientations)
            b.place(a, items[idx], self.cfg.orientations)
            foot.append((idx, (x, y, l, w)))
            total += self._reward(items[idx])
        if last_item is not None:
            total += self._value(b.hmap, last_item)
        return total

    def _best_child(self, node, c, qmin, qmax):
        best, best_v = None, -1e18
        span = max(qmax - qmin, 1e-8)
        for ch in node.children:
            q = (ch.Q - qmin) / span
            u = q + c * math.sqrt(2.0 * math.log(max(node.N, 1)) / max(ch.N, 1))
            if u > best_v:
                best, best_v = ch, u
        return best

    # -- public API --------------------------------------------------------
    def search(self, bin_, lookahead, last_item=None, n_sim=None):
        """Return the action for the *current* item (lookahead index 0)."""
        cfg = self.cfg
        n_sim = n_sim or cfg.mcts_simulations
        items = [tuple(int(v) for v in it) for it in lookahead]
        k = len(items)
        if k == 1:
            a, mask = self._policy_action(bin_, items[0], bin_.hmap)
            return a

        if cfg.mcts_last_item == "mean" or last_item is None:
            last_item = self.mean_item

        root = _Node(-1, None, bin_.copy(), [], 0.0, None, None)
        root.untried = list(range(k))
        qmin, qmax = 1e18, -1e18

        for _ in range(n_sim):
            node, path, remaining = root, [root], set(range(k))
            # --- TREE POLICY ---
            while True:
                expanded = False
                while node.untried:
                    idx = node.untried.pop(
                        int(np.random.randint(len(node.untried))))
                    child = self._place(node, idx, items)
                    if child is None:
                        continue          # this item cannot be placed here
                    child.untried = sorted(remaining - {idx})
                    node.children.append(child)
                    node = child
                    path.append(node)
                    remaining.discard(idx)
                    expanded = True
                    break
                if expanded or not node.children:
                    break                 # newly expanded, or a dead/terminal leaf
                node = self._best_child(node, cfg.mcts_c, qmin, qmax)
                path.append(node)
                remaining.discard(node.item_idx)

            # --- DEFAULT POLICY + accumulated tree reward ---
            g = sum(n.reward for n in path)
            g += self._default_policy(node, items, remaining, last_item)

            # --- BACKUP (max return) ---
            qmin, qmax = min(qmin, g), max(qmax, g)
            for n in path:
                n.N += 1
                n.Q = max(n.Q, g)

        # descend the best path until we reach the node for the current item
        node = root
        while node.item_idx != 0:
            if not node.children:
                a, _ = self._policy_action(bin_, items[0], bin_.hmap)
                return a
            node = self._best_child(node, 0.0, qmin, qmax)
        return node.action
