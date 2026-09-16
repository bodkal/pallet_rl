"""Prioritized experience replay (Schaul et al., ref [23])."""
import numpy as np


class PER:
    def __init__(self, cap, cfg, alpha=0.6, seed=0):
        self.cap, self.cfg, self.alpha = cap, cfg, alpha
        self.hm = np.zeros((cap, cfg.W, cfg.L), np.int16)
        self.buf = np.zeros((cap, cfg.b, 3), np.int8)
        self.pi = np.zeros((cap, cfg.A), np.float32)
        self.z = np.zeros(cap, np.float32)
        self.prio = np.zeros(cap, np.float32)
        self.n, self.ptr = 0, 0
        self.max_prio = 1.0
        self.rng = np.random.default_rng(seed)

    def add_many(self, hm, buf, pi, z):
        k = len(z)
        for i in range(k):
            p = self.ptr
            self.hm[p] = hm[i]; self.buf[p] = buf[i]
            self.pi[p] = pi[i]; self.z[p] = z[i]
            self.prio[p] = self.max_prio
            self.ptr = (self.ptr + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, k, beta=1.0):
        n = self.n
        k = min(k, n)
        pr = self.prio[:n] ** self.alpha
        pr = pr / pr.sum()
        idx = self.rng.choice(n, size=k, replace=True, p=pr)
        w = (n * pr[idx]) ** (-beta)
        w = (w / w.max()).astype(np.float32)
        return idx, w

    def update(self, idx, prios):
        prios = np.maximum(prios, 1e-6)
        self.prio[idx] = prios
        self.max_prio = max(self.max_prio, float(prios.max()))
