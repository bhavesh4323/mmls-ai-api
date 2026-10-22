"""
predictor.py - two demand policies the simulator can run:
  StaticPolicy  : planning factors x planned casualty rate (what exists today)
  DynamicPolicy : live casualty feed + known evacuation pipeline + learned
                  per-item correction factors + surge detection
Both return an (N, I) array: forecast THROUGHPUT per hour, i.e. own consumption
plus everything the node must pass down to its children.
"""
import numpy as np
from .config import Config

C_MIN, C_MAX = 0.5, 3.0     # bounds on the learned correction factor

def roll_up(world, own):
    """Add each node's rate to its parent, bottom-up (R1 -> R2E -> R3 -> DEP)."""
    tot = own.copy()
    for L in ["R1", "R2E", "R3", "DEP"]:
        idx = world.idx[L]
        np.add.at(tot, world.parent[idx], tot[idx])
    return tot

class StaticPolicy:
    name = "static"
    def __init__(self, world, cfg):
        self.w = world
        self.own = world.plan_arr[:, None] * world.F            # planned, never updated
        self.tot = roll_up(world, self.own)
    def observe(self, t, demand): pass
    def rate(self, t): return self.tot


class SeasonalNaivePolicy:
    """Forecast each node/item from the same hour one week earlier."""
    name = "seasonal_naive"

    def __init__(self, world, cfg):
        self.w = world
        self.static = StaticPolicy(world, cfg)

    def observe(self, t, demand):
        pass

    def rate(self, t):
        previous_week = t - 168
        if previous_week < 0:
            return self.static.tot
        return roll_up(self.w, self.w.demand[previous_week])


class CrostonPolicy:
    """SBA Croston baseline for intermittent demand, updated online by node/item."""
    name = "croston_sba"

    def __init__(self, world, cfg, alpha=0.1):
        self.w = world
        self.alpha = alpha
        self.static = StaticPolicy(world, cfg)
        shape = self.static.own.shape
        self.size = np.zeros(shape)
        self.period = np.ones(shape)
        self.elapsed = np.zeros(shape)
        self.seen = np.zeros(shape, dtype=bool)
        self.own = self.static.own.copy()

    def observe(self, t, demand):
        values = np.asarray(demand)
        self.elapsed += 1
        nonzero = values > 0
        first = nonzero & ~self.seen
        repeated = nonzero & self.seen
        self.size[first] = values[first]
        self.period[first] = self.elapsed[first]
        self.size[repeated] = (
            (1 - self.alpha) * self.size[repeated] + self.alpha * values[repeated]
        )
        self.period[repeated] = (
            (1 - self.alpha) * self.period[repeated] + self.alpha * self.elapsed[repeated]
        )
        self.seen |= nonzero
        self.elapsed[nonzero] = 0
        estimate = self.size / np.maximum(self.period, 1) * (1 - self.alpha / 2)
        self.own = np.where(self.seen, estimate, self.static.own)

    def rate(self, t):
        return roll_up(self.w, self.own)


class DynamicPolicy:
    name = "dynamic"
    def __init__(self, world, cfg, a_short=0.30, a_long=0.02, decay=0.98, prior=5.0):
        self.w, self.cfg = world, cfg
        n1 = len(world.r1)
        self.short = world.base_arr1.copy(); self.long = world.base_arr1.copy()
        self.a_s, self.a_l, self.decay, self.prior = a_short, a_long, decay, prior
        self.num = np.zeros((world.N, world.F.shape[1])); self.den = np.zeros_like(self.num)
        self.c = np.ones_like(self.num)                          # learned correction factors
    def observe(self, t, demand):
        w = self.w
        a1 = w.arr1[t]                                           # casualty feed (Rapid-EPR / MCI)
        self.short += self.a_s * (a1 - self.short); self.long += self.a_l * (a1 - self.long)
        self.num = self.decay * self.num + demand               # observed consumption (incl. logged unmet)
        self.den = self.decay * self.den + w.arr[t][:, None] * w.F
        self.c = np.clip((self.num + self.prior) / (self.den + self.prior), C_MIN, C_MAX)   # fail-safe: never forecast ~zero demand because a feed is missing
    def rate(self, t):
        w, H = self.w, self.cfg.horizon_h
        lam1 = np.maximum(self.short, self.long)                 # surge-aware arrival rate per R1
        lam = w.M @ lam1                                         # per node (subtree sum)
        E = np.zeros(w.N)
        for L in ["R1", "R2E", "R3"]:
            idx, D = w.idx[L], w.D[L]
            k = min(D, H)
            known = w.arr[t + 1: t + 1 + k, :][:, idx].sum(axis=0) if k > 0 else 0.0   # casualties already en route
            E[idx] = known + (H - k) * lam[idx]
        own = self.c * w.F * (E / H)[:, None]
        return roll_up(w, own)