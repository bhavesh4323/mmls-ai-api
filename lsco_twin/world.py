"""world.py - topology + ground-truth casualty arrivals and item demand."""
import numpy as np
from .config import Config, ITEMS, FACTORS, TRUE_MULT, LEVELS, I

class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        # ---------- topology ----------
        level, parent, name = ["WH", "DEP"], [-1, 0], ["WH", "DEPOT"]
        r3 = []
        for k in range(cfg.n_r3):
            r3.append(len(level)); level.append("R3"); parent.append(1); name.append(f"R3-{k+1:02d}")
        r2e = []
        for g in range(cfg.n_groups):
            r2e.append(len(level)); level.append("R2E"); parent.append(r3[g % cfg.n_r3]); name.append(f"R2E-{g+1:02d}")
        r1 = []
        for g in range(cfg.n_groups):
            for j in range(cfg.r1_per_group):
                r1.append(len(level)); level.append("R1"); parent.append(r2e[g]); name.append(f"R1-{g+1:02d}.{j+1}")
        self.level = np.array(level); self.parent = np.array(parent); self.name = name
        self.N = len(level)
        self.idx = {L: np.where(self.level == L)[0] for L in LEVELS}
        self.r1 = np.array(r1)
        # membership matrix: which R1 nodes sit under each node
        M = np.zeros((self.N, len(r1)))
        for k, r in enumerate(r1):
            n = r
            while n != -1:
                M[n, k] = 1; n = self.parent[n]
        self.M = M
        # ---------- casualty arrivals at Role 1 (hourly) ----------
        T, base = cfg.T, cfg.theatre_casualties_per_day / 24.0
        w = rng.dirichlet(np.full(len(r1), 20.0)) * len(r1)       # uneven axes of advance
        w /= w.sum()
        hrs = np.arange(T)
        diurnal = 1 + 0.25 * np.sin(2 * np.pi * (hrs % 24 - 6) / 24)
        surge = np.ones(T)
        surge[cfg.surge_start_h: cfg.surge_start_h + cfg.surge_len_h] = cfg.surge_mult
        profile = diurnal * surge
        profile /= profile.mean()                                  # keep the daily mean at the planning figure
        self.w = w
        self.base_arr1 = base * w                                  # planned per-hour rate per R1
        self.arr1 = rng.poisson(profile[:, None] * self.base_arr1[None, :]).astype(float)   # (T, nR1)
        # ---------- arrivals at every node (delayed by evacuation) ----------
        self.D = {L: int(round(cfg.evac_delay.get(L, 0) * cfg.evac_factor)) for L in ["R1", "R2E", "R3"]}
        self.arr = np.zeros((T, self.N))
        for L in ["R1", "R2E", "R3"]:
            for n in self.idx[L]:
                s = self.arr1[:, self.M[n] > 0].sum(axis=1)
                d = self.D[L]
                self.arr[d:, n] = s[: T - d] if d else s
        # planned (static) arrivals per hour per node
        self.plan_arr = self.M @ self.base_arr1
        # ---------- factor matrix and ground-truth demand ----------
        self.F = np.zeros((self.N, len(ITEMS)))
        for L, d in FACTORS.items():
            for it, v in d.items():
                self.F[self.idx[L], I[it]] = v
        self.true_mult = np.array([TRUE_MULT.get(it, 1.0) for it in ITEMS])
        mean = self.arr[:, :, None] * self.F[None, :, :] * self.true_mult[None, None, :]
        self.demand = rng.poisson(mean).astype(float)              # (T, N, I)
