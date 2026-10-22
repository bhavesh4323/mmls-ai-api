"""simulator.py - hourly stock / shipment simulation across WH -> DEPOT -> R3 -> R2E -> R1.

Models the LSCO failure modes from BST PID v0.2 Part A / Digital Twin PDF s4-s6:
depot loss, cold-chain failure, blood spoilage (short shelf life) and optional
walking blood bank inflow at Role 2E.
"""
import numpy as np, pandas as pd
from .config import Config, ITEMS, I, LEVELS, BLOOD_ITEMS
from .world import World
from .predictor import CrostonPolicy, DynamicPolicy, SeasonalNaivePolicy, StaticPolicy

PARENT_LINK = {"R1": "R1", "R2E": "R2E", "R3": "R3", "DEP": "DEP"}   # delay/loss keyed by child level

# Hours-of-Supply tiers with the system action each triggers (PID s9.3).
TIER_ACTIONS = {
    "GREEN": "Monitor (full readiness)",
    "TEAL":  "Draft resupply for logistician review (12-24h)",
    "AMBER": "Escalate to FST commander / MO (6-12h)",
    "RED":   "Emergency resupply; Role 3 CO alerted (2-6h)",
    "PURPLE":"Commander alert; lateral redistribution (<2h)",
}

def hos_tier(h):
    if h >= 24: return "GREEN"
    if h >= 12: return "TEAL"
    if h >= 6:  return "AMBER"
    if h >= 2:  return "RED"
    return "PURPLE"

def run(cfg: Config, policy: str = "dynamic", world: World = None):
    w = world or World(cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    T, N, K = cfg.T, w.N, len(ITEMS)
    policies = {
        "static": StaticPolicy,
        "dynamic": DynamicPolicy,
        "seasonal_naive": SeasonalNaivePolicy,
        "croston_sba": CrostonPolicy,
    }
    if policy not in policies:
        raise ValueError(f"unknown policy: {policy}")
    pol = policies[policy](w, cfg)
    planned = StaticPolicy(w, cfg).tot                                   # planned throughput per hour
    blood_k = [I[it] for it in ITEMS if it in BLOOD_ITEMS]
    fwd = np.concatenate([w.idx["R1"], w.idx["R2E"]])                    # forward cold storage (PID s6.2)
    # ---- initial stock ----
    stock = np.zeros((N, K))
    for L in ["R1", "R2E", "R3", "DEP"]:
        stock[w.idx[L]] = cfg.init_h[L] * planned[w.idx[L]]
    inflow = planned[1] * 24                                             # planned daily theatre demand per item
    frac = np.array([cfg.wh_inflow_frac.get(it, 1.0) for it in ITEMS])
    stock[0] = cfg.wh_init_days * inflow
    wh_in_h = inflow * frac / 24.0
    delay = {L: int(round(cfg.link_delay[L] * cfg.lift_factor)) for L in cfg.link_delay}
    maxd = max(delay.values()) + 2
    pipe, pipe_nom = np.zeros((T + maxd, N, K)), np.zeros((T + maxd, N, K))
    transit = np.zeros((N, K))
    unmet, stock_hist = np.zeros((T, N, K)), np.zeros((T, N, K), dtype=np.float32)
    rate_hist = np.zeros((T, N, K), dtype=np.float32); rate = planned.copy()
    for t in range(T):
        stock += pipe[t]; transit -= pipe_nom[t]; transit = np.maximum(transit, 0)
        stock[0] += wh_in_h
        if t == cfg.depot_loss_h: stock[1] = 0.0                         # depot struck
        if cfg.wbb_units_per_day > 0 and t % 24 == 0:
            stock[w.idx["R2E"], blood_k] += cfg.wbb_units_per_day         # walking blood bank (Nye benchmark)
        if blood_k and cfg.blood_spoil_h > 0:
            stock[np.ix_(fwd, blood_k)] *= (1.0 - cfg.blood_spoil_h)      # short shelf life wastage
        if t == cfg.cold_chain_loss_h and blood_k:
            stock[np.ix_(w.idx["R2E"], blood_k)] = 0.0                    # forward cold storage fails
        d = w.demand[t]; used = np.minimum(stock, d); stock -= used; unmet[t] = d - used
        pol.observe(t, d)
        if t % cfg.order_cadence_h == 0:
            rate = pol.rate(t)
            for L in ["R1", "R2E", "R3", "DEP"]:
                idx = w.idx[L]
                r = rate[idx]
                pos = stock[idx] + transit[idx]
                need = np.where((pos < cfg.reorder_h[L] * r) & (r > 0), cfg.target_h[L] * r - pos, 0.0)
                need = np.maximum(need, 0)
                if cfg.order_fail_p > 0:                                  # DDIL: some nodes cannot transmit
                    need[rng.random(len(idx)) < cfg.order_fail_p] = 0
                par = w.parent[idx]
                req = np.zeros((N, K)); np.add.at(req, par, need)
                ratio = np.where(req > 0, np.minimum(1.0, stock / np.maximum(req, 1e-9)), 0.0)
                ship = need * ratio[par]
                np.add.at(stock, par, -ship)
                lost = rng.random(len(idx)) < cfg.link_loss_p[L] * cfg.lift_factor
                arrive = ship * (~lost)[:, None]
                ta = t + delay[L]
                pipe[ta, idx] += arrive; pipe_nom[ta, idx] += ship; transit[idx] += ship
        stock_hist[t] = stock; rate_hist[t] = rate
    return Result(cfg, w, policy, unmet, stock_hist, rate_hist)

class Result:
    def __init__(self, cfg, w, policy, unmet, stock, rate):
        self.cfg, self.w, self.policy, self.unmet, self.stock, self.rate = cfg, w, policy, unmet, stock, rate
    def summary(self) -> pd.DataFrame:
        """Fill rate and first stock-out day per item (consuming nodes only)."""
        w, rows = self.w, []
        cons = np.concatenate([w.idx["R1"], w.idx["R2E"], w.idx["R3"]])
        for k, it in enumerate(ITEMS):
            dem, um = w.demand[:, cons, k], self.unmet[:, cons, k]
            if dem.sum() == 0: continue
            daily_um = um.sum(axis=1).reshape(-1, 24).sum(axis=1)
            first = int(np.argmax(daily_um > 0)) + 1 if (daily_um > 0).any() else None
            rows.append(dict(item=it, demand=int(dem.sum()), unmet=int(um.sum()),
                             fill_rate_pct=round(100 * (1 - um.sum() / dem.sum()), 2), first_stockout_day=first))
        return pd.DataFrame(rows)
    def by_level(self) -> pd.DataFrame:
        w, rows = self.w, []
        for L in ["R1", "R2E", "R3"]:
            idx = w.idx[L]
            dem, um = w.demand[:, idx, :].sum(), self.unmet[:, idx, :].sum()
            rows.append(dict(level=L, fill_rate_pct=round(100 * (1 - um / max(dem, 1)), 2)))
        return pd.DataFrame(rows)
    def hos_snapshot(self, t: int, level: str = None) -> pd.DataFrame:
        """Hours-of-Supply per node/item at hour t, using the policy's forecast rate."""
        w, rows = self.w, []
        for n in range(2, w.N):
            if level and w.level[n] != level: continue
            for k, it in enumerate(ITEMS):
                r = self.rate[t, n, k]
                if r <= 0: continue
                h = float(self.stock[t, n, k] / r)
                rows.append(dict(node=w.name[n], level=w.level[n], item=it, stock=round(float(self.stock[t, n, k]), 1),
                                 hos=round(min(h, 999), 1), tier=hos_tier(h)))
        return pd.DataFrame(rows)
