"""simulator.py - hourly stock / shipment simulation across WH -> DEPOT -> R3 -> R2E -> R1.

Models the LSCO failure modes from BST PID v0.2 Part A / Digital Twin PDF s4-s6:
depot loss, cold-chain failure, blood spoilage (short shelf life), and the
Walking Blood Bank (WBB) with donor attrition and ABO/Rh grouping constraints
(Nye et al. 2014 benchmark).
"""
import numpy as np, pandas as pd
from .config import (Config, ITEMS, I, LEVELS, BLOOD_ITEMS,
                     BLOOD_GROUPS, DONOR_POOL_INIT, DONOR_ATTRITION_RATE,
                     WBB_COLLECTION_PER_DONOR, WBB_COLLECTION_INTERVAL_H)
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
        if cfg.wbb_units_per_day > 0 and t % WBB_COLLECTION_INTERVAL_H == 0:
            # Walking blood bank: compute actual WBB output considering attrition
            # and blood group compatibility.
            if cfg.use_donor_attrition:
                # Donor pool decays exponentially: N(t) = N0 * (1 - attrition)^(t/24)
                day_frac = t / 24.0
                active_donors = DONOR_POOL_INIT * ((1 - DONOR_ATTRITION_RATE) ** day_frac)
                # Casualties among donors: stochastic Poisson loss
                donor_loss = rng.poisson(DONOR_POOL_INIT * DONOR_ATTRITION_RATE * 0.3)
                active_donors = max(0, int(active_donors) - donor_loss)
            else:
                active_donors = int(cfg.wbb_units_per_day)  # legacy fixed inflow
            if cfg.use_blood_grouping:
                # Only ~93% of collected units are usable (AB- rare, group mismatch)
                # Universal donor (O-) fraction limits emergency distribution
                usable_frac = 1.0 - BLOOD_GROUPS.get("AB-", 0.01) - BLOOD_GROUPS.get("B-", 0.02)
                wbb_units = active_donors * WBB_COLLECTION_PER_DONOR * usable_frac
            else:
                wbb_units = active_donors * WBB_COLLECTION_PER_DONOR
            if wbb_units > 0:
                stock[w.idx["R2E"], blood_k] += wbb_units
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


def monte_carlo(cfg: Config, n_runs: int = None, policy: str = "dynamic") -> dict:
    """Nye Monte Carlo validation: run the simulation N times with different
    random seeds, collect key metrics, and return the distribution.

    Returns dict with:
      - runs: list of per-run metrics (fill_rate, first_stockout_day, blood_stockout_day)
      - summary: percentiles (p5, p25, median, p75, p95) for each metric
      - config_used: the scenario config dict for reproducibility
    """
    n = n_runs or cfg.mc_runs
    results = []
    for i in range(n):
        run_cfg = Config(**{k: v for k, v in cfg.__dict__.items() if k != 'seed'})
        run_cfg.seed = cfg.mc_seed_base + i
        res = run(run_cfg, policy=policy)
        summary = res.summary()
        blood_rows = summary[summary["item"].isin(BLOOD_ITEMS)] if not summary.empty else pd.DataFrame()
        blood_stockout = int(blood_rows["first_stockout_day"].dropna().min()) if not blood_rows.empty and blood_rows["first_stockout_day"].notna().any() else None
        overall_fill = float(summary["fill_rate_pct"].mean()) if not summary.empty else 0.0
        results.append({
            "run": i + 1,
            "overall_fill_rate_pct": round(overall_fill, 2),
            "first_stockout_day": int(summary["first_stockout_day"].dropna().min()) if not summary.empty and summary["first_stockout_day"].notna().any() else None,
            "blood_stockout_day": blood_stockout,
        })
    # Compute distribution summary
    fills = [r["overall_fill_rate_pct"] for r in results]
    stockouts = [r["first_stockout_day"] for r in results if r["first_stockout_day"] is not None]
    blood_sos = [r["blood_stockout_day"] for r in results if r["blood_stockout_day"] is not None]
    def percentiles(arr):
        if not arr: return {}
        a = np.array(arr)
        return {"p5": round(float(np.percentile(a, 5)), 1),
                "p25": round(float(np.percentile(a, 25)), 1),
                "median": round(float(np.median(a)), 1),
                "p75": round(float(np.percentile(a, 75)), 1),
                "p95": round(float(np.percentile(a, 95)), 1),
                "mean": round(float(np.mean(a)), 1)}
    summary_stats = {
        "fill_rate": percentiles(fills),
        "first_stockout_day": percentiles(stockouts),
        "blood_stockout_day": percentiles(blood_sos),
    }
    return {
        "classification": "SYNTHETIC PLANNING ESTIMATE — Monte Carlo validation",
        "n_runs": n,
        "policy": policy,
        "results": results,
        "summary": summary_stats,
        "config_used": {"theatre_casualties_per_day": cfg.theatre_casualties_per_day,
                        "days": cfg.days, "surge_mult": cfg.surge_mult,
                        "wbb_units_per_day": cfg.wbb_units_per_day,
                        "use_donor_attrition": cfg.use_donor_attrition,
                        "use_blood_grouping": cfg.use_blood_grouping},
    }
