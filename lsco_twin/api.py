"""Digital-twin routes for the MMLS-AI Flask application."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import math

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from lsco_twin.config import Config, FACTORS, ITEMS, TRIAGE_MIX
from lsco_twin.predictor import C_MAX, C_MIN
from lsco_twin.simulator import TIER_ACTIONS, hos_tier, run

twin_bp = Blueprint("twin", __name__, url_prefix="/twin")
ALLOWED = {
    "theatre_casualties_per_day": float,
    "days": int,
    "surge_mult": float,
    "surge_start_h": int,
    "evac_factor": float,
    "lift_factor": float,
    "depot_loss_h": int,
    "cold_chain_loss_h": int,
    "blood_spoil_h": float,
    "wbb_units_per_day": float,
    "order_fail_p": float,
    "wh_init_days": float,
    "seed": int,
}


def _cfg_from(args):
    values = {key: cast(args[key]) for key, cast in ALLOWED.items() if key in args}
    cfg = Config(**values)
    if not 100 <= cfg.theatre_casualties_per_day <= 10000:
        raise ValueError("theatre_casualties_per_day must be 100-10000")
    if not 1 <= cfg.days <= 60:
        raise ValueError("days must be 1-60")
    if not 1 <= cfg.surge_mult <= 10:
        raise ValueError("surge_mult must be 1-10")
    if not 0 <= cfg.surge_start_h <= cfg.T:
        raise ValueError("surge_start_h must be within the simulation")
    if not 0 <= cfg.evac_factor <= 5:
        raise ValueError("evac_factor must be 0-5")
    if not 0 <= cfg.lift_factor <= 5:
        raise ValueError("lift_factor must be 0-5")
    if not -1 <= cfg.depot_loss_h < cfg.T:
        raise ValueError("depot_loss_h must be -1 or within the simulation")
    if not -1 <= cfg.cold_chain_loss_h < cfg.T:
        raise ValueError("cold_chain_loss_h must be -1 or within the simulation")
    if not 0 <= cfg.blood_spoil_h <= 0.5:
        raise ValueError("blood_spoil_h must be 0-0.5 (fractional hourly loss)")
    if not 0 <= cfg.wbb_units_per_day <= 200:
        raise ValueError("wbb_units_per_day must be 0-200 units per Role 2E node")
    if not 0 <= cfg.order_fail_p <= 1:
        raise ValueError("order_fail_p must be 0-1")
    if not 0 <= cfg.wh_init_days <= 365:
        raise ValueError("wh_init_days must be 0-365")
    if not 0 <= cfg.seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("numeric parameters must be finite")
    return cfg, tuple(sorted(values.items()))


@lru_cache(maxsize=32)
def _run_cached(key, policy):
    return run(Config(**dict(key)), policy)


def _parse_timestamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _live_predictions(items, events, now, horizon_h):
    """Forecast hourly consumption with bounded short/long EWMA rates."""
    start = now - timedelta(hours=72)
    by_item = {}
    for event in events:
        item_id = event.get("item_id")
        timestamp = _parse_timestamp(event.get("timestamp"))
        if not item_id or timestamp is None or not start <= timestamp <= now:
            continue
        try:
            quantity = float(event.get("qty_used", 0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(quantity) or quantity <= 0:
            continue
        age_h = (now - timestamp).total_seconds() / 3600
        hour = min(71, max(0, int(age_h)))
        by_item.setdefault(item_id, np.zeros(72))[71 - hour] += quantity

    rows = []
    for item in items:
        item_id = item.get("item_id")
        if not item_id:
            continue
        try:
            baseline = float(item.get("avg_consumption_per_hour", 0))
            stock = float(item.get("current_stock", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid stock or consumption rate for item {item_id}") from exc
        if not math.isfinite(baseline) or not math.isfinite(stock) or baseline < 0 or stock < 0:
            raise ValueError(f"invalid stock or consumption rate for item {item_id}")

        hourly = by_item.get(item_id, np.zeros(72))
        short = baseline
        long = baseline
        if hourly.sum() > 0:
            for observed in hourly:
                short += 0.30 * (observed - short)
                long += 0.02 * (observed - long)
        if baseline > 0 and hourly.sum() > 0:
            correction = np.clip(max(short, long) / baseline, C_MIN, C_MAX)
            predicted_rate = baseline * float(correction)
        elif baseline > 0:
            predicted_rate = baseline
        else:
            predicted_rate = max(short, long)
        supply_hours = stock / predicted_rate if predicted_rate > 0 else 999.0
        rows.append(
            {
                "item_id": item_id,
                "item_name": item.get("item_name", item_id),
                "current_stock": stock,
                "baseline_rate": round(baseline, 4),
                "predicted_rate": round(predicted_rate, 4),
                "expected_qty": round(predicted_rate * horizon_h, 2),
                "horizon_h": horizon_h,
                "hours_of_supply": round(min(supply_hours, 999), 1),
                "hos_tier": hos_tier(supply_hours),
                "observed_consumption_72h": round(float(hourly.sum()), 2),
                "predictor": "dynamic-ema-v1",
                "source": "recent_audit_activity" if hourly.sum() else "inventory_baseline",
            }
        )
    return sorted(rows, key=lambda row: row["hours_of_supply"])


@twin_bp.get("/simulate")
def simulate():
    try:
        cfg, key = _cfg_from(request.args)
        results = {}
        for policy in ("static", "dynamic"):
            result = _run_cached(key, policy)
            results[policy] = {
                "summary": result.summary().to_dict("records"),
                "by_level": result.by_level().to_dict("records"),
            }
        # Planned daily consumption per item at the configured casualty rate.
        # At 2,000/day this reproduces the PID s4.2 table (1,660 packs, 1,440
        # anaesthetic sets, 4,000 blood units, 1,000 haem/antibiotic, 2,000+ analgesia/dressings).
        per_casualty = {
            item: sum(level_factors.get(item, 0.0) for level_factors in FACTORS.values())
            for item in ITEMS
        }
        daily_demand = {
            item: round(factor * cfg.theatre_casualties_per_day, 1)
            for item, factor in per_casualty.items()
        }
        return jsonify(
            {
                "classification": "synthetic scenario result; not real-demand evidence",
                "assumptions": {
                    "theatre_casualties_per_day": cfg.theatre_casualties_per_day,
                    "days": cfg.days,
                    "triage_mix": TRIAGE_MIX,
                },
                "planning_basis": "BST PID v0.2 s4.2 / Digital Twin PDF s2 casualty-to-consumption placeholders",
                "daily_demand_at_casualty_rate": daily_demand,
                "tier_actions": TIER_ACTIONS,
                **results,
            }
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400


@twin_bp.get("/hos")
def hos():
    try:
        cfg, key = _cfg_from(request.args)
        result = _run_cached(key, "dynamic")
        try:
            hour = int(request.args.get("t", 24))
            limit = int(request.args.get("limit", 100))
        except ValueError:
            raise ValueError("t and limit must be integers") from None
        level = request.args.get("level")
        if not 0 <= hour < cfg.T:
            raise ValueError(f"t must be between 0 and {cfg.T - 1}")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if level is not None and level not in {"R1", "R2E", "R3"}:
            raise ValueError("level must be R1, R2E, or R3")
        snapshot = result.hos_snapshot(hour, level)
        return jsonify(snapshot.sort_values("hos").head(limit).to_dict("records"))
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400


@twin_bp.get("/live-forecasts")
def live_forecasts():
    try:
        horizon_h = int(request.args.get("horizon_h", 24))
        if not 1 <= horizon_h <= 168:
            raise ValueError("horizon_h must be between 1 and 168")
        collection_provider = current_app.config.get("MMLS_AI_COLLECTIONS")
        if not callable(collection_provider):
            return jsonify({"error": "inventory and audit collections are not configured"}), 503
        inventory_col, audit_col = collection_provider()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        items = list(inventory_col.find({}, {"_id": 0}))
        events = list(
            audit_col.find(
                {"timestamp": {"$gte": now - timedelta(hours=72)}},
                {"_id": 0, "item_id": 1, "qty_used": 1, "timestamp": 1},
            ).limit(5000)
        )
        predictions = _live_predictions(items, events, now, horizon_h)
        return jsonify(
            {
                "generated_at": now.isoformat(),
                "window_h": 72,
                "classification": "live estimate from inventory baselines and recent usage audit events",
                "predictions": predictions,
            }
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400
