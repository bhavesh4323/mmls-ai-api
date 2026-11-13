"""
app.py — MMLS-AI Flask API (Render.com deployment)
Reads MONGO_URI from environment variable
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.errors import InvalidId
from lsco_twin.api import twin_bp

# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client     = MongoClient(MONGO_URI)
db         = client["mmls_ai"]
inventory_col = db["inventory"]
audit_col     = db["audit_log"]
resupply_col  = db["resupply_requests"]

def _inventory_collections():
    return inventory_col, audit_col


app.config["MMLS_AI_COLLECTIONS"] = _inventory_collections
app.register_blueprint(twin_bp)

# Load XGBoost model
try:
    model = joblib.load("model.pkl")
    print("XGBoost model loaded")
except Exception as exc:
    model = None
    print(f"XGBoost model unavailable; using baseline predictor: {exc}")

# ── Hours of Supply Engine ────────────────────────────────────────────────────
def get_hos_tier(hours):
    if hours == 999 or hours >= 24: return "GREEN"
    if hours >= 12: return "TEAL"
    if hours >= 6:  return "AMBER"
    if hours >= 2:  return "RED"
    return "PURPLE"

def compute_hos(stock, rate):
    if rate <= 0: return 999, "GREEN"
    hours = round(stock / rate, 2)
    return hours, get_hos_tier(hours)

# ── ENDPOINT 1 — GET /inventory ───────────────────────────────────────────────
@app.route("/inventory", methods=["GET"])
def get_inventory():
    try:
        items = list(inventory_col.find({}, {"_id": 0}))
        now   = datetime.now()
        for item in items:
            rate  = item.get("avg_consumption_per_hour", 1.0)
            stock = item.get("current_stock", 0)
            hos, tier = compute_hos(stock, rate)
            item["hours_of_supply"] = hos
            item["hos_tier"]        = tier
            item["alert"]           = stock <= item.get("alert_threshold", 5)
            exp = item.get("expiry_date")
            if exp:
                if isinstance(exp, str): exp = datetime.fromisoformat(exp)
                item["expiry_warning"] = (exp - now).days <= 30
                item["expiry_date"]    = exp.strftime("%Y-%m-%d")
            else:
                item["expiry_warning"] = False
        return jsonify(items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 2 — POST /scan ───────────────────────────────────────────────────
@app.route("/scan", methods=["POST"])
def scan_item():
    try:
        data     = request.json
        item_id  = data.get("item_id", "")
        qty      = int(data.get("quantity_used", 1))
        user     = data.get("user", "unknown")
        item     = inventory_col.find_one({"item_id": item_id})
        if not item:
            return jsonify({"error": f"{item_id} not found"}), 404
        new_stock = max(0, item["current_stock"] - qty)
        inventory_col.update_one({"item_id": item_id},
            {"$set": {"current_stock": new_stock, "last_updated": datetime.now()}})
        alert = new_stock <= item.get("alert_threshold", 5)
        audit_col.insert_one({
            "event_type": "SCAN", "item_id": item_id,
            "item_name": item["item_name"], "qty_used": qty,
            "timestamp": datetime.now(), "user": user,
            "controlled": item.get("controlled_drug", False)
        })
        return jsonify({
            "item_id": item_id, "item_name": item["item_name"],
            "new_stock": new_stock, "alert": alert,
            "qty_used": qty, "message": "OK"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 3 — POST /clinical_event ────────────────────────────────────────
@app.route("/clinical_event", methods=["POST"])
def clinical_event():
    try:
        data  = request.json
        etype = data.get("event_type","")
        user  = data.get("user","unknown")
        EVENT_MAP = {
            "GSW":        [("ITM004",2),("ITM006",1),("ITM009",2),("ITM005",1)],
            "BLAST":      [("ITM004",3),("ITM006",2),("ITM008",1),("ITM010",1)],
            "HAEMORRHAGE":[("ITM004",2),("ITM006",1),("ITM007",1)],
            "INTUBATION": [("ITM010",1),("ITM009",1),("ITM012",1)],
            "BURN_MINOR":  [("ITM016",2),("ITM015",1)],
            "BURN_MAJOR":  [("ITM016",4),("ITM015",2),("ITM008",1)],
            "FRACTURE":    [("ITM018",1),("ITM015",1),("ITM009",1)],
            "CARDIAC":     [("ITM008",1),("ITM009",2),("ITM013",1)],
            "ANAPHYLAXIS": [("ITM008",1),("ITM013",1)],
            "CRUSH":       [("ITM004",2),("ITM008",1),("ITM009",2)],
            "AMPUTATION":  [("ITM004",3),("ITM006",2),("ITM007",1)],
            "CHEST_TRAUMA":[("ITM005",1),("ITM011",1),("ITM009",1)],
            "TBI":         [("ITM008",1),("ITM009",2),("ITM013",1)],
            "CBRN":        [("ITM015",2),("ITM009",2),("ITM008",1)],
            "POLYTRAUMA":  [("ITM004",3),("ITM006",2),("ITM007",1),("ITM008",1)],
        }
        items_used = EVENT_MAP.get(etype, [])
        results = []
        for item_id, qty in items_used:
            item = inventory_col.find_one({"item_id": item_id})
            if item:
                new_stock = max(0, item["current_stock"] - qty)
                inventory_col.update_one({"item_id": item_id},
                    {"$set": {"current_stock": new_stock, "last_updated": datetime.now()}})
                audit_col.insert_one({
                    "event_type": etype, "item_id": item_id,
                    "item_name": item["item_name"], "qty_used": qty,
                    "timestamp": datetime.now(), "user": user,
                    "controlled": item.get("controlled_drug", False)
                })
                results.append({"item_id": item_id, "qty_used": qty, "new_stock": new_stock})
        return jsonify({"event": etype, "items_updated": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 4 — GET /forecast ────────────────────────────────────────────────
@app.route("/forecast", methods=["GET"])
def forecast():
    try:
        tempo = float(request.args.get("operational_tempo", 0.5))
        t1    = float(request.args.get("t1", 0.3))
        t2    = float(request.args.get("t2", 0.2))
        t3    = float(request.args.get("t3", 0.1))
        items = list(inventory_col.find({}, {"_id": 0}))
        predictions = []
        ECHELON_MAP = {"Role1": 0, "Role2": 1, "Role3": 2}
        PACK_MAP    = {"Role1_Pack": 0, "Role2_Pack": 1, "Burns_Pack": 2,
                       "CBRN_Pack": 3, "Standalone": 4}
        for item in items:
            enc = list(range(len(items)))
            item_enc = enc[items.index(item)] if item in items else 0
            ech_enc  = ECHELON_MAP.get(item.get("echelon","Role1"), 0)
            pack_enc = PACK_MAP.get(item.get("pack_type","Standalone"), 4)
            base_rate = item.get("avg_consumption_per_hour", 1.0)
            stock     = item.get("current_stock", 0)
            exp_days  = 365
            try:
                exp = item.get("expiry_date")
                if exp:
                    if isinstance(exp, str): exp = datetime.fromisoformat(exp)
                    exp_days = max(0, (exp - datetime.now()).days)
            except: pass
            features = np.array([[
                item_enc, ech_enc, tempo, t1, t2, t3,
                1 if item.get("controlled_drug") else 0,
                stock, base_rate, 25.0, 3, exp_days, pack_enc
            ]])
            if model:
                try:
                    predicted = float(model.predict(features)[0])
                except:
                    predicted = base_rate * (1 + tempo)
            else:
                predicted = base_rate * (1 + tempo)
            predicted = max(0.0, predicted)
            hos, tier = compute_hos(stock, predicted) if predicted > 0 else (999, "GREEN")
            predictions.append({
                "item_id": item["item_id"], "item_name": item["item_name"],
                "current_stock": stock, "predicted_rate": round(predicted, 4),
                "hours_of_supply": hos, "hos_tier": tier,
                "echelon": item.get("echelon", "Role1")
            })
        return jsonify({"predictions": predictions, "tempo": tempo, "t1": t1, "t2": t2, "t3": t3})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 5 — GET /hours_of_supply ────────────────────────────────────────
@app.route("/hours_of_supply", methods=["GET"])
def hours_of_supply():
    try:
        items = list(inventory_col.find({}, {"_id": 0}))
        result = []
        for item in items:
            rate  = item.get("avg_consumption_per_hour", 1.0)
            stock = item.get("current_stock", 0)
            hos, tier = compute_hos(stock, rate)
            echelon = item.get("echelon", "Role1")
            result.append({
                "item_id": item["item_id"], "item_name": item["item_name"],
                "hours_of_supply": hos, "hos_tier": tier,
                "current_stock": stock, "echelon": echelon,
                "reorder_point_h": REORDER_H.get(echelon, 12),
                "target_h": TARGET_H.get(echelon, 48),
                "needs_reorder": hos <= REORDER_H.get(echelon, 12),
                "suggested_qty": max(suggested_resupply_qty(item), 0),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 6 — GET /audit_log ──────────────────────────────────────────────
@app.route("/audit_log", methods=["GET"])
def audit_log():
    try:
        logs = list(audit_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(100))
        for l in logs:
            if "timestamp" in l and isinstance(l["timestamp"], datetime):
                l["timestamp"] = l["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 7 — POST /register_item ─────────────────────────────────────────
@app.route("/register_item", methods=["POST"])
def register_item():
    try:
        data    = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400
        item_id = str(data.get("item_id", "")).strip()
        if not item_id:
            return jsonify({"error": "item_id is required"}), 400
        if inventory_col.find_one({"item_id": item_id}):
            return jsonify({"error": f"{item_id} already exists"}), 409
        
        expiry_days = int(data.get("expiry_days", 365))
        
        doc = {
            # Basic info
            "item_id":                  item_id,
            "item_name":                str(data.get("item_name", item_id)),
            "category":                 str(data.get("category", "Other")),
            "echelon":                  str(data.get("echelon", "Role1")),
            "unit":                     str(data.get("unit", "ea")),
            
            # Stock & AI variables
            "controlled_drug":          bool(data.get("controlled_drug", False)),
            "current_stock":            int(data.get("current_stock", 0)),
            "initial_stock":            int(data.get("current_stock", 0)),
            "alert_threshold":          int(data.get("alert_threshold", 5)),
            "avg_consumption_per_hour": float(data.get("avg_consumption_per_hour", 1.0)),
            "expiry_date":              datetime.now() + timedelta(days=expiry_days),
            "pack_type":                str(data.get("pack_type", "Standalone")),
            
            # Medical specifications & NATO/DLA details
            "role1":                    str(data.get("role1", "—")),
            "role2":                    str(data.get("role2", "—")),
            "role2_plus":               str(data.get("role2_plus", "—")),
            "fst":                      str(data.get("fst", "—")),
            "role3":                    str(data.get("role3", "—")),
            "fsc":                      str(data.get("fsc", "—")),
            "contents":                 str(data.get("contents", "—")),
            "cotccc":                   str(data.get("cotccc", "—")),
            "sterile":                  str(data.get("sterile", "YES")),
            "notes":                    str(data.get("notes", "—")),
            "nsn":                      str(data.get("nsn", "—")),
            
            # Metadata
            "last_updated":             datetime.now(),
            "registered_via":           "QR_SCAN_OR_MANUAL",
        }
        
        inventory_col.insert_one(doc)
        return jsonify({"message": f"{item_id} registered successfully", "item_id": item_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 8 — POST /delete_items ──────────────────────────────────────────
@app.route("/delete_items", methods=["POST"])
def delete_items():
    try:
        data = request.json
        item_ids_to_delete = data.get("item_ids", [])
        
        if not item_ids_to_delete:
            return jsonify({"error": "No items provided"}), 400

        # Delete all documents matching the provided item IDs
        result = inventory_col.delete_many({"item_id": {"$in": item_ids_to_delete}})

        return jsonify({
            "message": "Items deleted successfully", 
            "deleted_count": result.deleted_count
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 9 — POST /reset ─────────────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset_database():
    try:
        # Wipe the entire inventory collection
        inv_result = inventory_col.delete_many({})
        # Wipe the audit log so you start with a completely clean slate
        audit_result = audit_col.delete_many({})
        resupply_result = resupply_col.delete_many({})
        
        return jsonify({
            "message": "Database wiped successfully",
            "inventory_deleted": inv_result.deleted_count,
            "audit_deleted": audit_result.deleted_count,
            "resupply_deleted": resupply_result.deleted_count
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 10 — Resupply requests (how a medic reorders) ───────────────────
# Mirrors the digital twin's reorder policy: a node orders when its hours of
# supply fall below the reorder point, and orders enough to reach its target
# hours of supply. The medic only presses a button; the quantity is computed
# here, a logistician (Role 3) approves, and approval credits the stock.
TARGET_H = {"Role1": 48, "Role2": 48, "Role3": 72}
REORDER_H = {"Role1": 12, "Role2": 12, "Role3": 36}

def suggested_resupply_qty(item, pending=0):
    """Units needed to lift `item` from current stock back to its target HoS.

    Zero or negative means the target is already covered (including by requests
    still awaiting approval).
    """
    echelon = item.get("echelon", "Role1")
    rate    = float(item.get("avg_consumption_per_hour", 0) or 0)
    stock   = float(item.get("current_stock", 0) or 0)
    target  = TARGET_H.get(echelon, 48) * rate
    return int(np.ceil(target - stock - pending))

@app.route("/resupply_request", methods=["POST"])
def resupply_request():
    try:
        data    = request.json or {}
        item_id = str(data.get("item_id", "")).strip()
        item    = inventory_col.find_one({"item_id": item_id})
        if not item:
            return jsonify({"error": f"{item_id} not found"}), 404

        pending = sum(r.get("qty", 0) for r in resupply_col.find(
            {"item_id": item_id, "status": "PENDING"}, {"qty": 1}))
        qty = data.get("qty")
        if qty is None:
            qty = suggested_resupply_qty(item, pending)
            if qty <= 0:
                return jsonify({
                    "error": "no resupply needed — current stock and pending requests already "
                             f"cover the {TARGET_H.get(item.get('echelon','Role1'), 48)}h target",
                    "suggested_qty": qty, "pending_qty": pending,
                }), 409
        qty = int(qty)
        if not 1 <= qty <= 100000:
            return jsonify({"error": "qty must be between 1 and 100000"}), 400

        echelon = item.get("echelon", "Role1")
        rate    = float(item.get("avg_consumption_per_hour", 0) or 0)
        stock   = float(item.get("current_stock", 0) or 0)
        hos, tier = compute_hos(stock, rate)
        doc = {
            "item_id":      item_id,
            "item_name":    item.get("item_name", item_id),
            "echelon":      echelon,
            "qty":          qty,
            "status":       "PENDING",
            "requested_by": str(data.get("requested_by", "unknown")),
            "reason":       str(data.get("reason", ""))[:300],
            "hos_at_request":   hos,
            "tier_at_request":  tier,
            "stock_at_request": stock,
            "created_at":   datetime.now(),
            "decided_by":   None,
            "decided_at":   None,
            "decision_note": "",
        }
        resupply_col.insert_one(doc)
        audit_col.insert_one({
            "event_type": "RESUPPLY_REQUEST", "item_id": item_id,
            "item_name": doc["item_name"], "qty_used": 0,
            "timestamp": doc["created_at"], "user": doc["requested_by"],
            "controlled": item.get("controlled_drug", False),
            "detail": f"requested {qty} units (HoS {hos}h, {tier})",
        })
        doc["_id"] = str(doc["_id"])
        return jsonify(doc), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/resupply_requests", methods=["GET"])
def resupply_requests():
    try:
        query = {}
        status = request.args.get("status")
        if status:
            query["status"] = status.upper()
        echelon = request.args.get("echelon")
        if echelon:
            query["echelon"] = echelon
        try:
            limit = min(max(int(request.args.get("limit", 100)), 1), 500)
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400
        rows = list(resupply_col.find(query).sort("created_at", -1).limit(limit))
        for row in rows:
            row["_id"] = str(row["_id"])
            for field in ("created_at", "decided_at"):
                if isinstance(row.get(field), datetime):
                    row[field] = row[field].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/resupply_request/<req_id>/decision", methods=["POST"])
def resupply_decision(req_id):
    try:
        data   = request.json or {}
        action = str(data.get("action", "")).lower()
        if action not in ("approve", "reject"):
            return jsonify({"error": "action must be approve or reject"}), 400
        try:
            oid = ObjectId(req_id)
        except (InvalidId, TypeError):
            return jsonify({"error": "invalid request id"}), 400
        doc = resupply_col.find_one({"_id": oid})
        if not doc:
            return jsonify({"error": "resupply request not found"}), 404
        if doc["status"] != "PENDING":
            return jsonify({"error": f"request is already {doc['status']}"}), 409

        decided_by = str(data.get("decided_by", "unknown"))
        new_stock  = None
        if action == "approve":
            item = inventory_col.find_one({"item_id": doc["item_id"]})
            if not item:
                return jsonify({"error": f"{doc['item_id']} no longer in inventory"}), 404
            new_stock = int(item.get("current_stock", 0)) + int(doc["qty"])
            inventory_col.update_one({"item_id": doc["item_id"]},
                {"$set": {"current_stock": new_stock, "last_updated": datetime.now()}})
        resupply_col.update_one({"_id": oid}, {"$set": {
            "status": "APPROVED" if action == "approve" else "REJECTED",
            "decided_by": decided_by,
            "decided_at": datetime.now(),
            "decision_note": str(data.get("note", ""))[:300],
        }})
        audit_col.insert_one({
            "event_type": "RESUPPLY_APPROVED" if action == "approve" else "RESUPPLY_REJECTED",
            "item_id": doc["item_id"], "item_name": doc["item_name"], "qty_used": 0,
            "timestamp": datetime.now(), "user": decided_by,
            "controlled": False,
            "detail": (f"approved {doc['qty']} units, stock now {new_stock}"
                       if action == "approve" else f"rejected {doc['qty']} units"),
        })
        return jsonify({"message": "Request approved" if action == "approve" else "Request rejected",
                        "item_id": doc["item_id"],
                        "qty": doc["qty"], "new_stock": new_stock}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "MMLS-AI API running", "version": "2.0"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)