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

# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client     = MongoClient(MONGO_URI)
db         = client["mmls_ai"]
inventory_col = db["inventory"]
audit_col     = db["audit_log"]

# Load XGBoost model
try:
    model = joblib.load("model.pkl")
    print("✅ XGBoost model loaded")
except:
    model = None
    print("⚠️ model.pkl not found")

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
            result.append({
                "item_id": item["item_id"], "item_name": item["item_name"],
                "hours_of_supply": hos, "hos_tier": tier,
                "current_stock": stock, "echelon": item.get("echelon","Role1")
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
            "item_id":                  item_id,
            "item_name":                str(data.get("item_name", item_id)),
            "category":                 str(data.get("category", "Other")),
            "echelon":                  str(data.get("echelon", "Role1")),
            "controlled_drug":          bool(data.get("controlled_drug", False)),
            "current_stock":            int(data.get("current_stock", 0)),
            "initial_stock":            int(data.get("current_stock", 0)),
            "alert_threshold":          int(data.get("alert_threshold", 5)),
            "avg_consumption_per_hour": float(data.get("avg_consumption_per_hour", 1.0)),
            "expiry_date":              datetime.now() + timedelta(days=expiry_days),
            "pack_type":                str(data.get("pack_type", "Standalone")),
            "last_updated":             datetime.now(),
            "registered_via":           "QR_SCAN",
        }
        inventory_col.insert_one(doc)
        return jsonify({"message": f"{item_id} registered successfully", "item_id": item_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "MMLS-AI API running", "version": "2.0"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)