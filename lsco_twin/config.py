"""
config.py - every planning assumption lives here so it can be replaced with
clinical planning factors later.

Casualty-to-consumption factors are DERIVED from the triage mix and per-casualty
multipliers in BST_LSCO_Medical_Logistics_Digital_Twin.pdf Section 2 and
BST_Rapid-MMLS_Combined_Problem_Analysis_and_PID.pdf Section 4.2:
  - triage mix 20% T1 / 30% T2 / 50% T3 (placeholder, PID open question Q2)
  - surgical cases = all T1 + half of T2
  - blood = 10 units per T1 casualty (RUSI ratio; Digital Twin PDF ref [1])
  - haemorrhage control sets and antibiotic courses = one per T1+T2 casualty
  - analgesia doses and wound dressings = at least one per casualty

Planning factor validation sources (for replacement of placeholders):
  - NATO AMedP-13(B) Allied Joint Medical Logistics Doctrine
  - US Borden Institute JMALL Table 14-2 (Joint Medical Logistics Planning Factors)
  - UK DCDC Joint Force Health Action (JFHA) planning ratios
  - RUSI "Blood and Military Medicine" (2022) — 10 units per serious casualty
  - Nye B et al. "Walking Blood Bank in Afghanistan" (Mil Med, 2014)
  - BST PID v0.2 §4.2 (brigade annex × 20, current basis)

All numbers remain placeholders pending end-user validation (mentor action).
"""
from dataclasses import dataclass, field
import numpy as np

ITEMS = ["THEATRE_PACK", "ANAES_SET", "BLOOD_UNIT", "HAEM_SET",
         "ANTIBIOTIC", "ANALGESIA", "DRESSING"]
I = {name: k for k, name in enumerate(ITEMS)}
CONTROLLED = {"ANALGESIA"}            # morphine / ketamine / fentanyl lines
BLOOD_ITEMS = {"BLOOD_UNIT"}          # cold chain, short shelf life (PID s6)

# ---- Blood group distribution (UK/NATO military population approximations) ----
# Source: NHS Blood Transfusion statistics + military donor registries (young adult
# skew toward O+). Used by the Walking Blood Bank model to constrain cross-match.
BLOOD_GROUPS = {
    "O+": 0.35, "O-": 0.07, "A+": 0.32, "A-": 0.06,
    "B+": 0.09, "B-": 0.02, "AB+": 0.03, "AB-": 0.01,
}
# Universal-donor fraction (O- can give to all; O+ to all Rh+):
UNIVERSAL_DONOR_GROUPS = {"O-"}
RHEUS_POSITIVE = {"O+", "A+", "B+", "AB+"}  # ~85% of population

# ---- Walking Blood Bank (WBB) model parameters (Nye et al. 2014) ----
# Source: Nye B et al. "Walking blood bank: the experience at a deployed Role 2
# in Afghanistan." Military Medicine 179(7):365-370.
DONOR_POOL_INIT = 120          # pre-typed donors available per R2E
DONOR_ATTRITION_RATE = 0.08    # daily fractional loss (casualty, rotation, illness)
WBB_COLLECTION_PER_DONOR = 1   # units per active donor per collection cycle
WBB_COLLECTION_INTERVAL_H = 24 # hours between collection rounds

# ---- casualty-to-consumption, derived from triage (Digital Twin PDF s2) ----
TRIAGE_MIX = {"T1": 0.20, "T2": 0.30, "T3": 0.50}
SURGICAL_PER_CASUALTY = TRIAGE_MIX["T1"] + 0.5 * TRIAGE_MIX["T2"]   # 0.35 -> 700/day at 2,000
HAEM_AB_PER_CASUALTY = TRIAGE_MIX["T1"] + TRIAGE_MIX["T2"]          # 0.50 -> 1,000/day
BLOOD_PER_CASUALTY = 10.0 * TRIAGE_MIX["T1"]                        # 2.0  -> 4,000/day
BLOOD_SPLIT_R2E = 0.6          # 1.2/0.8 R2E/R3 split in the PID s4.2 table

# Units consumed per THEATRE casualty (counted at Role 1 entry), by echelon.
# Triage-derived rows match the Digital Twin PDF s2 multipliers exactly
# (haemorrhage/antibiotic 1 per T1+T2 = 1,000/day; analgesia and dressings
# >= 1 per casualty = 2,000/day; blood 10 per T1 = 4,000/day, split 60/40).
# Pack and anaesthetic rows use the PID s4.2 brigade-annex x 20 figures
# (R2E 660 + R3 1,000 packs/day; 600 + 840 anaesthetic sets/day), which scale
# linearly from the annex - itself an assumption, and higher than one pack per
# surgical case because Role 3 re-operations are included.
FACTORS = {
    "R1":  {"HAEM_SET": HAEM_AB_PER_CASUALTY, "ANTIBIOTIC": HAEM_AB_PER_CASUALTY,
            "ANALGESIA": 1.0, "DRESSING": 1.0},
    "R2E": {"THEATRE_PACK": 660.0 / 2000.0, "ANAES_SET": 600.0 / 2000.0,
            "BLOOD_UNIT": BLOOD_PER_CASUALTY * BLOOD_SPLIT_R2E},
    "R3":  {"THEATRE_PACK": 1000.0 / 2000.0, "ANAES_SET": 840.0 / 2000.0,
            "BLOOD_UNIT": BLOOD_PER_CASUALTY * (1 - BLOOD_SPLIT_R2E)},
}
# "Reality" the static plan does not know about (prolonged field care,
# higher blood use). The twin uses these to generate ground-truth demand.
TRUE_MULT = {"BLOOD_UNIT": 1.30, "ANALGESIA": 1.25, "DRESSING": 1.15,
             "HAEM_SET": 1.10, "THEATRE_PACK": 1.05}

LEVELS = ["WH", "DEP", "R3", "R2E", "R1"]

@dataclass
class Config:
    # ---- demand ----
    theatre_casualties_per_day: float = 2000.0
    days: int = 30
    surge_start_h: int = 72           # hour a mass-casualty surge begins
    surge_len_h: int = 6
    surge_mult: float = 2.5
    # ---- topology ----
    n_groups: int = 20                # brigade-equivalents (annex x 20)
    r1_per_group: int = 4
    n_r3: int = 10
    # ---- evacuation delays (hours, cumulative from R1) ----
    evac_delay: dict = field(default_factory=lambda: {"R1": 0, "R2E": 4, "R3": 16})
    evac_factor: float = 1.0          # >1 = slower evacuation (Ukraine-type)
    # ---- supply links (hours): child <- parent ----
    link_delay: dict = field(default_factory=lambda: {"R1": 4, "R2E": 8, "R3": 12, "DEP": 48})
    link_loss_p: dict = field(default_factory=lambda: {"R1": 0.05, "R2E": 0.03, "R3": 0.02, "DEP": 0.01})
    lift_factor: float = 1.0          # >1 = contested lift, slower resupply
    # ---- stock policy (hours of supply) ----
    init_h: dict = field(default_factory=lambda: {"R1": 48, "R2E": 48, "R3": 72, "DEP": 168})
    target_h: dict = field(default_factory=lambda: {"R1": 48, "R2E": 48, "R3": 72, "DEP": 120})
    # Forward nodes draft resupply before dropping below 12h of supply (PID s8).
    reorder_h: dict = field(default_factory=lambda: {"R1": 12, "R2E": 12, "R3": 36, "DEP": 72})
    order_cadence_h: int = 4          # 4-hourly node report (PID s9.1)
    horizon_h: int = 24               # predictor look-ahead
    # ---- national supply (PLACEHOLDERS) ----
    wh_init_days: float = 30.0
    wh_inflow_frac: dict = field(default_factory=lambda: {"BLOOD_UNIT": 0.5})  # of planned daily demand; default 1.0
    # ---- scenario switches ----
    depot_loss_h: int = -1            # hour the theatre depot is destroyed (-1 = never)
    cold_chain_loss_h: int = -1       # hour forward cold storage fails at R2E/R1 (-1 = never, PID s5/s6.2)
    blood_spoil_h: float = 0.005      # hourly fractional loss of forward blood stock (short shelf life)
    wbb_units_per_day: float = 0.0    # walking blood bank: fresh units/day added at each Role 2E (Nye benchmark)
    order_fail_p: float = 0.0         # DDIL: prob a node cannot send its order this cycle
    # ---- Monte Carlo validation (Nye model benchmark) ----
    mc_runs: int = 100                # number of stochastic iterations for validation
    mc_seed_base: int = 1000          # base seed; each run uses mc_seed_base + i
    # ---- Blood grouping & donor attrition ----
    use_blood_grouping: bool = False  # enable ABO/Rh compatibility constraint
    use_donor_attrition: bool = False # enable WBB donor pool depletion over time
    seed: int = 42

    @property
    def T(self): return self.days * 24
