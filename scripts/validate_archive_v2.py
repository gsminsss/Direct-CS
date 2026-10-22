#!/usr/bin/env python3
"""Consistency checks for the v2 (distance-3 interface) archive. Run from the repository root."""
from pathlib import Path
import json, math
ROOT = Path(__file__).resolve().parents[1]; V = ROOT / "v2"
J = lambda p: json.loads((V / p).read_text(encoding="utf-8"))

# detector / matrix contract of the distance-3 interface
s = J("escwork/gen/c1rep_g3w0/s21opt3c1_summary.json"); m = s["matrix"]
assert m["H_shape"] == [136, 1067] and m["H_rank"] == 136
assert m["H_aug_shape"] == [138, 1067] and m["H_aug_rank"] == 138
assert m["raw_effect_classes"] == 1071 and m["effect_columns"] == 1067
for r in s["evaluation_rows"]:
    assert r["downstream_locations_per_patch"] == 1436 and r["full_factory_potential_locations"] == 3762
    assert r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"] == 0

# ten-point sweep at both thresholds
sw = {(r["p"], int(r["gap"])): r for r in J("escwork/sweep_d3.json")}
exp = {(5e-4, 10): (0.527, 5227), (1e-3, 10): (0.2896, 7344), (5e-4, 4): (0.5448, 5056), (1e-3, 4): (0.3354, 6341)}
for k, (acc, cost) in exp.items():
    assert abs(sw[k]["acc"] - acc) < 6e-4 and abs(sw[k]["cost"] - cost) < 1, (k, sw[k]["acc"], sw[k]["cost"])
assert all(r["le2"] == 0 for r in sw.values())

# order-three coefficients (300,001 stratified samples, p_dec = 5e-4)
o3 = J("escwork/gen/a3main_g3w0_r2/multigap_order3.json")
A = lambda g: sum(r["A3_closed_boundary_binary_harmful_estimate"] for r in o3 if int(r["gap"]) == g)
assert math.isclose(A(10), 9.18e5, rel_tol=2e-3) and math.isclose(A(4), 1.7006e6, rel_tol=2e-3)

# threshold selection and final comparison
assert J("escwork/gap_selection.json")["selected"]["gap"] == 4
fc = {(r["p"], r["gap"]): r for r in J("escwork/final_compare.json")}
for (p, g), ratio in {(5e-4, 4): 0.979, (1e-3, 4): 1.276, (5e-4, 10): 1.012, (1e-3, 10): 1.418}.items():
    assert abs(fc[(p, g)]["ratio"] - ratio) < 2e-3, (p, g, fc[(p, g)]["ratio"])

# idle-memory noise (common rule, LP-retimed schedules)
ic = {(r["p"], r["gap"], r["A3_factor"]): r for r in J("idle/idle_compare.json")}
r5, r1 = ic[(5e-4, 4, 1.6)], ic[(1e-3, 4, 1.6)]
assert abs(r5["ratio_idle_lp"] - 1.075) < 0.01 and abs(r1["ratio_idle_lp"] - 1.462) < 0.01
assert r5["le2_harmful"] == 0 and r1["le2_harmful"] == 0
print("PASS: v2 archive contracts internally consistent")
print("136 operational / 168 validation bits; H 136x1067 rank 136; H_aug rank 138; 1,071 raw effect classes")
print("3,762 potential locations; A3_bin 9.18e5 (g=10), 1.70e6 (g*=4); g* = 4 selected on calibration data")
print("cost ratio vs three-T at g*: 0.98 (5e-4), 1.28 (1e-3); with idle noise 1.08, 1.46")
