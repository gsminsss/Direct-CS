"""Direct CS vs three cultivated T with idle-memory noise, both routes under the same rule:
every qubit within its lifetime receives DEPOLARIZE1(p) per idle layer (measured-then-reset gaps
excluded), schedules retimed by the same LP (per-qubit order fixed).  Active-location ledger."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import json, math, os, sys
import numpy as np
SC = str(_V2) + "/"
os.chdir(SC + "gapdec"); exec(open("ledger2.py").read().split("rows = []")[0])
AL = json.load(open(SC + "idle/abort_lp.json"))
A3 = {10: 9.18e5, 4: 1.70e6}                      # idle-free A3 (300k stratified)
NOIDLE = {(r["p"], r["gap"]): r for r in json.load(open(SC + "escwork/final_compare.json"))}
def direct(p):
    f = SC + f"escwork/gen/idlex_{'5e4' if p == 5e-4 else '1e3'}/multigap.json"
    if not os.path.exists(f): return {}
    return {int(r["gap"]): r for r in json.load(open(f)) if abs(r["p"] - p) < 1e-12}
def t3(p, sched, target):
    r = load(p, 0, "notail0") if sched == "none" else json.load(open(f"runs/r_{p:.0e}_s{sched}_notail0.json".replace("e-04", "e-4").replace("e-03", "e-3")))
    if sched == "none": ab = AB[f"{p:.0e}_noidle"]
    else: ab = AL[f"{p:.0e}_{sched}"]
    hd = r["discards"] / r["shots"]; cshot = ab["rho"] * PATH
    cand = [(3 * cshot * att + SYN, g, e / k * 3, att) for g, k, e, att, res in curve(r, "XC") if e / k * 3 <= target]
    if not cand: return None
    C, g, L, att = min(cand)
    st, _ = storage_sync3(att, hd, ab["mean_abort_tick"], FULLT)
    return dict(C=C, gap=g, L=L, hd=hd, sync=C + st)
out = []
for p in (5e-4, 1e-3):
    d = direct(p)
    for g in (4, 10):
        if g not in d: continue
        r = d[g]; acc = r["operational_acceptance"]; CD = r["stage_level_active_location_proxy_per_operational_accept"]
        se = CD * math.sqrt(acc * (1 - acc) / r["shots"]) / acc
        base = NOIDLE[(p, g)]
        for k in (1.0, 1.6, 2.5):
            tgt = A3[g] * k * p ** 3
            row = dict(p=p, gap=g, A3_factor=k, acc_noidle=base["acceptance"], acc_idle=acc, C_D_noidle=base["C_Direct"], C_D_idle=CD, C_D_idle_se=se,
                       le2_harmful=r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"])
            for s in ("none", "lp", "orig"):
                x = t3(p, s, A3[g] * (1.0 if s == "none" else k) * p ** 3)
                if x: row[f"C3T_{s}"] = x["C"]; row[f"C3T_{s}_sync"] = x["sync"]; row[f"gap3T_{s}"] = x["gap"]
            row["ratio_noidle"] = base["C_Direct"] / row["C3T_none"]
            row["ratio_idle_lp"] = CD / row["C3T_lp"]; row["ratio_idle_lp_sync3"] = CD / row["C3T_lp_sync"]
            row["ratio_idle_vs_public_sched"] = CD / row["C3T_orig"]
            out.append(row)
            print(f"p={p:.0e} g={g:2d} k={k:3.1f} | Direct acc {base['acceptance']:.3f}->{acc:.3f}  C {base['C_Direct']:5.0f}->{CD:5.0f}±{se:3.0f} | "
                  f"3T none {row['C3T_none']:5.0f}  LP {row['C3T_lp']:5.0f}  orig {row['C3T_orig']:5.0f} | ratio {row['ratio_noidle']:.2f} -> {row['ratio_idle_lp']:.2f} "
                  f"(sync3 {row['ratio_idle_lp_sync3']:.2f}; vs orig-sched 3T {row['ratio_idle_vs_public_sched']:.2f}) le2={row['le2_harmful']}")
json.dump(out, open(SC + "idle/idle_compare.json", "w"), indent=1)
