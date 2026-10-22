"""Final d3-protocol comparison against the re-decoded 3T comparator, in active locations
and in qubit-ticks (spacetime), at gap 10 (frozen) and gap g* (selected on calibration data)."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import json, math, os, sys
SC = str(_V2) + "/"
os.chdir(SC + "gapdec"); exec(open("ledger2.py").read().split("rows = []")[0])
fin = json.load(open(SC + "escwork/gen/c1rep_g3w0/multigap.json"))
o3 = json.load(open(SC + "escwork/gen/a3main_g3w0_r2/multigap_order3.json"))
reach = json.load(open(SC + "escwork/reach_fractions.json"))
st = json.load(open(SC + "spacetime_abort.json"))
GSTAR = json.load(open(SC + "escwork/gap_selection.json"))["selected"]["gap"]
Q_FRONT, Q_CORE, Q_DOWN = 356, 2040, 2950          # qubit-ticks: front, core module, one d3 downstream patch
Q_ROUND, Q_CONV = 337 * 8, 3 * 337 * 4               # d13 patch-round; 3 patches x 4 ticks for 3 CX + M
def A3(g):
    rows = [r for r in o3 if r["gap"] == g]
    return sum(r["A3_closed_boundary_binary_harmful_estimate"] for r in rows), math.sqrt(sum(r["A3_closed_boundary_binary_harmful_se"] ** 2 for r in rows))
out = []
for g in (10, GSTAR):
    a3, a3se = A3(g)
    for p in (5e-4, 8e-4, 9e-4, 1e-3):
        f = [r for r in fin if abs(r["p"] - p) < 1e-12 and r["gap"] == g]
        if not f: continue
        f = f[0]; acc = f["operational_acceptance"]; CD = f["stage_level_active_location_proxy_per_operational_accept"]
        CD_se = CD * math.sqrt(acc * (1 - acc) / f["shots"]) / acc
        ab = AB[f"{p:.0e}_noidle"]; r = load(p, 0, "notail0"); hd = r["discards"] / r["shots"]; cshot = ab["rho"] * PATH
        cand = [(3 * cshot * att + SYN, gg, att) for gg, k, e, att, res in curve(r, "XC") if e / k * 3 <= a3 * p ** 3]
        C3, g3, att = min(cand)
        sync_loc, ET = storage_sync3(att, hd, ab["mean_abort_tick"], FULLT)
        rr = reach[str(p)]
        QD = (Q_FRONT + rr["front"] * Q_CORE + rr["core"] * 2 * Q_DOWN) / acc
        Q3 = 3 * st[str(p)]["qticks_per_shot"] * att + Q_CONV
        Q3s = Q3 + sync_loc / ROUND * Q_ROUND
        be_rounds_loc = (CD - C3) / ROUND; be_rounds_q = (QD - Q3) / Q_ROUND
        QD_rect = (Q_FRONT + rr["front"] * Q_CORE + rr["core"] * 2 * 3752) / acc
        N0 = 1.0 / (1.0 - hd)
        Q3_pub = 3 * 3690 * att / N0 + (5 * 2 * 169 + 3 * 169)   # lower bracket: rejected attempts free
        row = dict(sync_storage=sync_loc, pipeline_k1=3 * ET / TPR * ROUND, QD_rect=QD_rect, Q3_pub=Q3_pub, ratio_Q_worst=QD_rect / Q3_pub, gap=g, p=p, A3_bin=a3, A3_se=a3se, LER_target=a3 * p ** 3, acceptance=acc, C_Direct=CD, C_Direct_se=CD_se,
                   gap3T=g3, C_3T=C3, ratio=CD / C3, ratio_sync3=CD / (C3 + sync_loc), breakeven_rounds=be_rounds_loc,
                   Q_Direct=QD, Q_3T=Q3, ratio_Q=QD / Q3, ratio_Q_sync3=QD / Q3s, breakeven_rounds_Q=be_rounds_q,
                   harmful_le2=f["sampled_closed_boundary_harmful_with_total_fault_locations_le2"])
        out.append(row)
        print(f"g={g:>2} p={p:.0e} acc={acc:.3f} C_D={CD:6.0f}±{CD_se:3.0f} C_3T={C3:5.0f}(gap{g3:2d}) ratio={CD/C3:.3f} sync3={row['ratio_sync3']:.3f} BE={be_rounds_loc:5.2f}r | "
              f"Q_D={QD:6.0f} Q_3T={Q3:6.0f} ratioQ={QD/Q3:.3f} sync3Q={row['ratio_Q_sync3']:.3f} BE_Q={be_rounds_q:5.2f}r | worst-conv ratioQ={QD_rect/Q3_pub:.3f} | A3={a3:.3g}")
json.dump(out, open(SC + "escwork/final_compare.json", "w"), indent=1)
