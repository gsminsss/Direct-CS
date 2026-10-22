"""Apply the frozen gap-selection rule (GAP_POLICY_PLAN.md) to calibration data."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import json, math, sys
sys.path.insert(0, str(_V2) + "/gapdec")
SC = str(_V2) + "/"
src = open(SC + "gapdec/ledger2.py").read().split("rows = []")[0]
import os; os.chdir(SC + "gapdec"); exec(src)
fin = json.load(open(SC + "escwork/gen/c1cal_g3w0/multigap.json"))
o3 = json.load(open(SC + "escwork/gen/a3cal_g3w0/multigap_order3.json"))
def c3t(p, A3):
    ab = AB[f"{p:.0e}_noidle"]; r = load(p, 0, "notail0"); cshot = ab["rho"] * PATH
    cand = [(3 * cshot * att + SYN) for g, k, e, att, res in curve(r, "XC") if e / k * 3 <= A3 * p ** 3]
    return min(cand) if cand else float("nan")
out = []
for g in sorted({r["gap"] for r in o3}):
    rows = [r for r in o3 if r["gap"] == g]
    A3 = sum(r["A3_closed_boundary_binary_harmful_estimate"] for r in rows)
    se = math.sqrt(sum(r["A3_closed_boundary_binary_harmful_se"] ** 2 for r in rows))
    h = sum(r["closed_boundary_harmful_samples"] for r in rows)
    ratios = {}; le2 = 0
    for p in (5e-4, 1e-3):
        f = [r for r in fin if abs(r["p"] - p) < 1e-12 and r["gap"] == g][0]
        le2 += f["sampled_closed_boundary_harmful_with_total_fault_locations_le2"]
        ratios[p] = f["stage_level_active_location_proxy_per_operational_accept"] / c3t(p, A3)
    R = sum(ratios.values()) / 2
    out.append(dict(gap=g, A3_bin=A3, A3_se=se, harmful=h, ratio_5e4=ratios[5e-4], ratio_1e3=ratios[1e-3], R=R, le2=le2))
    print(f"g={g:>4}: A3_bin={A3:.3g}±{se:.2g} (harm {h})  C_D/C_3T: 5e-4 {ratios[5e-4]:.3f}  1e-3 {ratios[1e-3]:.3f}  R={R:.3f}  le2={le2}")
ok = [r for r in out if r["le2"] == 0]
best = min(ok, key=lambda r: (round(r["R"], 6), -r["gap"]))
print("SELECTED g* =", best["gap"])
json.dump(dict(rows=out, selected=best), open(SC + "escwork/gap_selection.json", "w"), indent=1)
