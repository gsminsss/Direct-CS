import json, math, sys
C3T = {5e-4: 5027.0, 1e-3: 4969.0}
SYNC = {5e-4: 3798.0, 1e-3: 4468.0}
names = {"c1_base": "d5 escape + 2 d5 rounds (current)", "c1_w0": "d5 escape, no wait round", "c1_g3w0": "d3 escape -> d13", "c1_g3w1": "d3 escape + 1 d3 round", "c1_g3w2": "d3 escape + 2 d3 rounds"}
out = {}
for k, lab in names.items():
    try: s = json.load(open(f"gen/{k}/s21opt3c1_summary.json"))
    except FileNotFoundError: continue
    for r in s["evaluation_rows"]:
        p = float(r["p"]); a = r["operational_acceptance"]; n = r["shots"]
        se = math.sqrt(a * (1 - a) / n)
        C = r["stage_level_active_location_proxy_per_operational_accept"]
        out[(k, p)] = dict(label=lab, p=p, acceptance=a, acc_se=se, cost=C, cost_per_attempt=r["stage_level_expected_active_locations_per_attempt"],
                           resid=r["detectable_residual_fraction_among_accepted"], closed_harmful=r["closed_boundary_harmful"],
                           truth_closed=r["truth_closed_accepted"], le2_harmful=r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"],
                           potential=r["full_factory_potential_locations"], vs3T=C / C3T[p] - 1, vs3T_sync=C / (C3T[p] + SYNC[p]) - 1)
        x = out[(k, p)]
        print(f"{lab:36s} p={p:.0e} pot={x['potential']} acc={a:.4f}±{se:.4f} C={C:7.0f} vs3T {100*x['vs3T']:+5.1f}% (sync {100*x['vs3T_sync']:+5.1f}%) resid={x['resid']:.3f} closed_harm={x['closed_harmful']}/{x['truth_closed']} le2={x['le2_harmful']}")
json.dump({f"{k}|{p}": v for (k, p), v in out.items()}, open("compare_escape.json", "w"), indent=1)
