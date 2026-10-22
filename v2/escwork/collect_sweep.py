import json, glob, math
rows = []
for d in ("c1rep_lowA", "c1rep_lowB", "c1rep_g3w0"):
    mg = json.load(open(f"gen/{d}/multigap.json"))
    aud = json.load(open(f"gen/{d}/s21opt3c1_decoder_audit.json"))
    for r in mg:
        a = [x for x in aud if abs(x["p"] - r["p"]) < 1e-12][0]
        rows.append(dict(p=r["p"], gap=r["gap"], acc=r["operational_acceptance"], shots=r["shots"],
                         cost=r["stage_level_active_location_proxy_per_operational_accept"],
                         per_attempt=r["stage_level_expected_active_locations_per_attempt"],
                         attempts=1 / r["operational_acceptance"],
                         resid=r["detectable_residual_accepted"] / r["operational_accepted"],
                         harm=r["closed_boundary_harmful"], closed=r["truth_closed_accepted"], le2=r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"],
                         observed_records=a["observed_records"], d2_records=a["d2_unknown"], d2_fail=a["d2_failure"], gap_median=a["gap_median"]))
rows.sort(key=lambda r: (r["gap"], r["p"]))
json.dump(rows, open("sweep_d3.json", "w"), indent=1)
for r in rows:
    if r["gap"] in (4, 10):
        print(f"g={r['gap']:>2} p={r['p']:.0e} acc={r['acc']:.4f} cost={r['cost']:6.0f} per_att={r['per_attempt']:5.0f} att={r['attempts']:.3f} resid={r['resid']:.3f} harm={r['harm']}/{r['closed']} le2={r['le2']} obs={r['observed_records']} D2={r['d2_records']} fail={r['d2_fail']} gapmed={r['gap_median']:.2f}")
