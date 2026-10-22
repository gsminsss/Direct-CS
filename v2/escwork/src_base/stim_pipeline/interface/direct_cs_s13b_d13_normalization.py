#!/usr/bin/env python3
"""S13-B: append a common d=5 -> d=7 -> d=13 output-growth normalization."""
from pathlib import Path
import argparse, json

PER_PATCH_GROWTH = 3*(97*8) + 169*4  # 3004, Sahay f=3 Table A4
TWO_PATCH_GROWTH = 2*PER_PATCH_GROWTH # 6008
RAW_T_PATH = 3690

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s13a-summary", required=True)
    ap.add_argument("--out", default="s13b_d13_normalization_summary.json")
    a=ap.parse_args()
    s=json.loads(Path(a.s13a_summary).read_text())
    native=s["resource_model"]["native_schedule_proxy"]["successful_path_total_qsteps"]
    rows=[]
    for r in s["finite_p_resource_rows"]:
        q=r["expected_native_qsteps_per_accepted_output_with_early_abort"]
        rows.append({
            "p":r["p"],
            "acceptance":r["whitelist_acceptance"],
            "conditional_logical_lower_before_growth":r["conditional_logical_lower"],
            "conditional_logical_upper_before_growth":r["conditional_logical_upper"],
            "d5_internal_qsteps_per_accepted":q,
            "accepted_only_two_patch_d5_to_d13_growth_qsteps":TWO_PATCH_GROWTH,
            "d13_hybrid_lower_proxy_qsteps_per_accepted":q+TWO_PATCH_GROWTH,
        })
    out={
        "stage":"S13-B",
        "per_patch_growth_qsteps":PER_PATCH_GROWTH,
        "two_patch_growth_qsteps":TWO_PATCH_GROWTH,
        "pre_growth_success_path_native_qsteps":native,
        "d13_success_path_hybrid_proxy":native+TWO_PATCH_GROWTH,
        "three_Sahay_T_raw_success_path":3*RAW_T_PATH,
        "hybrid_ratio_vs_threeT_raw":(native+TWO_PATCH_GROWTH)/(3*RAW_T_PATH),
        "rows":rows,
        "absolute_STO_ranking_ready":False,
        "warning":"Hybrid proxy: pre-growth and appended-growth step conventions are not yet fully harmonized; appended growth noise/yield is not simulated."
    }
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
