#!/usr/bin/env python3
"""S13-D: break-even analysis for the 3T->CS conversion cost."""
from pathlib import Path
import argparse, json, math

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s13c-summary", required=True)
    ap.add_argument("--conversion-qsteps", type=float, nargs="*", default=[0,1000,2000,4000,6000,8000,10000])
    ap.add_argument("--out", default="s13d_recomputed.json")
    a=ap.parse_args()
    s=json.loads(Path(a.s13c_summary).read_text())
    m=s["matched_cost_result"]
    direct=float(m["direct_CS_d13_hybrid_lower_proxy_qsteps"])
    prod=float(m["three_T_production_expected_STO_qsteps"])
    one=float(m["one_T_expected_STO_qsteps"])
    be=direct-prod
    rows=[]
    for c in a.conversion_qsteps:
        total=prod+c
        rows.append({
            "conversion_qsteps":c,
            "threeT_total":total,
            "direct":direct,
            "winner":"3T_route" if total<direct else ("direct_CS" if total>direct else "tie")
        })
    out={
        "direct":direct,
        "threeT_production":prod,
        "break_even_conversion":be,
        "break_even_T_equivalents":be/one,
        "rows":rows,
        "absolute_physical_ranking_ready":False,
    }
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
