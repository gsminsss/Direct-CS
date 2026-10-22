#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

DEFAULT_P = [1e-3, 1.5e-3, 2e-3]
DEFAULT_SHOTS = [100000, 100000, 100000]

def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return (None, None)
    phat = k / n
    den = 1 + z*z/n
    center = (phat + z*z/(2*n))/den
    half = z*math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/den
    return max(0.0, center-half), min(1.0, center+half)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, required=True)
    ap.add_argument("--opt3b-result-dir", type=Path, required=True)
    ap.add_argument("--opt3c0-result-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--driver", type=Path,
        default=Path(__file__).with_name("direct_cs_s21opt3c1_practical_decoder_finitep_validation.py"))
    ap.add_argument("--p-values", type=float, nargs="+", default=DEFAULT_P)
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=DEFAULT_SHOTS)
    ap.add_argument("--prior-seed", type=int, default=20271201)
    ap.add_argument("--evaluation-seed", type=int, default=20271231)
    ap.add_argument("--prior-calibration-shots", type=int, default=30000)
    args = ap.parse_args()

    if len(args.p_values) != len(args.shots_per_p):
        raise SystemExit("--p-values and --shots-per-p length mismatch")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(args.driver),
        "--search-root", str(args.search_root),
        "--opt3b-result-dir", str(args.opt3b_result_dir),
        "--opt3c0-result-dir", str(args.opt3c0_result_dir),
        "--out-dir", str(args.out_dir),
        "--p-values", *[str(x) for x in args.p_values],
        "--shots-per-p", *[str(x) for x in args.shots_per_p],
        "--prior-calibration-shots", str(args.prior_calibration_shots),
        "--prior-seed", str(args.prior_seed),
        "--evaluation-seed", str(args.evaluation_seed),
        "--frozen-gap", "10",
        "--sector-candidates", "8",
        "--bp-max-iter", "30",
        "--bp-method", "minimum_sum",
        "--ms-scaling-factor", "0.625",
        "--osd-method", "osd_cs",
        "--osd-order", "2",
        "--max-correction-effects", "8",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    d = json.loads((args.out_dir / "s21opt3c1_summary.json").read_text(encoding="utf-8"))
    rows = []
    for r in d["evaluation_rows"]:
        p = float(r["p"])
        n = int(r["shots"])
        k = int(r["closed_boundary_harmful"])
        rate = k / n
        lo, hi = wilson(k, n)
        rows.append({
            "p": p,
            "shots": n,
            "operational_accepted": int(r["operational_accepted"]),
            "truth_closed_accepted": int(r["truth_closed_accepted"]),
            "closed_boundary_harmful": k,
            "binary_harmful_per_attempt": rate,
            "binary_harmful_per_attempt_wilson95": [lo, hi],
            "A3_empirical_binary": rate / (p**3),
            "A3_empirical_binary_wilson95": [lo/(p**3), hi/(p**3)],
            "operational_acceptance": float(r["operational_acceptance"]),
            "detectable_residual_fraction_among_accepted":
                float(r["detectable_residual_fraction_among_accepted"]),
            "sampled_harmful_le2":
                int(r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"]),
        })

    xy = [(math.log(x["p"]), math.log(x["binary_harmful_per_attempt"]))
          for x in rows if x["closed_boundary_harmful"] > 0]
    slope = None
    if len(xy) >= 2:
        xs = [a for a,b in xy]; ys = [b for a,b in xy]
        xb = sum(xs)/len(xs); yb = sum(ys)/len(ys)
        slope = sum((x-xb)*(y-yb) for x,y in zip(xs,ys))/sum((x-xb)**2 for x in xs)

    out = {
        "metric_contract": {
            "primary": "closed_boundary_harmful / total physical attempts",
            "comparison": "binary A3 * p^3",
            "not_primary": "closed_boundary_harmful / truth_closed_accepted",
        },
        "frozen_decoder": d["frozen_decoder"],
        "main_gate": d["main_gate"],
        "rows": rows,
        "unweighted_loglog_slope": slope,
    }
    (args.out_dir / "highstat_binary_ler_validation_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
