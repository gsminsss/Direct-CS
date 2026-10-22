#!/usr/bin/env python3
"""
S13-A: Direct-|CS> resource accounting and comparison-normalization audit.

This script intentionally does NOT claim an apples-to-apples literature STO ranking.
It computes:
  1) exact active-location/output accounting from S12 acceptance,
  2) a frozen native qubit-step schedule proxy,
  3) early-abort expected qsteps/output using paired S12 rejection logs,
  4) optional k=2 ZXZ escape catalogue consistency checks,
  5) comparison-readiness flags.

The native qstep proxy is specific to the present active-location model:
PREP/M/H/1Q/CX/CZZ/MX/MZ each occupy one native primitive layer, idles are not
charged/noised, and the two output escapes are assumed parallel.
"""
from __future__ import annotations
import argparse, csv, io, json, zipfile
from pathlib import Path

MODELED_LOCATIONS = 2946
OUTPUT_DISTANCE = 5
FRONT = 356.0
CATPREP = 150.0
BODIES = [168.0,168.0,84.0,168.0,168.0,84.0]
MIDED = 300.0
DUAL_ESCAPE = 5808.0
FULL_PATH = 8204.0

def expected_qsteps(row):
    n = int(row["shots"])
    rr = row["rejection_reasons"]
    cost = n * FRONT
    reach = n - rr.get("initialED", 0)
    for i in range(3):
        cost += reach * CATPREP
        reach -= rr.get(f"CAT{i}", 0)
        cost += reach * BODIES[i]
    cost += reach * MIDED
    reach -= rr.get("midED", 0)
    for i in range(3,6):
        cost += reach * CATPREP
        reach -= rr.get(f"CAT{i}", 0)
        cost += reach * BODIES[i]
    reach -= rr.get("record", 0)
    # Both ZXZ escapes must execute before the 171-bit old/new live policies can be evaluated.
    if reach != int(row["core_midED_accepted"]):
        raise AssertionError(("preescape reach mismatch", row["p"], reach, row["core_midED_accepted"]))
    cost += reach * DUAL_ESCAPE
    return cost / n

def audit_escape(path: Path):
    z = zipfile.ZipFile(path)
    name = "s7b3_results/s7b3_faultable2_tail1_single_catalog.csv"
    with z.open(name) as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f)))
    locs = {}
    tick_qubits = {}
    unique = set()
    for r in rows:
        lid = int(r["location_id"]); tick = int(r["tick"])
        qs = tuple(int(x) for x in r["qubits"].split()) if r["qubits"].strip() else ()
        locs.setdefault(lid, (tick, r["gate"], qs))
        unique.update(qs)
    for tick, gate, qs in locs.values():
        tick_qubits.setdefault(tick,set()).update(qs)
    out = {
        "locations": len(locs),
        "ticks": max(tick_qubits)-min(tick_qubits)+1,
        "unique_qubit_labels": len(unique),
        "max_acted_qubits_per_tick": max(map(len,tick_qubits.values())),
        "sum_distinct_acted_qubits_over_ticks": sum(map(len,tick_qubits.values())),
    }
    if out["locations"] != 1028 or out["ticks"] != 44 or out["unique_qubit_labels"] != 66:
        raise AssertionError(out)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s12-sweep", required=True)
    ap.add_argument("--s7b3-results-zip")
    ap.add_argument("--out", default="s13a_resource_summary.json")
    args=ap.parse_args()
    sweep=json.loads(Path(args.s12_sweep).read_text())
    rows=[]
    for r in sweep:
        pacc=float(r["live171_whitelist_policy"]["acceptance"])
        qpa=expected_qsteps(r)
        rows.append({
            "p":float(r["p"]),
            "acceptance":pacc,
            "conditional_logical_lower":r["live171_whitelist_policy"]["conditional_logical_infidelity_lower"],
            "conditional_logical_upper":r["live171_whitelist_policy"]["conditional_logical_infidelity_upper"],
            "active_locations_per_accepted_output":MODELED_LOCATIONS/pacc,
            "expected_native_qsteps_per_attempt":qpa,
            "expected_native_qsteps_per_accepted_output":qpa/pacc,
        })
    out={
        "stage":"S13-A",
        "output_distance":OUTPUT_DISTANCE,
        "modeled_locations":MODELED_LOCATIONS,
        "successful_path_native_qsteps":FULL_PATH,
        "rows":rows,
        "absolute_literature_STO_ranking_ready":False,
        "reason":"d=5 output and model-specific native timing are not normalized to comparison final distance/noise conventions."
    }
    if args.s7b3_results_zip:
        out["escape_audit"]=audit_escape(Path(args.s7b3_results_zip))
    Path(args.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
