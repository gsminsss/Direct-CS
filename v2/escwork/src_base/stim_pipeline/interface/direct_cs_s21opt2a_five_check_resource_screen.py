#!/usr/bin/env python3
"""S21-OPT2A: deterministic 5-check resource screen.

Purpose
-------
Decide whether it is worth paying for a new full S18/B3 decoder run after
replacing the six-check [6,2,4] record with the minimal five-check [5,2,3]
record:
    C1, C2 | D_mid | C12, C1, C2.

This is deliberately NOT a new logical-performance claim. It uses:
  * exact gate/location inventories,
  * exact record-only leading terms,
  * the previously measured p=1e-3 module-level 5-check/6-check acceptance
    ratio as an optimistic projection only,
  * the frozen B3 p=1e-3 reach counts/resources.

Stop rule
---------
If even the optimistic projection remains above the 3T fresh-patch upper
spacetime benchmark and below the target acceptance needed for competitiveness,
do not spend hours rerunning the full D2 decoder merely for resource rescue.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

SIX_CORE = {"CZZ":42, "CX":56, "1Q":224, "total":322}
FIVE_CORE = {"CZZ":35, "CX":56, "1Q":196, "total":287}
CAT_PER_CHECK_ACTIVE = 35  # 28 prep/verify + 7 readout
SIX_CHECKS = 6
FIVE_CHECKS = 5

# Frozen successful-path qstep schedule from S20-A2/S13-A.
FRONT_Q = 356
SIX_COREMODULE_Q = 2040
FIVE_COREMODULE_Q = (
    5*150 +                    # CAT prep/verify per check
    (168+168+84+168+168) +   # C1,C2,C12,C1,C2 body+readout
    300                        # dual midpoint ED
)
DOWNSTREAM_Q = 17208
SIX_FULL_Q = FRONT_Q + SIX_COREMODULE_Q + DOWNSTREAM_Q
FIVE_FULL_Q = FRONT_Q + FIVE_COREMODULE_Q + DOWNSTREAM_Q

# Frozen active-location ledger.
SIX_TOTAL_LOC = 7960
FIVE_TOTAL_LOC = SIX_TOTAL_LOC - ((SIX_CORE["total"]-FIVE_CORE["total"]) + CAT_PER_CHECK_ACTIVE)

# Historical million-shot module screen at p=1e-3.
P5_MODULE = 0.672969
P6_MODULE = 0.657747
MODULE_ACCEPT_RATIO = P5_MODULE / P6_MODULE

THREET_FRESH_UPPER_Q = 29289.388734683263
TARGET_ACC_UPPER = 0.3451688286013432

def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def dump(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

def recover_b3_p1e3(summary):
    rows = summary["evaluation_rows"]
    row = min(rows, key=lambda r: abs(float(r["p"])-1e-3))
    shots = int(row["shots"])
    pacc = float(row["operational_acceptance"])
    expected_loc = float(row["stage_level_expected_active_locations_per_attempt"])

    # Same integer reconstruction used in S20-A2.
    FRONT_LOC = 194
    COREMODULE_LOC = 696
    DOWNSTREAM_LOC = 7070
    target_extra = int(round((expected_loc-FRONT_LOC)*shots))
    sols=[]
    for core in range(shots+1):
        rem = target_extra - DOWNSTREAM_LOC*core
        if rem < 0 or rem % COREMODULE_LOC:
            continue
        front = rem // COREMODULE_LOC
        if core <= front <= shots:
            sols.append((front,core))
    if len(sols) != 1:
        raise RuntimeError(("cannot uniquely recover B3 reach counts", sols[:20], len(sols)))
    front,core=sols[0]
    return row, shots, front, core, pacc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s18b3-summary", type=Path,
                    default=Path("stim_pipeline/interface/s18b3_validation_results/s18b3_summary.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("stim_pipeline/interface/s21opt2a_results"))
    ap.add_argument("--refined-a3", type=float, default=1211811.2163500572)
    args=ap.parse_args()

    if not args.s18b3_summary.exists():
        raise FileNotFoundError(args.s18b3_summary)
    b3=load(args.s18b3_summary)
    row,shots,front,core,pacc6=recover_b3_p1e3(b3)

    # This is intentionally optimistic: assume the historical module-level
    # 5/6 acceptance ratio transfers multiplicatively to both core reach and
    # final operational acceptance, with all downstream behavior unchanged.
    pacc5_proj=min(1.0,pacc6*MODULE_ACCEPT_RATIO)
    core5_proj=min(shots, core*MODULE_ACCEPT_RATIO)

    q_attempt6 = FRONT_Q + (front/shots)*SIX_COREMODULE_Q + (core/shots)*DOWNSTREAM_Q
    q_accept6 = q_attempt6/pacc6

    q_attempt5 = FRONT_Q + (front/shots)*FIVE_COREMODULE_Q + (core5_proj/shots)*DOWNSTREAM_Q
    q_accept5 = q_attempt5/pacc5_proj

    a3=float(args.refined_a3)
    record5_A3=686.0
    record6_at_p1e3=7203*(1e-3)**4
    record5_at_p1e3=686*(1e-3)**3
    leading_A3_fraction = record5_A3/a3
    optimistic_a3_with_record_only = a3 + record5_A3

    worthwhile = (pacc5_proj >= TARGET_ACC_UPPER) or (q_accept5 <= THREET_FRESH_UPPER_Q)

    result={
      "stage":"S21-OPT2A deterministic five-check resource screen",
      "status":"SCREEN_COMPLETE",
      "scientific_scope":"resource triage only; not a replacement for a 5-check <=2 certificate or finite-p validation",
      "five_check_protocol":{
        "sequence":["C1","C2","MID_ED","C12","C1","C2"],
        "record":"g5(a,b)=(a,b,a^b,a,b)",
        "record_distance":3,
        "core_inventory":FIVE_CORE,
        "CAT_active_locations":FIVE_CHECKS*CAT_PER_CHECK_ACTIVE,
        "total_potential_locations_full_factory":FIVE_TOTAL_LOC,
        "successful_path_qsteps":FIVE_FULL_Q,
      },
      "six_check_reference":{
        "core_inventory":SIX_CORE,
        "CAT_active_locations":SIX_CHECKS*CAT_PER_CHECK_ACTIVE,
        "total_potential_locations_full_factory":SIX_TOTAL_LOC,
        "successful_path_qsteps":SIX_FULL_Q,
      },
      "exact_savings":{
        "active_locations_per_full_attempt":SIX_TOTAL_LOC-FIVE_TOTAL_LOC,
        "qsteps_per_successful_path":SIX_FULL_Q-FIVE_FULL_Q,
        "fraction_full_qstep_reduction":(SIX_FULL_Q-FIVE_FULL_Q)/SIX_FULL_Q,
      },
      "record_only_error":{
        "five_check_leading":"686 p_m^3 + O(p_m^4)",
        "six_check_leading":"7203 p_m^4 + O(p_m^5)",
        "p1e3_five_check":record5_at_p1e3,
        "p1e3_six_check":record6_at_p1e3,
        "five_check_A3_coefficient_686_fraction_of_current_refined_A3":leading_A3_fraction,
        "optimistic_additive_A3_if_only_new_leading_term_is_record_only":optimistic_a3_with_record_only,
        "warning":"A true 5-check A3 requires re-auditing mixed physical mechanisms; 686 is only the exact record-only leading subset."
      },
      "p1e3_projection":{
        "B3_shots":shots,
        "B3_frontend_reach":front,
        "B3_core_reach_six_check":core,
        "B3_operational_acceptance_six_check":pacc6,
        "historical_module_P5":P5_MODULE,
        "historical_module_P6":P6_MODULE,
        "historical_module_acceptance_ratio_P5_over_P6":MODULE_ACCEPT_RATIO,
        "optimistically_projected_core_reach_five_check":core5_proj,
        "optimistically_projected_operational_acceptance_five_check":pacc5_proj,
        "six_check_qsteps_per_accept_reconstructed":q_accept6,
        "optimistically_projected_five_check_qsteps_per_accept":q_accept5,
        "acceptance_needed_to_beat_3T_fresh_upper":TARGET_ACC_UPPER,
        "threeT_fresh_upper_qsteps":THREET_FRESH_UPPER_Q,
      },
      "decision":{
        "full_5check_B3_rerun_for_resource_rescue_worthwhile":bool(worthwhile),
        "recommendation":(
            "A full 5-check B3 run is not justified solely as a resource-rescue step; "
            "preserve it as a theorem/resource-optimized appendix candidate and move to growth optimization."
            if not worthwhile else
            "The optimistic screen reaches the competitiveness gate; proceed to a full 5-check structural+finite-p validation."
        )
      }
    }
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    dump(out/"s21opt2a_summary.json",result)
    print(json.dumps(result,indent=2))
    print("S21-OPT2A COMPLETE")

if __name__=="__main__":
    main()
