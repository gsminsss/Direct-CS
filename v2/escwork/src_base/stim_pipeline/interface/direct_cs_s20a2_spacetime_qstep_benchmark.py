#!/usr/bin/env python3
"""S20-A2: direct-|CS| qubit-step / spacetime-volume benchmark.

This stage performs no new logical-error Monte Carlo.  It converts the frozen
S20-A1 active-location ledger into the project's native qubit-step schedule and
compares it with a target-matched 3T->CS route.

Key design choices
------------------
1. Direct-CS uses the frozen S13-A native schedule for the d=5 front/cultivation
   core, the S13-B d5->d13 growth schedule, and the now-physical first d13 close.
2. The same coarse early-abort contract as S20-A1 is used:
      FRONT is always charged;
      the remaining pre-downstream core is charged iff both front blocks pass;
      the dual downstream path is charged iff the pre-downstream core passes.
   The exact frontend/core reach counts are recovered from the S18-B3 location
   equation, so no new sampling is needed.
3. Sahay f=3 target matching is inherited from S20-A1.  At p=5e-4, the public
   data give final hard/gap survival but do not expose every intermediate-stage
   survival needed to reconstruct exact expected STO.  We therefore report a
   rigorous production bracket from the published successful-path volume V:

      lower = V * attempts_at_gap / attempts_no_gap
      upper = V * attempts_at_gap

   Lower assumes hard-rejected attempts cost zero; upper assumes every raw
   attempt costs a full successful path.  The true expected STO lies between.
   At p=1e-3, the preserved S13-C digitized expected-STO calibration is also
   rescaled to the new target gap as a diagnostic point inside the bracket.
4. 3T->CS conversion and fresh data-patch preparation remain explicitly
   optimistic lower proxies.  Storage, idles, routing and post-conversion QEC
   are not charged.

Primary paper-safe conclusion:
- p=5e-4: fresh-patch spacetime ranking is unresolved within the defensible
  Sahay early-abort bracket (near break-even).
- p=1e-3: direct-CS is more expensive across the full bracket under the current
  proof-first/postselected implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "S20-A2-2026-09-04"
TARGET_P = (5e-4, 1e-3)

# S20-A1 coarse active-location stage contract.
FRONT_LOC = 194
COREMODULE_LOC = 696
DOWNSTREAM_LOC = 7070
TOTAL_LOC = FRONT_LOC + COREMODULE_LOC + DOWNSTREAM_LOC

# Rotated-surface-code active footprint used in the preserved S13-B convention.
D5_ACTIVE_QUBITS = 49       # 25 data + 24 syndrome ancillas
D13_ACTIVE_QUBITS = 337     # 169 data + 168 syndrome ancillas
SYNDROME_STEPS = 8
D13_CLOSE_QSTEPS_PER_PATCH = D13_ACTIVE_QUBITS * SYNDROME_STEPS

# Optimistic flexible-connectivity 3T->CS conversion from S13-E:
# 5 logical transversal CNOTs + destructive measurement of 3 magic patches.
D13_DATA = 169
CONVERSION_QSTEPS_OPTIMISTIC = 5 * (2 * D13_DATA) + 3 * D13_DATA


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def find_one(root: Path, explicit: str | None, names: list[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p.resolve()
    hits: list[Path] = []
    for n in names:
        hits.extend(root.rglob(n))
    if not hits:
        raise FileNotFoundError((str(root), names))
    hits = list(dict.fromkeys(x.resolve() for x in hits))
    hits.sort(key=lambda p: (
        0 if "stim_pipeline" in str(p) else 1,
        0 if "reference" in str(p) else 1,
        len(str(p)), str(p),
    ))
    return hits[0]


def b3_rows(summary: dict[str, Any]) -> dict[float, dict[str, Any]]:
    rows = summary.get("evaluation_rows")
    if not isinstance(rows, list):
        raise RuntimeError("S18-B3 evaluation_rows missing")
    out = {float(r["p"]): r for r in rows}
    for p in (2e-4, 5e-4, 1e-3):
        if p not in out:
            raise RuntimeError(("missing B3 p", p, sorted(out)))
    return out


def recover_reach_counts(row: dict[str, Any]) -> dict[str, Any]:
    """Recover exact frontend/core reach counts from the frozen A1 resource equation.

    B3 stored only the aggregate stage-level expected location count, but that
    quantity was generated as
        FRONT_LOC + f*COREMODULE_LOC + c*DOWNSTREAM_LOC,
    with integer reach counts f,c.  Search the integer solution and require it
    to be unique.
    """
    shots = int(row["shots"])
    expected = float(row["stage_level_expected_active_locations_per_attempt"])
    # Values were produced from integer counts, so rounding the total executed
    # location count is exact up to binary-float serialization noise.
    target_extra = int(round((expected - FRONT_LOC) * shots))
    sols: list[tuple[int, int]] = []
    for core in range(shots + 1):
        rem = target_extra - DOWNSTREAM_LOC * core
        if rem < 0:
            continue
        if rem % COREMODULE_LOC:
            continue
        front = rem // COREMODULE_LOC
        if core <= front <= shots:
            sols.append((int(front), int(core)))
    if len(sols) != 1:
        raise RuntimeError(("non-unique B3 reach reconstruction", row.get("p"), sols[:20], len(sols)))
    front, core = sols[0]
    reconstructed = FRONT_LOC + (front / shots) * COREMODULE_LOC + (core / shots) * DOWNSTREAM_LOC
    if abs(reconstructed - expected) > 1e-8:
        raise RuntimeError(("B3 reach reconstruction drift", row.get("p"), reconstructed, expected))
    return {
        "shots": shots,
        "frontend_accepted": front,
        "core_accepted": core,
        "frontend_reach_fraction": front / shots,
        "core_reach_fraction": core / shots,
        "reconstructed_expected_locations_per_attempt": reconstructed,
    }


def direct_schedule(s13a: dict[str, Any], s13b: dict[str, Any]) -> dict[str, int]:
    ns = s13a["resource_model"]["native_schedule_proxy"]
    front = int(ns["front_total_qsteps"])
    old_dual_escape = int(ns["dual_escape_qsteps"])
    old_success = int(ns["successful_path_total_qsteps"])
    pre_downstream = old_success - old_dual_escape
    coremodule = pre_downstream - front
    growth_two = int(s13b["normalization_basis"]["two_patch_growth_qubit_steps"])
    d13_close_two = 2 * D13_CLOSE_QSTEPS_PER_PATCH
    downstream = old_dual_escape + growth_two + d13_close_two
    full = front + coremodule + downstream

    expected = {
        "front": 356,
        "coremodule_after_front": 2040,
        "dual_k2_escape": 5808,
        "two_patch_d5_to_d13_growth": 6008,
        "two_patch_first_d13_close": 5392,
        "downstream_after_core": 17208,
        "full_successful_path": 19604,
    }
    got = {
        "front": front,
        "coremodule_after_front": coremodule,
        "dual_k2_escape": old_dual_escape,
        "two_patch_d5_to_d13_growth": growth_two,
        "two_patch_first_d13_close": d13_close_two,
        "downstream_after_core": downstream,
        "full_successful_path": full,
    }
    if got != expected:
        raise RuntimeError(("direct qstep schedule regression", got, expected))
    return got


def direct_cost(row: dict[str, Any], sched: dict[str, int]) -> dict[str, Any]:
    reach = recover_reach_counts(row)
    q_attempt = (
        sched["front"]
        + reach["frontend_reach_fraction"] * sched["coremodule_after_front"]
        + reach["core_reach_fraction"] * sched["downstream_after_core"]
    )
    pacc = float(row["operational_acceptance"])
    q_accept = q_attempt / pacc
    full_per_accept = sched["full_successful_path"] / pacc
    return {
        **reach,
        "operational_acceptance": pacc,
        "expected_qsteps_per_attempt_coarse_early_abort": q_attempt,
        "qsteps_per_operational_accept": q_accept,
        "full_run_qsteps_per_accept_no_early_abort": full_per_accept,
        "early_abort_saving_fraction_vs_full_run": 1.0 - q_accept / full_per_accept,
    }


def load_sahay_row(csv_path: Path, p: float) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) < 8:
                continue
            try:
                meta = json.loads(r[6])
            except Exception:
                continue
            if (
                abs(float(meta.get("p", -1)) - p) <= 1e-15
                and int(meta.get("f", -1)) == 3
                and meta.get("prep") == "unitstab"
                and int(meta.get("dfinal", -1)) == 13
                and int(meta.get("ghz_size", -1)) == 3
                and meta.get("c") == "e2eY-unitstab-g3-3ps1_mid7x3-ue13x0-uniform"
            ):
                return {
                    "shots": int(r[0].strip()),
                    "hard_discards": int(r[2].strip()),
                    "counts": json.loads(r[7]),
                    "meta": meta,
                }
    raise RuntimeError(("Sahay row not found", p, str(csv_path)))


def gap_curve(row: dict[str, Any]) -> dict[int, dict[str, float]]:
    gc: dict[int, dict[str, int]] = {}
    for k, v in row["counts"].items():
        if not k or k[0] not in ("I", "X"):
            continue
        g = int(k[1:])
        gc.setdefault(g, {"I": 0, "X": 0})[k[0]] += int(v)
    gs = sorted(gc)
    out: dict[int, dict[str, float]] = {}
    for t in range(min(gs), max(gs) + 1):
        I = sum(gc[g]["I"] for g in gs if g >= t)
        X = sum(gc[g]["X"] for g in gs if g >= t)
        kept = I + X
        if not kept:
            continue
        out[t] = {
            "gap_threshold_dB": t,
            "kept_shots": kept,
            "logical_errors": X,
            "survival_fraction": kept / row["shots"],
            "expected_attempts": row["shots"] / kept,
            "T_LER": X / kept,
        }
    return out


def sahay_production_bracket(row: dict[str, Any], target_gap: int, successful_path_qsteps: float) -> dict[str, Any]:
    curve = gap_curve(row)
    if 0 not in curve or target_gap not in curve:
        raise RuntimeError(("missing Sahay gap", target_gap, min(curve), max(curve)))
    nogap = curve[0]
    gap = curve[target_gap]
    no_gap_attempts = float(nogap["expected_attempts"])
    gap_attempts = float(gap["expected_attempts"])

    # The complementary-gap selection is final, with no subsequent quantum
    # circuit.  Every hard-surviving attempt therefore executes the full
    # successful-path volume.  Hard-rejected attempts cost between 0 and V.
    lower_one = successful_path_qsteps * (gap_attempts / no_gap_attempts)
    upper_one = successful_path_qsteps * gap_attempts
    if not (successful_path_qsteps <= lower_one <= upper_one + 1e-9):
        raise RuntimeError(("bad Sahay bracket", lower_one, upper_one))
    return {
        "no_gap": nogap,
        "target_gap": gap,
        "successful_path_qsteps": successful_path_qsteps,
        "one_T_expected_qsteps_lower": lower_one,
        "one_T_expected_qsteps_upper": upper_one,
        "three_T_production_qsteps_lower": 3 * lower_one,
        "three_T_production_qsteps_upper": 3 * upper_one,
        "bracket_logic": "hard-surviving attempts execute full V; hard-rejected attempts cost in [0,V]; final complementary-gap filter adds no subsequent circuit",
    }


def fresh_output_patch_qsteps(s13b: dict[str, Any]) -> dict[str, Any]:
    growth = int(s13b["normalization_basis"]["per_patch_growth_qubit_steps"])
    # Sahay convention notes |+> preparation as two steps.  Use a deliberately
    # simple conventional d=5 FT prep: 25 data preparations followed by five
    # d=5 syndrome rounds, then the audited d5->d13 growth.
    prep = 25 * 2
    five_rounds = 5 * D5_ACTIVE_QUBITS * SYNDROME_STEPS
    one = prep + five_rounds + growth
    two = 2 * one
    if (growth, one, two) != (3004, 5014, 10028):
        raise RuntimeError(("fresh patch qstep regression", growth, one, two))
    return {
        "one_patch_data_prep_qsteps": prep,
        "one_patch_five_d5_syndrome_rounds_qsteps": five_rounds,
        "one_patch_d5_to_d13_growth_qsteps": growth,
        "one_patch_total_qsteps": one,
        "two_patch_total_qsteps": two,
        "optimism_warning": "No storage/idles and no extra post-conversion d13 QEC close are charged to the 3T route.",
    }


def pct(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", default=".")
    ap.add_argument("--s20a1-summary")
    ap.add_argument("--s18b3-summary")
    ap.add_argument("--s13a-summary")
    ap.add_argument("--s13b-summary")
    ap.add_argument("--s13c-summary")
    ap.add_argument("--sahay-csv")
    ap.add_argument("--out-dir", default="stim_pipeline/interface/s20a2_results")
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.search_root).resolve()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    p_a1 = find_one(root, args.s20a1_summary, ["s20a1_summary.json", "s20a1_summary(1).json"])
    p_b3 = find_one(root, args.s18b3_summary, ["s18b3_summary.json"])
    p_s13a = find_one(root, args.s13a_summary, ["s13a_resource_summary.json"])
    p_s13b = find_one(root, args.s13b_summary, ["s13b_d13_normalization_summary.json"])
    p_s13c = find_one(root, args.s13c_summary, ["s13c_summary.json"])
    p_sahay = find_one(root, args.sahay_csv, ["sahay_all_paper_data_v2.csv"])

    a1 = load_json(p_a1)
    b3 = load_json(p_b3)
    s13a = load_json(p_s13a)
    s13b = load_json(p_s13b)
    s13c = load_json(p_s13c)

    if a1.get("status") != "COMPLETE_ACTIVE_LOCATION_BENCHMARK":
        raise RuntimeError(("S20-A1 not complete", a1.get("status")))
    if int(a1["direct_CS"]["total_potential_locations_if_full_run"]) != TOTAL_LOC:
        raise RuntimeError("S20-A1 location inventory drift")

    sched = direct_schedule(s13a, s13b)
    br = b3_rows(b3)
    direct = {f"{p:.4g}": direct_cost(br[p], sched) for p in (2e-4, 5e-4, 1e-3)}

    # Regression: the integer reaches implied by the frozen B3 resource rows.
    expected_reaches = {
        2e-4: (4829, 4247),
        5e-4: (4622, 3427),
        1e-3: (2585, 1394),
    }
    for p, (ef, ec) in expected_reaches.items():
        r = direct[f"{p:.4g}"]
        if (r["frontend_accepted"], r["core_accepted"]) != (ef, ec):
            raise RuntimeError(("reach regression", p, r, (ef, ec)))

    successful_T = float(s13b["three_T_raw_diagnostic"]["one_Sahay_f3_T_successful_path_qsteps_d13"])
    if successful_T != 3690:
        raise RuntimeError(("Sahay successful-path volume drift", successful_T))
    fresh = fresh_output_patch_qsteps(s13b)

    a1_cmp = {float(r["p"]): r for r in a1["target_matched_comparisons"]}
    comparisons: list[dict[str, Any]] = []
    for p in TARGET_P:
        if p not in a1_cmp:
            raise RuntimeError(("missing S20-A1 comparison", p))
        c1 = a1_cmp[p]
        gap = int(c1["Sahay_gap_threshold_dB"])
        srow = load_sahay_row(p_sahay, p)
        sb = sahay_production_bracket(srow, gap, successful_T)
        d = direct[f"{p:.4g}"]
        dq = float(d["qsteps_per_operational_accept"])

        free_lo = sb["three_T_production_qsteps_lower"] + CONVERSION_QSTEPS_OPTIMISTIC
        free_hi = sb["three_T_production_qsteps_upper"] + CONVERSION_QSTEPS_OPTIMISTIC
        fresh_lo = free_lo + fresh["two_patch_total_qsteps"]
        fresh_hi = free_hi + fresh["two_patch_total_qsteps"]

        row: dict[str, Any] = {
            "p": p,
            "direct_target_PL_A3p3": float(c1["direct_CS_leading_target_error_A3p3"]),
            "Sahay_gap_threshold_dB": gap,
            "Sahay_T_LER": float(c1["Sahay_T_LER"]),
            "direct_qsteps_per_accepted_CS": dq,
            "Sahay_one_T_qsteps_lower": sb["one_T_expected_qsteps_lower"],
            "Sahay_one_T_qsteps_upper": sb["one_T_expected_qsteps_upper"],
            "threeT_production_qsteps_lower": sb["three_T_production_qsteps_lower"],
            "threeT_production_qsteps_upper": sb["three_T_production_qsteps_upper"],
            "conversion_qsteps_optimistic": CONVERSION_QSTEPS_OPTIMISTIC,
            "fresh_two_output_patches_qsteps_optimistic": fresh["two_patch_total_qsteps"],
            "threeT_route_free_patches_qsteps_lower": free_lo,
            "threeT_route_free_patches_qsteps_upper": free_hi,
            "threeT_route_fresh_patches_qsteps_lower": fresh_lo,
            "threeT_route_fresh_patches_qsteps_upper": fresh_hi,
            "direct_percent_vs_free_lower": pct(dq, free_lo),
            "direct_percent_vs_free_upper": pct(dq, free_hi),
            "direct_percent_vs_fresh_lower": pct(dq, fresh_lo),
            "direct_percent_vs_fresh_upper": pct(dq, fresh_hi),
            "fresh_patch_ranking": (
                "DIRECT_DEFINITELY_CHEAPER" if dq < fresh_lo else
                "DIRECT_DEFINITELY_MORE_EXPENSIVE" if dq > fresh_hi else
                "UNRESOLVED_WITHIN_SAHAY_EARLY_ABORT_BRACKET"
            ),
            "Sahay_bracket": sb,
        }

        # Preserved digitized expected-STO calibration exists only at p=1e-3.
        if abs(p - 1e-3) < 1e-15:
            cal = s13c["sahay_expected_STO_digitization"]
            no_gap_q = float(cal["no_gap_expected_STO_qsteps"])
            no_gap_attempts = float(sb["no_gap"]["expected_attempts"])
            gap_attempts = float(sb["target_gap"]["expected_attempts"])
            one_cal = no_gap_q * (gap_attempts / no_gap_attempts)
            route_cal_free = 3 * one_cal + CONVERSION_QSTEPS_OPTIMISTIC
            route_cal_fresh = route_cal_free + fresh["two_patch_total_qsteps"]
            row["p1e3_digitized_Sahay_expected_STO_diagnostic"] = {
                "one_T_qsteps": one_cal,
                "threeT_route_free_patches_qsteps": route_cal_free,
                "threeT_route_fresh_patches_qsteps": route_cal_fresh,
                "direct_percent_vs_fresh": pct(dq, route_cal_fresh),
                "digitization_relative_uncertainty": float(cal["estimated_relative_digitization_uncertainty"]),
            }
        comparisons.append(row)

    # Assertions encode the scientific stop-rule we expect from this stage.
    c5 = next(x for x in comparisons if abs(x["p"] - 5e-4) < 1e-15)
    c10 = next(x for x in comparisons if abs(x["p"] - 1e-3) < 1e-15)
    if c5["fresh_patch_ranking"] != "UNRESOLVED_WITHIN_SAHAY_EARLY_ABORT_BRACKET":
        raise RuntimeError(("unexpected p=5e-4 ranking", c5))
    if c10["fresh_patch_ranking"] != "DIRECT_DEFINITELY_MORE_EXPENSIVE":
        raise RuntimeError(("unexpected p=1e-3 ranking", c10))

    self_test = {
        "stage": "S20-A2",
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "direct_successful_path_qsteps": sched["full_successful_path"],
        "direct_stage_qsteps": sched,
        "B3_recovered_reach_counts": {f"{p:.4g}": list(expected_reaches[p]) for p in expected_reaches},
        "Sahay_successful_path_qsteps": successful_T,
        "conversion_qsteps_optimistic": CONVERSION_QSTEPS_OPTIMISTIC,
        "fresh_two_output_patches_qsteps": fresh["two_patch_total_qsteps"],
        "p5e4_fresh_ranking": c5["fresh_patch_ranking"],
        "p1e3_fresh_ranking": c10["fresh_patch_ranking"],
    }
    dump_json(out / "s20a2_self_test.json", self_test)
    if args.self_test_only:
        print(json.dumps(self_test, indent=2))
        return

    summary = {
        "stage": "S20-A2 qubit-step/spacetime benchmark with target-matched 3T bracket",
        "status": "COMPLETE_QSTEP_BRACKET_BENCHMARK",
        "script_version": SCRIPT_VERSION,
        "inputs": {
            "s20a1_summary": str(p_a1),
            "s18b3_summary": str(p_b3),
            "s13a_resource_summary": str(p_s13a),
            "s13b_d13_normalization_summary": str(p_s13b),
            "s13c_summary": str(p_s13c),
            "sahay_public_data": str(p_sahay),
        },
        "metric": "qubit x native-gate-step / active-qubit gate-step proxy; idles/storage not charged",
        "metric_scope_warning": "This is a normalized spacetime proxy, not a hardware-clock-cycle compiler. CZZ, preparation, measurement, and connectivity timing remain architecture dependent.",
        "direct_CS": {
            "schedule": sched,
            "coarse_early_abort_contract": "front always; remaining pre-downstream core iff frontend passes; full dual downstream iff pre-downstream core passes",
            "cost_by_p": direct,
            "important": "The d13 close is now included physically: 337 active qubits x 8 syndrome steps per output patch.",
        },
        "threeT_baseline": {
            "Sahay_successful_path_qsteps": successful_T,
            "expected_STO_bracket_method": "V*(attempts_gap/attempts_no_gap) <= one-T expected STO <= V*attempts_gap",
            "conversion_qsteps_optimistic": CONVERSION_QSTEPS_OPTIMISTIC,
            "fresh_output_patch_qsteps": fresh,
            "important": "The bracket is required because exact intermediate hard-postselection survival by stage is not available at every p from the public aggregate row. At p=1e-3 a digitized published expected-STO calibration is included as a diagnostic.",
        },
        "target_matched_comparisons": comparisons,
        "paper_safe_interpretation": {
            "p_5e4": "Fresh-patch direct-vs-3T spacetime ranking is near break-even and unresolved within the defensible Sahay early-abort bracket; do not claim lower STO at this point.",
            "p_1e3": "Direct CS is more expensive even against the upper end of the optimistic fresh-patch 3T bracket; the current postselection overhead dominates.",
            "free_patch_scenario": "If two encoded output data patches are pre-existing/free, the 3T route is decisively cheaper at both common p points.",
            "claim": "Direct CS is resource-competitive only in an architecture/error-regime-dependent sense. The paper should lead with direct entangled-resource preparation and certification, not universal spacetime superiority.",
        },
        "next_stage": "S20-B: freeze the resource table/figure and paper wording; no additional large Monte Carlo is justified. Optionally add one correlated/hypergraph decoder note only if a reviewer-facing appendix needs it.",
    }
    dump_json(out / "s20a2_summary.json", summary)

    csv_path = out / "s20a2_target_matched_qsteps.csv"
    fields = [
        "p", "direct_target_PL_A3p3", "Sahay_gap_threshold_dB", "Sahay_T_LER",
        "direct_qsteps_per_accepted_CS",
        "Sahay_one_T_qsteps_lower", "Sahay_one_T_qsteps_upper",
        "threeT_route_free_patches_qsteps_lower", "threeT_route_free_patches_qsteps_upper",
        "threeT_route_fresh_patches_qsteps_lower", "threeT_route_fresh_patches_qsteps_upper",
        "direct_percent_vs_free_lower", "direct_percent_vs_free_upper",
        "direct_percent_vs_fresh_lower", "direct_percent_vs_fresh_upper",
        "fresh_patch_ranking",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in comparisons:
            w.writerow({k: r[k] for k in fields})

    status = f"""# S20-A2 — Qubit-step / spacetime benchmark\n\nStatus: **COMPLETE_QSTEP_BRACKET_BENCHMARK**\n\n## 1. Direct-CS current physicalized schedule\n\nSuccessful-path normalized qubit-step proxy:\n\n- front: **{sched['front']:,}**\n- post-front cultivation/check/midpoint core: **{sched['coremodule_after_front']:,}**\n- dual k=2 d=5 escape: **{sched['dual_k2_escape']:,}**\n- two-patch d5->d13 growth: **{sched['two_patch_d5_to_d13_growth']:,}**\n- two physical first-d13 close rounds: **{sched['two_patch_first_d13_close']:,}**\n- **full successful path: {sched['full_successful_path']:,} qsteps**\n\nUsing the exact B3 reach counts implied by the frozen S20-A1 resource equation:\n\n| p | frontend reach | core/downstream reach | Pacc | expected qsteps/attempt | qsteps/accepted CS |\n|---:|---:|---:|---:|---:|---:|\n| 2e-4 | {direct['0.0002']['frontend_accepted']}/{direct['0.0002']['shots']} | {direct['0.0002']['core_accepted']}/{direct['0.0002']['shots']} | {direct['0.0002']['operational_acceptance']:.4f} | {direct['0.0002']['expected_qsteps_per_attempt_coarse_early_abort']:.1f} | **{direct['0.0002']['qsteps_per_operational_accept']:.1f}** |\n| 5e-4 | {direct['0.0005']['frontend_accepted']}/{direct['0.0005']['shots']} | {direct['0.0005']['core_accepted']}/{direct['0.0005']['shots']} | {direct['0.0005']['operational_acceptance']:.4f} | {direct['0.0005']['expected_qsteps_per_attempt_coarse_early_abort']:.1f} | **{direct['0.0005']['qsteps_per_operational_accept']:.1f}** |\n| 1e-3 | {direct['0.001']['frontend_accepted']}/{direct['0.001']['shots']} | {direct['0.001']['core_accepted']}/{direct['0.001']['shots']} | {direct['0.001']['operational_acceptance']:.4f} | {direct['0.001']['expected_qsteps_per_attempt_coarse_early_abort']:.1f} | **{direct['0.001']['qsteps_per_operational_accept']:.1f}** |\n\n## 2. Target-matched 3T->CS spacetime bracket\n\nThe public Sahay aggregate row does not expose every intermediate hard-postselection survival at every p.  Therefore the exact expected STO is bracketed using the published f=3 successful-path volume **3,690 qsteps/T**:\n\n`V * attempts_gap / attempts_no_gap <= expected one-T STO <= V * attempts_gap`.\n\nConversion is an optimistic flexible-connectivity proxy of **{CONVERSION_QSTEPS_OPTIMISTIC:,} qsteps**.  Two fresh d=13 output patches are charged **{fresh['two_patch_total_qsteps']:,} qsteps** with no storage/idles/post-conversion QEC.\n\n| p | direct qsteps | 3T free-patch bracket | 3T fresh-patch bracket | conclusion |\n|---:|---:|---:|---:|---|\n| 5e-4 | **{c5['direct_qsteps_per_accepted_CS']:.1f}** | {c5['threeT_route_free_patches_qsteps_lower']:.1f}–{c5['threeT_route_free_patches_qsteps_upper']:.1f} | **{c5['threeT_route_fresh_patches_qsteps_lower']:.1f}–{c5['threeT_route_fresh_patches_qsteps_upper']:.1f}** | near break-even; **unresolved** within T early-abort bracket |\n| 1e-3 | **{c10['direct_qsteps_per_accepted_CS']:.1f}** | {c10['threeT_route_free_patches_qsteps_lower']:.1f}–{c10['threeT_route_free_patches_qsteps_upper']:.1f} | **{c10['threeT_route_fresh_patches_qsteps_lower']:.1f}–{c10['threeT_route_fresh_patches_qsteps_upper']:.1f}** | direct **definitely more expensive** |\n\nAt p=1e-3, the preserved digitized Sahay expected-STO calibration gives a target-gap fresh-patch route of about **{c10['p1e3_digitized_Sahay_expected_STO_diagnostic']['threeT_route_fresh_patches_qsteps']:.1f} qsteps**, versus direct **{c10['direct_qsteps_per_accepted_CS']:.1f}**, i.e. direct is about **{c10['p1e3_digitized_Sahay_expected_STO_diagnostic']['direct_percent_vs_fresh']:.1f}% higher** under that diagnostic.\n\n## 3. Paper-safe conclusion\n\n- **Do not claim universal lower spacetime overhead.**\n- At `p=5e-4`, the stand-alone fresh-patch comparison is genuinely close enough that the ranking depends on the T factory's internal early-abort accounting.\n- At `p=1e-3`, current direct-CS postselection is the dominant resource weakness.\n- If output data patches are already available, the 3T route is decisively cheaper at both common p points.\n- The strongest paper story remains **direct entangled |CS> cultivation + proof-first order-3 certification + native-CZZ compilation**, with resource competitiveness stated conditionally.\n\n## 4. Stop rule\n\nS20 resource simulation is now sufficiently closed for the main paper.  The next useful work is **S20-B paper table/figure + manuscript wording**, not another large Monte Carlo sweep.\n"""
    (out / "S20A2_STATUS.md").write_text(status, encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "direct_full_qsteps": sched["full_successful_path"],
        "p5e4_direct_qsteps": c5["direct_qsteps_per_accepted_CS"],
        "p5e4_fresh_3T_bracket": [c5["threeT_route_fresh_patches_qsteps_lower"], c5["threeT_route_fresh_patches_qsteps_upper"]],
        "p5e4_ranking": c5["fresh_patch_ranking"],
        "p1e3_direct_qsteps": c10["direct_qsteps_per_accepted_CS"],
        "p1e3_fresh_3T_bracket": [c10["threeT_route_fresh_patches_qsteps_lower"], c10["threeT_route_fresh_patches_qsteps_upper"]],
        "p1e3_ranking": c10["fresh_patch_ranking"],
    }, indent=2))


if __name__ == "__main__":
    main()
