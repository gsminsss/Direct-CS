#!/usr/bin/env python3
"""S20-A1: deterministic resource ledger and target-matched 3T->CS benchmark.

No new physical Monte Carlo is performed.  The script consumes the frozen
S18-B3 integrated validation, S19-A2A/A3-Lite leading-order result, the audited
S13-E physical Sahay circuit count, and the public Sahay gap histogram.

Primary metric in S20-A1:
    expected active physical locations per operationally accepted |CS>

This is *not* yet qubit-round/spacetime volume.  S20-A2 should close the actual
qubit-step convention after S20-A1 fixes the gate/location ledger.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "S20-A1-2026-09-04"
TARGET_P = (5e-4, 1e-3)
TOL = 1e-12


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def find_one(root: Path, explicit: str | None, names: list[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(p)
    hits: list[Path] = []
    for name in names:
        hits.extend(root.rglob(name))
    if not hits:
        raise FileNotFoundError((str(root), names))
    hits = list(dict.fromkeys(x.resolve() for x in hits))
    hits.sort(key=lambda p: (0 if "stim_pipeline" in str(p) else 1, len(str(p)), str(p)))
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


def locate_hirano3_a2a(a2a: dict[str, Any]) -> float:
    for r in a2a.get("top_active_strata", []):
        if r.get("key") == "DOWN_HIRANO_D53":
            return float(r["A3_closed_boundary_infidelity_estimate"])
    raise RuntimeError("DOWN_HIRANO_D53 not found in A2A top_active_strata")


def refined_a3(a2a: dict[str, Any], a3: dict[str, Any]) -> dict[str, float]:
    primary = float(a2a["aggregate"]["A3_closed_boundary_infidelity_estimate"])
    old_h = locate_hirano3_a2a(a2a)
    per_prior = a3["per_prior"]
    key = "0.0002" if "0.0002" in per_prior else 0.0002
    fresh_h = float(per_prior[key]["A3_selected"])
    fresh_h_se = float(per_prior[key]["A3_selected_naive_se"])
    refined = primary - old_h + fresh_h
    stable_remainder = primary - old_h
    # Remove the old Hirano variance from A2A in quadrature only when numerically possible.
    old_global_se = float(a2a["aggregate"]["A3_closed_boundary_infidelity_naive_se"])
    old_h_se = None
    for r in a2a.get("top_active_strata", []):
        if r.get("key") == "DOWN_HIRANO_D53":
            old_h_se = float(r["A3_closed_boundary_infidelity_se"])
            break
    if old_h_se is None:
        stable_se = float("nan")
        refined_se = float("nan")
    else:
        stable_var = max(0.0, old_global_se**2 - old_h_se**2)
        stable_se = math.sqrt(stable_var)
        refined_se = math.sqrt(stable_var + fresh_h_se**2)
    return {
        "A2A_primary": primary,
        "A2A_old_Hirano3": old_h,
        "fresh_Hirano3": fresh_h,
        "fresh_Hirano3_se": fresh_h_se,
        "stable_remainder": stable_remainder,
        "stable_remainder_se_quadrature": stable_se,
        "refined_global_A3": refined,
        "refined_global_A3_se_quadrature": refined_se,
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
                    "raw_errors": int(r[1].strip()),
                    "hard_discards": int(r[2].strip()),
                    "meta": meta,
                    "counts": json.loads(r[7]),
                }
    raise RuntimeError(("Sahay row not found", p, str(csv_path)))


def gap_curve(row: dict[str, Any]) -> list[dict[str, float]]:
    gc: dict[int, dict[str, int]] = {}
    for k, v in row["counts"].items():
        typ = k[0]
        if typ not in ("I", "X"):
            continue
        g = int(k[1:])
        gc.setdefault(g, {"I": 0, "X": 0})[typ] += int(v)
    gs = sorted(gc)
    out: list[dict[str, float]] = []
    for t in range(min(gs), max(gs) + 1):
        I = sum(gc[g]["I"] for g in gs if g >= t)
        X = sum(gc[g]["X"] for g in gs if g >= t)
        kept = I + X
        if kept == 0:
            continue
        out.append({
            "gap_threshold_dB": int(t),
            "kept_shots": int(kept),
            "logical_errors": int(X),
            "survival_fraction": kept / row["shots"],
            "expected_attempts": row["shots"] / kept,
            "T_logical_error_rate": X / kept,
        })
    return out


def target_match(curve: list[dict[str, float]], cs_target_error: float) -> dict[str, float]:
    t_budget = cs_target_error / 3.0
    candidates = [x for x in curve if x["T_logical_error_rate"] <= t_budget]
    if not candidates:
        raise RuntimeError(("no Sahay gap meets target", cs_target_error, t_budget))
    best = min(candidates, key=lambda x: x["expected_attempts"])
    out = dict(best)
    out["T_error_budget"] = t_budget
    out["leading_3T_to_CS_error"] = 3.0 * best["T_logical_error_rate"]
    return out


def wilson_accept_upper_cost(row: dict[str, Any], total_locations: int) -> dict[str, float]:
    acc = float(row["operational_acceptance"])
    full_run = total_locations / acc
    stage = float(row["stage_level_active_location_proxy_per_operational_accept"])
    return {
        "operational_acceptance": acc,
        "attempts_per_accept": float(row["attempts_per_operational_accept"]),
        "full_run_no_early_abort_locations_per_accept": full_run,
        "stage_level_expected_locations_per_attempt": float(row["stage_level_expected_active_locations_per_attempt"]),
        "stage_level_locations_per_accept": stage,
        "early_abort_saving_fraction_vs_full_run": 1.0 - stage / full_run,
    }


def pct_delta(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", default=".")
    ap.add_argument("--s18b3-summary")
    ap.add_argument("--s19a2a-summary")
    ap.add_argument("--s19a3-summary")
    ap.add_argument("--s13e-summary")
    ap.add_argument("--sahay-csv")
    ap.add_argument("--out-dir", default="stim_pipeline/interface/s20a1_results")
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.search_root).resolve()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    p_b3 = find_one(root, args.s18b3_summary, ["s18b3_summary.json"])
    p_a2a = find_one(root, args.s19a2a_summary, ["s19a2a_summary.json"])
    p_a3 = find_one(root, args.s19a3_summary, ["s19a3_lite_summary.json"])
    p_s13e = find_one(root, args.s13e_summary, ["s13e_summary.json"])
    p_sahay = find_one(root, args.sahay_csv, ["sahay_all_paper_data_v2.csv"])

    b3 = load_json(p_b3)
    a2a = load_json(p_a2a)
    a3 = load_json(p_a3)
    s13e = load_json(p_s13e)

    br = b3_rows(b3)
    loc_inv = a2a["location_inventory"]
    total_locations = int(loc_inv["total"])
    if total_locations != 7960:
        raise RuntimeError(("unexpected total locations", total_locations))

    a3r = refined_a3(a2a, a3)
    if abs(a3r["refined_global_A3"] - 1211811.2163500572) > 1e-6:
        raise RuntimeError(("refined A3 regression", a3r))

    t_locations = int(s13e["public_Sahay_circuit"]["physical_active_locations_through_d13_excluding_ideal_decoder_and_terminal_readout"])
    conv = int(s13e["threeT_to_CS_conversion"]["explicit_magic_teleportation_model"]["optimized_conversion_active_locations"])
    fresh_patches = int(s13e["fresh_output_data_patch_models"]["two_output_patches"])
    if (t_locations, conv, fresh_patches) != (1662, 1352, 4328):
        raise RuntimeError(("S13E baseline regression", t_locations, conv, fresh_patches))

    direct_rows: dict[str, Any] = {}
    for p in (2e-4, 5e-4, 1e-3):
        direct_rows[f"{p:.4g}"] = wilson_accept_upper_cost(br[p], total_locations)

    comparisons: list[dict[str, Any]] = []
    for p in TARGET_P:
        direct_target = a3r["refined_global_A3"] * p**3
        row = load_sahay_row(p_sahay, p)
        match = target_match(gap_curve(row), direct_target)
        one_t = t_locations * match["expected_attempts"]
        three_t = 3.0 * one_t
        route_free = three_t + conv
        route_fresh = route_free + fresh_patches
        direct = float(br[p]["stage_level_active_location_proxy_per_operational_accept"])
        comparisons.append({
            "p": p,
            "direct_CS_leading_target_error_A3p3": direct_target,
            "direct_refined_A3": a3r["refined_global_A3"],
            "direct_operational_acceptance": float(br[p]["operational_acceptance"]),
            "direct_stage_level_locations_per_accepted_CS": direct,
            "Sahay_gap_threshold_dB": int(match["gap_threshold_dB"]),
            "Sahay_T_LER": float(match["T_logical_error_rate"]),
            "Sahay_expected_attempts_per_T": float(match["expected_attempts"]),
            "threeT_leading_CS_error": float(match["leading_3T_to_CS_error"]),
            "one_T_active_locations": one_t,
            "three_T_production_active_locations": three_t,
            "conversion_active_locations_optimistic_flexible": conv,
            "fresh_two_output_patches_active_locations": fresh_patches,
            "threeT_route_data_patches_free": route_free,
            "threeT_route_fresh_output_patches": route_fresh,
            "direct_over_threeT_free_ratio": direct / route_free,
            "direct_over_threeT_fresh_ratio": direct / route_fresh,
            "direct_percent_vs_threeT_free": pct_delta(direct, route_free),
            "direct_percent_vs_threeT_fresh": pct_delta(direct, route_fresh),
        })

    self_test = {
        "stage": "S20-A1",
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "total_potential_locations": total_locations,
        "location_inventory": loc_inv,
        "refined_global_A3": a3r["refined_global_A3"],
        "Sahay_physical_locations_per_attempt": t_locations,
        "conversion_locations": conv,
        "fresh_output_patch_locations": fresh_patches,
        "common_physical_p_points": list(TARGET_P),
    }
    dump_json(out / "s20a1_self_test.json", self_test)
    if args.self_test_only:
        print(json.dumps(self_test, indent=2))
        return

    summary = {
        "stage": "S20-A1 deterministic resource ledger + target-matched 3T->CS active-location benchmark",
        "status": "COMPLETE_ACTIVE_LOCATION_BENCHMARK",
        "script_version": SCRIPT_VERSION,
        "inputs": {
            "s18b3_summary": str(p_b3),
            "s19a2a_summary": str(p_a2a),
            "s19a3_summary": str(p_a3),
            "s13e_summary": str(p_s13e),
            "sahay_public_data": str(p_sahay),
        },
        "metric": "expected active physical locations per operationally accepted state",
        "metric_warning": "Not yet qubit-round/spacetime volume. S20-A2 should close qubit-step/STO accounting.",
        "direct_CS": {
            "location_inventory": loc_inv,
            "total_potential_locations_if_full_run": total_locations,
            "refined_A3": a3r,
            "B3_stage_level_resource": direct_rows,
        },
        "threeT_baseline": {
            "Sahay_physical_active_locations_per_attempt": t_locations,
            "optimized_flexible_3T_to_CS_conversion_locations": conv,
            "fresh_two_output_patch_locations": fresh_patches,
            "target_matching_rule": "choose minimum Sahay complementary gap with T_LER <= direct_CS_leading_error/3; 3T->CS leading output error = 3 p_T",
            "early_abort_credit_for_T_factory": False,
        },
        "target_matched_comparisons": comparisons,
        "interpretation": {
            "p_5e4": "At p=5e-4, direct CS is more expensive than the optimistic free-output-patch 3T route, but cheaper than the stand-alone 3T route when two fresh output patches are charged.",
            "p_1e3": "At p=1e-3, direct CS is more expensive in both scenarios under the present proof-first/postselected implementation.",
            "claim": "No architecture-independent resource supremacy. The resource ranking depends on physical error rate and whether fresh output patches/conversion resources are charged.",
        },
        "next_stage": "S20-A2: qubit-step/spacetime ledger using the same postselection segments and final d=13 normalization; no new logical-error Monte Carlo required.",
    }
    dump_json(out / "s20a1_summary.json", summary)

    csv_path = out / "s20a1_target_matched_3t.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(comparisons[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(comparisons)

    # Compact paper-facing status note.
    c5, c10 = comparisons
    md = f"""# S20-A1 — Resource ledger and target-matched 3T→CS benchmark

Status: **COMPLETE_ACTIVE_LOCATION_BENCHMARK**

## 1. Direct-CS ledger

The complete physicalized S19 factory has **{total_locations:,} potential active fault locations** per full attempt.
S18-B3 already includes stage-level early-abort accounting, so the paper-facing active-location proxy is taken directly from B3 rather than using {total_locations}/Pacc.

| p | Pacc | expected active locations / attempt | active locations / accepted CS | full-run {total_locations}/Pacc |
|---:|---:|---:|---:|---:|
| 2e-4 | {br[2e-4]['operational_acceptance']:.4f} | {br[2e-4]['stage_level_expected_active_locations_per_attempt']:.1f} | {br[2e-4]['stage_level_active_location_proxy_per_operational_accept']:.1f} | {total_locations/br[2e-4]['operational_acceptance']:.1f} |
| 5e-4 | {br[5e-4]['operational_acceptance']:.4f} | {br[5e-4]['stage_level_expected_active_locations_per_attempt']:.1f} | {br[5e-4]['stage_level_active_location_proxy_per_operational_accept']:.1f} | {total_locations/br[5e-4]['operational_acceptance']:.1f} |
| 1e-3 | {br[1e-3]['operational_acceptance']:.4f} | {br[1e-3]['stage_level_expected_active_locations_per_attempt']:.1f} | {br[1e-3]['stage_level_active_location_proxy_per_operational_accept']:.1f} | {total_locations/br[1e-3]['operational_acceptance']:.1f} |

Refined leading coefficient:

`A3 = {a3r['refined_global_A3']:.6g}` (A2A stable remainder + fresh 10k-shot Hirano^3 precision).

## 2. Target-matched 3T→CS baseline

Sahay f=3 d13 public data are matched at the same physical p.  The smallest complementary-gap threshold satisfying `3*T_LER <= direct A3*p^3` is used.  The audited physical T circuit costs {t_locations} active locations per full attempt.  No T-factory early-abort credit is used.  The optimistic conversion costs {conv} active locations; the stand-alone resource-generator comparison additionally charges {fresh_patches} locations for two fresh logical output patches.

| p | direct target PL | Sahay gap | 3T→CS error | direct / accepted CS | 3T free patches | 3T fresh patches | direct vs free | direct vs fresh |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | {c5['direct_CS_leading_target_error_A3p3']:.3e} | {c5['Sahay_gap_threshold_dB']} dB | {c5['threeT_leading_CS_error']:.3e} | {c5['direct_stage_level_locations_per_accepted_CS']:.1f} | {c5['threeT_route_data_patches_free']:.1f} | {c5['threeT_route_fresh_output_patches']:.1f} | {c5['direct_percent_vs_threeT_free']:+.1f}% | {c5['direct_percent_vs_threeT_fresh']:+.1f}% |
| 1e-3 | {c10['direct_CS_leading_target_error_A3p3']:.3e} | {c10['Sahay_gap_threshold_dB']} dB | {c10['threeT_leading_CS_error']:.3e} | {c10['direct_stage_level_locations_per_accepted_CS']:.1f} | {c10['threeT_route_data_patches_free']:.1f} | {c10['threeT_route_fresh_output_patches']:.1f} | {c10['direct_percent_vs_threeT_free']:+.1f}% | {c10['direct_percent_vs_threeT_fresh']:+.1f}% |

## 3. Interpretation

- At **p=5e-4**, direct CS is **{abs(c5['direct_percent_vs_threeT_fresh']):.1f}% lower** than the stand-alone 3T route when fresh output patches are charged, but **{c5['direct_percent_vs_threeT_free']:.1f}% higher** if the two output patches are assumed free/pre-existing.
- At **p=1e-3**, direct CS is **{c10['direct_percent_vs_threeT_fresh']:.1f}% higher** even in the fresh-output-patch comparison, reflecting the current proof-first protocol's strong postselection overhead.
- Therefore the paper should **not** claim universal resource superiority.  The defensible claim is architecture- and error-rate-dependent competitiveness, with a favorable stand-alone-generator point at p=5e-4.

## 4. Next

S20-A2 should convert the same ledger into **qubit-steps / spacetime volume** using explicit stage durations and survival weighting.  No new logical-error Monte Carlo is needed.
"""
    (out / "S20A1_STATUS.md").write_text(md, encoding="utf-8")

    print(json.dumps({
        "stage": summary["stage"],
        "status": summary["status"],
        "refined_A3": a3r["refined_global_A3"],
        "p5e-4": comparisons[0],
        "p1e-3": comparisons[1],
        "out_dir": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
