#!/usr/bin/env python3
"""S17-C/B: sector-aware 470-bit postselection certificate.

Purpose
=======
S17-C/A maps the 64 canonical Steane boundary sectors into the same 470-bit
continuous operational record used by S17-B v13d, together with the raw terminal
logical Pauli transported to d=13.  S17-B v13d independently certifies the clean
boundary sector for all location-compatible downstream fault sets of weight <=2.

This stage combines the two results under the inherited S8-C total low-order
budget

    boundary_weight + downstream_fault_locations <= 2.

The exact configurations covered are:
    w_boundary = 0: downstream order 0, 1, 2
    w_boundary = 1: downstream order 0, 1
    w_boundary = 2: downstream order 0

For every reachable 470-bit operational record, the possible final joint Pauli
labels are collected.  Nonzero ambiguous records are rejected.  The zero record
must remain uniquely identity; otherwise postselection cannot preserve the
no-fault branch.

Important scope
===============
This is a structural low-order bridge certificate.  It does NOT yet assign
finite-p probabilities to the 64 boundary sectors.  Those probabilities come
from the non-Clifford direct-CS core trajectory model in the next S17-C stage.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

PAULI_NAMES = {0: "I", 1: "X", 2: "Z", 3: "Y"}
EXPECTED_SECTORS = 64
EXPECTED_DETECTORS = 470
EXPECTED_LOCATIONS = 2407


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _find(root: Path, name: str) -> Path:
    xs = sorted(root.rglob(name))
    if not xs:
        raise FileNotFoundError(f"Could not find {name!r} under {root.resolve()}")
    # Prefer canonical result/source locations when duplicates exist.
    xs.sort(key=lambda p: (
        0 if "s17b_v11_order2_results" in str(p) else 1,
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)),
        str(p),
    ))
    return xs[0]


def _import_module(path: Path):
    spec = importlib.util.spec_from_file_location("s17b_v13d", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Can't import {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses expects the module to be visible in sys.modules during exec.
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sector_rows(path: Path) -> list[dict[str, Any]]:
    rows = _load(path)
    if not isinstance(rows, list) or len(rows) != EXPECTED_SECTORS:
        raise RuntimeError(f"Expected {EXPECTED_SECTORS} sector rows, got {type(rows).__name__}/{len(rows) if isinstance(rows, list) else 'n/a'}")
    return rows


def _validate_sector_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_s = {int(r["steane_syndrome"]): r for r in rows}
    if set(by_s) != set(range(64)):
        raise RuntimeError("Sector map does not contain syndromes 0..63 exactly once")
    sigs = [int(r["detector_470_hex"], 16) for r in rows]
    if len(set(sigs)) != 64:
        raise RuntimeError("470-bit sector signatures are not injective")
    if any(s.bit_length() > EXPECTED_DETECTORS for s in sigs):
        raise RuntimeError("Sector signature exceeds 470 detector bits")
    z = by_s[0]
    if int(z["detector_470_hex"], 16) != 0 or int(z["raw_terminal_pauli_int"]) != 0:
        raise RuntimeError("Zero sector is not zero-record/identity")
    if any(int(r["steane_syndrome"]) != 0 and int(r["detector_470_hex"], 16) == 0 for r in rows):
        raise RuntimeError("Nontrivial Steane sector maps to zero 470-bit record")
    weight_hist = Counter(int(r["canonical_weight"]) for r in rows)
    if dict(weight_hist) != {0: 1, 1: 21, 2: 42}:
        raise RuntimeError(f"Unexpected canonical sector weight histogram: {dict(weight_hist)}")
    # S17-C/A showed the sector offset lives entirely in the inherited Hirano/live
    # prefix.  Re-check instead of assuming this property.
    prefix_mismatch = [
        int(r["steane_syndrome"]) for r in rows
        if int(r["detector_470_hex"], 16) != int(r["hirano_prefix_171_hex"], 16)
    ]
    return {
        "sector_count": len(rows),
        "weight_histogram": dict(sorted(weight_hist.items())),
        "signature_unique_count": len(set(sigs)),
        "zero_sector_zero_record_identity": True,
        "nontrivial_zero_record_sectors": 0,
        "full470_equals_historical171_prefix_for_all_sectors": len(prefix_mismatch) == 0,
        "prefix_mismatch_examples": prefix_mismatch[:10],
    }


def _build_v13d_effects(v13d, x_path: Path, z_path: Path):
    import stim

    x_det = v13d._detector_lines(x_path)
    z_det = v13d._detector_lines(z_path)
    if x_det != z_det or len(x_det) != EXPECTED_DETECTORS:
        raise RuntimeError("X/Z detector definitions differ or detector count != 470")

    x_locs, x_native_to_ord, x_meta = v13d._noise_locations_from_text(x_path)
    z_locs, z_native_to_ord, z_meta = v13d._noise_locations_from_text(z_path)
    if len(x_locs) != EXPECTED_LOCATIONS or len(z_locs) != EXPECTED_LOCATIONS:
        raise RuntimeError((len(x_locs), len(z_locs), EXPECTED_LOCATIONS))
    x_phys = [(x_meta[i]["tick"], x_meta[i]["gate"], tuple(x_meta[i]["qubits"])) for i in x_locs]
    z_phys = [(z_meta[i]["tick"], z_meta[i]["gate"], tuple(z_meta[i]["qubits"])) for i in z_locs]
    if x_phys != z_phys:
        raise RuntimeError("X/Z stochastic physical schedules differ")

    v13d._LOCATION_META.clear()
    v13d._LOCATION_META.update(x_meta)
    xc = stim.Circuit(x_path.read_text(encoding="utf-8"))
    zc = stim.Circuit(z_path.read_text(encoding="utf-8"))

    print("S17-C/B: recovering concrete downstream Pauli mechanisms in X probe...", flush=True)
    xpartial, xmeta_out = v13d._recover_mechanism_map(xc, v13d._raw_dem(xc), "X", x_native_to_ord)
    print("S17-C/B: recovering concrete downstream Pauli mechanisms in Z probe...", flush=True)
    zpartial, zmeta_out = v13d._recover_mechanism_map(zc, v13d._raw_dem(zc), "Z", z_native_to_ord)

    universe = v13d._all_mechanism_keys(x_locs)
    if set(xpartial) - universe or set(zpartial) - universe:
        raise RuntimeError("Stim explanation contains a mechanism outside the 2407-location physical universe")
    xmap = {m: xpartial.get(m, (0, 0)) for m in universe}
    zmap = {m: zpartial.get(m, (0, 0)) for m in universe}
    effects, joint_meta = v13d._build_joint_effects(xmap, zmap)

    first_states, _w, first_meta = v13d._joint_first_order(effects)
    if not first_meta["joint_first_order_information_gate_pass"]:
        raise RuntimeError(f"Inherited v13d first-order gate failed: {first_meta}")

    return effects, {
        "stim_version": getattr(stim, "__version__", "unknown"),
        "physical_locations": EXPECTED_LOCATIONS,
        "complete_physical_mechanisms": len(universe),
        "joint_unique_downstream_effects": len(effects),
        "joint_meta": joint_meta,
        "joint_first_order": first_meta,
        "x_probe_explained_mechanisms": len(xpartial),
        "z_probe_explained_mechanisms": len(zpartial),
    }


def _add_state(states: dict[int, int], witnesses: dict[tuple[int, int], dict[str, Any]], syndrome: int, pauli: int, witness: dict[str, Any]) -> None:
    bit = 1 << int(pauli)
    if not (states.get(int(syndrome), 0) & bit):
        witnesses[(int(syndrome), int(pauli))] = witness
    states[int(syndrome)] = states.get(int(syndrome), 0) | bit


def _sector_witness(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "steane_syndrome": int(row["steane_syndrome"]),
        "boundary_weight": int(row["canonical_weight"]),
        "boundary_record_hex": str(row["detector_470_hex"]),
        "boundary_raw_terminal_pauli": str(row["raw_terminal_pauli"]),
        "canonical_pauli": str(row.get("canonical_pauli", "")),
    }


def _effect_witness(v13d, e: Any, effect_index: int) -> dict[str, Any]:
    return {
        "effect_index": int(effect_index),
        "downstream_syndrome_hex": hex(int(e.syndrome)),
        "downstream_pauli": PAULI_NAMES[int(e.pauli)],
        "example_mechanism": v13d._mechanism_json(e.example_mechanism),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector-map", type=Path, default=None)
    ap.add_argument("--v13d-script", type=Path, default=None)
    ap.add_argument("--v13d-reject-policy", type=Path, default=None)
    ap.add_argument("--x-circuit", type=Path, default=None)
    ap.add_argument("--z-circuit", type=Path, default=None)
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=40)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.search_root.resolve()

    sector_path = (args.sector_map or _find(root, "s17c_a_sector_map.json")).resolve()
    v13d_path = (args.v13d_script or _find(root, "direct_cs_s17b_v13d_joint_postselection_certificate.py")).resolve()
    reject_path = (args.v13d_reject_policy or _find(root, "s17b_v13d_reject_policy.json")).resolve()
    x_path = (args.x_circuit or _find(root, "s17b_v11_X_transition_circuit.stim")).resolve()
    z_path = (args.z_circuit or _find(root, "s17b_v11_Z_transition_circuit.stim")).resolve()

    print(f"S17-C/B sector map: {sector_path}", flush=True)
    print(f"S17-C/B inherited v13d reject policy: {reject_path}", flush=True)
    print(f"S17-C/B X circuit: {x_path}", flush=True)
    print(f"S17-C/B Z circuit: {z_path}", flush=True)

    sectors = _sector_rows(sector_path)
    sector_guard = _validate_sector_map(sectors)
    v13d = _import_module(v13d_path)
    effects, downstream_meta = _build_v13d_effects(v13d, x_path, z_path)

    reject_obj = _load(reject_path)
    clean_reject = {int(x, 16) for x in reject_obj["rejected_syndromes_hex"]}
    if len(clean_reject) != int(reject_obj["rejected_syndrome_count"]):
        raise RuntimeError("v13d reject policy contains duplicate/malformed syndrome entries")
    if 0 in clean_reject:
        raise RuntimeError("Inherited clean v13d policy rejects zero syndrome")

    # Build all configurations in the total budget <=2 EXCEPT clean-boundary
    # downstream-order2.  Those 23.9M clean pairs are streamed later only against
    # the ~O(1e5) queried records, avoiding a giant 23M-entry state dictionary.
    states: dict[int, int] = {}
    witnesses: dict[tuple[int, int], dict[str, Any]] = {}
    branch_counts = Counter()
    t0 = time.time()

    for row in sectors:
        bw = int(row["canonical_weight"])
        rs = int(row["detector_470_hex"], 16)
        ps = int(row["raw_terminal_pauli_int"])
        sw = _sector_witness(row)

        # downstream order 0 is allowed for every boundary weight 0/1/2.
        _add_state(states, witnesses, rs, ps, {
            "total_budget": bw,
            "boundary": sw,
            "downstream_order": 0,
            "downstream": [],
        })
        branch_counts[f"boundary_w{bw}+down0"] += 1

        # downstream order 1 allowed only when boundary weight <=1.
        if bw <= 1:
            for ei, e in enumerate(effects):
                r = rs ^ int(e.syndrome)
                p = ps ^ int(e.pauli)
                _add_state(states, witnesses, r, p, {
                    "total_budget": bw + 1,
                    "boundary": sw,
                    "downstream_order": 1,
                    "downstream": [_effect_witness(v13d, e, ei)],
                })
                branch_counts[f"boundary_w{bw}+down1_effects"] += 1

    query_syndromes = set(states)
    foreground_ambiguous_before_clean_pairs = {s for s, bits in states.items() if bits.bit_count() > 1}

    # Clean-boundary downstream-order2 scan.  We only need Pauli labels for
    # syndromes already touched by non-clean boundary branches.  All other clean
    # pair records retain the already-certified v13d policy unchanged.
    print(
        f"S17-C/B: {len(query_syndromes)} sector-aware query records; streaming clean-boundary location-compatible downstream order-2 pairs...",
        flush=True,
    )
    pair_structural = 0
    pair_query_hits = 0
    pair_query_new_labels = 0
    n = len(effects)
    tscan = time.time()
    for i in range(n):
        a = effects[i]
        for j in range(i, n):
            b = effects[j]
            pw = v13d._pair_weight_distinct_locations(a, b, i == j)
            if pw <= 1e-15:
                continue
            pair_structural += 1
            s = int(a.syndrome) ^ int(b.syndrome)
            if s not in query_syndromes:
                continue
            pair_query_hits += 1
            p = int(a.pauli) ^ int(b.pauli)
            old = states.get(s, 0)
            if not (old & (1 << p)):
                loc_pair = v13d._find_distinct_location_witness(a, b)
                if loc_pair is None:
                    raise AssertionError((i, j, s, p))
                _add_state(states, witnesses, s, p, {
                    "total_budget": 2,
                    "boundary": _sector_witness(next(r for r in sectors if int(r["steane_syndrome"]) == 0)),
                    "downstream_order": 2,
                    "downstream": [
                        _effect_witness(v13d, a, i),
                        _effect_witness(v13d, b, j),
                    ],
                    "distinct_location_pair": [
                        v13d._location_json(loc_pair[0]),
                        v13d._location_json(loc_pair[1]),
                    ],
                })
                pair_query_new_labels += 1

    # Soft compatibility check against the sibling v13d summary when present.
    # A mismatch is scientifically serious, so fail rather than silently continue.
    sibling_summary = Path(str(reject_path).replace("reject_policy", "summary"))
    if sibling_summary.exists():
        expected = int(_load(sibling_summary)["joint_order_le2"]["location_compatible_joint_effect_pairs"])
        if pair_structural != expected:
            raise RuntimeError(("clean order2 structural pair count mismatch", pair_structural, expected))

    # Final sector-aware reject policy.  Clean-v13d ambiguous records remain
    # rejected even if no non-clean branch touches them.  New ambiguity is any
    # queried record with >1 possible joint Pauli label.
    new_ambiguous = {s for s, bits in states.items() if bits.bit_count() > 1}
    sector_reject = set(clean_reject) | new_ambiguous
    added_reject = sector_reject - clean_reject
    zero_bits = states.get(0, 0)
    zero_labels = [p for p in range(4) if zero_bits & (1 << p)]
    zero_identity_only = zero_labels == [0]
    if 0 in sector_reject:
        zero_identity_only = False

    # Build representative new ambiguity examples.
    examples = []
    for s in sorted(added_reject):
        labels = [p for p in range(4) if states.get(s, 0) & (1 << p)]
        if len(labels) <= 1:
            # This can only happen if future policy logic changes; added_reject is
            # currently exactly the new ambiguous query set minus clean rejects.
            continue
        examples.append({
            "syndrome_hex": hex(s),
            "possible_paulis": [PAULI_NAMES[p] for p in labels],
            "witnesses": {PAULI_NAMES[p]: witnesses.get((s, p)) for p in labels},
        })
        if len(examples) >= args.max_examples:
            break

    # Accepted query records must have exactly one label; all untouched clean
    # records inherit the v13d uniqueness certificate by construction.
    bad_accepted_query = [
        (s, bits) for s, bits in states.items()
        if s not in sector_reject and bits.bit_count() != 1
    ]
    if bad_accepted_query:
        raise RuntimeError(f"Accepted sector-aware query records with non-unique Pauli: {bad_accepted_query[:10]}")

    pass_gate = bool(zero_identity_only and not bad_accepted_query)

    # Classify which budget branches participate in newly ambiguous records.
    ambiguity_label_hist = Counter()
    for s in added_reject:
        bits = states.get(s, 0)
        labels = tuple(PAULI_NAMES[p] for p in range(4) if bits & (1 << p))
        ambiguity_label_hist["/".join(labels)] += 1

    summary = {
        "stage": "S17-C/B sector-aware 470-bit selective-rejection low-order certificate",
        "inputs": {
            "sector_map": str(sector_path),
            "v13d_script": str(v13d_path),
            "v13d_reject_policy": str(reject_path),
            "X_circuit": str(x_path),
            "Z_circuit": str(z_path),
        },
        "scope": {
            "total_fault_budget": "canonical_boundary_weight + downstream_fault_locations <= 2",
            "covered_budget_classes": [
                "boundary w0 + downstream order0/1/2",
                "boundary w1 + downstream order0/1",
                "boundary w2 + downstream order0",
            ],
            "finite_p_core_probabilities_included": False,
            "interpretation": "PASS means every covered low-order boundary+handoff configuration is either assigned a unique final joint Pauli frame or mapped to a nonzero rejected 470-bit operational record.",
        },
        "sector_guard": sector_guard,
        "downstream_guard": downstream_meta,
        "enumeration": {
            "foreground_unique_query_records": len(query_syndromes),
            "foreground_ambiguous_before_clean_order2_scan": len(foreground_ambiguous_before_clean_pairs),
            "branch_counts": dict(sorted(branch_counts.items())),
            "clean_order2_location_compatible_effect_pairs_scanned": pair_structural,
            "clean_order2_pair_query_hits": pair_query_hits,
            "clean_order2_new_pauli_labels_added_on_queries": pair_query_new_labels,
            "clean_order2_scan_seconds": time.time() - tscan,
            "total_seconds": time.time() - t0,
        },
        "policy": {
            "inherited_clean_v13d_rejected_records": len(clean_reject),
            "new_sector_aware_rejected_records": len(added_reject),
            "total_sector_aware_rejected_records": len(sector_reject),
            "new_ambiguity_pauli_label_histogram": dict(sorted(ambiguity_label_hist.items())),
            "zero_record_possible_paulis": [PAULI_NAMES[p] for p in zero_labels],
            "zero_record_identity_only": zero_identity_only,
            "all_accepted_query_records_have_unique_joint_pauli": not bad_accepted_query,
        },
        "main_gate": {
            "s17c_b_sector_aware_postselected_df_ge_3_bridge_gate_pass": pass_gate,
            "zero_record_identity_only_for_total_budget_le2": zero_identity_only,
            "all_accepted_covered_records_have_unique_joint_pauli": not bad_accepted_query,
        },
        "next_if_pass": "Integrate actual non-Clifford direct-CS core trajectory probabilities/correlated two-patch boundary classes with this sector-aware 470-bit policy, then run finite-p E2E sampling (S17-C/C).",
    }

    policy_out = {
        "detector_bits": EXPECTED_DETECTORS,
        "inherited_clean_reject_count": len(clean_reject),
        "added_sector_aware_reject_count": len(added_reject),
        "total_reject_count": len(sector_reject),
        "added_sector_aware_rejected_syndromes_hex": [hex(s) for s in sorted(added_reject)],
        "total_sector_aware_rejected_syndromes_hex": [hex(s) for s in sorted(sector_reject)],
        "new_ambiguity_examples": examples,
    }

    _dump(args.out_dir / "s17c_b_summary.json", summary)
    _dump(args.out_dir / "s17c_b_sector_aware_reject_policy.json", policy_out)
    _dump(args.out_dir / "s17c_b_new_ambiguity_examples.json", examples)

    print(json.dumps({
        "s17c_b_sector_aware_postselected_df_ge_3_bridge_gate_pass": pass_gate,
        "zero_record_identity_only_for_total_budget_le2": zero_identity_only,
        "inherited_clean_reject_count": len(clean_reject),
        "added_sector_aware_reject_count": len(added_reject),
        "total_sector_aware_reject_count": len(sector_reject),
        "foreground_query_records": len(query_syndromes),
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
