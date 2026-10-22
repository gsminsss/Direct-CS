#!/usr/bin/env python3
"""S17-B v11: complete deterministic missing-detector basis.

Motivation
==========
S17-B v9 established a very specific failure mode:
  * raw first-order syndrome labels are unique (no ambiguity),
  * but allowing Sahay's first d5->7 single-outcome DETECTOR annotations as
    gauge detectors introduces a rank-14 p=1/2 gauge space,
  * quotienting that gauge space collapses physically distinct logical labels.
  * adding extra dead-end d=5 rounds does not change the result.

The first d5->7 round in the public Sahay circuit contains 35 single-outcome
DETECTOR annotations. These annotations are valid for Sahay's own fresh/known
input boundary, but they are not the correct detector basis after splicing an
arbitrary *live* d=5 code state produced by the Hirano escape.

This script removes those 35 first-growth detector annotations and reconstructs
the *complete* deterministic complement returned by Stim missing_detectors().
Importantly, missing_detectors returns a row-reduced generator basis; support of
an individual generator is basis-dependent. Therefore v11 does not filter rows
by whether they explicitly touch the first growth batch. The resulting stitched
circuit is required to produce a detector error model with
allow_gauge_detectors=False.

This is an information/boundary audit. No Monte Carlo and no PyMatching are
used. If first-order uniqueness passes, run the optional --run-order2 audit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("s17b_v7", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v7 module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _bool_attr(x: Any, name: str) -> bool:
    v = getattr(x, name, False)
    return bool(v() if callable(v) else v)


def _target_value(t: Any) -> int:
    if hasattr(t, "val"):
        v = getattr(t, "val")
        return int(v() if callable(v) else v)
    if hasattr(t, "value"):
        v = getattr(t, "value")
        return int(v() if callable(v) else v)
    s = str(t)
    if s.startswith("rec[") and s.endswith("]"):
        return int(s[4:-1])
    if len(s) >= 2 and s[0] in "DLS" and s[1:].lstrip("-").isdigit():
        return int(s[1:])
    raise TypeError(f"Can't extract target value from {t!r}")


def _annotation_dependencies(circuit: Any) -> tuple[list[list[int]], list[list[int]]]:
    """Returns detector/observable dependencies as absolute measurement ids."""
    dets: list[list[int]] = []
    obs: dict[int, list[int]] = {}
    m = 0
    for inst in circuit:
        if not hasattr(inst, "name"):
            raise AssertionError("repeat blocks are not expected here")
        # Targets are resolved relative to the measurement count *before* the
        # annotation itself. Annotation instructions produce no measurements.
        if inst.name == "DETECTOR":
            row = []
            for t in inst.targets_copy():
                if _bool_attr(t, "is_measurement_record_target"):
                    row.append(m + _target_value(t))
            dets.append(sorted(set(row)))
        elif inst.name == "OBSERVABLE_INCLUDE":
            args = list(inst.gate_args_copy())
            if len(args) != 1:
                raise AssertionError(inst)
            k = int(args[0])
            row = obs.setdefault(k, [])
            for t in inst.targets_copy():
                if _bool_attr(t, "is_measurement_record_target"):
                    row.append(m + _target_value(t))
        # Count produced measurements after handling annotations.
        try:
            m += int(inst.num_measurements)
        except Exception:
            # Stable fallback for Stim versions without instruction.num_measurements.
            if inst.name in {"M", "MZ", "MX", "MY", "MR", "MRX", "MRY"}:
                m += len(inst.targets_copy())
            elif inst.name in {"MPP", "MXX", "MYY", "MZZ"}:
                # Not expected in the Sahay fragment; avoid silently guessing.
                raise AssertionError(f"Unsupported measurement op for fallback: {inst}")
    out_obs = [sorted(set(obs.get(k, []))) for k in range(max(obs.keys(), default=-1) + 1)]
    return dets, out_obs


def _strip_first_growth_detectors(g57: Any) -> tuple[Any, dict[str, Any]]:
    """Remove the first consecutive DETECTOR block after the first M round."""
    import stim

    out = stim.Circuit()
    seen_first_measurement = False
    in_first_detector_block = False
    completed = False
    removed: list[str] = []
    first_measurement_count = None

    for inst in g57:
        if not hasattr(inst, "name"):
            raise AssertionError("repeat block in g57")
        name = inst.name
        if not seen_first_measurement and name in {"M", "MZ", "MX", "MY"}:
            seen_first_measurement = True
            first_measurement_count = len(inst.targets_copy())
            out.append(inst)
            continue

        if seen_first_measurement and not completed:
            if name == "DETECTOR":
                in_first_detector_block = True
                removed.append(str(inst))
                continue
            if in_first_detector_block:
                # The public source immediately follows the first detector block
                # with SHIFT_COORDS. Preserve this and everything after it.
                completed = True

        out.append(inst)

    if first_measurement_count != 48:
        raise AssertionError(("first g57 measurement count", first_measurement_count))
    if len(removed) != 35:
        raise AssertionError(("first growth detectors removed", len(removed), removed[:5]))
    return out, {
        "first_growth_measurements": int(first_measurement_count),
        "removed_first_growth_detectors": len(removed),
        "removed_detector_examples": removed[:10],
    }


def _missing_detectors(circuit: Any) -> tuple[Any, str]:
    """Use Stim's native missing_detectors; require it in user Stim 1.16+."""
    if not hasattr(circuit, "missing_detectors"):
        raise RuntimeError(
            "This audit requires stim.Circuit.missing_detectors(), available in the "
            "Stim version used by the current project. Upgrade Stim if this method is missing."
        )
    return circuit.missing_detectors(unknown_input=True), "stim.Circuit.missing_detectors(unknown_input=True)"


def _complete_missing_detector_basis(
    missing: Any,
    *,
    total_measurements: int,
    pre_growth_measurements: int,
    first_growth_count: int,
) -> tuple[Any, dict[str, Any]]:
    """Append the complete row-reduced deterministic complement from Stim.

    ``Circuit.missing_detectors`` returns a generator basis after Gaussian
    elimination. Individual row support is not invariant under basis changes, so
    filtering rows by whether they explicitly touch the first-growth measurements
    can discard the entire missing subspace (the v10 bug).
    """
    import stim

    first_lo = int(pre_growth_measurements)
    first_hi = first_lo + int(first_growth_count)
    out = stim.Circuit()
    rows_meta = []

    for inst in missing:
        if not hasattr(inst, "name") or inst.name != "DETECTOR":
            continue
        abs_ids = []
        for t in inst.targets_copy():
            if not _bool_attr(t, "is_measurement_record_target"):
                continue
            k = int(total_measurements + _target_value(t))
            if not (0 <= k < total_measurements):
                raise AssertionError(("bad missing-detector rec", k, total_measurements, inst))
            abs_ids.append(k)
        out.append("DETECTOR", inst.targets_copy(), inst.gate_args_copy())
        rows_meta.append({
            "measurement_count": len(abs_ids),
            "absolute_measurement_ids": abs_ids,
            "pre_growth_terms": sum(k < first_lo for k in abs_ids),
            "first_growth_terms": sum(first_lo <= k < first_hi for k in abs_ids),
            "later_terms": sum(k >= first_hi for k in abs_ids),
            "touches_first_growth": any(first_lo <= k < first_hi for k in abs_ids),
            "instruction": str(inst),
        })

    return out, {
        "candidate_missing_detectors_total": int(missing.num_detectors),
        "selected_complete_basis_detectors": int(out.num_detectors),
        "rows_touching_first_growth": sum(bool(r["touches_first_growth"]) for r in rows_meta),
        "rows_not_touching_first_growth": sum(not bool(r["touches_first_growth"]) for r in rows_meta),
        "rows": rows_meta,
    }


def _raw_first_order_information(raw_dem: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    labels: dict[int, set[int]] = {}
    gauge_rows = 0
    for i, inst in enumerate(raw_dem.flattened()):
        if getattr(inst, "type", None) != "error":
            continue
        args = list(inst.args_copy()) if hasattr(inst, "args_copy") else []
        p = float(args[0]) if args else None
        det_mask = 0
        logical = 0
        for t in inst.targets_copy():
            if _bool_attr(t, "is_relative_detector_id"):
                det_mask ^= 1 << _target_value(t)
            elif _bool_attr(t, "is_logical_observable_id"):
                if _target_value(t) != 0:
                    raise AssertionError(t)
                logical ^= 1
            elif _bool_attr(t, "is_separator"):
                raise AssertionError("raw DEM unexpectedly decomposed")
        is_gauge = p is not None and abs(p - 0.5) <= 1e-12
        gauge_rows += int(is_gauge)
        row = {
            "row_index": i,
            "probability": p,
            "detector_mask": det_mask,
            "detector_mask_hex": hex(det_mask),
            "detector_weight": det_mask.bit_count(),
            "logical": logical,
            "is_gauge_p_half": bool(is_gauge),
            "instruction": str(inst),
        }
        rows.append(row)
        if not is_gauge:
            labels.setdefault(det_mask, set()).add(logical)

    amb = {k: v for k, v in labels.items() if len(v) > 1}
    zero = labels.get(0, set())
    meta = {
        "raw_error_instructions": len(rows),
        "p_half_gauge_error_instructions": gauge_rows,
        "physical_error_instructions": len(rows) - gauge_rows,
        "unique_physical_syndrome_keys": len(labels),
        "ambiguous_first_order_syndrome_keys": len(amb),
        "ambiguous_examples_hex": [hex(k) for k in sorted(amb)[:30]],
        "zero_syndrome_logical_labels": sorted(int(x) for x in zero),
        "zero_syndrome_nontrivial_first_order": 1 in zero,
        "first_order_information_gate_pass": len(amb) == 0 and 1 not in zero and gauge_rows == 0,
    }
    return rows, meta


def _order2_information(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Conservative order<=2 uniqueness over unique raw DEM effects, with replacement."""
    effects_mult: dict[tuple[int, int], int] = {}
    for r in rows:
        if r["is_gauge_p_half"]:
            continue
        key = (int(r["detector_mask"]), int(r["logical"]))
        effects_mult[key] = effects_mult.get(key, 0) + 1
    effects = sorted(effects_mult)

    # syndrome -> bitset of logical labels (bit 0 for I, bit 1 for L)
    states: dict[int, int] = {0: 1}
    witnesses: dict[tuple[int, int], Any] = {(0, 0): {"order": 0, "effects": []}}

    # order 1
    for i, (s, l) in enumerate(effects):
        bit = 1 << l
        if not (states.get(s, 0) & bit):
            witnesses[(s, l)] = {"order": 1, "effects": [{"effect_index": i, "syndrome_hex": hex(s), "logical": l, "multiplicity": effects_mult[(s,l)]}]}
        states[s] = states.get(s, 0) | bit

    # order 2 over unique effects with replacement: conservative overapprox.
    n = len(effects)
    t0 = time.time()
    for i in range(n):
        s1, l1 = effects[i]
        for j in range(i, n):
            s2, l2 = effects[j]
            s = s1 ^ s2
            l = l1 ^ l2
            bit = 1 << l
            if not (states.get(s, 0) & bit):
                witnesses[(s, l)] = {"order": 2, "effects": [i, j]}
            states[s] = states.get(s, 0) | bit

    amb = [s for s, bits in states.items() if bits == 3]
    zero_bits = states.get(0, 0)
    return {
        "unique_physical_effects": n,
        "effect_pairs_with_replacement": n * (n + 1) // 2,
        "unique_order_le2_syndromes": len(states),
        "ambiguous_order_le2_syndromes": len(amb),
        "ambiguity_examples_hex": [hex(x) for x in sorted(amb)[:30]],
        "zero_syndrome_logical_labels": [x for x in (0,1) if zero_bits & (1 << x)],
        "zero_syndrome_nontrivial_order_le2": bool(zero_bits & 2),
        "order_le2_information_gate_pass": len(amb) == 0 and not (zero_bits & 2),
        "enumeration_seconds": time.time() - t0,
    }


def _build_channel(v7: Any, hirano_src: Path, sahay_stim: Path, channel: str, p: float) -> tuple[Any, dict[str, Any]]:
    import stim

    lines = v7._read_source(sahay_stim)
    geo = v7._source_geometry(lines)
    wait_rounds = 2

    full, _wrapper, init_range, probe_start, online_ids, terminal_ids = v7.s7b._build_with_wait_rounds(
        hirano_src, channel, wait_rounds
    )
    pre_clean = full[:probe_start]

    # Recover Hirano's measurement-dependent observable frame, as in v7.
    _full_det_deps, full_obs_deps = v7.s7b._annotation_dependencies(full)
    if len(full_obs_deps) != 1:
        raise AssertionError(("Hirano observables", len(full_obs_deps), full.num_observables))
    pre_m = int(pre_clean.num_measurements)
    obs_all = [int(x) for x in full_obs_deps[0]]
    hirano_preprobe_obs = [m for m in obs_all if m < pre_m]
    hirano_terminal_obs = [m for m in obs_all if m >= pre_m]
    if len(hirano_terminal_obs) != 5 or not hirano_preprobe_obs:
        raise AssertionError((channel, hirano_preprobe_obs, hirano_terminal_obs, pre_m))

    if len(online_ids) != 171 or int(pre_clean.num_detectors) != 171:
        raise AssertionError((len(online_ids), pre_clean.num_detectors))
    if len(terminal_ids) != 12:
        raise AssertionError(terminal_ids)

    noisy_pre, hloc, _noise_map = v7.s7b._instrument_before_probe(pre_clean, init_range, len(pre_clean), p)
    if len(hloc) != 1028:
        raise AssertionError(("Hirano locations", len(hloc)))

    qmap, map_meta = v7._build_qmap(pre_clean, geo)
    transport_flow = v7._solve_transport_flow(lines, geo, channel)

    g57_src = v7._clean_source_range(lines, v7.G57_START, v7.G57_END)
    g57_src, strip_meta = _strip_first_growth_detectors(g57_src)
    g713_src = v7._clean_source_range(lines, v7.G713_START, v7.G713_END)
    close_src = v7._clean_source_range(lines, v7.CLOSE_START, v7.CLOSE_END)

    g57_clean = v7._remap_circuit(g57_src, qmap)
    g713_clean = v7._remap_circuit(g713_src, qmap)
    close_clean = v7._remap_circuit(close_src, qmap)
    g57_noisy, c57 = v7._instrument_growth(g57_clean, p)
    g713_noisy, c713 = v7._instrument_growth(g713_clean, p)
    if c57["total"] != 971 or c713["total"] != 408:
        raise AssertionError((c57, c713))

    # Existing deterministic history, but without the invalid first-growth
    # single-outcome detector block.
    clean_preterminal = pre_clean + g57_clean + g713_clean + close_clean
    noisy_preterminal = noisy_pre + g57_noisy + g713_noisy + close_clean
    if int(clean_preterminal.num_measurements) != int(noisy_preterminal.num_measurements):
        raise AssertionError("noise changed measurement count")

    # Synthesise missing deterministic measurement invariants from the *clean*
    # Clifford circuit, then keep only checks that touch the first growth batch.
    missing, method = _missing_detectors(clean_preterminal)
    transition, trans_meta = _complete_missing_detector_basis(
        missing,
        total_measurements=int(clean_preterminal.num_measurements),
        pre_growth_measurements=pre_m,
        first_growth_count=48,
    )

    # Append exactly the same annotation rows to clean and noisy circuits.
    clean_preterminal += transition
    noisy_preterminal += transition

    # Check determinism *before* adding the terminal logical scorer.
    clean_dem_error = None
    try:
        clean_preterminal.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    except Exception as ex:
        clean_dem_error = repr(ex)

    transport_measurement_offset = int(noisy_pre.num_measurements)
    c = noisy_preterminal
    terminal_meta = v7._append_terminal_boundary(
        c,
        channel,
        geo,
        qmap,
        transport_measurement_offset=transport_measurement_offset,
        transport_flow=transport_flow,
        hirano_preprobe_observable_measurements=hirano_preprobe_obs,
    )

    meta = {
        "channel": channel,
        "p": float(p),
        "wait_rounds": wait_rounds,
        "faultable_locations_per_patch": len(hloc) + c57["total"] + c713["total"],
        "original_hirano_detectors": 171,
        "removed_first_growth_detectors": strip_meta["removed_first_growth_detectors"],
        "synthesized_complete_basis_detectors": trans_meta["selected_complete_basis_detectors"],
        "detectors_before_transition_synthesis": 171 + (131 - 35) + 168,
        "detectors_after_transition_synthesis": int(c.num_detectors),
        "measurements_before_growth": pre_m,
        "first_growth_measurement_interval": [pre_m, pre_m + 48],
        "missing_detector_method": method,
        "missing_detector_meta": trans_meta,
        "strip_meta": strip_meta,
        "clean_preterminal_determinism_error": clean_dem_error,
        "mapping": map_meta,
        "transport_flow": transport_flow,
        "terminal": terminal_meta,
        "hirano_observable_transport": {
            "preprobe_frame_measurement_ids": hirano_preprobe_obs,
            "terminal_d5_boundary_measurement_ids": hirano_terminal_obs,
        },
        "active_locations": {
            "hirano_escape_plus_two_d5_rounds": len(hloc),
            "sahay_d5_to_d7": c57["total"],
            "sahay_d7_to_d13": c713["total"],
            "total": len(hloc) + c57["total"] + c713["total"],
        },
    }
    return c, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-script", type=Path, required=True)
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--sahay-stim", type=Path, required=True)
    ap.add_argument("--signature-p", type=float, default=0.001)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-order2", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    v7 = _load_module(args.v7_script.resolve())

    import stim
    summary = {
        "stage": "S17-B v11 complete deterministic missing-detector basis",
        "stim_version": getattr(stim, "__version__", "unknown"),
        "signature_p": float(args.signature_p),
        "channels": {},
        "order2_was_run": bool(args.run_order2),
    }

    for ch in ("X", "Z"):
        print(f"S17-B v11: building {ch} transition-detector circuit...", flush=True)
        c, build = _build_channel(v7, args.hirano_src.resolve(), args.sahay_stim.resolve(), ch, float(args.signature_p))
        c.to_file(args.out_dir / f"s17b_v11_{ch}_transition_circuit.stim")
        _dump(args.out_dir / f"s17b_v11_{ch}_build_meta.json", build)

        dem_error = None
        raw = None
        try:
            raw = c.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
        except Exception as ex:
            dem_error = repr(ex)

        if raw is None:
            out = {
                "channel": ch,
                "build_meta": build,
                "detector_error_model_error": dem_error,
                "deterministic_detector_gate_pass": False,
            }
            _dump(args.out_dir / f"s17b_v11_{ch}_first_order.json", out)
            summary["channels"][ch] = out
            continue

        rows, first = _raw_first_order_information(raw)
        out = {
            "channel": ch,
            "build_meta": build,
            "detector_error_model_error": None,
            "deterministic_detector_gate_pass": first["p_half_gauge_error_instructions"] == 0,
            "first_order_information": first,
        }
        if args.run_order2 and first["first_order_information_gate_pass"]:
            print(f"S17-B v11: enumerating {ch} exact order<=2 effects...", flush=True)
            out["order_le2_information"] = _order2_information(rows)
        _dump(args.out_dir / f"s17b_v11_{ch}_first_order.json", out)
        summary["channels"][ch] = out

    both_det = all(summary["channels"].get(ch, {}).get("deterministic_detector_gate_pass", False) for ch in ("X","Z"))
    both_first = all(summary["channels"].get(ch, {}).get("first_order_information", {}).get("first_order_information_gate_pass", False) for ch in ("X","Z"))
    summary["deterministic_complete_basis_gate_pass_both_channels"] = both_det
    summary["exact_first_order_information_gate_pass_both_channels"] = both_first
    if args.run_order2:
        summary["exact_order_le2_information_gate_pass_both_channels"] = all(
            summary["channels"].get(ch, {}).get("order_le2_information", {}).get("order_le2_information_gate_pass", False)
            for ch in ("X","Z")
        )
    summary["decision_rule"] = [
        "PASS requires allow_gauge_detectors=False DEM construction in both channels: no p=1/2 gauge mechanism is permitted.",
        "If first-order uniqueness passes, the v8/v9 ambiguity was caused by using the wrong detector basis at the live-to-growth splice; locality optimization of this complete basis can be done afterward.",
        "If deterministic DEM construction succeeds but first-order uniqueness still fails, the added invariant basis is still insufficient; inspect the synthesized rows and physical witnesses before any Monte Carlo.",
        "If first-order passes, rerun with --run-order2 before finite-p simulation.",
    ]
    _dump(args.out_dir / "s17b_v11_summary.json", summary)
    print(json.dumps({
        "deterministic_complete_basis_gate_pass_both_channels": both_det,
        "exact_first_order_information_gate_pass_both_channels": both_first,
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
