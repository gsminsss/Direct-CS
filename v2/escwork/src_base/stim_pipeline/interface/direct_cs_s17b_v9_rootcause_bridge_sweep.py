#!/usr/bin/env python3
"""S17-B v9: root-cause explanation + d=5 bridge-round first-order sweep.

Purpose
-------
S17-B v8 found first-order information ambiguities in the 470-bit operational
record, including canonical-zero logical effects, while the earlier S15/S16
physical Pauli audits found no hidden single-fault logical for the corresponding
growth/composed interface.  This script diagnoses that discrepancy before any
protocol redesign.

It performs two independent checks:

1. Root-cause explanation at wait_rounds=2
   * Builds the exact v7 stitched X/Z circuits and raw Stim DEMs.
   * Separates raw-syndrome ambiguity from ambiguity introduced only after
     quotienting by Stim's p=1/2 gauge subspace.
   * Locates every physical DEM row with canonical syndrome 0 and logical 1.
   * Uses Circuit.explain_detector_error_model_errors() to map representative
     DEM mechanisms back to actual circuit error locations/ticks.
   * Classifies representative errors into Hirano / d5->7 / d7->13 stages.

2. Bridge-round sweep
   * Rebuilds the same stitched circuit with 2, 3, and 4 faultable d=5 rounds
     between the Hirano escape and Sahay growth.
   * Audits exact first-order syndrome->logical uniqueness, both before and
     after gauge quotient.
   * This tests whether the v8 first-order ambiguity is a terminal-boundary
     effect cured by one additional d=5 bridge round.

No Monte Carlo is run.  No graphlike approximation is used for the information
checks.  The output is diagnostic; it does not by itself certify order<=2.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _set_wait_rounds(v7: Any, rounds: int) -> None:
    # Preserved S7-B fingerprints:
    #   rounds=2 -> 1028 locations, 171 live detectors
    #   rounds=3 -> 1156 locations, 195 live detectors
    # Therefore each extra full d=5 surface round contributes 128 active
    # locations and 24 live detectors.  These formulas also reproduce the
    # original rounds=2 constants exactly.
    v7.WAIT_ROUNDS = int(rounds)
    v7.EXPECTED_HIRANO_LOCATIONS = 772 + 128 * int(rounds)
    v7.EXPECTED_HIRANO_LIVE_DETECTORS = 123 + 24 * int(rounds)
    v7.EXPECTED_TOTAL_LOCATIONS = v7.EXPECTED_HIRANO_LOCATIONS + v7.EXPECTED_GROWTH_LOCATIONS
    v7.EXPECTED_TOTAL_DETECTORS = (
        v7.EXPECTED_HIRANO_LIVE_DETECTORS
        + v7.EXPECTED_GROWTH_DETECTORS
        + v7.EXPECTED_D13_CLOSE_DETECTORS
    )


def _raw_rows_with_inst(v7: Any, raw_dem: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, inst in enumerate(raw_dem.flattened()):
        if getattr(inst, "type", None) != "error":
            continue
        mask = 0
        obs = 0
        for t in inst.targets_copy():
            if v7._bool_attr(t, "is_relative_detector_id"):
                mask ^= 1 << v7._target_value(t)
            elif v7._bool_attr(t, "is_logical_observable_id"):
                if v7._target_value(t) != 0:
                    raise AssertionError(t)
                obs ^= 1
            elif v7._bool_attr(t, "is_separator"):
                raise AssertionError("raw DEM unexpectedly contains separator")
        pr = v7._dem_inst_probability(inst)
        if pr is None:
            raise AssertionError(inst)
        rows.append({
            "row_index": len(rows),
            "dem_instruction_index": idx,
            "mask": int(mask),
            "logical": int(obs),
            "probability": float(pr),
            "is_gauge": bool(abs(float(pr) - 0.5) <= 1e-12),
            "instruction": str(inst),
        })
    return rows


def _first_order_info(v7: Any, raw_dem: Any) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    rows = _raw_rows_with_inst(v7, raw_dem)
    gauge_rows = [r for r in rows if r["is_gauge"]]
    physical = [r for r in rows if not r["is_gauge"]]
    reducer = v7._GaugeReducer([r["mask"] for r in gauge_rows])

    raw_labels: dict[int, set[int]] = {0: {0}}
    canon_labels: dict[int, set[int]] = {0: {0}}
    for r in physical:
        raw_labels.setdefault(r["mask"], set()).add(r["logical"])
        k = reducer.reduce_mask(r["mask"])
        r["canonical_mask"] = int(k)
        canon_labels.setdefault(k, set()).add(r["logical"])

    raw_amb = {k: sorted(v) for k, v in raw_labels.items() if len(v) > 1}
    canon_amb = {k: sorted(v) for k, v in canon_labels.items() if len(v) > 1}
    zero_rows = [r for r in physical if r["canonical_mask"] == 0 and r["logical"] == 1]

    gauge_obs = collections.Counter(r["logical"] for r in gauge_rows)
    meta = {
        "raw_error_rows": len(rows),
        "physical_error_rows": len(physical),
        "gauge_error_rows": len(gauge_rows),
        "gauge_rank": reducer.rank,
        "gauge_logical_label_counts": {str(k): int(v) for k, v in sorted(gauge_obs.items())},
        "raw_unique_syndrome_keys": len(raw_labels),
        "raw_ambiguous_first_order_keys": len(raw_amb),
        "raw_zero_nontrivial_first_order": 1 in raw_labels.get(0, set()),
        "canonical_unique_syndrome_keys": len(canon_labels),
        "canonical_ambiguous_first_order_keys": len(canon_amb),
        "canonical_zero_nontrivial_first_order": 1 in canon_labels.get(0, set()),
        "canonical_zero_logical_physical_rows": len(zero_rows),
        "canonical_ambiguous_examples_hex": [hex(k) for k in sorted(canon_amb)[:30]],
        "raw_ambiguous_examples_hex": [hex(k) for k in sorted(raw_amb)[:30]],
    }
    return meta, zero_rows, reducer


def _stage_boundaries(v7: Any, hirano_src: Path, sahay_stim: Path, channel: str, rounds: int) -> dict[str, int]:
    full, _wrapper, _init_range, probe_start, _online, _terminal = v7.s7b._build_with_wait_rounds(
        hirano_src, channel, rounds
    )
    pre_clean = full[:probe_start]
    lines = v7._read_source(sahay_stim)
    g57 = v7._clean_source_range(lines, v7.G57_START, v7.G57_END)
    g713 = v7._clean_source_range(lines, v7.G713_START, v7.G713_END)
    close = v7._clean_source_range(lines, v7.CLOSE_START, v7.CLOSE_END)
    h = int(pre_clean.num_ticks)
    a = h + int(g57.num_ticks)
    b = a + int(g713.num_ticks)
    c = b + int(close.num_ticks)
    return {
        "hirano_end_tick_exclusive": h,
        "g57_end_tick_exclusive": a,
        "g713_end_tick_exclusive": b,
        "ideal_close_end_tick_exclusive": c,
    }


def _classify_tick(tick: int, bounds: dict[str, int]) -> str:
    if tick < bounds["hirano_end_tick_exclusive"]:
        return "hirano_escape_plus_d5_rounds"
    if tick < bounds["g57_end_tick_exclusive"]:
        return "sahay_d5_to_d7"
    if tick < bounds["g713_end_tick_exclusive"]:
        return "sahay_d7_to_d13"
    if tick < bounds["ideal_close_end_tick_exclusive"]:
        return "ideal_d13_close_unexpected_fault"
    return "terminal_or_unknown"


def _explain_inst(circuit: Any, inst_text: str, bounds: dict[str, int], max_locations: int = 8) -> dict[str, Any]:
    import stim
    # Preserve the exact target set.  Probability is irrelevant to the explanation
    # lookup, but using the original instruction is the least surprising filter.
    filt = stim.DetectorErrorModel(inst_text + ("\n" if not inst_text.endswith("\n") else ""))
    try:
        explained = circuit.explain_detector_error_model_errors(
            dem_filter=filt,
            reduce_to_one_representative_error=False,
        )
    except Exception as ex:
        return {"explain_error": repr(ex), "explained_errors": []}

    out = []
    for ee in explained:
        locs = []
        for loc in list(ee.circuit_error_locations)[:max_locations]:
            s = str(loc)
            m = re.search(r"resolving to\s+([A-Z0-9_]+)", s)
            gate = m.group(1) if m else None
            tick = int(loc.tick_offset)
            frames = []
            for fr in loc.stack_frames:
                frames.append({
                    "instruction_offset": int(fr.instruction_offset),
                    "iteration_index": int(fr.iteration_index),
                    "instruction_repetitions_arg": int(fr.instruction_repetitions_arg),
                })
            locs.append({
                "tick_offset": tick,
                "stage": _classify_tick(tick, bounds),
                "gate_guess": gate,
                "flipped_pauli_product": [str(x) for x in loc.flipped_pauli_product],
                "flipped_measurement": None if loc.flipped_measurement is None else str(loc.flipped_measurement),
                "stack_frames": frames,
                "text": s,
            })
        out.append({
            "dem_error_terms": [str(x) for x in ee.dem_error_terms],
            "circuit_error_location_count": len(ee.circuit_error_locations),
            "representative_locations": locs,
        })
    return {"explained_errors": out}


def _diagnose_channel(v7: Any, hirano_src: Path, sahay_stim: Path, channel: str, p: float, rounds: int, max_explanations: int) -> dict[str, Any]:
    _set_wait_rounds(v7, rounds)
    circuit, build_meta = v7._build_channel(hirano_src, sahay_stim, channel, p)
    raw = circuit.detector_error_model(decompose_errors=False, allow_gauge_detectors=True)
    info, zero_rows, _reducer = _first_order_info(v7, raw)
    bounds = _stage_boundaries(v7, hirano_src, sahay_stim, channel, rounds)

    explained = []
    for r in zero_rows[:max_explanations]:
        x = {
            "row_index": r["row_index"],
            "raw_mask_hex": hex(r["mask"]),
            "canonical_mask_hex": hex(r["canonical_mask"]),
            "logical": r["logical"],
            "probability": r["probability"],
            "instruction": r["instruction"],
        }
        x.update(_explain_inst(circuit, r["instruction"], bounds))
        explained.append(x)

    stage_counts: dict[str, int] = {}
    for e in explained:
        stages = {loc["stage"] for ee in e.get("explained_errors", []) for loc in ee.get("representative_locations", [])}
        if not stages:
            stage_counts["unexplained"] = stage_counts.get("unexplained", 0) + 1
        else:
            for s in stages:
                stage_counts[s] = stage_counts.get(s, 0) + 1

    return {
        "channel": channel,
        "wait_rounds": rounds,
        "faultable_locations_per_patch": int(build_meta["active_locations"]["total_faultable_clifford_interface"]),
        "detectors": int(build_meta["total_detectors"]),
        "first_order_information": info,
        "tick_stage_boundaries": bounds,
        "explained_canonical_zero_logical_rows": explained,
        "explained_stage_presence_counts": stage_counts,
    }


def main() -> None:
    import collections  # local to keep top-level dependency minimal
    globals()["collections"] = collections

    ap = argparse.ArgumentParser(description="S17-B v9 root-cause explanation and bridge-round sweep")
    ap.add_argument("--v7-script", type=Path, required=True)
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--sahay-stim", type=Path, required=True)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--wait-rounds", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--max-explanations", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s17b_v9_results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    v7 = _load_module(args.v7_script.resolve(), "s17b_v7_for_v9")
    hirano = args.hirano_src.resolve()
    sahay = args.sahay_stim.resolve()

    sweep: dict[str, Any] = {}
    for rounds in args.wait_rounds:
        print(f"S17-B v9: wait_rounds={rounds} first-order sweep...", flush=True)
        sweep[str(rounds)] = {}
        for ch in ["X", "Z"]:
            _set_wait_rounds(v7, rounds)
            c, meta = v7._build_channel(hirano, sahay, ch, float(args.signature_p))
            raw = c.detector_error_model(decompose_errors=False, allow_gauge_detectors=True)
            info, _zero, _red = _first_order_info(v7, raw)
            sweep[str(rounds)][ch] = {
                "faultable_locations_per_patch": int(meta["active_locations"]["total_faultable_clifford_interface"]),
                "detectors": int(meta["total_detectors"]),
                **info,
            }

    # Full physical explanations only for the baseline two-round stitched circuit.
    rootcause = {}
    for ch in ["X", "Z"]:
        print(f"S17-B v9: explaining {ch} canonical-zero logical mechanisms at wait_rounds=2...", flush=True)
        rootcause[ch] = _diagnose_channel(
            v7, hirano, sahay, ch, float(args.signature_p), 2, int(args.max_explanations)
        )
        _dump(args.out_dir / f"s17b_v9_{ch}_rootcause.json", rootcause[ch])

    summary = {
        "stage": "S17-B v9 root-cause + bridge-round first-order sweep",
        "signature_p": float(args.signature_p),
        "wait_rounds_tested": [int(x) for x in args.wait_rounds],
        "sweep": sweep,
        "rootcause_summary": {
            ch: {
                "canonical_zero_logical_physical_rows": rootcause[ch]["first_order_information"]["canonical_zero_logical_physical_rows"],
                "raw_ambiguous_first_order_keys": rootcause[ch]["first_order_information"]["raw_ambiguous_first_order_keys"],
                "canonical_ambiguous_first_order_keys": rootcause[ch]["first_order_information"]["canonical_ambiguous_first_order_keys"],
                "gauge_logical_label_counts": rootcause[ch]["first_order_information"]["gauge_logical_label_counts"],
                "explained_stage_presence_counts": rootcause[ch]["explained_stage_presence_counts"],
            }
            for ch in ["X", "Z"]
        },
        "decision_rule": [
            "If raw ambiguity is zero but canonical ambiguity is nonzero, the information loss is specifically associated with the p=1/2 gauge quotient at the growth boundary.",
            "If wait_rounds=3 removes first-order canonical ambiguity in both channels, use the third d=5 round as a bridge round and proceed to exact order<=2 audit for that architecture.",
            "If ambiguity persists for 3 and 4 rounds, do not keep adding dead-end d=5 rounds; redesign the growth-boundary detector/frame stitching using the explained physical locations.",
            "If gauge_logical_label_counts contains logical=1 gauge rows, the previous gauge reducer is invalid and must be upgraded to quotient joint (syndrome, logical) vectors before interpreting v8.",
        ],
        "scope": [
            "No Monte Carlo.",
            "No graphlike approximation in the first-order information sweep.",
            "The explanation API maps DEM mechanisms back to actual circuit errors; it is diagnostic, not itself a fault-distance certificate.",
        ],
    }
    _dump(args.out_dir / "s17b_v9_summary.json", summary)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "sweep": {
            r: {
                ch: {
                    "raw_amb": sweep[r][ch]["raw_ambiguous_first_order_keys"],
                    "canon_amb": sweep[r][ch]["canonical_ambiguous_first_order_keys"],
                    "canon_zero_L": sweep[r][ch]["canonical_zero_nontrivial_first_order"],
                } for ch in ["X", "Z"]
            } for r in sweep
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
