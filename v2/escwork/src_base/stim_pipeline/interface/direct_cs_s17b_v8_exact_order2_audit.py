from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_v7(path: Path):
    spec = importlib.util.spec_from_file_location("s17b_v7", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import v7: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _labels_to_names(mask: int) -> list[str]:
    out = []
    if mask & 1:
        out.append("I")
    if mask & 2:
        out.append("L")
    return out


def _exact_order2_audit(v7: Any, raw_dem: Any, max_witnesses: int = 20) -> dict[str, Any]:
    rows = v7._raw_error_rows(raw_dem)
    gauge_masks = [int(m) for m, _o, _p, g in rows if g]
    reducer = v7._GaugeReducer(gauge_masks)

    # Collapse raw physical mechanisms only by their canonical syndrome + logical label.
    effect_counts: dict[tuple[int, int], int] = {}
    effect_prob_sum: dict[tuple[int, int], float] = {}
    for mask, obs, pr, is_gauge in rows:
        if is_gauge:
            continue
        key = reducer.reduce_mask(int(mask))
        e = (key, int(obs))
        effect_counts[e] = effect_counts.get(e, 0) + 1
        effect_prob_sum[e] = effect_prob_sum.get(e, 0.0) + float(pr)

    effects = sorted(effect_counts)

    # Label bit mask: bit0 = logical 0 exists, bit1 = logical 1 exists.
    labels: dict[int, int] = {0: 1}
    witnesses: dict[int, dict[str, Any]] = {0: {"order": 0, "effects": []}}

    def add(key: int, logical: int, witness: dict[str, Any]) -> None:
        bit = 1 << int(logical)
        old = labels.get(key, 0)
        new = old | bit
        labels[key] = new
        if old == 0:
            witnesses[key] = witness
        elif new == 3 and old != 3:
            # Keep first ambiguity witness pair: previous and new.
            witnesses[key] = {
                "previous": witnesses.get(key),
                "new": witness,
            }

    for i, (k, o) in enumerate(effects):
        add(k, o, {"order": 1, "effects": [{"effect_index": i, "key_hex": hex(k), "logical": o, "multiplicity": effect_counts[(k, o)]}]})

    n = len(effects)
    pair_count = n * (n + 1) // 2
    t0 = time.time()
    for i in range(n):
        k1, o1 = effects[i]
        for j in range(i, n):
            k2, o2 = effects[j]
            k = k1 ^ k2
            o = o1 ^ o2
            if labels.get(k, 0) == 3:
                # Already ambiguous; no need to update witness.
                continue
            add(k, o, {
                "order": 2,
                "effects": [
                    {"effect_index": i, "key_hex": hex(k1), "logical": o1, "multiplicity": effect_counts[(k1, o1)]},
                    {"effect_index": j, "key_hex": hex(k2), "logical": o2, "multiplicity": effect_counts[(k2, o2)]},
                ],
            })

    ambiguous_keys = [k for k, m in labels.items() if m == 3]
    zero_mask = labels.get(0, 0)
    amb_examples = []
    for k in ambiguous_keys[:max_witnesses]:
        amb_examples.append({"canonical_syndrome_hex": hex(k), "witness": witnesses.get(k)})

    single_label_map: dict[int, int] = {0: 1}
    for k, o in effects:
        single_label_map[k] = single_label_map.get(k, 0) | (1 << o)
    single_ambiguous = [k for k, m in single_label_map.items() if m == 3]

    return {
        "gauge_rank": reducer.rank,
        "raw_error_instructions": len(rows),
        "raw_gauge_error_instructions": len(gauge_masks),
        "raw_physical_error_instructions": len(rows) - len(gauge_masks),
        "unique_physical_effects_canonical_syndrome_plus_logical": n,
        "unique_first_order_canonical_syndromes": len(single_label_map),
        "ambiguous_first_order_canonical_syndromes": len(single_ambiguous),
        "zero_canonical_nontrivial_first_order": bool(single_label_map.get(0, 1) & 2),
        "order2_effect_pairs_with_replacement_overapprox": pair_count,
        "unique_order_le2_canonical_syndromes": len(labels),
        "ambiguous_order_le2_canonical_syndromes": len(ambiguous_keys),
        "zero_canonical_logical_labels_order_le2": _labels_to_names(zero_mask),
        "zero_canonical_nontrivial_order_le2": bool(zero_mask & 2),
        "exact_first_order_information_gate_pass": len(single_ambiguous) == 0 and not bool(single_label_map.get(0, 1) & 2),
        "exact_order_le2_information_gate_pass": len(ambiguous_keys) == 0 and not bool(zero_mask & 2),
        "ambiguity_examples": amb_examples,
        "enumeration_seconds": time.time() - t0,
        "scope": [
            "This audit works on raw Stim DEM mechanisms modulo p=0.5 gauge freedom.",
            "Pairs are enumerated over unique canonical physical DEM effects with replacement, so the order-2 set is a conservative over-approximation of location-compatible two-fault events.",
            "PASS establishes information-theoretic syndrome->logical uniqueness for this DEM mechanism model; it does not assert that MWPM is the decoder achieving it.",
            "Use together with the preserved S16 physical fault-class audit for the strongest d_f>=3 composability claim.",
        ],
    }


def _classify_first_order_decoder(v7: Any, raw_dem: Any, decoder: Any, lookup_meta: dict[str, Any]) -> dict[str, Any]:
    rows = v7._raw_error_rows(raw_dem)
    gauge_masks = [int(m) for m, _o, _p, g in rows if g]
    reducer = v7._GaugeReducer(gauge_masks)

    # Reconstruct exact labels directly to classify hybrid failures.
    label_sets: dict[int, set[int]] = {0: {0}}
    for mask, obs, _p, is_gauge in rows:
        if is_gauge:
            continue
        key = reducer.reduce_mask(int(mask))
        label_sets.setdefault(key, set()).add(int(obs))

    physical_rows = [(m, o, p) for m, o, p, g in rows if not g]
    mat = np.zeros((len(physical_rows), v7.EXPECTED_TOTAL_DETECTORS), dtype=np.uint8)
    actual = np.zeros((len(physical_rows), 1), dtype=np.uint8)
    canonical_keys = []
    for i, (mask, obs, _p) in enumerate(physical_rows):
        x = int(mask)
        for d in range(v7.EXPECTED_TOTAL_DETECTORS):
            if (x >> d) & 1:
                mat[i, d] = 1
        actual[i, 0] = int(obs)
        canonical_keys.append(reducer.reduce_mask(x))

    pred = np.asarray(decoder.decode_batch(mat), dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    bad = np.flatnonzero(np.any(pred != actual, axis=1))

    information_limited = 0
    decoder_limited = 0
    examples = []
    for idx in bad:
        k = canonical_keys[int(idx)]
        labs = label_sets.get(k, set())
        if len(labs) > 1:
            information_limited += 1
            cls = "information_limited_ambiguous_key"
        else:
            decoder_limited += 1
            cls = "decoder_limited_unique_key"
        if len(examples) < 30:
            mask, obs, pr = physical_rows[int(idx)]
            examples.append({
                "row_index": int(idx),
                "canonical_syndrome_hex": hex(k),
                "raw_detector_weight": int(mask).bit_count(),
                "actual_logical": int(obs),
                "predicted_logical": int(pred[int(idx), 0]),
                "exact_labels_for_key": sorted(labs),
                "classification": cls,
                "probability": float(pr),
            })

    return {
        "physical_raw_rows": len(physical_rows),
        "hybrid_first_order_failures": len(bad),
        "information_limited_failure_rows": information_limited,
        "decoder_limited_failure_rows": decoder_limited,
        "all_hybrid_failures_are_decoder_limited": len(bad) > 0 and information_limited == 0,
        "exact_lookup_meta_from_v7": lookup_meta,
        "failure_examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="S17-B v8 exact information audit through DEM order <=2")
    ap.add_argument("--v7-script", type=Path, default=Path(__file__).with_name("direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"))
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--sahay-stim", type=Path, required=True)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s17b_v8_results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    v7 = _load_v7(args.v7_script.resolve())
    all_results = {}
    overall_first = True
    overall_order2 = True

    for channel in ["X", "Z"]:
        print(f"S17-B v8: building {channel} stitched circuit and raw DEM...", flush=True)
        c, meta = v7._build_channel(args.hirano_src.resolve(), args.sahay_stim.resolve(), channel, float(args.signature_p))
        raw, graph, decoder, dem_meta = v7._make_matching(c)
        print(f"S17-B v8: {channel} exact DEM order<=2 enumeration...", flush=True)
        exact = _exact_order2_audit(v7, raw)
        cls = _classify_first_order_decoder(v7, raw, decoder, dem_meta["exact_single_lookup"])
        result = {
            "channel": channel,
            "build_meta": meta,
            "dem_matching_meta": dem_meta,
            "first_order_failure_classification": cls,
            "exact_information_audit": exact,
        }
        _dump(args.out_dir / f"s17b_v8_{channel}_exact_order2.json", result)
        all_results[channel] = result
        overall_first &= bool(exact["exact_first_order_information_gate_pass"])
        overall_order2 &= bool(exact["exact_order_le2_information_gate_pass"])
        print(json.dumps({
            "channel": channel,
            "exact_first_order_information_gate_pass": exact["exact_first_order_information_gate_pass"],
            "exact_order_le2_information_gate_pass": exact["exact_order_le2_information_gate_pass"],
            "ambiguous_order_le2": exact["ambiguous_order_le2_canonical_syndromes"],
            "hybrid_failures": cls["hybrid_first_order_failures"],
            "decoder_limited": cls["decoder_limited_failure_rows"],
            "information_limited": cls["information_limited_failure_rows"],
        }, indent=2), flush=True)

    summary = {
        "stage": "S17-B v8 exact operational information audit",
        "exact_first_order_information_gate_pass_both_channels": overall_first,
        "exact_order_le2_information_gate_pass_both_channels": overall_order2,
        "interpretation": (
            "If order<=2 PASSes while v7 hybrid/PyMatching first-order gate fails, the stitched 470-bit operational record is information-sufficient through order 2 and the remaining problem is decoder-limited, not a fault-tolerant interface ambiguity."
        ),
        "channels": all_results,
    }
    _dump(args.out_dir / "s17b_v8_summary.json", summary)
    print(json.dumps({
        "exact_first_order_information_gate_pass_both_channels": overall_first,
        "exact_order_le2_information_gate_pass_both_channels": overall_order2,
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
