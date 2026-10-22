#!/usr/bin/env python3
"""S17-B v12: physical-location-aware order<=2 information audit.

Purpose
=======
S17-B v11 established that, after replacing the invalid fresh-input Sahay
boundary annotations by the complete deterministic missing-detector basis,
all first-order raw DEM signatures are unique in both X and Z channels.

The optional v11 order<=2 audit deliberately over-approximated by XORing every
pair of unique DEM effects *with replacement*.  This includes combinations of
two mutually-exclusive Pauli outcomes from the same physical noise location.
Therefore a v11 order-2 FAIL is not a physical d_f<3 certificate.

This script uses Stim's explain_detector_error_model_errors() mapping to recover,
for each unique DEM effect, the set of actual circuit noise locations capable of
producing it.  It then enumerates effect pairs only when there exists a pair of
*distinct* physical noise locations supporting the two effects.

No Monte Carlo. No MWPM. No gauge quotient. The input v11 circuits must already
admit allow_gauge_detectors=False detector error models.

Interpretation
==============
* PASS: every location-compatible order<=2 physical effect has a unique logical
  label given the 470-bit deterministic history.  This is the correct next
  structural certificate before a source-level physical fault-class audit.
* FAIL with zero-syndrome logical: serious hidden order-2 logical mechanism.
* FAIL only at nonzero syndromes: correction-only uniqueness fails, but a
  postselected policy may still recover d_f>=3 by rejecting those operational
  records.  Such a policy must then be audited for physical realizability and
  finite-p acceptance cost.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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
    m = re.match(r"^[DL](-?\d+)", s)
    if m:
        return int(m.group(1))
    raise TypeError(f"Can't read target value: {t!r}")


def _dem_effects(raw_dem: Any) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict[str, Any]]]:
    effects = []
    meta = {}
    for inst in raw_dem.flattened():
        if getattr(inst, "type", None) != "error":
            continue
        mask = 0
        logical = 0
        for t in inst.targets_copy():
            if _bool_attr(t, "is_relative_detector_id"):
                mask ^= 1 << _target_value(t)
            elif _bool_attr(t, "is_logical_observable_id"):
                if _target_value(t) != 0:
                    raise AssertionError(t)
                logical ^= 1
            elif _bool_attr(t, "is_separator"):
                raise AssertionError("raw DEM unexpectedly decomposed")
        args = list(inst.args_copy()) if hasattr(inst, "args_copy") else []
        p = float(args[0]) if args else None
        if p is not None and abs(p - 0.5) <= 1e-12:
            raise AssertionError("v12 requires a gauge-free v11 circuit; found p=0.5 DEM row")
        key = (int(mask), int(logical))
        if key in meta:
            # v11 currently has one unique DEM row per syndrome/effect.  Duplicates
            # would still be logically harmless for this audit but should be made
            # explicit instead of silently changing semantics.
            meta[key]["dem_row_multiplicity"] += 1
            meta[key]["probabilities"].append(p)
        else:
            effects.append(key)
            meta[key] = {
                "dem_row_multiplicity": 1,
                "probabilities": [p],
                "instruction": str(inst),
            }
    effects.sort()
    return effects, meta


def _effect_from_explained_terms(terms: Any) -> tuple[int, int]:
    mask = 0
    logical = 0
    for t in terms:
        s = str(t)
        md = re.match(r"^D(\d+)", s)
        if md:
            mask ^= 1 << int(md.group(1))
            continue
        ml = re.match(r"^L(\d+)", s)
        if ml:
            if int(ml.group(1)) != 0:
                raise AssertionError(s)
            logical ^= 1
            continue
        raise AssertionError(f"Unexpected explained DEM term: {s}")
    return int(mask), int(logical)


def _location_id(loc: Any) -> tuple[Any, ...]:
    """Canonical identity of one physical stochastic noise opportunity.

    A batched Stim error instruction can represent many independent active
    locations.  CircuitErrorLocation's stack frame identifies the instruction;
    the textual target range identifies the specific target or target pair.
    """
    s = str(loc)
    mt = re.search(r"at targets #(\d+)(?: to #(\d+))? of the instruction", s)
    if mt:
        a = int(mt.group(1))
        b = int(mt.group(2) if mt.group(2) is not None else mt.group(1))
    else:
        # Stable fallback.  Keep the full 'resolving to' line so two target
        # groups in the same instruction are not accidentally merged.
        mr = re.search(r"resolving to\s+([^\n]+)", s)
        a = b = mr.group(1).strip() if mr else s

    frames = []
    for fr in loc.stack_frames:
        frames.append((
            int(fr.instruction_offset),
            int(fr.iteration_index),
            int(fr.instruction_repetitions_arg),
        ))
    return (int(loc.tick_offset), tuple(frames), a, b)


def _location_json(loc_id: tuple[Any, ...]) -> dict[str, Any]:
    tick, frames, a, b = loc_id
    return {
        "tick_offset": int(tick),
        "stack_frames": [
            {
                "instruction_offset": int(x[0]),
                "iteration_index": int(x[1]),
                "instruction_repetitions_arg": int(x[2]),
            }
            for x in frames
        ],
        "target_range": [a, b],
    }


def _recover_effect_location_support(circuit: Any, raw_dem: Any, effects: list[tuple[int, int]]) -> tuple[dict[tuple[int, int], set[tuple[Any, ...]]], dict[str, Any]]:
    print("  mapping DEM effects back to circuit fault locations...", flush=True)
    explained = circuit.explain_detector_error_model_errors(
        dem_filter=raw_dem,
        reduce_to_one_representative_error=False,
    )

    effect_set = set(effects)
    support: dict[tuple[int, int], set[tuple[Any, ...]]] = {e: set() for e in effects}
    circuit_error_mechanisms = 0
    duplicate_explained_effects = 0
    seen_explained = set()
    location_gate_counts = Counter()
    location_fault_examples: dict[tuple[Any, ...], str] = {}

    for ee in explained:
        e = _effect_from_explained_terms(ee.dem_error_terms)
        if e not in effect_set:
            # The filter should prevent this.  Fail loudly because accepting an
            # unexplained mismatch would invalidate the compatibility audit.
            raise AssertionError(("explained effect absent from raw DEM", e, [str(x) for x in ee.dem_error_terms]))
        if e in seen_explained:
            duplicate_explained_effects += 1
        seen_explained.add(e)
        for loc in ee.circuit_error_locations:
            circuit_error_mechanisms += 1
            lid = _location_id(loc)
            support[e].add(lid)
            txt = str(loc)
            mg = re.search(r"resolving to\s+([A-Z0-9_]+)", txt)
            if mg:
                location_gate_counts[mg.group(1)] += 1
            location_fault_examples.setdefault(lid, txt.splitlines()[-1].strip() if txt.splitlines() else txt)

    missing = [e for e in effects if not support[e]]
    if missing:
        raise RuntimeError(
            f"{len(missing)} raw DEM effects had no one-circuit-fault explanation; "
            "location-aware pair completeness cannot be certified. "
            f"First examples: {[(hex(s), l) for s,l in missing[:10]]}"
        )

    all_locations = set()
    for ls in support.values():
        all_locations.update(ls)

    multiplicities = Counter(len(v) for v in support.values())
    meta = {
        "explained_error_objects": len(explained),
        "raw_unique_effects": len(effects),
        "explained_unique_effects": len(seen_explained),
        "duplicate_explained_effect_objects": duplicate_explained_effects,
        "circuit_error_mechanisms_in_explanations": circuit_error_mechanisms,
        "unique_nontrivial_physical_noise_locations_seen": len(all_locations),
        "effect_location_support_size_histogram": {str(k): int(v) for k, v in sorted(multiplicities.items())},
        "location_gate_mechanism_counts": dict(sorted(location_gate_counts.items())),
        "all_raw_effects_have_location_support": not missing,
    }
    return support, meta


def _compatible_distinct_locations(a: set[tuple[Any, ...]], b: set[tuple[Any, ...]]) -> tuple[bool, tuple[tuple[Any, ...], tuple[Any, ...]] | None]:
    """Returns whether two effects can occur at distinct physical locations."""
    if not a or not b:
        return False, None
    # Fast path: take arbitrary representatives.
    la = next(iter(a))
    lb = next(iter(b))
    if la != lb:
        return True, (la, lb)
    # If the first representatives coincide, any second location in either set
    # gives a valid distinct pair.
    for x in a:
        if x != lb:
            return True, (x, lb)
    for y in b:
        if y != la:
            return True, (la, y)
    return False, None


def _location_aware_order2(effects: list[tuple[int, int]], support: dict[tuple[int, int], set[tuple[Any, ...]]]) -> dict[str, Any]:
    # syndrome -> logical-label bitset; include no-fault and all order-1 effects.
    states: dict[int, int] = {0: 1}
    witnesses: dict[tuple[int, int], dict[str, Any]] = {(0, 0): {"order": 0, "effects": []}}
    for i, (s, l) in enumerate(effects):
        bit = 1 << l
        if not (states.get(s, 0) & bit):
            witnesses[(s, l)] = {
                "order": 1,
                "effects": [{"effect_index": i, "syndrome_hex": hex(s), "logical": l}],
            }
        states[s] = states.get(s, 0) | bit

    n = len(effects)
    total_pairs = n * (n + 1) // 2
    compatible_pairs = 0
    incompatible_pairs = 0
    t0 = time.time()

    # Keep first witness for each newly-created logical label.  This gives exact
    # physical-location examples for ambiguous operational records.
    for i in range(n):
        e1 = effects[i]
        s1, l1 = e1
        a = support[e1]
        for j in range(i, n):
            e2 = effects[j]
            ok, loc_pair = _compatible_distinct_locations(a, support[e2])
            if not ok:
                incompatible_pairs += 1
                continue
            compatible_pairs += 1
            s2, l2 = e2
            s = s1 ^ s2
            l = l1 ^ l2
            bit = 1 << l
            if not (states.get(s, 0) & bit):
                assert loc_pair is not None
                witnesses[(s, l)] = {
                    "order": 2,
                    "effects": [
                        {"effect_index": i, "syndrome_hex": hex(s1), "logical": l1},
                        {"effect_index": j, "syndrome_hex": hex(s2), "logical": l2},
                    ],
                    "distinct_location_pair": [_location_json(loc_pair[0]), _location_json(loc_pair[1])],
                }
            states[s] = states.get(s, 0) | bit

    amb = sorted(s for s, bits in states.items() if bits == 3)
    zero_bits = states.get(0, 0)

    examples = []
    for s in amb[:30]:
        examples.append({
            "syndrome_hex": hex(s),
            "logical_0_witness": witnesses.get((s, 0)),
            "logical_1_witness": witnesses.get((s, 1)),
        })

    order1_vs_order2 = 0
    order2_vs_order2 = 0
    for s in amb:
        w0 = witnesses.get((s, 0), {})
        w1 = witnesses.get((s, 1), {})
        o0 = int(w0.get("order", -1))
        o1 = int(w1.get("order", -1))
        if 1 in (o0, o1) and 2 in (o0, o1):
            order1_vs_order2 += 1
        elif o0 == 2 and o1 == 2:
            order2_vs_order2 += 1

    return {
        "unique_physical_effects": n,
        "all_effect_pairs_with_replacement": total_pairs,
        "location_compatible_distinct_location_effect_pairs": compatible_pairs,
        "excluded_same_location_only_effect_pairs": incompatible_pairs,
        "unique_order_le2_syndromes": len(states),
        "ambiguous_order_le2_syndromes": len(amb),
        "ambiguous_order1_vs_order2_syndromes": order1_vs_order2,
        "ambiguous_order2_vs_order2_syndromes": order2_vs_order2,
        "zero_syndrome_logical_labels": [x for x in (0, 1) if zero_bits & (1 << x)],
        "zero_syndrome_nontrivial_order_le2": bool(zero_bits & 2),
        "location_aware_order_le2_information_gate_pass": len(amb) == 0 and not (zero_bits & 2),
        "ambiguity_examples": examples,
        "enumeration_seconds": time.time() - t0,
    }


def _analyze(path: Path, channel: str) -> dict[str, Any]:
    import stim
    c = stim.Circuit(path.read_text(encoding="utf-8"))
    raw = c.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    effects, dem_meta = _dem_effects(raw)

    # v11 first-order PASS should imply one logical label per syndrome and no
    # hidden zero-syndrome logical. Recheck independently here.
    labels: dict[int, set[int]] = defaultdict(set)
    for s, l in effects:
        labels[s].add(l)
    amb1 = [s for s, vs in labels.items() if len(vs) > 1]
    if amb1 or 1 in labels.get(0, set()):
        raise RuntimeError(f"{channel}: input circuit is not a v11 first-order PASS circuit")

    support, explain_meta = _recover_effect_location_support(c, raw, effects)
    print(f"  {channel}: {len(effects)} effects; enumerating location-compatible pairs...", flush=True)
    order2 = _location_aware_order2(effects, support)
    return {
        "channel": channel,
        "stim_version": getattr(stim, "__version__", "unknown"),
        "circuit_file": str(path),
        "num_detectors": int(c.num_detectors),
        "num_observables": int(c.num_observables),
        "raw_dem_unique_effects": len(effects),
        "dem_effect_meta": {
            "max_dem_row_multiplicity": max((x["dem_row_multiplicity"] for x in dem_meta.values()), default=0),
            "dem_rows_with_multiplicity_gt1": sum(x["dem_row_multiplicity"] > 1 for x in dem_meta.values()),
        },
        "first_order_recheck": {
            "ambiguous_syndromes": len(amb1),
            "zero_syndrome_nontrivial": 1 in labels.get(0, set()),
            "pass": not amb1 and 1 not in labels.get(0, set()),
        },
        "explanation_mapping": explain_meta,
        "location_aware_order_le2": order2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-circuit", type=Path, required=True)
    ap.add_argument("--z-circuit", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": "S17-B v12 physical-location-aware order<=2 information audit",
        "channels": {},
        "scope": [
            "Input circuits are the deterministic 470-detector v11 stitched circuits.",
            "Stim explain_detector_error_model_errors is used to map each raw DEM effect to every one-circuit-fault location that can produce it.",
            "Order-2 effect pairs are included only when the two effects can be realized at distinct physical stochastic noise locations.",
            "This removes mutually-exclusive same-location Pauli-pair artifacts from the v11 with-replacement over-approximation.",
            "No Monte Carlo, no MWPM, no gauge quotient.",
        ],
    }

    for ch, p in (("X", args.x_circuit), ("Z", args.z_circuit)):
        print(f"S17-B v12: analyzing {ch}...", flush=True)
        out = _analyze(p.resolve(), ch)
        summary["channels"][ch] = out
        _dump(args.out_dir / f"s17b_v12_{ch}_location_order2.json", out)

    summary["location_aware_order_le2_information_gate_pass_both_channels"] = all(
        summary["channels"][ch]["location_aware_order_le2"]["location_aware_order_le2_information_gate_pass"]
        for ch in ("X", "Z")
    )
    summary["zero_syndrome_nontrivial_order_le2_both_channels_absent"] = all(
        not summary["channels"][ch]["location_aware_order_le2"]["zero_syndrome_nontrivial_order_le2"]
        for ch in ("X", "Z")
    )
    summary["decision_rule"] = [
        "If both-channel location-aware order<=2 uniqueness PASSes, proceed to source-level physical fault-class cross-check and then finite-p E2E.",
        "If uniqueness FAILs but zero-syndrome nontrivial remains absent, inspect physical ambiguity witnesses; selective rejection may still certify a postselected d_f>=3 factory.",
        "If a location-compatible zero-syndrome nontrivial order-2 mechanism exists, the current handoff has a genuine hidden two-fault logical failure and must be redesigned before Monte Carlo.",
    ]
    _dump(args.out_dir / "s17b_v12_summary.json", summary)
    print(json.dumps({
        "location_aware_order_le2_information_gate_pass_both_channels": summary["location_aware_order_le2_information_gate_pass_both_channels"],
        "zero_syndrome_nontrivial_order_le2_both_channels_absent": summary["zero_syndrome_nontrivial_order_le2_both_channels_absent"],
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
