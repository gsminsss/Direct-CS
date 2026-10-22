#!/usr/bin/env python3
"""S17-B v13c: joint-Pauli selective-rejection certificate.

Scientific purpose
==================
S17-B v11 established a deterministic 470-detector continuous handoff and
first-order uniqueness in separate X- and Z-reference probes.  S17-B v12 then
showed that correction-only order<=2 uniqueness fails even after restricting to
pairs of distinct physical noise locations, but no hidden zero-syndrome logical
pair exists in either probe.

This script asks the correct question for a postselected magic-state factory:
can one define an *operational selective-rejection policy* such that every
location-compatible physical fault set of weight <=2 is either

    (a) accepted with a uniquely determined final logical Pauli frame, or
    (b) rejected from its nonzero detector record?

Unlike v12, v13 does not combine X and Z only at the DEM-effect level.  It first
matches the SAME underlying physical Pauli fault mechanism across the X-probe
(|+> / X_L scoring) and Z-probe (|0> / Z_L scoring) circuits.  The two input
circuits are required to have the same detector definitions and the same 2407
physical stochastic noise locations.  For every concrete circuit fault
mechanism, the detector syndrome must agree between probes.

Probe-to-Pauli convention
=========================
* X-reference probe observable X_L flips under logical Z/Y, so its logical bit
  is the final logical Z-component.
* Z-reference probe observable Z_L flips under logical X/Y, so its logical bit
  is the final logical X-component.

Thus joint Pauli label = x_bit | (z_bit << 1):
    0=I, 1=X, 2=Z, 3=Y.

Outputs
=======
* exact joint first-order audit;
* exact location-compatible order<=2 joint syndrome->Pauli label set;
* reject set = all nonzero syndromes with more than one possible Pauli label;
* certificate that all accepted order<=2 records have a unique Pauli frame;
* leading low-weight rejection enumerators W1 and W2 under the implemented
  independent-location noise model;
* stage-resolved first-order rejection burden;
* representative physical witnesses.

Important scope
===============
This is a low-order structural/postselection certificate, not a finite-p E2E
performance simulation.  In particular, at p~1e-3 the weight>=3 probability is
not negligible for N=2407.  The W1/W2 values quantify low-weight rejection
burden and are inputs to the later S17C finite-p simulation.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PAULI_NAMES = {0: "I", 1: "X", 2: "Z", 3: "Y"}
EXPECTED_LOCATIONS = 2407
EXPECTED_DETECTORS = 470
# Boundaries already audited in S17-B v9/v11 for wait_rounds=2.
HIRANO_END_TICK_EXCLUSIVE = 44
G57_END_TICK_EXCLUSIVE = 71
G713_END_TICK_EXCLUSIVE = 77

# Canonical physical-location identity is Stim instruction offset + target range.
# Metadata is populated independently from the flat v11 circuit text.
_LOCATION_META: dict[tuple[int, int, int], dict[str, Any]] = {}


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


def _effect_from_terms(terms: Iterable[Any]) -> tuple[int, int]:
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
        raise AssertionError(f"Unexpected DEM term: {s}")
    return int(mask), int(logical)


def _detector_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip().startswith("DETECTOR")]


def _noise_locations_from_text(path: Path) -> tuple[list[int], dict[tuple[int, int, int], int], dict[int, dict[str, Any]]]:
    """Parse flat stochastic locations and canonicalize them by physical ordinal.

    Stim's native CircuitErrorLocation key (instruction_offset, target_range) is
    probe-specific because X/Z reference circuits can differ in noiseless Clifford
    instruction count.  The physical stochastic schedule, however, is identical.
    Therefore each probe maps its native key to the ordinal of that stochastic
    opportunity (0..2406), and joint X/Z auditing uses that ordinal.
    """
    tick = 0
    ordinals: list[int] = []
    native_to_ord: dict[tuple[int, int, int], int] = {}
    meta_by_ord: dict[int, dict[str, Any]] = {}
    instruction_offset = -1
    lines = path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        instruction_offset += 1
        if s.startswith("REPEAT") or s == "}":
            raise RuntimeError("v13c flat textual location parser does not accept REPEAT blocks")
        if s == "TICK":
            tick += 1
            continue
        m = re.match(r"(DEPOLARIZE2|DEPOLARIZE1|X_ERROR|Z_ERROR)\([^)]*\)\s+(.*)$", s)
        if not m:
            continue
        gate = m.group(1)
        qs = [int(x) for x in re.findall(r"(?<![.\d])(\d+)(?:\[[^]]*\])?", m.group(2))]
        if gate == "DEPOLARIZE2":
            if len(qs) % 2:
                raise AssertionError((gate, qs, s))
            for k in range(0, len(qs), 2):
                a, b = k + 1, k + 2
                native = (instruction_offset, a, b)
                ordinal = len(ordinals)
                if native in native_to_ord:
                    raise AssertionError(("duplicate native location", native))
                native_to_ord[native] = ordinal
                ordinals.append(ordinal)
                meta_by_ord[ordinal] = {
                    "tick": tick,
                    "gate": gate,
                    "qubits": (qs[k], qs[k + 1]),
                    "native_instruction_offset": instruction_offset,
                    "native_target_range": (a, b),
                }
        else:
            for k, q in enumerate(qs):
                a = b = k + 1
                native = (instruction_offset, a, b)
                ordinal = len(ordinals)
                if native in native_to_ord:
                    raise AssertionError(("duplicate native location", native))
                native_to_ord[native] = ordinal
                ordinals.append(ordinal)
                meta_by_ord[ordinal] = {
                    "tick": tick,
                    "gate": gate,
                    "qubits": (q,),
                    "native_instruction_offset": instruction_offset,
                    "native_target_range": (a, b),
                }
    return ordinals, native_to_ord, meta_by_ord

def _all_mechanism_keys(location_sequence: list[tuple[int, int, int]]) -> set[tuple[Any, ...]]:
    """Enumerate every concrete non-identity Pauli outcome at each noise location."""
    out: set[tuple[Any, ...]] = set()
    paulis = ("I", "X", "Y", "Z")
    for loc in location_sequence:
        meta = _LOCATION_META[loc]
        gate = str(meta["gate"])
        qs = tuple(int(x) for x in meta["qubits"])
        if gate == "DEPOLARIZE2":
            q1, q2 = qs
            for p1 in paulis:
                for p2 in paulis:
                    if p1 == p2 == "I":
                        continue
                    prod = []
                    if p1 != "I":
                        prod.append((q1, p1))
                    if p2 != "I":
                        prod.append((q2, p2))
                    out.add((loc, tuple(sorted(prod))))
        elif gate == "DEPOLARIZE1":
            for pp in ("X", "Y", "Z"):
                out.add((loc, ((qs[0], pp),)))
        elif gate == "X_ERROR":
            out.add((loc, ((qs[0], "X"),)))
        elif gate == "Z_ERROR":
            out.add((loc, ((qs[0], "Z"),)))
        else:
            raise ValueError((loc, meta))
    return out


def _location_gate(loc: tuple[int, int, int]) -> str:
    return str(_LOCATION_META[loc]["gate"])


def _location_tick(loc: tuple[int, int, int]) -> int:
    return int(_LOCATION_META[loc]["tick"])


def _stage_from_tick(tick: int) -> str:
    if tick < HIRANO_END_TICK_EXCLUSIVE:
        return "hirano_escape_plus_d5_rounds"
    if tick < G57_END_TICK_EXCLUSIVE:
        return "sahay_d5_to_d7"
    if tick < G713_END_TICK_EXCLUSIVE:
        return "sahay_d7_to_d13"
    return "unexpected_or_ideal_close"


def _parse_flipped_pauli_product(text: str) -> tuple[tuple[int, str], ...]:
    m = re.search(r"flipped_pauli_product:\s*([^\n]+)", text)
    if not m:
        raise ValueError(f"Missing flipped_pauli_product in CircuitErrorLocation:\n{text}")
    expr = m.group(1).strip()
    if expr in {"I", "1", ""}:
        return tuple()
    terms = []
    for part in expr.split("*"):
        part = part.strip()
        mm = re.match(r"([XYZ])(\d+)", part)
        if not mm:
            raise ValueError((expr, part, text))
        terms.append((int(mm.group(2)), mm.group(1)))
    # Operators are on distinct qubits for Stim's local noise mechanisms; sorting
    # gives a channel-independent canonical representation.
    return tuple(sorted(terms))


def _parse_resolving_noise(text: str) -> tuple[str, tuple[int, ...]]:
    m = re.search(r"resolving to\s+([A-Z0-9_]+)\([^)]*\)\s+([^\n]+)", text)
    if not m:
        raise ValueError(f"Missing resolving-to noise line:\n{text}")
    gate = m.group(1)
    rhs = m.group(2)
    qs = tuple(int(x) for x in re.findall(r"(?<![.\d])(\d+)(?:\[[^]]*\])?", rhs))
    if gate == "DEPOLARIZE2" and len(qs) != 2:
        raise AssertionError((gate, qs, text))
    if gate in {"DEPOLARIZE1", "X_ERROR", "Z_ERROR"} and len(qs) != 1:
        raise AssertionError((gate, qs, text))
    return gate, qs


def _mechanism_weight(gate: str) -> float:
    if gate == "DEPOLARIZE2":
        return 1.0 / 15.0
    if gate == "DEPOLARIZE1":
        return 1.0 / 3.0
    if gate in {"X_ERROR", "Z_ERROR"}:
        return 1.0
    raise ValueError(gate)


def _target_range_from_error_location(loc: Any) -> tuple[int, int]:
    txt = str(loc)
    mt = re.search(r"at targets #(\d+)(?: to #(\d+))? of the instruction", txt)
    if not mt:
        raise ValueError(f"Missing target range in CircuitErrorLocation:\n{txt}")
    a = int(mt.group(1))
    b = int(mt.group(2) if mt.group(2) is not None else mt.group(1))
    return a, b


def _location_key_from_error_location(loc: Any, native_to_ord: dict[tuple[int, int, int], int]) -> int:
    """Map Stim-native location identity to the common stochastic ordinal."""
    frames = list(loc.stack_frames)
    if not frames:
        raise ValueError(f"Missing stack frame in CircuitErrorLocation: {loc}")
    fr = frames[-1]
    if int(fr.iteration_index) != 0 or int(fr.instruction_repetitions_arg) != 0:
        raise RuntimeError(f"Unexpected repeated-circuit stack frame in v13c: {fr}")
    a, b = _target_range_from_error_location(loc)
    native = (int(fr.instruction_offset), int(a), int(b))
    if native not in native_to_ord:
        raise RuntimeError(
            f"Stim explanation refers to a native noise location absent from this probe schedule: {native}\n{loc}"
        )
    return int(native_to_ord[native])


def _mechanism_key_from_error_location(loc: Any, native_to_ord: dict[tuple[int, int, int], int]) -> tuple[Any, ...]:
    txt = str(loc)
    lkey = _location_key_from_error_location(loc, native_to_ord)
    pauli = _parse_flipped_pauli_product(txt)
    allowed = set(int(q) for q in _LOCATION_META[lkey]["qubits"])
    if any(int(q) not in allowed for q, _p in pauli):
        raise RuntimeError(
            f"Explained Pauli product acts outside canonical noise targets at ordinal {lkey}: "
            f"allowed={sorted(allowed)} pauli={pauli}\n{txt}"
        )
    return (lkey, pauli)

def _location_json(loc_key: int) -> dict[str, Any]:
    meta = _LOCATION_META[loc_key]
    return {
        "canonical_stochastic_ordinal": int(loc_key),
        "reference_native_instruction_offset": int(meta.get("native_instruction_offset", -1)),
        "reference_native_target_range": [int(x) for x in meta.get("native_target_range", (-1, -1))],
        "tick_offset_textual": int(meta["tick"]),
        "noise_gate": str(meta["gate"]),
        "qubits": [int(x) for x in meta["qubits"]],
        "stage": _stage_from_tick(int(meta["tick"])),
    }


def _mechanism_json(mkey: tuple[Any, ...]) -> dict[str, Any]:
    loc, pp = mkey
    return {
        "location": _location_json(loc),
        "flipped_pauli_product": [{"qubit": int(q), "pauli": str(p)} for q, p in pp],
        "conditional_weight_given_location_fault": _mechanism_weight(_location_gate(loc)),
    }


def _raw_dem(circuit: Any) -> Any:
    return circuit.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)


def _recover_mechanism_map(circuit: Any, raw_dem: Any, probe_name: str, native_to_ord: dict[tuple[int, int, int], int]) -> tuple[dict[tuple[Any, ...], tuple[int, int]], dict[str, Any]]:
    """Map concrete physical Pauli mechanism -> (detector syndrome, probe logical bit)."""
    explained = circuit.explain_detector_error_model_errors(
        dem_filter=raw_dem,
        reduce_to_one_representative_error=False,
    )
    out: dict[tuple[Any, ...], tuple[int, int]] = {}
    locs: set[tuple[Any, ...]] = set()
    duplicate_same = 0
    gate_counts = Counter()
    stage_counts = Counter()

    for ee in explained:
        effect = _effect_from_terms(ee.dem_error_terms)
        for loc in ee.circuit_error_locations:
            mkey = _mechanism_key_from_error_location(loc, native_to_ord)
            lkey = mkey[0]
            locs.add(lkey)
            gate_counts[_location_gate(lkey)] += 1
            stage_counts[_stage_from_tick(_location_tick(lkey))] += 1
            if mkey in out:
                if out[mkey] != effect:
                    raise RuntimeError(f"{probe_name}: same physical mechanism mapped to two DEM effects: {mkey} {out[mkey]} vs {effect}")
                duplicate_same += 1
            else:
                out[mkey] = effect

    if len(locs) != EXPECTED_LOCATIONS:
        raise RuntimeError(f"{probe_name}: recovered {len(locs)} physical locations, expected {EXPECTED_LOCATIONS}")
    return out, {
        "probe": probe_name,
        "explained_error_objects": len(explained),
        "unique_physical_pauli_mechanisms": len(out),
        "unique_physical_noise_locations": len(locs),
        "duplicate_same_mechanism_mappings": duplicate_same,
        "mechanism_gate_counts": dict(sorted(gate_counts.items())),
        "mechanism_stage_counts": dict(sorted(stage_counts.items())),
    }


@dataclass
class JointEffectData:
    syndrome: int
    pauli: int
    # location -> total conditional mechanism weight at that location producing effect
    loc_weight: dict[tuple[Any, ...], float]
    # location -> number of concrete Pauli mechanisms at that location producing effect
    loc_count: dict[tuple[Any, ...], int]
    example_mechanism: tuple[Any, ...]

    @property
    def total_weight(self) -> float:
        return float(sum(self.loc_weight.values()))

    @property
    def total_mechanisms(self) -> int:
        return int(sum(self.loc_count.values()))


def _build_joint_effects(
    x_map: dict[tuple[Any, ...], tuple[int, int]],
    z_map: dict[tuple[Any, ...], tuple[int, int]],
) -> tuple[list[JointEffectData], dict[str, Any]]:
    kx = set(x_map)
    kz = set(z_map)
    if kx != kz:
        only_x = list(kx - kz)[:10]
        only_z = list(kz - kx)[:10]
        raise RuntimeError(f"Physical mechanism sets differ across probes: X-only={only_x}, Z-only={only_z}")

    grouped: dict[tuple[int, int], JointEffectData] = {}
    syndrome_mismatch = []
    location_weight_sum = defaultdict(float)
    location_mech_count = Counter()
    joint_pauli_mech_counts = Counter()

    for mkey in sorted(kx, key=str):
        sx, lx_probe = x_map[mkey]
        sz, lz_probe = z_map[mkey]
        if sx != sz:
            if len(syndrome_mismatch) < 20:
                syndrome_mismatch.append({
                    "mechanism": _mechanism_json(mkey),
                    "x_syndrome_hex": hex(sx),
                    "z_syndrome_hex": hex(sz),
                })
            continue

        # X-probe logical flip = logical Z component.
        # Z-probe logical flip = logical X component.
        xbit = int(lz_probe)
        zbit = int(lx_probe)
        pauli = xbit | (zbit << 1)
        ekey = (int(sx), int(pauli))
        loc = mkey[0]
        w = _mechanism_weight(_location_gate(loc))
        location_weight_sum[loc] += w
        location_mech_count[loc] += 1
        joint_pauli_mech_counts[PAULI_NAMES[pauli]] += 1
        if ekey not in grouped:
            grouped[ekey] = JointEffectData(
                syndrome=int(sx),
                pauli=int(pauli),
                loc_weight={},
                loc_count={},
                example_mechanism=mkey,
            )
        g = grouped[ekey]
        g.loc_weight[loc] = g.loc_weight.get(loc, 0.0) + w
        g.loc_count[loc] = g.loc_count.get(loc, 0) + 1

    if syndrome_mismatch:
        raise RuntimeError(
            f"Same physical fault mechanism produced probe-dependent detector syndromes; joint operational record is invalid. "
            f"Examples: {syndrome_mismatch}"
        )

    bad_location_weights = [(loc, w) for loc, w in location_weight_sum.items() if abs(w - 1.0) > 1e-10]
    if bad_location_weights:
        raise RuntimeError(f"Conditional Pauli weights at physical locations do not sum to 1. Examples: {bad_location_weights[:10]}")
    if len(location_weight_sum) != EXPECTED_LOCATIONS:
        raise RuntimeError((len(location_weight_sum), EXPECTED_LOCATIONS))

    effects = sorted(grouped.values(), key=lambda e: (e.syndrome, e.pauli))
    return effects, {
        "joint_unique_effects": len(effects),
        "joint_physical_pauli_mechanisms": len(kx),
        "physical_locations": len(location_weight_sum),
        "all_same_mechanism_detector_syndromes_match_across_probes": True,
        "per_location_conditional_weights_sum_to_one": True,
        "mechanism_count_per_location_histogram": dict(sorted(Counter(location_mech_count.values()).items())),
        "joint_pauli_mechanism_counts": dict(sorted(joint_pauli_mech_counts.items())),
        "total_first_order_conditional_weight": float(sum(location_weight_sum.values())),
    }


def _pair_weight_distinct_locations(a: JointEffectData, b: JointEffectData, same_effect: bool) -> float:
    A = a.total_weight
    B = b.total_weight
    shared = set(a.loc_weight) & set(b.loc_weight)
    overlap = sum(a.loc_weight[l] * b.loc_weight[l] for l in shared)
    if same_effect:
        return 0.5 * (A * A - overlap)
    return A * B - overlap


def _pair_mechanism_count_distinct_locations(a: JointEffectData, b: JointEffectData, same_effect: bool) -> int:
    A = a.total_mechanisms
    B = b.total_mechanisms
    shared = set(a.loc_count) & set(b.loc_count)
    overlap = sum(a.loc_count[l] * b.loc_count[l] for l in shared)
    if same_effect:
        return (A * A - overlap) // 2
    return A * B - overlap


def _find_distinct_location_witness(a: JointEffectData, b: JointEffectData) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    if not a.loc_weight or not b.loc_weight:
        return None
    la = next(iter(a.loc_weight))
    lb = next(iter(b.loc_weight))
    if la != lb:
        return la, lb
    for x in a.loc_weight:
        if x != lb:
            return x, lb
    for y in b.loc_weight:
        if y != la:
            return la, y
    return None


def _joint_first_order(effects: list[JointEffectData]) -> tuple[dict[int, int], dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    states: dict[int, int] = {0: 1 << 0}
    witnesses: dict[tuple[int, int], dict[str, Any]] = {(0, 0): {"order": 0, "effects": []}}
    for i, e in enumerate(effects):
        bit = 1 << e.pauli
        if not (states.get(e.syndrome, 0) & bit):
            witnesses[(e.syndrome, e.pauli)] = {
                "order": 1,
                "effect_index": i,
                "syndrome_hex": hex(e.syndrome),
                "pauli": PAULI_NAMES[e.pauli],
                "example_mechanism": _mechanism_json(e.example_mechanism),
            }
        states[e.syndrome] = states.get(e.syndrome, 0) | bit
    amb = [s for s, bits in states.items() if bits.bit_count() > 1]
    zero_labels = [p for p in range(4) if states.get(0, 0) & (1 << p)]
    return states, witnesses, {
        "ambiguous_first_order_syndromes": len(amb),
        "zero_syndrome_pauli_labels": [PAULI_NAMES[x] for x in zero_labels],
        "zero_syndrome_nonidentity_first_order": any(x != 0 for x in zero_labels),
        "joint_first_order_information_gate_pass": len(amb) == 0 and zero_labels == [0],
    }


def _enumerate_joint_order2(effects: list[JointEffectData]) -> tuple[dict[int, int], dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    states, witnesses, first = _joint_first_order(effects)
    n = len(effects)
    structural_pairs = 0
    weighted_pairs = 0.0
    mechanism_pairs = 0
    t0 = time.time()

    for i in range(n):
        a = effects[i]
        for j in range(i, n):
            b = effects[j]
            pw = _pair_weight_distinct_locations(a, b, i == j)
            if pw <= 1e-15:
                continue
            structural_pairs += 1
            weighted_pairs += pw
            mechanism_pairs += _pair_mechanism_count_distinct_locations(a, b, i == j)
            s = a.syndrome ^ b.syndrome
            p = a.pauli ^ b.pauli
            bit = 1 << p
            if not (states.get(s, 0) & bit):
                loc_pair = _find_distinct_location_witness(a, b)
                if loc_pair is None:
                    raise AssertionError((i, j, pw))
                witnesses[(s, p)] = {
                    "order": 2,
                    "effects": [
                        {"effect_index": i, "syndrome_hex": hex(a.syndrome), "pauli": PAULI_NAMES[a.pauli]},
                        {"effect_index": j, "syndrome_hex": hex(b.syndrome), "pauli": PAULI_NAMES[b.pauli]},
                    ],
                    "distinct_location_pair": [_location_json(loc_pair[0]), _location_json(loc_pair[1])],
                }
            states[s] = states.get(s, 0) | bit

    expected_pair_weight = math.comb(EXPECTED_LOCATIONS, 2)
    if abs(weighted_pairs - expected_pair_weight) > 1e-6 * max(1.0, expected_pair_weight):
        raise RuntimeError(("order2 conditional weight completeness", weighted_pairs, expected_pair_weight))

    ambiguous = sorted(s for s, bits in states.items() if bits.bit_count() > 1)
    zero_bits = states.get(0, 0)
    zero_labels = [p for p in range(4) if zero_bits & (1 << p)]

    examples = []
    class_counts = Counter()
    for s in ambiguous:
        labels = [p for p in range(4) if states[s] & (1 << p)]
        orders = sorted({int(witnesses[(s, p)]["order"]) for p in labels if (s, p) in witnesses})
        if 1 in orders and 2 in orders:
            class_counts["order1_vs_order2"] += 1
        elif orders == [2]:
            class_counts["order2_vs_order2"] += 1
        else:
            class_counts["other"] += 1
        if len(examples) < 30:
            examples.append({
                "syndrome_hex": hex(s),
                "possible_paulis": [PAULI_NAMES[p] for p in labels],
                "witnesses": {PAULI_NAMES[p]: witnesses.get((s, p)) for p in labels},
            })

    return states, witnesses, {
        "joint_unique_effects": n,
        "location_compatible_joint_effect_pairs": structural_pairs,
        "physical_mechanism_pair_count_distinct_locations": mechanism_pairs,
        "total_order2_conditional_weight": weighted_pairs,
        "expected_total_order2_conditional_weight": expected_pair_weight,
        "unique_order_le2_syndromes": len(states),
        "ambiguous_order_le2_syndromes": len(ambiguous),
        "ambiguity_class_counts": dict(sorted(class_counts.items())),
        "zero_syndrome_pauli_labels": [PAULI_NAMES[x] for x in zero_labels],
        "zero_syndrome_nonidentity_order_le2": any(x != 0 for x in zero_labels),
        "correction_only_joint_order_le2_gate_pass": len(ambiguous) == 0 and zero_labels == [0],
        "ambiguity_examples": examples,
        "enumeration_seconds": time.time() - t0,
        "first_order": first,
    }


def _build_rejection_policy(states: dict[int, int]) -> tuple[set[int], dict[str, Any]]:
    reject = {s for s, bits in states.items() if bits.bit_count() > 1}
    if 0 in reject:
        raise RuntimeError("Zero syndrome is ambiguous; selective rejection cannot preserve the no-fault branch.")
    accepted = {s: bits for s, bits in states.items() if s not in reject}
    bad = [(s, bits) for s, bits in accepted.items() if bits.bit_count() != 1]
    if bad:
        raise AssertionError(bad[:10])
    zero_bits = states.get(0, 0)
    if zero_bits != 1:
        raise RuntimeError(("zero syndrome not uniquely identity", zero_bits))
    return reject, {
        "rejected_ambiguous_syndromes": len(reject),
        "accepted_unique_syndromes": len(accepted),
        "zero_syndrome_is_accepted_and_identity_only": True,
        "all_accepted_order_le2_syndromes_have_unique_joint_pauli": True,
        "postselected_detect_or_correct_df_ge_3_low_order_gate_pass": True,
    }


def _rejection_enumerators(
    effects: list[JointEffectData],
    reject: set[int],
    p_values: list[float],
) -> dict[str, Any]:
    # First-order rejected physical mechanisms / conditional weight.
    W1 = 0.0
    M1 = 0
    first_reject_locations: dict[tuple[Any, ...], float] = defaultdict(float)
    stage_w1 = Counter()
    stage_m1 = Counter()
    rejected_effects_1 = 0
    for e in effects:
        if e.syndrome not in reject:
            continue
        rejected_effects_1 += 1
        W1 += e.total_weight
        M1 += e.total_mechanisms
        for loc, w in e.loc_weight.items():
            first_reject_locations[loc] += w
            stage_w1[_stage_from_tick(_location_tick(loc))] += w
        for loc, c in e.loc_count.items():
            stage_m1[_stage_from_tick(_location_tick(loc))] += c

    # Second-order rejected pair conditional weight.
    W2 = 0.0
    M2 = 0
    rejected_effect_pairs_2 = 0
    t0 = time.time()
    n = len(effects)
    for i in range(n):
        a = effects[i]
        for j in range(i, n):
            b = effects[j]
            s = a.syndrome ^ b.syndrome
            if s not in reject:
                continue
            pw = _pair_weight_distinct_locations(a, b, i == j)
            if pw <= 1e-15:
                continue
            W2 += pw
            M2 += _pair_mechanism_count_distinct_locations(a, b, i == j)
            rejected_effect_pairs_2 += 1

    N = EXPECTED_LOCATIONS
    # Series coefficient of p^2 after expanding the exactly-one-fault term.
    B1 = W1
    B2_series = W2 - (N - 1) * W1

    low_weight = {}
    for p in p_values:
        q = 1.0 - p
        p1 = W1 * p * (q ** (N - 1))
        p2 = W2 * (p ** 2) * (q ** (N - 2))
        total_le2 = (q ** N) + N * p * (q ** (N - 1)) + math.comb(N, 2) * p * p * (q ** (N - 2))
        low_weight[f"{p:.12g}"] = {
            "rejection_probability_from_exactly_one_fault": p1,
            "rejection_probability_from_exactly_two_faults": p2,
            "rejection_probability_from_weight_le2_only": p1 + p2,
            "total_probability_mass_weight_le2": total_le2,
            "fraction_of_weight_le2_mass_rejected": (p1 + p2) / total_le2 if total_le2 else None,
            "weight_ge3_probability": 1.0 - total_le2,
        }

    max_loc_rej_weight = max(first_reject_locations.values(), default=0.0)
    return {
        "W1_rejected_single_fault_conditional_weight": W1,
        "W2_rejected_two_fault_conditional_weight": W2,
        "B1_series_coefficient": B1,
        "B2_series_coefficient": B2_series,
        "note_on_B2": "B2_series is the p^2 Taylor coefficient of rejection probability, including expansion of (1-p)^(N-1) from rejected single-fault trajectories; W2 is usually the more interpretable positive two-fault enumerator.",
        "rejected_concrete_single_pauli_mechanisms": M1,
        "rejected_joint_first_order_effects": rejected_effects_1,
        "physical_locations_with_some_rejected_single_fault_mechanism": len(first_reject_locations),
        "max_rejected_conditional_probability_at_one_location": max_loc_rej_weight,
        "first_order_rejection_weight_by_stage": {k: float(v) for k, v in sorted(stage_w1.items())},
        "first_order_rejected_mechanism_count_by_stage": {k: int(v) for k, v in sorted(stage_m1.items())},
        "rejected_location_compatible_joint_effect_pairs": rejected_effect_pairs_2,
        "rejected_concrete_two_location_pauli_mechanism_pairs": M2,
        "low_weight_rejection_estimates": low_weight,
        "second_pass_enumeration_seconds": time.time() - t0,
        "scope_warning": "These probabilities include only exactly 1- and 2-location faults. They are not the final finite-p factory acceptance because weight>=3 faults and the non-Clifford core acceptance are not included.",
    }


def _find_circuit(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Could not find {name} under {root.resolve()}")
    # Prefer v11_order2_results / v11_results paths, then most recently modified.
    def score(p: Path) -> tuple[int, float]:
        s = str(p).lower()
        pref = 2 if "s17b_v11_order2_results" in s else (1 if "s17b_v11_results" in s else 0)
        return pref, p.stat().st_mtime
    hits.sort(key=score, reverse=True)
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-circuit", type=Path, default=None)
    ap.add_argument("--z-circuit", type=Path, default=None)
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--p-values", nargs="*", type=float, default=[1e-4, 2e-4, 5e-4, 1e-3])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import stim

    x_path = args.x_circuit or _find_circuit(args.search_root, "s17b_v11_X_transition_circuit.stim")
    z_path = args.z_circuit or _find_circuit(args.search_root, "s17b_v11_Z_transition_circuit.stim")
    x_path = x_path.resolve()
    z_path = z_path.resolve()
    print(f"S17-B v13c X circuit: {x_path}", flush=True)
    print(f"S17-B v13c Z circuit: {z_path}", flush=True)

    # Independent schedule/basis guard before invoking Stim explanations.
    x_det = _detector_lines(x_path)
    z_det = _detector_lines(z_path)
    if x_det != z_det or len(x_det) != EXPECTED_DETECTORS:
        raise RuntimeError(f"X/Z detector definitions are not identical 470-bit operational records: {len(x_det)} vs {len(z_det)}")
    x_locs_text, x_native_to_ord, x_meta_native = _noise_locations_from_text(x_path)
    z_locs_text, z_native_to_ord, z_meta_native = _noise_locations_from_text(z_path)
    if len(x_locs_text) != EXPECTED_LOCATIONS or len(z_locs_text) != EXPECTED_LOCATIONS:
        raise RuntimeError((len(x_locs_text), len(z_locs_text), EXPECTED_LOCATIONS))
    # Compare only physical stochastic schedule fields. Native instruction offsets
    # are intentionally probe-specific and are not part of joint physical identity.
    x_phys = [(x_meta_native[i]["tick"], x_meta_native[i]["gate"], tuple(x_meta_native[i]["qubits"])) for i in x_locs_text]
    z_phys = [(z_meta_native[i]["tick"], z_meta_native[i]["gate"], tuple(z_meta_native[i]["qubits"])) for i in z_locs_text]
    if x_phys != z_phys:
        for i, (a, b) in enumerate(zip(x_phys, z_phys)):
            if a != b:
                raise RuntimeError(f"X/Z physical stochastic schedule differs at ordinal {i}: X={a} Z={b}")
        raise RuntimeError("X/Z physical stochastic schedules differ in length")
    # Use X metadata as the common canonical ordinal metadata; retain X native
    # offsets only for human-readable diagnostics.
    _LOCATION_META.clear()
    _LOCATION_META.update(x_meta_native)

    xc = stim.Circuit(x_path.read_text(encoding="utf-8"))
    zc = stim.Circuit(z_path.read_text(encoding="utf-8"))
    if int(xc.num_detectors) != EXPECTED_DETECTORS or int(zc.num_detectors) != EXPECTED_DETECTORS:
        raise RuntimeError((xc.num_detectors, zc.num_detectors))

    print("S17-B v13c: mapping concrete physical Pauli mechanisms in X probe...", flush=True)
    xpartial, xmeta = _recover_mechanism_map(xc, _raw_dem(xc), "X", x_native_to_ord)
    print("S17-B v13c: mapping concrete physical Pauli mechanisms in Z probe...", flush=True)
    zpartial, zmeta = _recover_mechanism_map(zc, _raw_dem(zc), "Z", z_native_to_ord)

    # DEMs omit exactly-trivial fault mechanisms. Reconstruct the complete physical
    # mechanism universe from the common 2407-location noise schedule and fill each
    # omitted probe effect with (syndrome=0, logical=0).
    universe = _all_mechanism_keys(x_locs_text)
    bad_x = set(xpartial) - universe
    bad_z = set(zpartial) - universe
    if bad_x or bad_z:
        raise RuntimeError(f"Explained mechanisms absent from textual physical universe: X={list(bad_x)[:5]} Z={list(bad_z)[:5]}")
    xmap = {m: xpartial.get(m, (0, 0)) for m in universe}
    zmap = {m: zpartial.get(m, (0, 0)) for m in universe}
    xmeta["complete_physical_mechanism_universe"] = len(universe)
    zmeta["complete_physical_mechanism_universe"] = len(universe)
    xmeta["exactly_trivial_mechanisms_omitted_by_raw_dem"] = len(universe) - len(xpartial)
    zmeta["exactly_trivial_mechanisms_omitted_by_raw_dem"] = len(universe) - len(zpartial)

    print("S17-B v13c: joining X/Z mechanisms and verifying detector-syndrome identity...", flush=True)
    effects, joint_meta = _build_joint_effects(xmap, zmap)

    states1, _w1, first = _joint_first_order(effects)
    if not first["joint_first_order_information_gate_pass"]:
        raise RuntimeError(f"Joint first-order gate unexpectedly failed: {first}")

    print(f"S17-B v13c: {len(effects)} joint effects; enumerating location-compatible order<=2 joint Pauli records...", flush=True)
    states, witnesses, order2 = _enumerate_joint_order2(effects)
    reject, policy = _build_rejection_policy(states)
    print(f"S17-B v13c: {len(reject)} ambiguous joint records -> selective reject; computing W1/W2 rejection burden...", flush=True)
    rej = _rejection_enumerators(effects, reject, list(args.p_values))

    # Representative reject-list output. Full masks are compact as hex strings.
    reject_hex = [hex(s) for s in sorted(reject)]
    reject_examples = []
    for s in sorted(reject)[:50]:
        labels = [p for p in range(4) if states[s] & (1 << p)]
        reject_examples.append({
            "syndrome_hex": hex(s),
            "possible_paulis": [PAULI_NAMES[p] for p in labels],
            "witnesses": {PAULI_NAMES[p]: witnesses.get((s, p)) for p in labels},
        })

    summary = {
        "stage": "S17-B v13c joint-Pauli selective-rejection low-order certificate",
        "stim_version": getattr(stim, "__version__", "unknown"),
        "inputs": {"X": str(x_path), "Z": str(z_path)},
        "schedule_guard": {
            "identical_detector_definition_lines": True,
            "detectors": EXPECTED_DETECTORS,
            "identical_physical_noise_location_sequence": True,
            "physical_noise_locations": EXPECTED_LOCATIONS,
        },
        "probe_semantics": {
            "X_probe_logical_bit_is": "final logical Z-component",
            "Z_probe_logical_bit_is": "final logical X-component",
            "joint_label_encoding": {"0": "I", "1": "X", "2": "Z", "3": "Y"},
        },
        "mechanism_mapping": {"X": xmeta, "Z": zmeta, "joint": joint_meta},
        "joint_first_order": first,
        "joint_order_le2": order2,
        "selective_rejection_policy": policy,
        "rejection_enumerators": rej,
        "main_gate": {
            "zero_syndrome_identity_only_through_order2": not order2["zero_syndrome_nonidentity_order_le2"] and order2["zero_syndrome_pauli_labels"] == ["I"],
            "all_accepted_order_le2_records_have_unique_joint_pauli": policy["all_accepted_order_le2_syndromes_have_unique_joint_pauli"],
            "postselected_detect_or_correct_df_ge_3_low_order_gate_pass": policy["postselected_detect_or_correct_df_ge_3_low_order_gate_pass"] and not order2["zero_syndrome_nonidentity_order_le2"],
        },
        "interpretation": [
            "A PASS is a low-order structural certificate for the continuous Clifford handoff under the specified active-location Pauli noise model: every physical fault set of weight <=2 is either correctable by a unique Pauli frame or mapped to a nonzero rejected operational record.",
            "This does not yet certify the full non-Clifford direct-CS factory or finite-p performance; S17C must combine the non-Clifford core, two output patches, this rejection policy, noisy growth, and downstream QEC.",
            "W1/W2 quantify the additional low-weight rejection burden. At p near 1e-3, do not use the weight<=2 truncation as the final acceptance because weight>=3 probability is substantial for N=2407.",
        ],
    }

    _dump(args.out_dir / "s17b_v13c_summary.json", summary)
    _dump(args.out_dir / "s17b_v13c_reject_policy.json", {
        "detector_bits": EXPECTED_DETECTORS,
        "rejected_syndrome_count": len(reject),
        "rejected_syndromes_hex": reject_hex,
        "examples": reject_examples,
    })
    _dump(args.out_dir / "s17b_v13c_mechanism_mapping_meta.json", summary["mechanism_mapping"])

    print(json.dumps({
        "postselected_detect_or_correct_df_ge_3_low_order_gate_pass": summary["main_gate"]["postselected_detect_or_correct_df_ge_3_low_order_gate_pass"],
        "joint_ambiguous_order_le2_records_rejected": policy["rejected_ambiguous_syndromes"],
        "W1_rejected_single_fault_conditional_weight": rej["W1_rejected_single_fault_conditional_weight"],
        "W2_rejected_two_fault_conditional_weight": rej["W2_rejected_two_fault_conditional_weight"],
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
