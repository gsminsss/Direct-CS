#!/usr/bin/env python3
"""S3/S4 exact physical-fault audit for the Hirano d=5 full-ZXZ escape.

Purpose
-------
Close the missing Step-4 reproducibility gap after S1 (circuit fingerprint)
and S2 (DETECTOR/OBSERVABLE semantics) have passed.

For each complementary logical-transfer channel X/Z this program:
  * rebuilds the public Hirano ZXZ circuit with full postselection;
  * keeps the ideal Steane input preparation in the physical circuit but
    marks it NON-FAULTABLE, matching the historical encoded-input Step-4 scope;
  * instruments exactly the canonical active-location stochastic-Pauli model:
      CX  -> DEPOLARIZE2(p) after the gate, representing all 15 non-II Paulis;
      RZ  -> X_ERROR(p) after reset;
      RX  -> Z_ERROR(p) after reset;
      MZ  -> X_ERROR(p) before measurement (outcome flip);
      MX  -> Z_ERROR(p) before measurement (outcome flip);
    with no idle, leakage, coherent, or other gate noise;
  * uses Stim's detector-error-model explanation API to map every *individual*
    circuit fault mechanism to its detector signature and observable flip;
  * explicitly fills fault mechanisms with zero symptoms, which are omitted
    from a detector error model;
  * verifies the exact inventory 797 locations / 7,279 single fault classes /
    26,439,666 distinct-location two-fault class pairs;
  * verifies historical single-fault acceptance targets (X:450, Z:466) and
    zero accepted malignant single faults;
  * hashes detector signatures to count malignant two-fault pairs exactly,
    without simulating 26,439,666 separate circuits.

Malignant condition under full postselection
--------------------------------------------
For single fault f:
    accepted iff D(f)=0;
    malignant iff D(f)=0 and O(f)=1.

For two faults f,g at DISTINCT physical locations, Clifford/Pauli linearity gives
    D(f,g)=D(f) XOR D(g), O(f,g)=O(f) XOR O(g).
Because every one of the 135 declared detectors is postselected in the public
full-postselection circuit (verified in S2), a pair is malignant iff
    D(f)=D(g) and O(f) != O(g).

The pair audit therefore runs in O(number of fault classes), grouped by detector
signature, while retaining exact distinct-location counting.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hirano_zxz_generator as s1  # noqa: E402

Channel = Literal["X", "Z"]

CANONICAL_LOCATIONS = 797
CANONICAL_SINGLE_CLASSES = 7_279
CANONICAL_PAIR_CLASSES = 26_439_666
CANONICAL_ACCEPTED_SINGLES = {"X": 450, "Z": 466}
CANONICAL_MALIGNANT_SINGLES = {"X": 0, "Z": 0}
CANONICAL_MALIGNANT_PAIRS = {"X": 0, "Z": 0}
EXPECTED_DETECTORS = 135
EXPECTED_OBSERVABLES = 1

PAULIS = ("I", "X", "Y", "Z")
CX_FAULT_CLASSES = tuple(a + b for a in PAULIS for b in PAULIS if a + b != "II")
assert len(CX_FAULT_CLASSES) == 15


@dataclass(frozen=True)
class PhysicalLocation:
    location_id: int
    original_instruction_index: int
    tick: int
    gate: str
    qubits: tuple[int, ...]
    class_count: int
    noise_instruction_index: int
    noise_target_range_start: int


@dataclass
class FaultRow:
    fault_id: int
    location_id: int
    tick: int
    gate: str
    qubits: tuple[int, ...]
    fault_class: str
    detector_mask: int = 0
    detector_weight: int = 0
    observable_flip: int = 0
    explained_by_dem: bool = False

    @property
    def accepted(self) -> bool:
        return self.detector_mask == 0

    @property
    def malignant_single(self) -> bool:
        return self.detector_mask == 0 and self.observable_flip == 1


@dataclass(frozen=True)
class ChannelSummary:
    channel: Channel
    ideal_init_instruction_start: int
    ideal_init_instruction_end: int
    num_detectors: int
    num_observables: int
    postselection_detector_count: int
    postselection_ids_are_all_detectors: bool
    location_counts: dict[str, int]
    physical_locations: int
    single_fault_classes: int
    two_fault_class_pairs: int
    dem_num_detectors: int
    dem_num_observables: int
    dem_explained_signature_groups: int
    faults_explained_by_dem: int
    zero_symptom_faults_filled: int
    unique_detector_signatures: int
    accepted_single_faults: int
    malignant_single_faults: int
    accepted_single_target: int
    malignant_single_target: int
    malignant_two_fault_pairs: int
    malignant_two_fault_target: int
    malignant_pair_witnesses: list[dict[str, Any]]


def _bool_attr(x: Any, name: str) -> bool:
    v = getattr(x, name)
    return bool(v() if callable(v) else v)


def _qubit_value(t: Any) -> int:
    q = getattr(t, "qubit_value", None)
    if callable(q):
        q = q()
    if q is None:
        if not _bool_attr(t, "is_qubit_target"):
            raise AssertionError(f"expected qubit target, got {t!r}")
        q = int(t.value)
    return int(q)


def _pauli_type(gt: Any) -> str:
    p = getattr(gt, "pauli_type", None)
    if callable(p):
        p = p()
    p = str(p)
    if p not in {"I", "X", "Y", "Z"}:
        raise AssertionError(f"unexpected pauli type {p!r} from {gt!r}")
    return p


def _dem_target_flag(t: Any, name: str) -> bool:
    v = getattr(t, name)
    return bool(v() if callable(v) else v)


def _append_noise(c: Any, gate: str, targets: list[int], p: float) -> tuple[int, int]:
    """Append one noise segment and return (instruction_index, target_start).

    Stim is allowed to fuse adjacent instructions having the same gate and the
    same arguments, e.g. two consecutive X_ERROR(p) appends can become one
    X_ERROR(p) instruction with a longer target list.  CircuitErrorLocation
    identifies an error by both instruction_offset and target_range_start, so
    the bookkeeping must preserve the target offset within a fused instruction.
    """
    before = len(c)
    prev_name = None
    prev_args: tuple[float, ...] = ()
    prev_target_count = 0
    if before:
        prev = c[before - 1]
        if hasattr(prev, "name"):
            prev_name = prev.name
            prev_args = tuple(float(x) for x in prev.gate_args_copy())
            prev_target_count = len(prev.targets_copy())

    c.append(gate, targets, p)
    after = len(c)

    if after == before + 1:
        # A new top-level instruction was created.
        inst = c[before]
        if inst.name != gate:
            raise AssertionError(
                f"noise append created unexpected gate: requested={gate}, got={inst.name}"
            )
        return before, 0

    if after == before:
        # Stim fused the append into the previous compatible instruction.
        if before == 0:
            raise AssertionError("Stim reported fusion into an empty circuit")
        inst = c[before - 1]
        args = tuple(float(x) for x in inst.gate_args_copy())
        now_targets = inst.targets_copy()
        if prev_name != gate or inst.name != gate:
            raise AssertionError(
                f"unexpected fusion gate: prev={prev_name}, requested={gate}, now={inst.name}"
            )
        if prev_args != (float(p),) or args != (float(p),):
            raise AssertionError(
                f"unexpected fusion args for {gate}: prev={prev_args}, now={args}, requested={(float(p),)}"
            )
        if len(now_targets) != prev_target_count + len(targets):
            raise AssertionError(
                f"fused target count mismatch for {gate}: before={prev_target_count}, "
                f"added={len(targets)}, after={len(now_targets)}"
            )
        return before - 1, prev_target_count

    raise AssertionError(
        f"noise append unexpectedly changed instruction count: {gate}, len {before}->{after}"
    )


def _self_test_noise_append_fusion() -> None:
    """Verify bookkeeping against Stim's adjacent-instruction fusion behavior."""
    import stim

    c = stim.Circuit()
    i0, s0 = _append_noise(c, "X_ERROR", [0], 0.01)
    i1, s1 = _append_noise(c, "X_ERROR", [1, 2], 0.01)
    if (i0, s0) != (0, 0):
        raise AssertionError(f"X_ERROR self-test first segment wrong: {(i0, s0)}")
    if (i1, s1) != (0, 1):
        raise AssertionError(
            f"X_ERROR fusion bookkeeping wrong: expected (0,1), got {(i1, s1)}"
        )
    if len(c) != 1 or c[0].name != "X_ERROR" or len(c[0].targets_copy()) != 3:
        raise AssertionError(f"X_ERROR fusion self-test unexpected circuit: {c}")

    c.append("TICK")
    j0, t0 = _append_noise(c, "DEPOLARIZE2", [0, 1], 0.02)
    j1, t1 = _append_noise(c, "DEPOLARIZE2", [2, 3], 0.02)
    if (j0, t0) != (2, 0):
        raise AssertionError(f"DEPOLARIZE2 self-test first segment wrong: {(j0, t0)}")
    if (j1, t1) != (2, 2):
        raise AssertionError(
            f"DEPOLARIZE2 fusion bookkeeping wrong: expected (2,2), got {(j1, t1)}"
        )
    if len(c) != 3 or c[2].name != "DEPOLARIZE2" or len(c[2].targets_copy()) != 4:
        raise AssertionError(f"DEPOLARIZE2 fusion self-test unexpected circuit: {c}")


def _count_active_targets(circuit: Any, start: int, end: int) -> dict[str, int]:
    out = {"CX": 0, "RX": 0, "RZ": 0, "MX": 0, "MZ": 0}
    for idx in range(start, end):
        inst = circuit[idx]
        name = inst.name
        ts = inst.targets_copy()
        if name in {"CX", "CNOT"}:
            out["CX"] += len(ts) // 2
        elif name == "RX":
            out["RX"] += len(ts)
        elif name in {"R", "RZ"}:
            out["RZ"] += len(ts)
        elif name == "MX":
            out["MX"] += len(ts)
        elif name in {"M", "MZ"}:
            out["MZ"] += len(ts)
    return out


def build_public_zxz_with_init_range(
    hirano_src: Path,
    *,
    channel: Channel,
) -> tuple[Any, Any, tuple[int, int]]:
    """Build exact public circuit and record the ideal Steane-init instruction range.

    The public wrapper inserts perfect encoded input preparation in the middle of
    the escape circuit. Historical Step-4 kept this preparation to define the
    input state but excluded these locations from the fault budget. We wrap the
    public initialization routine only to record its instruction interval; the
    original function still performs the exact original gates.
    """
    lsed, util = s1._load_hirano(hirano_src)
    steane_code = importlib.import_module("steane_code")
    mapping = util.QubitMapping(30, 30)
    initial_value = lsed.InitialValue.Plus if channel == "X" else lsed.InitialValue.Zero
    fn_name = (
        "perform_perfect_steane_plus_initialization"
        if channel == "X"
        else "perform_perfect_steane_zero_initialization"
    )
    original = getattr(steane_code, fn_name)
    rec: dict[str, tuple[int, int]] = {}

    def wrapped(stim_circuit: Any, mapping_obj: Any) -> None:
        if rec:
            raise AssertionError("ideal Steane initialization called more than once")
        before = len(stim_circuit)
        original(stim_circuit, mapping_obj)
        after = len(stim_circuit)
        rec["range"] = (before, after)

    setattr(steane_code, fn_name, wrapped)
    try:
        runner = lsed.SteanePlusSurfaceCode(
            mapping,
            5,
            initial_value,
            lsed.SteaneSyndromeExtractionPattern.ZXZ,
            0.0,
            True,
        )
        runner.run()
    finally:
        setattr(steane_code, fn_name, original)

    if "range" not in rec:
        raise AssertionError("failed to record ideal Steane initialization range")
    circuit = runner.circuit.circuit
    init_range = rec["range"]

    got = _count_active_targets(circuit, *init_range)
    expected = s1.IDEAL_STEANE_INIT[channel]
    if got != expected:
        raise AssertionError(
            f"{channel}: recorded init range has wrong fingerprint: got={got}, expected={expected}, "
            f"range={init_range}"
        )

    # Keep S1 closed.
    fp = s1.fingerprint(circuit, runner.circuit, channel=channel)
    s1.assert_canonical(fp)
    return circuit, runner.circuit, init_range


def instrument_canonical_fault_model(
    circuit: Any,
    *,
    init_range: tuple[int, int],
    p: float,
) -> tuple[Any, list[PhysicalLocation], dict[tuple[int, int], int]]:
    """Insert exactly the historical Step-4 active-location Pauli model.

    Returns:
        noisy_circuit,
        locations in canonical order,
        map[(noise_instruction_index, noise_target_range_start)] -> location_id.
    """
    import stim

    if not (0 < p < 1):
        raise ValueError("p must satisfy 0<p<1")

    out = stim.Circuit()
    locations: list[PhysicalLocation] = []
    noise_segment_to_location: dict[tuple[int, int], int] = {}
    tick = 0
    init_start, init_end = init_range

    for original_idx, inst in enumerate(circuit):
        # Public generator is Python-unrolled. Refuse hidden repeat blocks so our
        # instruction offsets remain one-to-one and auditable.
        if not hasattr(inst, "name"):
            raise AssertionError(
                f"unexpected non-instruction block at original index {original_idx}: {inst!r}"
            )
        name = inst.name
        targets = inst.targets_copy()
        in_ideal_init = init_start <= original_idx < init_end

        # Measurement outcome flips are represented by the anticommuting Pauli
        # immediately BEFORE the measurement, matching Hirano util.Circuit.
        if not in_ideal_init and name in {"MX", "M", "MZ"}:
            qids = [_qubit_value(t) for t in targets]
            noise_gate = "Z_ERROR" if name == "MX" else "X_ERROR"
            noise_idx, noise_base = _append_noise(out, noise_gate, qids, p)
            for k, q in enumerate(qids):
                loc_id = len(locations)
                gate = "MX" if name == "MX" else "MZ"
                locations.append(
                    PhysicalLocation(
                        loc_id, original_idx, tick, gate, (q,), 1, noise_idx, noise_base + k
                    )
                )
                noise_segment_to_location[(noise_idx, noise_base + k)] = loc_id
            out.append(inst)

        elif not in_ideal_init and name in {"CX", "CNOT"}:
            if len(targets) % 2:
                raise AssertionError(f"odd CX target count at instruction {original_idx}")
            qids = [_qubit_value(t) for t in targets]
            # Hirano's wrapper forbids a qubit participating twice within a tick;
            # assert the batched CX pairs are disjoint before moving their errors
            # after the batched instruction.
            if len(set(qids)) != len(qids):
                raise AssertionError(
                    f"overlapping CX targets in one Stim instruction at {original_idx}: {qids}"
                )
            out.append(inst)
            noise_idx, noise_base = _append_noise(out, "DEPOLARIZE2", qids, p)
            for k in range(0, len(qids), 2):
                loc_id = len(locations)
                pair = (qids[k], qids[k + 1])
                locations.append(
                    PhysicalLocation(
                        loc_id, original_idx, tick, "CX", pair, 15, noise_idx, noise_base + k
                    )
                )
                noise_segment_to_location[(noise_idx, noise_base + k)] = loc_id

        elif not in_ideal_init and name in {"RX", "R", "RZ"}:
            qids = [_qubit_value(t) for t in targets]
            out.append(inst)
            noise_gate = "Z_ERROR" if name == "RX" else "X_ERROR"
            noise_idx, noise_base = _append_noise(out, noise_gate, qids, p)
            for k, q in enumerate(qids):
                loc_id = len(locations)
                gate = "RX" if name == "RX" else "RZ"
                locations.append(
                    PhysicalLocation(
                        loc_id, original_idx, tick, gate, (q,), 1, noise_idx, noise_base + k
                    )
                )
                noise_segment_to_location[(noise_idx, noise_base + k)] = loc_id

        else:
            out.append(inst)

        if name == "TICK":
            tick += 1

    if out.without_noise() != circuit:
        raise AssertionError("instrumented circuit does not reduce exactly to public circuit when noise is removed")
    return out, locations, noise_segment_to_location


def build_fault_rows(locations: list[PhysicalLocation]) -> list[FaultRow]:
    rows: list[FaultRow] = []
    for loc in locations:
        if loc.gate == "CX":
            classes = CX_FAULT_CLASSES
        elif loc.gate == "RX":
            classes = ("Z",)
        elif loc.gate == "RZ":
            classes = ("X",)
        elif loc.gate == "MX":
            classes = ("Z",)
        elif loc.gate == "MZ":
            classes = ("X",)
        else:
            raise AssertionError(loc)
        if len(classes) != loc.class_count:
            raise AssertionError((loc, classes))
        for c in classes:
            rows.append(
                FaultRow(
                    fault_id=len(rows),
                    location_id=loc.location_id,
                    tick=loc.tick,
                    gate=loc.gate,
                    qubits=loc.qubits,
                    fault_class=c,
                )
            )
    return rows


def _signature_from_explained_error(ex: Any) -> tuple[int, int]:
    detector_mask = 0
    observable_flip = 0
    for twc in ex.dem_error_terms:
        t = twc.dem_target
        if _dem_target_flag(t, "is_relative_detector_id"):
            detector_mask ^= 1 << int(t.val)
        elif _dem_target_flag(t, "is_logical_observable_id"):
            oid = int(t.val)
            if oid != 0:
                raise AssertionError(f"unexpected observable id L{oid}")
            observable_flip ^= 1
        elif _dem_target_flag(t, "is_separator"):
            raise AssertionError(
                "separator appeared with decompose_errors=False; refusing ambiguous signature"
            )
        else:
            raise AssertionError(f"unrecognized DEM target {t!r}")
    return detector_mask, observable_flip


def _fault_class_from_error_location(cel: Any, loc: PhysicalLocation) -> str:
    by_qubit: dict[int, str] = {}
    for qwc in cel.flipped_pauli_product:
        gt = qwc.gate_target
        q = int(gt.value)
        p = _pauli_type(gt)
        if p == "I":
            continue
        if q in by_qubit:
            raise AssertionError(f"duplicate Pauli target on q{q}: {cel!r}")
        by_qubit[q] = p

    if loc.gate == "CX":
        a, b = loc.qubits
        label = by_qubit.get(a, "I") + by_qubit.get(b, "I")
        if label not in CX_FAULT_CLASSES:
            raise AssertionError(
                f"cannot map DEPOLARIZE2 mechanism to CX class at loc {loc.location_id}: "
                f"label={label}, error={cel}"
            )
        return label

    # We inject explicit X_ERROR/Z_ERROR for all one-class locations. If Stim
    # chooses to represent the immediate-before-measurement effect as a flipped
    # measurement instead of a Pauli product, the class is still unique.
    if loc.gate in {"RX", "MX"}:
        expected = "Z"
    elif loc.gate in {"RZ", "MZ"}:
        expected = "X"
    else:
        raise AssertionError(loc)
    if by_qubit:
        if by_qubit != {loc.qubits[0]: expected}:
            raise AssertionError(
                f"unexpected single-qubit error product at loc {loc.location_id}: "
                f"got={by_qubit}, expected={expected} on {loc.qubits[0]}"
            )
    elif cel.flipped_measurement is None:
        raise AssertionError(
            f"single-class error has neither Pauli nor flipped measurement: {cel}"
        )
    return expected


def _self_test_explainer_api() -> None:
    """Fail fast if the local Stim explanation API shape is incompatible."""
    import stim

    c = stim.Circuit(
        """
        R 0 1
        DEPOLARIZE2(0.01) 0 1
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    exs = c.explain_detector_error_model_errors(reduce_to_one_representative_error=False)
    if not exs:
        raise AssertionError("Stim explainer self-test produced no explained errors")
    found = False
    for ex in exs:
        _signature_from_explained_error(ex)
        for cel in ex.circuit_error_locations:
            if not cel.stack_frames:
                raise AssertionError("Stim explainer self-test returned empty stack_frames")
            frame = cel.stack_frames[-1]
            if int(frame.instruction_offset) == 1:
                r0 = int(cel.instruction_targets.target_range_start)
                r1 = int(cel.instruction_targets.target_range_end)
                if r0 != 0 or r1 != 2:
                    raise AssertionError(
                        f"unexpected DEPOLARIZE2 target range in self-test: [{r0},{r1})"
                    )
                found = True
    if not found:
        raise AssertionError("Stim explainer self-test could not locate DEPOLARIZE2 mechanism")


def populate_signatures_from_dem(
    noisy: Any,
    locations: list[PhysicalLocation],
    rows: list[FaultRow],
    noise_segment_to_location: dict[tuple[int, int], int],
) -> tuple[Any, int, int]:
    """Fill rows using one noisy-circuit DEM + all physical error explanations."""
    dem = noisy.detector_error_model(decompose_errors=False)
    if dem.num_detectors != EXPECTED_DETECTORS or dem.num_observables != EXPECTED_OBSERVABLES:
        raise AssertionError((dem.num_detectors, dem.num_observables))

    row_by_key = {(r.location_id, r.fault_class): r for r in rows}
    if len(row_by_key) != len(rows):
        raise AssertionError("duplicate fault keys")
    loc_by_id = {l.location_id: l for l in locations}

    explained_groups = noisy.explain_detector_error_model_errors(
        reduce_to_one_representative_error=False
    )
    seen_assignments = 0
    for ex in explained_groups:
        sig = _signature_from_explained_error(ex)
        for cel in ex.circuit_error_locations:
            if len(cel.stack_frames) != 1:
                raise AssertionError(
                    "unexpected repeat/nesting in instrumented circuit error location: "
                    + str(cel)
                )
            noise_instruction_index = int(cel.stack_frames[0].instruction_offset)
            start = int(cel.instruction_targets.target_range_start)
            key0 = (noise_instruction_index, start)
            if key0 not in noise_segment_to_location:
                raise AssertionError(
                    f"Stim explained an error outside canonical inserted noise: key={key0}\n{cel}"
                )
            loc_id = noise_segment_to_location[key0]
            loc = loc_by_id[loc_id]
            fault_class = _fault_class_from_error_location(cel, loc)
            row = row_by_key[(loc_id, fault_class)]
            if row.explained_by_dem:
                if (row.detector_mask, row.observable_flip) != sig:
                    raise AssertionError(
                        f"fault assigned conflicting signatures: loc={loc_id}, class={fault_class}, "
                        f"old={(row.detector_mask,row.observable_flip)}, new={sig}"
                    )
                # Exact duplicate explanation is harmless but should be rare.
                continue
            row.detector_mask, row.observable_flip = sig
            row.detector_weight = int(sig[0].bit_count())
            row.explained_by_dem = True
            seen_assignments += 1

    # A zero-symptom physical fault is intentionally absent from the DEM. Its
    # default row values (D=0,O=0) are the correct exact signature.
    zero_filled = len(rows) - seen_assignments
    return dem, len(explained_groups), zero_filled


def _location_counts(locations: Iterable[PhysicalLocation]) -> dict[str, int]:
    d = {"CX": 0, "RX": 0, "RZ": 0, "MX": 0, "MZ": 0}
    for loc in locations:
        d[loc.gate] += 1
    return d


def _pair_denominator(locations: list[PhysicalLocation]) -> int:
    n = sum(l.class_count for l in locations)
    same_location_ordered = sum(l.class_count * l.class_count for l in locations)
    return (n * n - same_location_ordered) // 2


def exact_malignant_pair_hash(rows: list[FaultRow], *, max_witnesses: int = 20) -> tuple[int, list[dict[str, Any]]]:
    """Count distinct-location pairs with same D but opposite O exactly."""
    groups: dict[int, dict[int, list[FaultRow]]] = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        groups[r.detector_mask][r.observable_flip].append(r)

    malignant = 0
    witnesses: list[dict[str, Any]] = []
    for mask, by_obs in groups.items():
        a = by_obs[0]
        b = by_obs[1]
        if not a or not b:
            continue
        # All opposite-observable cross pairs have D xor D = 0. Exclude pairs
        # whose two fault classes are at the same physical location.
        total = len(a) * len(b)
        a_by_loc: dict[int, list[FaultRow]] = defaultdict(list)
        b_by_loc: dict[int, list[FaultRow]] = defaultdict(list)
        for r in a:
            a_by_loc[r.location_id].append(r)
        for r in b:
            b_by_loc[r.location_id].append(r)
        same_loc = sum(len(a_by_loc[k]) * len(b_by_loc[k]) for k in a_by_loc.keys() & b_by_loc.keys())
        count = total - same_loc
        malignant += count

        if count and len(witnesses) < max_witnesses:
            for r0 in a:
                for r1 in b:
                    if r0.location_id == r1.location_id:
                        continue
                    witnesses.append(
                        {
                            "detector_mask_hex": hex(mask),
                            "fault_a": {
                                "fault_id": r0.fault_id,
                                "location_id": r0.location_id,
                                "tick": r0.tick,
                                "gate": r0.gate,
                                "qubits": list(r0.qubits),
                                "class": r0.fault_class,
                                "observable_flip": r0.observable_flip,
                            },
                            "fault_b": {
                                "fault_id": r1.fault_id,
                                "location_id": r1.location_id,
                                "tick": r1.tick,
                                "gate": r1.gate,
                                "qubits": list(r1.qubits),
                                "class": r1.fault_class,
                                "observable_flip": r1.observable_flip,
                            },
                        }
                    )
                    if len(witnesses) >= max_witnesses:
                        break
                if len(witnesses) >= max_witnesses:
                    break
    return malignant, witnesses


def _write_fault_csv(path: Path, rows: list[FaultRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "fault_id",
                "location_id",
                "tick",
                "gate",
                "qubits",
                "fault_class",
                "detector_mask_hex",
                "detector_weight",
                "detector_ids",
                "observable_flip",
                "accepted",
                "malignant_single",
                "explained_by_dem",
            ]
        )
        for r in rows:
            ids = [i for i in range(EXPECTED_DETECTORS) if (r.detector_mask >> i) & 1]
            w.writerow(
                [
                    r.fault_id,
                    r.location_id,
                    r.tick,
                    r.gate,
                    " ".join(map(str, r.qubits)),
                    r.fault_class,
                    hex(r.detector_mask),
                    r.detector_weight,
                    " ".join(map(str, ids)),
                    r.observable_flip,
                    int(r.accepted),
                    int(r.malignant_single),
                    int(r.explained_by_dem),
                ]
            )


def audit_channel(
    hirano_src: Path,
    *,
    channel: Channel,
    p: float,
    out_dir: Path | None,
    do_assert: bool,
) -> ChannelSummary:
    circuit, wrapper, init_range = build_public_zxz_with_init_range(
        hirano_src, channel=channel
    )
    post_ids = [int(d.id) for d in wrapper.detectors_for_post_selection]
    all_detectors = set(range(circuit.num_detectors))
    post_all = len(post_ids) == circuit.num_detectors and set(post_ids) == all_detectors
    if do_assert and not post_all:
        raise AssertionError(
            f"{channel}: full-postselection detector set is not all detectors: {post_ids}"
        )

    noisy, locations, noise_map = instrument_canonical_fault_model(
        circuit, init_range=init_range, p=p
    )
    rows = build_fault_rows(locations)

    counts = _location_counts(locations)
    singles = len(rows)
    pairs = _pair_denominator(locations)
    if do_assert:
        if len(locations) != CANONICAL_LOCATIONS:
            raise AssertionError(f"{channel}: locations={len(locations)} != {CANONICAL_LOCATIONS}")
        if singles != CANONICAL_SINGLE_CLASSES:
            raise AssertionError(f"{channel}: singles={singles} != {CANONICAL_SINGLE_CLASSES}")
        if pairs != CANONICAL_PAIR_CLASSES:
            raise AssertionError(f"{channel}: pairs={pairs} != {CANONICAL_PAIR_CLASSES}")

    dem, n_groups, zero_filled = populate_signatures_from_dem(
        noisy, locations, rows, noise_map
    )

    accepted = sum(r.accepted for r in rows)
    malignant_single = sum(r.malignant_single for r in rows)
    unique_sigs = len({(r.detector_mask, r.observable_flip) for r in rows})
    malignant_pairs, witnesses = exact_malignant_pair_hash(rows)

    summary = ChannelSummary(
        channel=channel,
        ideal_init_instruction_start=init_range[0],
        ideal_init_instruction_end=init_range[1],
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
        postselection_detector_count=len(post_ids),
        postselection_ids_are_all_detectors=post_all,
        location_counts=counts,
        physical_locations=len(locations),
        single_fault_classes=singles,
        two_fault_class_pairs=pairs,
        dem_num_detectors=dem.num_detectors,
        dem_num_observables=dem.num_observables,
        dem_explained_signature_groups=n_groups,
        faults_explained_by_dem=sum(r.explained_by_dem for r in rows),
        zero_symptom_faults_filled=zero_filled,
        unique_detector_signatures=unique_sigs,
        accepted_single_faults=accepted,
        malignant_single_faults=malignant_single,
        accepted_single_target=CANONICAL_ACCEPTED_SINGLES[channel],
        malignant_single_target=CANONICAL_MALIGNANT_SINGLES[channel],
        malignant_two_fault_pairs=malignant_pairs,
        malignant_two_fault_target=CANONICAL_MALIGNANT_PAIRS[channel],
        malignant_pair_witnesses=witnesses,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_fault_csv(out_dir / f"s3_{channel}_fault_catalog.csv", rows)
        (out_dir / f"s3_{channel}_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        (out_dir / f"s3_{channel}_instrumented_noise.stim").write_text(
            str(noisy), encoding="utf-8"
        )
        (out_dir / f"s3_{channel}_dem.dem").write_text(str(dem), encoding="utf-8")

    if do_assert:
        if accepted != CANONICAL_ACCEPTED_SINGLES[channel]:
            raise AssertionError(
                f"{channel}: accepted single faults={accepted}, historical target="
                f"{CANONICAL_ACCEPTED_SINGLES[channel]}. Outputs were written for diagnosis."
            )
        if malignant_single != 0:
            bad = [r for r in rows if r.malignant_single][:10]
            raise AssertionError(
                f"{channel}: malignant single faults={malignant_single}; examples={bad}"
            )
        if malignant_pairs != 0:
            raise AssertionError(
                f"{channel}: malignant two-fault pairs={malignant_pairs}; witnesses={witnesses[:5]}"
            )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--channel", choices=["X", "Z", "both"], default="both")
    ap.add_argument(
        "--noise-probability",
        type=float,
        default=1e-3,
        help="Arbitrary nonzero probability used only so Stim exposes each fault mechanism; signatures are p-independent.",
    )
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    _self_test_noise_append_fusion()
    _self_test_explainer_api()
    channels: list[Channel] = ["X", "Z"] if args.channel == "both" else [args.channel]
    results = []
    for ch in channels:
        try:
            s = audit_channel(
                args.hirano_src,
                channel=ch,
                p=args.noise_probability,
                out_dir=args.out_dir,
                do_assert=not args.no_assert,
            )
            results.append(asdict(s))
        except Exception:
            # If the first channel fails after outputs were produced, keep the
            # traceback. The user can rerun that channel with --no-assert to get
            # the full diagnostic summary instead of guessing at a patch.
            raise

    print(json.dumps(results, indent=2, sort_keys=True))
    if not args.no_assert:
        print("S3/S4 PASS: exact 7,279-class catalogue + 26,439,666-pair ZXZ audit")


if __name__ == "__main__":
    main()
