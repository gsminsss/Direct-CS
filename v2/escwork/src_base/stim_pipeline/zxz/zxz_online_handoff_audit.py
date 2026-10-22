#!/usr/bin/env python3
"""S7-B online/non-destructive handoff audit for the Hirano d=5 ZXZ escape.

Why this step exists
--------------------
S1-S4 reproduced the historical full-ZXZ audit by appending complementary
*destructive* X/Z readouts. S5.6 then used all 135 detector bits, including 12
terminal detectors produced by those destructive readouts, to build a very
useful selective-certified lookup decoder.

For a real magic-state factory, however, the surface-code output cannot be
measured destructively before it is released. Therefore S7-B asks a stricter
implementation question:

    Using only detector information available BEFORE the terminal destructive
    probe, can the ZXZ output be handed off while retaining a useful certified
    logical Pauli instrument?

The terminal readout is retained in this program ONLY as an ideal/noiseless
scoring probe. Its 25 measurement locations are excluded from the fault model,
and its 12 closing detectors are excluded from the online decision syndrome.

Two handoff depths are especially useful:
  wait_rounds=0 : the historical pre-readout boundary (123 online detectors)
  wait_rounds=1 : add one complete non-destructive surface stabilizer round
                  before the scoring probe (expected 147 online detectors)

For each wait depth, the X- and Z-complementary probe circuits are paired fault
class by fault class. Their observable flips jointly determine the residual
logical Pauli label:

    X-probe flip  Z-probe flip   residual Pauli
         0             0               I
         0             1               X
         1             0               Z
         1             1               Y

This is an exact characterization under the stated active-location stochastic
Pauli model for every single physical fault. Two-fault signatures compose by
GF(2) XOR, so syndrome->Pauli ambiguity through order 2 is audited exactly by
signature-group compression.

Important interpretation
------------------------
The terminal probe is a scoring device, not an available online detector. If a
physical error is only revealed by the terminal boundary, S7-B deliberately
counts that as information unavailable at release time. This makes the audit
conservative and directly tests whether an extra non-destructive handoff round
improves implementability.
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ZXZ_DIR = HERE.parent / "zxz"
for pth in (HERE, ZXZ_DIR):
    if str(pth) not in sys.path:
        sys.path.insert(0, str(pth))

import hirano_zxz_generator as s1  # type: ignore  # noqa: E402
import zxz_exact_fault_audit as s3  # type: ignore  # noqa: E402

D = 5
BASE_PRE_DETECTORS = 123
TERMINAL_DETECTORS = 12
TOTAL_BASE_DETECTORS = 135
HANDOFF_DETECTORS_PER_ROUND = 24
BASE_PRE_LOCATIONS = 772
BASE_PRE_SINGLE_CLASSES = 7254
BASE_PRE_PAIR_CLASSES = 26_258_016
SURFACE_ROUND_CX = 80
SURFACE_ROUND_1Q_LOCATIONS = 48  # 24 ancilla resets + 24 ancilla measurements
SURFACE_ROUND_LOCATIONS = SURFACE_ROUND_CX + SURFACE_ROUND_1Q_LOCATIONS
SURFACE_ROUND_SINGLE_CLASSES = SURFACE_ROUND_CX * 15 + SURFACE_ROUND_1Q_LOCATIONS

PAULI_LABELS = ("I", "X", "Z", "Y")


@dataclass(frozen=True)
class OnlineFault:
    fault_id: int
    location_id: int
    tick: int
    gate: str
    qubits: tuple[int, ...]
    fault_class: str
    syndrome: int
    residual_pauli: int  # 0=I,1=X,2=Z,3=Y; composition is XOR


@dataclass
class SigGroup:
    syndrome: int
    pauli: int
    count: int
    examples_by_location: dict[int, OnlineFault]
    counts_by_location: dict[int, int]

    @property
    def location_count(self) -> int:
        return len(self.examples_by_location)


@dataclass(frozen=True)
class OnlineSummary:
    wait_rounds: int
    online_detector_count: int
    terminal_detector_count: int
    total_detector_count: int
    physical_locations: int
    single_fault_classes: int
    two_fault_class_pairs: int
    location_counts: dict[str, int]
    xz_online_syndrome_mismatches: int
    single_syndromes: int
    ambiguous_single_syndromes: int
    ambiguous_single_fault_classes: int
    zero_syndrome_single_nontrivial_paulis: int
    full_prepost_malignant_singles: int
    full_prepost_malignant_pairs: int
    unique_syndromes_le2: int
    ambiguous_syndromes_le2: int
    zero_syndrome_pauli_set_le2: list[str]
    ideal_zero_syndrome_is_certifiably_acceptable: bool
    selective_rejected_single_fault_classes: int
    selective_single_reject_coefficient: float
    selective_exact_accepted_single_failures: int
    selective_exact_accepted_pair_failures: int
    ambiguity_witnesses: list[dict[str, Any]]


def _bool_attr(x: Any, name: str) -> bool:
    v = getattr(x, name)
    return bool(v() if callable(v) else v)


def _rec_offset(t: Any) -> int | None:
    flag = getattr(t, "is_measurement_record_target", False)
    flag = flag() if callable(flag) else flag
    if not flag:
        return None
    return int(t.value)


def _count_measurements(inst: Any) -> int:
    n = getattr(inst, "num_measurements", None)
    if n is not None:
        return int(n() if callable(n) else n)
    import stim
    one = stim.Circuit()
    one.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
    return int(one.num_measurements)


def _annotation_dependencies(circuit: Any) -> tuple[list[list[int]], list[list[int]]]:
    """Absolute measurement ids for detectors and observables."""
    mc = 0
    dets: list[list[int]] = []
    obs: dict[int, list[int]] = defaultdict(list)
    for inst in circuit.flattened():
        if inst.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            mids: list[int] = []
            for t in inst.targets_copy():
                off = _rec_offset(t)
                if off is not None:
                    mid = mc + off
                    if not (0 <= mid < mc):
                        raise AssertionError((mid, mc, inst))
                    mids.append(mid)
            if inst.name == "DETECTOR":
                dets.append(mids)
            else:
                args = list(inst.gate_args_copy())
                oid = int(args[0]) if args else 0
                obs[oid].extend(mids)
        mc += _count_measurements(inst)
    if len(dets) != circuit.num_detectors or mc != circuit.num_measurements:
        raise AssertionError((len(dets), circuit.num_detectors, mc, circuit.num_measurements))
    nobs = circuit.num_observables
    out_obs = [obs.get(i, []) for i in range(nobs)]
    return dets, out_obs


def _terminal_measurement_ids(circuit: Any, channel: str) -> list[int]:
    """Return final 25 simple one-qubit measurement ids, verifying basis."""
    meta: list[tuple[int, str, int]] = []  # mid, gate, qubit
    mid = 0
    for inst in circuit.flattened():
        n = _count_measurements(inst)
        if n == 0:
            continue
        name = inst.name
        ts = inst.targets_copy()
        if name in {"MX", "M", "MZ"} and len(ts) == n:
            for t in ts:
                meta.append((mid, name, int(t.value)))
                mid += 1
        else:
            for _ in range(n):
                meta.append((mid, name, -1))
                mid += 1
    if len(meta) != circuit.num_measurements:
        raise AssertionError((len(meta), circuit.num_measurements))
    term = meta[-25:]
    for _, g, _ in term:
        if channel == "X":
            if g != "MX":
                raise AssertionError(f"X terminal probe basis mismatch: {g}")
        else:
            if g not in {"M", "MZ"}:
                raise AssertionError(f"Z terminal probe basis mismatch: {g}")
    return [x[0] for x in term]


def _partition_detectors(circuit: Any, channel: str) -> tuple[list[int], list[int]]:
    det_deps, _ = _annotation_dependencies(circuit)
    terminal_mids = set(_terminal_measurement_ids(circuit, channel))
    terminal_det_ids = [
        i for i, mids in enumerate(det_deps) if any(m in terminal_mids for m in mids)
    ]
    online = [i for i in range(circuit.num_detectors) if i not in set(terminal_det_ids)]
    if len(terminal_det_ids) != TERMINAL_DETECTORS:
        raise AssertionError(f"{channel}: terminal detector ids={terminal_det_ids}")
    return online, terminal_det_ids


def _project_mask(mask: int, detector_ids: list[int]) -> int:
    out = 0
    for j, did in enumerate(detector_ids):
        if (mask >> did) & 1:
            out |= 1 << j
    return out


def _build_with_wait_rounds(hirano_src: Path, channel: str, wait_rounds: int):
    """Public Hirano ZXZ + optional non-destructive full surface rounds + terminal probe.

    Returns full scoring circuit, wrapper, ideal-init range, and probe-start
    instruction index. Operations before probe_start are physically faultable;
    terminal destructive measurements at/after probe_start are ideal scoring only.
    """
    if wait_rounds < 0:
        raise ValueError(wait_rounds)
    lsed, util = s1._load_hirano(hirano_src)
    steane_code = importlib.import_module("steane_code")
    mapping = util.QubitMapping(30, 30)
    initial_value = lsed.InitialValue.Plus if channel == "X" else lsed.InitialValue.Zero
    runner = lsed.SteanePlusSurfaceCode(
        mapping, 5, initial_value, lsed.SteaneSyndromeExtractionPattern.ZXZ, 0.0, True
    )

    init_name = (
        "perform_perfect_steane_plus_initialization"
        if channel == "X"
        else "perform_perfect_steane_zero_initialization"
    )
    init_original = getattr(steane_code, init_name)
    rec: dict[str, Any] = {}

    def init_wrapped(c, m):
        if "init_range" in rec:
            raise AssertionError("ideal init called twice")
        a = len(c)
        init_original(c, m)
        rec["init_range"] = (a, len(c))

    setattr(steane_code, init_name, init_wrapped)

    meth_name = "_perform_destructive_x_measurement" if channel == "X" else "_perform_destructive_z_measurement"
    probe_original = getattr(runner, meth_name)

    def probe_wrapped():
        # Each SurfaceSyndromeMeasurement has six circuit stages. Six global
        # substeps therefore make one complete non-destructive syndrome round.
        for _ in range(wait_rounds):
            for _stage in range(6):
                for m in runner.surface_syndrome_measurements.values():
                    m.run()
                runner.circuit.place_tick()
        for m in runner.surface_syndrome_measurements.values():
            if not m.is_complete():
                raise AssertionError("handoff round did not end on syndrome-round boundary")
        rec["probe_start"] = len(runner.circuit.circuit)
        probe_original()

    setattr(runner, meth_name, probe_wrapped)
    try:
        runner.run()
    finally:
        setattr(steane_code, init_name, init_original)

    if "init_range" not in rec or "probe_start" not in rec:
        raise AssertionError(rec)
    circuit = runner.circuit.circuit

    # Wait=0 must remain byte-for-byte/semantic-equivalent to S1 canonical circuit.
    if wait_rounds == 0:
        fp = s1.fingerprint(circuit, runner.circuit, channel=channel)
        s1.assert_canonical(fp)

    online_ids, terminal_ids = _partition_detectors(circuit, channel)
    expected_online = BASE_PRE_DETECTORS + wait_rounds * HANDOFF_DETECTORS_PER_ROUND
    if len(online_ids) != expected_online:
        raise AssertionError(
            f"{channel}, wait={wait_rounds}: online detectors={len(online_ids)} expected={expected_online}"
        )
    if len(terminal_ids) != TERMINAL_DETECTORS:
        raise AssertionError(terminal_ids)
    return circuit, runner.circuit, tuple(rec["init_range"]), int(rec["probe_start"]), online_ids, terminal_ids


def _instrument_before_probe(circuit: Any, init_range: tuple[int, int], probe_start: int, p: float):
    """Canonical active-location model before terminal scoring probe only."""
    import stim
    if not (0 < p < 1):
        raise ValueError(p)
    out = stim.Circuit()
    locations: list[s3.PhysicalLocation] = []
    noise_map: dict[tuple[int, int], int] = {}
    tick = 0
    init_start, init_end = init_range

    for original_idx, inst in enumerate(circuit):
        if not hasattr(inst, "name"):
            raise AssertionError("unexpected repeat block")
        name = inst.name
        targets = inst.targets_copy()
        faultable = original_idx < probe_start and not (init_start <= original_idx < init_end)

        if faultable and name in {"MX", "M", "MZ"}:
            qids = [s3._qubit_value(t) for t in targets]
            ng = "Z_ERROR" if name == "MX" else "X_ERROR"
            ni, nb = s3._append_noise(out, ng, qids, p)
            for k, q in enumerate(qids):
                lid = len(locations)
                gate = "MX" if name == "MX" else "MZ"
                locations.append(s3.PhysicalLocation(lid, original_idx, tick, gate, (q,), 1, ni, nb + k))
                noise_map[(ni, nb + k)] = lid
            out.append(inst)
        elif faultable and name in {"CX", "CNOT"}:
            qids = [s3._qubit_value(t) for t in targets]
            if len(qids) % 2 or len(set(qids)) != len(qids):
                raise AssertionError((original_idx, qids))
            out.append(inst)
            ni, nb = s3._append_noise(out, "DEPOLARIZE2", qids, p)
            for k in range(0, len(qids), 2):
                lid = len(locations)
                pair = (qids[k], qids[k+1])
                locations.append(s3.PhysicalLocation(lid, original_idx, tick, "CX", pair, 15, ni, nb + k))
                noise_map[(ni, nb + k)] = lid
        elif faultable and name in {"RX", "R", "RZ"}:
            qids = [s3._qubit_value(t) for t in targets]
            out.append(inst)
            ng = "Z_ERROR" if name == "RX" else "X_ERROR"
            ni, nb = s3._append_noise(out, ng, qids, p)
            for k, q in enumerate(qids):
                lid = len(locations)
                gate = "RX" if name == "RX" else "RZ"
                locations.append(s3.PhysicalLocation(lid, original_idx, tick, gate, (q,), 1, ni, nb + k))
                noise_map[(ni, nb + k)] = lid
        else:
            out.append(inst)
        if name == "TICK":
            tick += 1

    if out.without_noise() != circuit:
        raise AssertionError("noise-free reconstruction mismatch")
    return out, locations, noise_map


def _signature_from_explained_error(ex: Any) -> tuple[int, int]:
    d = 0
    o = 0
    for twc in ex.dem_error_terms:
        t = twc.dem_target
        if s3._dem_target_flag(t, "is_relative_detector_id"):
            d ^= 1 << int(t.val)
        elif s3._dem_target_flag(t, "is_logical_observable_id"):
            if int(t.val) != 0:
                raise AssertionError(t)
            o ^= 1
        elif s3._dem_target_flag(t, "is_separator"):
            raise AssertionError("separator with decompose_errors=False")
        else:
            raise AssertionError(t)
    return d, o


def _populate_dynamic(noisy: Any, locations: list[s3.PhysicalLocation], rows: list[s3.FaultRow], noise_map):
    dem = noisy.detector_error_model(decompose_errors=False)
    if dem.num_observables != 1:
        raise AssertionError(dem.num_observables)
    by_key = {(r.location_id, r.fault_class): r for r in rows}
    loc_by_id = {l.location_id: l for l in locations}
    for ex in noisy.explain_detector_error_model_errors(reduce_to_one_representative_error=False):
        sig = _signature_from_explained_error(ex)
        for cel in ex.circuit_error_locations:
            if len(cel.stack_frames) != 1:
                raise AssertionError(cel)
            ii = int(cel.stack_frames[0].instruction_offset)
            start = int(cel.instruction_targets.target_range_start)
            key = (ii, start)
            if key not in noise_map:
                raise AssertionError(f"explained error outside injected model: {key}")
            lid = noise_map[key]
            loc = loc_by_id[lid]
            fc = s3._fault_class_from_error_location(cel, loc)
            row = by_key[(lid, fc)]
            if row.explained_by_dem and (row.detector_mask, row.observable_flip) != sig:
                raise AssertionError((row, sig))
            row.detector_mask, row.observable_flip = sig
            row.detector_weight = sig[0].bit_count()
            row.explained_by_dem = True
    return dem


def _location_counts(locations):
    return s3._location_counts(locations)


def _pair_denominator(locations):
    return s3._pair_denominator(locations)


def _build_channel_rows(hirano_src: Path, channel: str, wait_rounds: int, p: float):
    circuit, wrapper, init_range, probe_start, online_ids, terminal_ids = _build_with_wait_rounds(
        hirano_src, channel, wait_rounds
    )
    noisy, locations, noise_map = _instrument_before_probe(circuit, init_range, probe_start, p)
    rows = s3.build_fault_rows(locations)
    dem = _populate_dynamic(noisy, locations, rows, noise_map)

    # Compress full detector masks into exactly the online detector coordinates.
    for r in rows:
        r.detector_mask = _project_mask(int(r.detector_mask), online_ids)
        r.detector_weight = int(r.detector_mask.bit_count())

    expected_loc = BASE_PRE_LOCATIONS + wait_rounds * SURFACE_ROUND_LOCATIONS
    expected_cls = BASE_PRE_SINGLE_CLASSES + wait_rounds * SURFACE_ROUND_SINGLE_CLASSES
    if len(locations) != expected_loc or len(rows) != expected_cls:
        raise AssertionError(
            f"{channel}, wait={wait_rounds}: loc/classes={(len(locations),len(rows))}, expected={(expected_loc,expected_cls)}"
        )
    if wait_rounds == 0:
        if _pair_denominator(locations) != BASE_PRE_PAIR_CLASSES:
            raise AssertionError(_pair_denominator(locations))
    return rows, locations, {
        "circuit": circuit,
        "noisy": noisy,
        "dem": dem,
        "probe_start": probe_start,
        "online_ids": online_ids,
        "terminal_ids": terminal_ids,
        "location_counts": _location_counts(locations),
    }


def _pair_complementary(xrows: list[s3.FaultRow], zrows: list[s3.FaultRow]) -> tuple[list[OnlineFault], int]:
    if len(xrows) != len(zrows):
        raise AssertionError((len(xrows), len(zrows)))
    out: list[OnlineFault] = []
    mismatches = 0
    for xr, zr in zip(xrows, zrows):
        meta_x = (xr.location_id, xr.tick, xr.gate, xr.qubits, xr.fault_class)
        meta_z = (zr.location_id, zr.tick, zr.gate, zr.qubits, zr.fault_class)
        if meta_x != meta_z:
            raise AssertionError((meta_x, meta_z))
        if xr.detector_mask != zr.detector_mask:
            mismatches += 1
            if mismatches <= 5:
                print("WARNING complementary online syndrome mismatch", meta_x, hex(xr.detector_mask), hex(zr.detector_mask))
        # X-channel observable detects logical Z component; Z-channel observable detects X component.
        zbit = int(xr.observable_flip)
        xbit = int(zr.observable_flip)
        pauli = xbit | (zbit << 1)
        out.append(OnlineFault(
            fault_id=len(out), location_id=xr.location_id, tick=xr.tick, gate=xr.gate,
            qubits=xr.qubits, fault_class=xr.fault_class,
            syndrome=int(xr.detector_mask), residual_pauli=pauli,
        ))
    if mismatches:
        raise AssertionError(f"X/Z complementary online syndrome mismatch count={mismatches}")
    return out, mismatches


def _build_groups(rows: list[OnlineFault]) -> list[SigGroup]:
    tmp: dict[tuple[int,int], dict[int, OnlineFault]] = defaultdict(dict)
    cnt: dict[tuple[int,int], int] = defaultdict(int)
    perloc: dict[tuple[int,int], dict[int,int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        k=(r.syndrome,r.residual_pauli)
        cnt[k]+=1
        tmp[k].setdefault(r.location_id,r)
        perloc[k][r.location_id]+=1
    return [SigGroup(k[0],k[1],cnt[k],tmp[k],dict(perloc[k])) for k in sorted(tmp)]


def _pair_legal(a: SigGroup, b: SigGroup, same: bool) -> bool:
    if same:
        return a.location_count >= 2
    if a.location_count == 1 and b.location_count == 1:
        return next(iter(a.examples_by_location)) != next(iter(b.examples_by_location))
    return True


def _pick_pair(a: SigGroup, b: SigGroup, same: bool):
    if same:
        vals=list(a.examples_by_location.values())
        return vals[0],vals[1]
    for la,ra in a.examples_by_location.items():
        for lb,rb in b.examples_by_location.items():
            if la!=lb:
                return ra,rb
    raise AssertionError("no distinct pair")


def _pauli_set(flags: int) -> list[int]:
    return [p for p in range(4) if flags & (1<<p)]


def _analyze_le2(rows: list[OnlineFault], locations: list[s3.PhysicalLocation], max_witnesses=20):
    # state_sets[D] is a 4-bit set of Pauli labels reachable at order <=2.
    states: dict[int,int] = {0:1<<0}
    order1_syndromes: dict[int,int] = defaultdict(int)
    for r in rows:
        states[r.syndrome]=states.get(r.syndrome,0)|(1<<r.residual_pauli)
        order1_syndromes[r.syndrome] |= 1<<r.residual_pauli

    groups=_build_groups(rows)
    pair_candidates=0
    pair_legal=0
    full_pair_malignant=0
    witnesses=[]
    # Exact pair existence/state audit; exact malignant zero-syndrome pair count is
    # weighted by physical class multiplicities with same-location subtraction.
    zero_groups_by_p = defaultdict(list)
    for i,a in enumerate(groups):
        for j in range(i,len(groups)):
            b=groups[j]
            pair_candidates+=1
            same=i==j
            if not _pair_legal(a,b,same):
                continue
            pair_legal+=1
            syn=a.syndrome^b.syndrome
            p=a.pauli^b.pauli
            states[syn]=states.get(syn,0)|(1<<p)
            if syn==0 and p!=0:
                # Exact number of physical pairs represented by these groups.
                if same:
                    total=a.count*(a.count-1)//2
                    same_loc=sum(v*(v-1)//2 for v in a.counts_by_location.values())
                    count=total-same_loc
                else:
                    total=a.count*b.count
                    same_loc=sum(
                        a.counts_by_location[lid]*b.counts_by_location[lid]
                        for lid in set(a.counts_by_location)&set(b.counts_by_location)
                    )
                    count=total-same_loc
                full_pair_malignant += count
                if count and len(witnesses)<max_witnesses:
                    r0,r1=_pick_pair(a,b,same)
                    witnesses.append({
                        "type":"zero_syndrome_nontrivial_pair",
                        "residual_pauli":PAULI_LABELS[p],
                        "fault_a":asdict(r0),"fault_b":asdict(r1),
                    })

    ambiguous={m:f for m,f in states.items() if f.bit_count()>1}
    ambiguous1={m:f for m,f in order1_syndromes.items() if f.bit_count()>1}
    rejected_single=0
    reject_coeff=0.0
    accepted_single_fail=0
    for r in rows:
        flags=states[r.syndrome]
        if flags.bit_count()!=1:
            rejected_single+=1
            reject_coeff += 1/15 if r.gate=="CX" else 1.0
        else:
            pred=_pauli_set(flags)[0]
            if pred!=r.residual_pauli:
                accepted_single_fail+=1

    # Accepted pair failures under unique<=2 lookup are zero by construction if
    # the state table is correct, but verify by group pass without enumerating all physical pairs.
    accepted_pair_fail=0
    for i,a in enumerate(groups):
        for j in range(i,len(groups)):
            b=groups[j]
            same=i==j
            if not _pair_legal(a,b,same):
                continue
            syn=a.syndrome^b.syndrome
            actual=a.pauli^b.pauli
            flags=states[syn]
            if flags.bit_count()==1:
                pred=_pauli_set(flags)[0]
                if pred!=actual:
                    accepted_pair_fail+=1
                    if len(witnesses)<max_witnesses:
                        r0,r1=_pick_pair(a,b,same)
                        witnesses.append({"type":"selective_pair_failure", "fault_a":asdict(r0),"fault_b":asdict(r1)})

    single_nontrivial_zero=sum(r.syndrome==0 and r.residual_pauli!=0 for r in rows)
    return {
        "states":states,
        "groups":groups,
        "single_syndromes":len(order1_syndromes),
        "ambiguous_single_syndromes":len(ambiguous1),
        "ambiguous_single_fault_classes":sum(
            1 for r in rows if order1_syndromes[r.syndrome].bit_count()>1
        ),
        "zero_syndrome_single_nontrivial_paulis":single_nontrivial_zero,
        "full_prepost_malignant_singles":single_nontrivial_zero,
        "full_prepost_malignant_pairs":full_pair_malignant,
        "unique_syndromes_le2":len(states),
        "ambiguous_syndromes_le2":len(ambiguous),
        "zero_syndrome_pauli_set_le2":[PAULI_LABELS[p] for p in _pauli_set(states.get(0,0))],
        "ideal_zero_syndrome_is_certifiably_acceptable":states.get(0,0).bit_count()==1 and bool(states.get(0,0)&1),
        "selective_rejected_single_fault_classes":rejected_single,
        "selective_single_reject_coefficient":reject_coeff,
        "selective_exact_accepted_single_failures":accepted_single_fail,
        "selective_exact_accepted_pair_failures":accepted_pair_fail,
        "ambiguity_witnesses":witnesses,
    }


def _mc(rows: list[OnlineFault], locations: list[s3.PhysicalLocation], states: dict[int,int], p: float, shots: int, seed: int):
    by_loc: dict[int,list[OnlineFault]]=defaultdict(list)
    for r in rows:
        by_loc[r.location_id].append(r)
    rng=random.Random(seed)
    stats={
        "full_prepost":{"accepted":0,"logical_errors":0,"pauli_counts":[0,0,0,0]},
        "selective_online":{"accepted":0,"logical_errors":0,"pauli_counts":[0,0,0,0],"unknown":0,"ambiguous":0},
    }
    for _ in range(shots):
        syn=0; actual=0
        for loc in locations:
            if rng.random() < p:
                opts=by_loc[loc.location_id]
                r=opts[rng.randrange(len(opts))]
                syn ^= r.syndrome
                actual ^= r.residual_pauli
        # Full online postselection: no correction.
        if syn==0:
            s=stats["full_prepost"]; s["accepted"]+=1; s["pauli_counts"][actual]+=1
            if actual!=0: s["logical_errors"]+=1
        # Selective exact <=2 lookup.
        flags=states.get(syn,0)
        s=stats["selective_online"]
        if flags==0:
            s["unknown"]+=1
        elif flags.bit_count()!=1:
            s["ambiguous"]+=1
        else:
            pred=_pauli_set(flags)[0]
            residual=actual^pred
            s["accepted"]+=1; s["pauli_counts"][residual]+=1
            if residual!=0: s["logical_errors"]+=1
    out={}
    for mode,s in stats.items():
        acc=s["accepted"]
        d=dict(s)
        d["shots"]=shots
        d["p"]=p
        d["acceptance"]=acc/shots
        d["conditional_logical_error"]=(s["logical_errors"]/acc if acc else None)
        d["conditional_residual_pauli_probabilities"]={
            PAULI_LABELS[i]:(s["pauli_counts"][i]/acc if acc else None) for i in range(4)
        }
        out[mode]=d
    return out


def audit_wait(hirano_src: Path, wait_rounds: int, signature_p: float, p_values: list[float], shots: int, seed: int, out_dir: Path):
    xrows,xloc,xmeta=_build_channel_rows(hirano_src,"X",wait_rounds,signature_p)
    zrows,zloc,zmeta=_build_channel_rows(hirano_src,"Z",wait_rounds,signature_p)
    if len(xloc)!=len(zloc): raise AssertionError((len(xloc),len(zloc)))
    for a,b in zip(xloc,zloc):
        if (a.location_id,a.tick,a.gate,a.qubits,a.class_count)!=(b.location_id,b.tick,b.gate,b.qubits,b.class_count):
            raise AssertionError((a,b))
    rows,mismatches=_pair_complementary(xrows,zrows)
    analysis=_analyze_le2(rows,xloc)

    summary=OnlineSummary(
        wait_rounds=wait_rounds,
        online_detector_count=len(xmeta["online_ids"]),
        terminal_detector_count=len(xmeta["terminal_ids"]),
        total_detector_count=xmeta["circuit"].num_detectors,
        physical_locations=len(xloc),
        single_fault_classes=len(rows),
        two_fault_class_pairs=_pair_denominator(xloc),
        location_counts=xmeta["location_counts"],
        xz_online_syndrome_mismatches=mismatches,
        single_syndromes=analysis["single_syndromes"],
        ambiguous_single_syndromes=analysis["ambiguous_single_syndromes"],
        ambiguous_single_fault_classes=analysis["ambiguous_single_fault_classes"],
        zero_syndrome_single_nontrivial_paulis=analysis["zero_syndrome_single_nontrivial_paulis"],
        full_prepost_malignant_singles=analysis["full_prepost_malignant_singles"],
        full_prepost_malignant_pairs=analysis["full_prepost_malignant_pairs"],
        unique_syndromes_le2=analysis["unique_syndromes_le2"],
        ambiguous_syndromes_le2=analysis["ambiguous_syndromes_le2"],
        zero_syndrome_pauli_set_le2=analysis["zero_syndrome_pauli_set_le2"],
        ideal_zero_syndrome_is_certifiably_acceptable=analysis["ideal_zero_syndrome_is_certifiably_acceptable"],
        selective_rejected_single_fault_classes=analysis["selective_rejected_single_fault_classes"],
        selective_single_reject_coefficient=analysis["selective_single_reject_coefficient"],
        selective_exact_accepted_single_failures=analysis["selective_exact_accepted_single_failures"],
        selective_exact_accepted_pair_failures=analysis["selective_exact_accepted_pair_failures"],
        ambiguity_witnesses=analysis["ambiguity_witnesses"],
    )

    out_dir.mkdir(parents=True,exist_ok=True)
    tag=f"wait{wait_rounds}"
    (out_dir/f"s7b_{tag}_summary.json").write_text(json.dumps(asdict(summary),indent=2,sort_keys=True),encoding="utf-8")

    mc=[]
    for i,p in enumerate(p_values):
        mc.append({"wait_rounds":wait_rounds, **_mc(rows,xloc,analysis["states"],p,shots,seed+1000*wait_rounds+i)})
    (out_dir/f"s7b_{tag}_mc.json").write_text(json.dumps(mc,indent=2,sort_keys=True),encoding="utf-8")

    # Compact single-fault online instrument catalogue.
    import csv
    with (out_dir/f"s7b_{tag}_online_fault_catalog.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["fault_id","location_id","tick","gate","qubits","fault_class","online_syndrome_hex","residual_pauli"])
        for r in rows:
            w.writerow([r.fault_id,r.location_id,r.tick,r.gate," ".join(map(str,r.qubits)),r.fault_class,hex(r.syndrome),PAULI_LABELS[r.residual_pauli]])
    return summary,mc


def _self_test_pauli_packing():
    # X-probe flip -> Z bit; Z-probe flip -> X bit.
    assert (0 | (0<<1))==0
    assert (1 | (0<<1))==1  # Zprobe=1 -> X
    assert (0 | (1<<1))==2  # Xprobe=1 -> Z
    assert (1 | (1<<1))==3  # both -> Y
    # Pauli composition modulo phase is XOR in this packing.
    assert 1^2==3 and 3^1==2 and 3^2==1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hirano-src",type=Path,required=True)
    ap.add_argument("--wait-rounds",type=int,nargs="+",default=[0,1])
    ap.add_argument("--signature-p",type=float,default=1e-3)
    ap.add_argument("--p-values",type=float,nargs="+",default=[2e-4,5e-4,1e-3,2e-3])
    ap.add_argument("--shots",type=int,default=100000)
    ap.add_argument("--seed",type=int,default=20260820)
    ap.add_argument("--out-dir",type=Path,default=Path("stim_pipeline/interface/s7b_results"))
    args=ap.parse_args()
    _self_test_pauli_packing()
    s3._self_test_noise_append_fusion()
    s3._self_test_explainer_api()
    all_s=[]
    for wr in args.wait_rounds:
        s,_=audit_wait(args.hirano_src,wr,args.signature_p,args.p_values,args.shots,args.seed,args.out_dir)
        all_s.append(asdict(s))
    print(json.dumps(all_s,indent=2,sort_keys=True))
    print("S7-B PASS: online/non-destructive handoff ambiguity + logical-Pauli instrument audit completed")

if __name__=="__main__":
    main()
