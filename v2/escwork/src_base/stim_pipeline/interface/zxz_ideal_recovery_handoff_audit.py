#!/usr/bin/env python3
"""S7-B2: exact live-handoff -> ideal-recovery composability audit.

Purpose
=======
The historical Hirano ZXZ certification appends a complementary destructive
surface readout. That readout is excellent for *scoring* logical transfer but
cannot be used as an online decision if the output surface state must remain
alive.

S7-B showed that simply dropping the 12 terminal detectors leaves pending
boundary information. This file asks the physically relevant composability
question instead:

    After the faultable ZXZ escape ends, do one or more PERFECT (noiseless)
    future surface syndrome rounds provide enough syndrome information for an
    ideal recovery to correct every audited <=2-fault residual before the
    state is used downstream?

The future syndrome rounds and final destructive X/Z probes are audit-only.
They are NOT included in the physical fault budget. Therefore all audit depths
use the same historical live-handoff fault inventory:

    772 physical fault locations
    7,254 single-fault classes
    26,258,016 distinct-location two-fault class pairs

For each physical fault class, complementary X/Z scoring circuits are paired.
The online+future-ideal detector syndrome is the recovery information. The two
terminal observable flips provide a 2-bit frame label. For a *fixed recovery
syndrome*, different frame labels mean that no single recovery chosen from the
syndrome can correct both errors. Conversely, if the frame label is unique for
every syndrome reachable by <=2 faults, a syndrome-only ideal recovery exists
for all audited <=2-fault residuals.

Important semantic point
========================
The 2-bit X/Z terminal probe label is not, by itself, asserted to be the live
state's logical Pauli. It becomes a recovery-coset label only *relative to a
fixed perfect recovery syndrome*. This audit therefore tests uniqueness of the
pair (recovery syndrome -> terminal frame), which is the correct condition for
existence of a syndrome-only recovery through the audited fault order.
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
from typing import Any

HERE = Path(__file__).resolve().parent
ZXZ_DIR = HERE.parent / "zxz"
for p in (HERE, ZXZ_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import hirano_zxz_generator as s1  # type: ignore  # noqa: E402
import zxz_exact_fault_audit as s3  # type: ignore  # noqa: E402
import zxz_online_handoff_audit as s7b  # type: ignore  # noqa: E402

D = 5
BASE_ONLINE_DETECTORS = 123
DETECTORS_PER_IDEAL_ROUND = 24
TERMINAL_DETECTORS = 12
HANDOFF_LOCATIONS = 772
HANDOFF_SINGLE_CLASSES = 7254
HANDOFF_PAIR_CLASSES = 26_258_016
FRAME_LABELS = ("I", "X", "Z", "Y")


@dataclass(frozen=True)
class RecoveryFault:
    fault_id: int
    location_id: int
    tick: int
    gate: str
    qubits: tuple[int, ...]
    fault_class: str
    recovery_syndrome: int
    frame: int  # 0=I,1=X,2=Z,3=Y; just a terminal-probe frame label


@dataclass
class SigGroup:
    syndrome: int
    frame: int
    count: int
    examples_by_location: dict[int, RecoveryFault]
    counts_by_location: dict[int, int]

    @property
    def location_count(self) -> int:
        return len(self.examples_by_location)


def _build_with_ideal_audit_rounds(hirano_src: Path, channel: str, audit_rounds: int):
    """Build public ZXZ, mark LIVE handoff, append noiseless future rounds, then terminal probe."""
    if audit_rounds < 1:
        raise ValueError("S7-B2 requires at least one future ideal syndrome round")

    lsed, util = s1._load_hirano(hirano_src)
    steane_code = importlib.import_module("steane_code")
    mapping = util.QubitMapping(30, 30)
    initial_value = lsed.InitialValue.Plus if channel == "X" else lsed.InitialValue.Zero
    runner = lsed.SteanePlusSurfaceCode(
        mapping,
        D,
        initial_value,
        lsed.SteaneSyndromeExtractionPattern.ZXZ,
        0.0,
        True,
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

    meth_name = (
        "_perform_destructive_x_measurement"
        if channel == "X"
        else "_perform_destructive_z_measurement"
    )
    probe_original = getattr(runner, meth_name)

    def probe_wrapped():
        # This is the real live-output boundary. Everything after this point is
        # an ideal/noiseless audit continuation and MUST NOT enter the fault budget.
        rec["handoff_start"] = len(runner.circuit.circuit)
        rec["detectors_at_handoff"] = int(runner.circuit.circuit.num_detectors)

        for _ in range(audit_rounds):
            for _stage in range(6):
                for m in runner.surface_syndrome_measurements.values():
                    m.run()
                runner.circuit.place_tick()
            for m in runner.surface_syndrome_measurements.values():
                if not m.is_complete():
                    raise AssertionError("ideal future round did not end on syndrome boundary")

        rec["probe_start"] = len(runner.circuit.circuit)
        rec["detectors_before_probe"] = int(runner.circuit.circuit.num_detectors)
        probe_original()

    setattr(runner, meth_name, probe_wrapped)
    try:
        runner.run()
    finally:
        setattr(steane_code, init_name, init_original)

    required = {"init_range", "handoff_start", "detectors_at_handoff", "probe_start", "detectors_before_probe"}
    if not required <= rec.keys():
        raise AssertionError(rec)

    circuit = runner.circuit.circuit
    online_ids, terminal_ids = s7b._partition_detectors(circuit, channel)
    expected_recovery = BASE_ONLINE_DETECTORS + audit_rounds * DETECTORS_PER_IDEAL_ROUND

    if rec["detectors_at_handoff"] != BASE_ONLINE_DETECTORS:
        raise AssertionError((channel, "detectors_at_handoff", rec["detectors_at_handoff"]))
    if rec["detectors_before_probe"] != expected_recovery:
        raise AssertionError((channel, "detectors_before_probe", rec["detectors_before_probe"], expected_recovery))
    if len(online_ids) != expected_recovery:
        raise AssertionError((channel, len(online_ids), expected_recovery))
    if len(terminal_ids) != TERMINAL_DETECTORS:
        raise AssertionError((channel, terminal_ids))
    if circuit.num_detectors != expected_recovery + TERMINAL_DETECTORS:
        raise AssertionError((circuit.num_detectors, expected_recovery + TERMINAL_DETECTORS))

    return (
        circuit,
        runner.circuit,
        tuple(rec["init_range"]),
        int(rec["handoff_start"]),
        online_ids,
        terminal_ids,
    )


def _instrument_only_before_handoff(circuit: Any, init_range: tuple[int, int], handoff_start: int, p: float):
    """Canonical active-location noise ONLY before the live handoff boundary."""
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
        faultable = original_idx < handoff_start and not (init_start <= original_idx < init_end)

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
                pair = (qids[k], qids[k + 1])
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

    if len(locations) != HANDOFF_LOCATIONS:
        raise AssertionError(("physical_locations", len(locations), HANDOFF_LOCATIONS))
    if s7b._pair_denominator(locations) != HANDOFF_PAIR_CLASSES:
        raise AssertionError(("pair_denominator", s7b._pair_denominator(locations), HANDOFF_PAIR_CLASSES))

    return out, locations, noise_map


def _build_channel_rows(hirano_src: Path, channel: str, audit_rounds: int, p: float):
    circuit, wrapper, init_range, handoff_start, recovery_ids, terminal_ids = _build_with_ideal_audit_rounds(
        hirano_src, channel, audit_rounds
    )
    noisy, locations, noise_map = _instrument_only_before_handoff(circuit, init_range, handoff_start, p)
    rows = s3.build_fault_rows(locations)
    dem = s7b._populate_dynamic(noisy, locations, rows, noise_map)

    # Keep only information available to an ideal future recovery BEFORE the
    # terminal destructive scoring probe. This includes the historical 123
    # detectors plus 24 perfect future-round detectors per audit round.
    for r in rows:
        r.detector_mask = s7b._project_mask(int(r.detector_mask), recovery_ids)
        r.detector_weight = int(r.detector_mask.bit_count())

    if len(rows) != HANDOFF_SINGLE_CLASSES:
        raise AssertionError(("single_fault_classes", len(rows), HANDOFF_SINGLE_CLASSES))

    return rows, locations, {
        "circuit": circuit,
        "noisy": noisy,
        "dem": dem,
        "recovery_ids": recovery_ids,
        "terminal_ids": terminal_ids,
        "handoff_start": handoff_start,
        "location_counts": s7b._location_counts(locations),
    }


def _pair_complementary(xrows: list[s3.FaultRow], zrows: list[s3.FaultRow]) -> tuple[list[RecoveryFault], int]:
    if len(xrows) != len(zrows):
        raise AssertionError((len(xrows), len(zrows)))
    out: list[RecoveryFault] = []
    mismatches = 0
    for xr, zr in zip(xrows, zrows):
        meta_x = (xr.location_id, xr.tick, xr.gate, xr.qubits, xr.fault_class)
        meta_z = (zr.location_id, zr.tick, zr.gate, zr.qubits, zr.fault_class)
        if meta_x != meta_z:
            raise AssertionError((meta_x, meta_z))
        if xr.detector_mask != zr.detector_mask:
            mismatches += 1
            if mismatches <= 5:
                print("WARNING X/Z recovery syndrome mismatch", meta_x, hex(xr.detector_mask), hex(zr.detector_mask))

        # X-probe observable flip is a Z-frame bit; Z-probe flip is an X-frame bit.
        zbit = int(xr.observable_flip)
        xbit = int(zr.observable_flip)
        frame = xbit | (zbit << 1)
        out.append(RecoveryFault(
            fault_id=len(out),
            location_id=xr.location_id,
            tick=xr.tick,
            gate=xr.gate,
            qubits=xr.qubits,
            fault_class=xr.fault_class,
            recovery_syndrome=int(xr.detector_mask),
            frame=frame,
        ))
    if mismatches:
        raise AssertionError(f"X/Z recovery syndrome mismatch count={mismatches}")
    return out, mismatches


def _build_groups(rows: list[RecoveryFault]) -> list[SigGroup]:
    examples: dict[tuple[int, int], dict[int, RecoveryFault]] = defaultdict(dict)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    perloc: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        k = (r.recovery_syndrome, r.frame)
        counts[k] += 1
        examples[k].setdefault(r.location_id, r)
        perloc[k][r.location_id] += 1
    return [
        SigGroup(k[0], k[1], counts[k], examples[k], dict(perloc[k]))
        for k in sorted(examples)
    ]


def _pair_legal(a: SigGroup, b: SigGroup, same: bool) -> bool:
    if same:
        return a.location_count >= 2
    if a.location_count == 1 and b.location_count == 1:
        return next(iter(a.examples_by_location)) != next(iter(b.examples_by_location))
    return True


def _pick_pair(a: SigGroup, b: SigGroup, same: bool) -> tuple[RecoveryFault, RecoveryFault]:
    if same:
        vals = list(a.examples_by_location.values())
        return vals[0], vals[1]
    for la, ra in a.examples_by_location.items():
        for lb, rb in b.examples_by_location.items():
            if la != lb:
                return ra, rb
    raise AssertionError("no distinct-location pair")


def _frames(flags: int) -> list[int]:
    return [p for p in range(4) if flags & (1 << p)]


def _analyze_exact_le2(rows: list[RecoveryFault], max_witnesses: int = 30):
    # maps syndrome -> 4-bit set of terminal frame labels reachable at <= given order
    order1: dict[int, int] = defaultdict(int)
    le2: dict[int, int] = {0: 1 << 0}  # fault-free: syndrome 0, frame I
    order_min: dict[tuple[int, int], int] = {(0, 0): 0}
    witnesses_by_key: dict[tuple[int, int], dict[str, Any]] = {
        (0, 0): {"order": 0, "kind": "fault_free"}
    }

    for r in rows:
        order1[r.recovery_syndrome] |= 1 << r.frame
        le2[r.recovery_syndrome] = le2.get(r.recovery_syndrome, 0) | (1 << r.frame)
        key = (r.recovery_syndrome, r.frame)
        if key not in order_min or order_min[key] > 1:
            order_min[key] = 1
            witnesses_by_key[key] = {"order": 1, "fault": asdict(r)}

    groups = _build_groups(rows)
    for i, a in enumerate(groups):
        for j in range(i, len(groups)):
            b = groups[j]
            same = i == j
            if not _pair_legal(a, b, same):
                continue
            syn = a.syndrome ^ b.syndrome
            frame = a.frame ^ b.frame
            le2[syn] = le2.get(syn, 0) | (1 << frame)
            key = (syn, frame)
            if key not in order_min or order_min[key] > 2:
                order_min[key] = 2
                r0, r1 = _pick_pair(a, b, same)
                witnesses_by_key[key] = {"order": 2, "fault_a": asdict(r0), "fault_b": asdict(r1)}

    ambiguous1 = {s: f for s, f in order1.items() if f.bit_count() > 1}
    ambiguous2 = {s: f for s, f in le2.items() if f.bit_count() > 1}

    witnesses: list[dict[str, Any]] = []
    type_counts: dict[str, int] = defaultdict(int)
    for syn, flags in ambiguous2.items():
        fr = _frames(flags)
        # classify by the lowest order at which each competing frame appears
        orders = sorted(order_min[(syn, f)] for f in fr)
        typ = "v".join(str(o) for o in orders[:2]) if len(orders) >= 2 else str(orders[0])
        type_counts[typ] += 1
        if len(witnesses) < max_witnesses:
            witnesses.append({
                "syndrome_hex": hex(syn),
                "frames": [FRAME_LABELS[f] for f in fr],
                "min_orders": {FRAME_LABELS[f]: order_min[(syn, f)] for f in fr},
                "representatives": {
                    FRAME_LABELS[f]: witnesses_by_key[(syn, f)] for f in fr
                },
            })

    zero_flags = le2.get(0, 0)
    return {
        "single_syndromes": len(order1),
        "ambiguous_single_syndromes": len(ambiguous1),
        "ambiguous_single_fault_classes": sum(1 for r in rows if order1[r.recovery_syndrome].bit_count() > 1),
        "unique_recovery_syndromes_le2": len(le2),
        "ambiguous_recovery_syndromes_le2": len(ambiguous2),
        "ambiguity_type_counts": dict(sorted(type_counts.items())),
        "zero_syndrome_frame_set_le2": [FRAME_LABELS[f] for f in _frames(zero_flags)],
        "zero_syndrome_unique_identity_le2": zero_flags == (1 << 0),
        "all_recovery_syndromes_logically_unique_le2": len(ambiguous2) == 0,
        "syndrome_only_ideal_recovery_exists_le2": len(ambiguous2) == 0,
        "ambiguity_witnesses": witnesses,
        "state_table": le2,
    }


def audit_round_count(hirano_src: Path, audit_rounds: int, signature_p: float, out_dir: Path):
    xrows, xloc, xmeta = _build_channel_rows(hirano_src, "X", audit_rounds, signature_p)
    zrows, zloc, zmeta = _build_channel_rows(hirano_src, "Z", audit_rounds, signature_p)

    if len(xloc) != len(zloc):
        raise AssertionError((len(xloc), len(zloc)))
    for a, b in zip(xloc, zloc):
        ma = (a.location_id, a.tick, a.gate, a.qubits, a.class_count)
        mb = (b.location_id, b.tick, b.gate, b.qubits, b.class_count)
        if ma != mb:
            raise AssertionError((ma, mb))

    rows, mismatches = _pair_complementary(xrows, zrows)
    analysis = _analyze_exact_le2(rows)

    summary = {
        "audit_rounds": audit_rounds,
        "faultable_boundary": "historical ZXZ pre-readout live handoff",
        "future_audit_rounds_are_noiseless": True,
        "physical_locations": len(xloc),
        "single_fault_classes": len(rows),
        "two_fault_class_pairs": s7b._pair_denominator(xloc),
        "location_counts": xmeta["location_counts"],
        "detectors_at_live_handoff": BASE_ONLINE_DETECTORS,
        "future_ideal_round_detectors": audit_rounds * DETECTORS_PER_IDEAL_ROUND,
        "recovery_detector_count": len(xmeta["recovery_ids"]),
        "terminal_scoring_detectors": len(xmeta["terminal_ids"]),
        "total_detector_count": xmeta["circuit"].num_detectors,
        "xz_recovery_syndrome_mismatches": mismatches,
        "single_syndromes": analysis["single_syndromes"],
        "ambiguous_single_syndromes": analysis["ambiguous_single_syndromes"],
        "ambiguous_single_fault_classes": analysis["ambiguous_single_fault_classes"],
        "unique_recovery_syndromes_le2": analysis["unique_recovery_syndromes_le2"],
        "ambiguous_recovery_syndromes_le2": analysis["ambiguous_recovery_syndromes_le2"],
        "ambiguity_type_counts": analysis["ambiguity_type_counts"],
        "zero_syndrome_frame_set_le2": analysis["zero_syndrome_frame_set_le2"],
        "zero_syndrome_unique_identity_le2": analysis["zero_syndrome_unique_identity_le2"],
        "all_recovery_syndromes_logically_unique_le2": analysis["all_recovery_syndromes_logically_unique_le2"],
        "syndrome_only_ideal_recovery_exists_le2": analysis["syndrome_only_ideal_recovery_exists_le2"],
        "ambiguity_witnesses": analysis["ambiguity_witnesses"],
    }

    # Core invariants: future audit rounds must never change the fault budget.
    if summary["physical_locations"] != HANDOFF_LOCATIONS:
        raise AssertionError(summary["physical_locations"])
    if summary["single_fault_classes"] != HANDOFF_SINGLE_CLASSES:
        raise AssertionError(summary["single_fault_classes"])
    if summary["two_fault_class_pairs"] != HANDOFF_PAIR_CLASSES:
        raise AssertionError(summary["two_fault_class_pairs"])
    if summary["recovery_detector_count"] != BASE_ONLINE_DETECTORS + audit_rounds * DETECTORS_PER_IDEAL_ROUND:
        raise AssertionError(summary["recovery_detector_count"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"s7b2_audit{audit_rounds}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Small single-fault catalogue for reproducibility/debugging.
    with (out_dir / f"s7b2_audit{audit_rounds}_single_catalog.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fault_id", "location_id", "tick", "gate", "qubits", "fault_class", "recovery_syndrome_hex", "frame"])
        for r in rows:
            w.writerow([
                r.fault_id,
                r.location_id,
                r.tick,
                r.gate,
                " ".join(map(str, r.qubits)),
                r.fault_class,
                hex(r.recovery_syndrome),
                FRAME_LABELS[r.frame],
            ])

    return summary


def _synthetic_self_test() -> None:
    # Two single faults with different syndromes are unambiguous; their pair
    # creates a third syndrome/frame. No ambiguity should appear.
    rows = [
        RecoveryFault(0, 0, 0, "RZ", (0,), "X", 0b001, 0),
        RecoveryFault(1, 1, 0, "RZ", (1,), "X", 0b010, 1),
    ]
    a = _analyze_exact_le2(rows)
    assert a["ambiguous_single_syndromes"] == 0
    assert a["ambiguous_recovery_syndromes_le2"] == 0
    assert a["syndrome_only_ideal_recovery_exists_le2"]

    # Add same syndrome with opposite frame -> single-level true ambiguity.
    rows.append(RecoveryFault(2, 2, 0, "RZ", (2,), "X", 0b001, 1))
    b = _analyze_exact_le2(rows)
    assert b["ambiguous_single_syndromes"] == 1
    assert not b["syndrome_only_ideal_recovery_exists_le2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", required=True, type=Path)
    ap.add_argument("--audit-rounds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--signature-p", type=float, default=1e-6, help="nonzero marker probability for exact DEM symptom extraction")
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s7b2_results"))
    args = ap.parse_args()

    _synthetic_self_test()

    results = []
    for r in args.audit_rounds:
        if r < 1:
            raise SystemExit("--audit-rounds values must be >=1")
        print(f"Running S7-B2 ideal-recovery audit with {r} noiseless future round(s)...")
        s = audit_round_count(args.hirano_src, r, args.signature_p, args.out_dir)
        results.append(s)
        print(json.dumps(s, indent=2, sort_keys=True))

    (args.out_dir / "s7b2_all_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )

    if any(s["syndrome_only_ideal_recovery_exists_le2"] for s in results):
        print("S7-B2 RESULT: IDEAL RECOVERY EXISTS through <=2 faults for at least one audited future depth")
    else:
        print("S7-B2 RESULT: genuine <=2 recovery ambiguity remains at all audited future depths")
    print("S7-B2 PASS: live-handoff -> noiseless-future-recovery exact ambiguity audit completed")


if __name__ == "__main__":
    main()
