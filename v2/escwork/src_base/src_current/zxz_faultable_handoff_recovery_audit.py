#!/usr/bin/env python3
"""S7-B3: faultable live-handoff + noiseless downstream-recovery exact audit.

Why this exists
===============
S7-B2 showed that, for the historical ZXZ pre-readout live handoff, one
*noiseless future* surface syndrome round is enough to make every single
fault syndrome logically unambiguous and to make zero recovery syndrome
identity-only through <=2 faults. But that future syndrome is not free in a
real device: obtaining it requires running a physical surface-QEC round that
can itself fail.

S7-B3 therefore makes 0/1/2 complete post-ZXZ surface syndrome rounds
FAULTABLE protocol operations, then appends exactly one NOISELESS future
surface round only for audit/recovery scoring:

  faultable ZXZ
    -> k faultable surface handoff rounds
    -> LIVE HANDOFF
    ------------------------------------ audit boundary
    -> 1 noiseless future surface round
    -> ideal destructive complementary X/Z scoring probes

Fault budget
============
Only operations before LIVE HANDOFF are faultable. The ideal future round and
terminal probe never enter the fault budget.

For each physical fault class, the X/Z complementary scoring circuits are
paired. The recovery syndrome consists of all detectors available through the
single noiseless future round. The terminal X/Z observable flips define a
4-valued recovery-frame label I/X/Z/Y.

For all distinct-location fault sets of order <=2 the script asks:
  * Are all single-fault recovery syndromes frame-unambiguous?
  * Is zero recovery syndrome identity-only?
  * Are all <=2 recovery syndromes frame-unambiguous?
  * If not, what fraction of single-fault weight must a selective-certified
    decoder reject in order to accept only <=2-unambiguous syndromes?

Interpretation
==============
If single faults are unambiguous and zero syndrome is identity-only, then the
handoff is compatible with normal downstream QEC at first order. If genuine
order-2 ambiguities remain, a correction-only syndrome decoder cannot correct
every <=2 fault set. A selective-certified policy can still preserve accepted
fault distance 3 by rejecting exactly the <=2-ambiguous recovery syndromes.

This is an exact active-location stochastic-Pauli audit, not Monte Carlo.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ZXZ_DIR = HERE.parent / "zxz"
for p in (HERE, ZXZ_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import hirano_zxz_generator as s1  # type: ignore
import zxz_exact_fault_audit as s3  # type: ignore
import zxz_online_handoff_audit as s7b  # type: ignore
import zxz_ideal_recovery_handoff_audit as s7b2  # type: ignore

D = 5
BASE_DETECTORS = 123
DETECTORS_PER_SURFACE_ROUND = 24
TERMINAL_DETECTORS = 12
BASE_LOCATIONS = 772
BASE_SINGLE_CLASSES = 7254
SURFACE_ROUND_CX = 80
SURFACE_ROUND_OTHER = 48
SURFACE_ROUND_LOCATIONS = SURFACE_ROUND_CX + SURFACE_ROUND_OTHER  # 128
SURFACE_ROUND_SINGLE_CLASSES = SURFACE_ROUND_CX * 15 + SURFACE_ROUND_OTHER  # 1248
FRAME_LABELS = ("I", "X", "Z", "Y")


def _run_full_surface_round(runner: Any) -> None:
    for _stage in range(6):
        for m in runner.surface_syndrome_measurements.values():
            m.run()
        runner.circuit.place_tick()
    for m in runner.surface_syndrome_measurements.values():
        if not m.is_complete():
            raise AssertionError("surface round did not end on a syndrome boundary")


def _build_circuit(hirano_src: Path, channel: str, faultable_rounds: int, ideal_tail_rounds: int):
    if faultable_rounds < 0 or ideal_tail_rounds < 1:
        raise ValueError((faultable_rounds, ideal_tail_rounds))

    lsed, util = s1._load_hirano(hirano_src)
    steane_code = importlib.import_module("steane_code")
    mapping = util.QubitMapping(30, 30)
    initial_value = lsed.InitialValue.Plus if channel == "X" else lsed.InitialValue.Zero
    runner = lsed.SteanePlusSurfaceCode(
        mapping, D, initial_value, lsed.SteaneSyndromeExtractionPattern.ZXZ, 0.0, True
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
        rec["detectors_after_zxz"] = int(runner.circuit.circuit.num_detectors)

        # These rounds are REAL protocol operations and enter the fault budget.
        for _ in range(faultable_rounds):
            _run_full_surface_round(runner)

        rec["detectors_at_live_handoff"] = int(runner.circuit.circuit.num_detectors)
        rec["faultable_end"] = len(runner.circuit.circuit)

        # These rounds are audit-only and must remain noiseless.
        for _ in range(ideal_tail_rounds):
            _run_full_surface_round(runner)

        rec["detectors_before_probe"] = int(runner.circuit.circuit.num_detectors)
        rec["probe_start"] = len(runner.circuit.circuit)
        probe_original()

    setattr(runner, meth_name, probe_wrapped)
    try:
        runner.run()
    finally:
        setattr(steane_code, init_name, init_original)

    required = {
        "init_range", "detectors_after_zxz", "detectors_at_live_handoff",
        "faultable_end", "detectors_before_probe", "probe_start",
    }
    if not required <= rec.keys():
        raise AssertionError(rec)

    circuit = runner.circuit.circuit
    online_ids, terminal_ids = s7b._partition_detectors(circuit, channel)

    exp_after_zxz = BASE_DETECTORS
    exp_handoff = BASE_DETECTORS + faultable_rounds * DETECTORS_PER_SURFACE_ROUND
    exp_recovery = exp_handoff + ideal_tail_rounds * DETECTORS_PER_SURFACE_ROUND

    if rec["detectors_after_zxz"] != exp_after_zxz:
        raise AssertionError(("detectors_after_zxz", rec["detectors_after_zxz"], exp_after_zxz))
    if rec["detectors_at_live_handoff"] != exp_handoff:
        raise AssertionError(("detectors_at_live_handoff", rec["detectors_at_live_handoff"], exp_handoff))
    if rec["detectors_before_probe"] != exp_recovery:
        raise AssertionError(("detectors_before_probe", rec["detectors_before_probe"], exp_recovery))
    if len(online_ids) != exp_recovery:
        raise AssertionError(("recovery detector ids", len(online_ids), exp_recovery))
    if len(terminal_ids) != TERMINAL_DETECTORS:
        raise AssertionError(("terminal detector ids", terminal_ids))
    if circuit.num_detectors != exp_recovery + TERMINAL_DETECTORS:
        raise AssertionError((circuit.num_detectors, exp_recovery + TERMINAL_DETECTORS))

    return (
        circuit,
        tuple(rec["init_range"]),
        int(rec["faultable_end"]),
        online_ids,
        terminal_ids,
        rec,
    )


def _instrument_until_faultable_end(
    circuit: Any,
    init_range: tuple[int, int],
    faultable_end: int,
    p: float,
):
    """Canonical noise only before the actual live-handoff boundary."""
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
        faultable = original_idx < faultable_end and not (init_start <= original_idx < init_end)

        if faultable and name in {"MX", "M", "MZ"}:
            qids = [s3._qubit_value(t) for t in targets]
            ng = "Z_ERROR" if name == "MX" else "X_ERROR"
            ni, nb = s3._append_noise(out, ng, qids, p)
            for k, q in enumerate(qids):
                lid = len(locations)
                gate = "MX" if name == "MX" else "MZ"
                locations.append(
                    s3.PhysicalLocation(lid, original_idx, tick, gate, (q,), 1, ni, nb + k)
                )
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
                locations.append(
                    s3.PhysicalLocation(lid, original_idx, tick, "CX", pair, 15, ni, nb + k)
                )
                noise_map[(ni, nb + k)] = lid
        elif faultable and name in {"RX", "R", "RZ"}:
            qids = [s3._qubit_value(t) for t in targets]
            out.append(inst)
            ng = "Z_ERROR" if name == "RX" else "X_ERROR"
            ni, nb = s3._append_noise(out, ng, qids, p)
            for k, q in enumerate(qids):
                lid = len(locations)
                gate = "RX" if name == "RX" else "RZ"
                locations.append(
                    s3.PhysicalLocation(lid, original_idx, tick, gate, (q,), 1, ni, nb + k)
                )
                noise_map[(ni, nb + k)] = lid
        else:
            out.append(inst)

        if name == "TICK":
            tick += 1

    if out.without_noise() != circuit:
        raise AssertionError("noise-free reconstruction mismatch")
    return out, locations, noise_map


def _build_channel_rows(
    hirano_src: Path,
    channel: str,
    faultable_rounds: int,
    ideal_tail_rounds: int,
    p: float,
):
    circuit, init_range, faultable_end, recovery_ids, terminal_ids, rec = _build_circuit(
        hirano_src, channel, faultable_rounds, ideal_tail_rounds
    )
    noisy, locations, noise_map = _instrument_until_faultable_end(
        circuit, init_range, faultable_end, p
    )
    rows = s3.build_fault_rows(locations)
    dem = s7b._populate_dynamic(noisy, locations, rows, noise_map)

    # Keep exactly the syndrome available after the noiseless downstream audit tail.
    for r in rows:
        r.detector_mask = s7b._project_mask(int(r.detector_mask), recovery_ids)
        r.detector_weight = int(r.detector_mask.bit_count())

    exp_loc = BASE_LOCATIONS + faultable_rounds * SURFACE_ROUND_LOCATIONS
    exp_cls = BASE_SINGLE_CLASSES + faultable_rounds * SURFACE_ROUND_SINGLE_CLASSES
    if len(locations) != exp_loc:
        raise AssertionError(("physical_locations", len(locations), exp_loc))
    if len(rows) != exp_cls:
        raise AssertionError(("single_fault_classes", len(rows), exp_cls))

    return rows, locations, {
        "circuit": circuit,
        "dem": dem,
        "recovery_ids": recovery_ids,
        "terminal_ids": terminal_ids,
        "location_counts": s7b._location_counts(locations),
        "rec": rec,
    }


def _class_probability_coefficient(r: s7b2.RecoveryFault) -> float:
    # Under the canonical model a CX location fails with probability p and,
    # conditioned on failure, each of 15 nonidentity 2q Paulis is equiprobable.
    return 1.0 / 15.0 if r.gate == "CX" else 1.0


def _selective_metrics(rows: list[s7b2.RecoveryFault], analysis: dict[str, Any]) -> dict[str, Any]:
    state_table: dict[int, int] = analysis["state_table"]
    ambiguous = {s for s, flags in state_table.items() if int(flags).bit_count() > 1}

    rejected_single_classes = sum(1 for r in rows if r.recovery_syndrome in ambiguous)
    rejected_single_coeff = sum(
        _class_probability_coefficient(r)
        for r in rows
        if r.recovery_syndrome in ambiguous
    )

    # Every accepted <=2 syndrome has exactly one reachable frame by construction.
    # Verify singles explicitly. Pair safety follows from the exact le2 state table,
    # but we also state this invariant in the output.
    accepted_single_bad = 0
    for r in rows:
        flags = int(state_table[r.recovery_syndrome])
        if flags.bit_count() == 1:
            only = (flags.bit_length() - 1)
            if only != r.frame:
                accepted_single_bad += 1

    return {
        "selective_ambiguous_syndrome_count_le2": len(ambiguous),
        "selective_rejected_single_fault_classes": rejected_single_classes,
        "selective_single_reject_coefficient": rejected_single_coeff,
        "selective_exact_accepted_single_failures": accepted_single_bad,
        "selective_exact_accepted_pair_failures": 0,
    }


def audit_depth(
    hirano_src: Path,
    faultable_rounds: int,
    ideal_tail_rounds: int,
    signature_p: float,
    out_dir: Path,
):
    xrows, xloc, xmeta = _build_channel_rows(
        hirano_src, "X", faultable_rounds, ideal_tail_rounds, signature_p
    )
    zrows, zloc, zmeta = _build_channel_rows(
        hirano_src, "Z", faultable_rounds, ideal_tail_rounds, signature_p
    )

    if len(xloc) != len(zloc):
        raise AssertionError((len(xloc), len(zloc)))
    for a, b in zip(xloc, zloc):
        ma = (a.location_id, a.tick, a.gate, a.qubits, a.class_count)
        mb = (b.location_id, b.tick, b.gate, b.qubits, b.class_count)
        if ma != mb:
            raise AssertionError((ma, mb))

    paired, mismatches = s7b2._pair_complementary(xrows, zrows)
    analysis = s7b2._analyze_exact_le2(paired)
    selective = _selective_metrics(paired, analysis)

    summary = {
        "faultable_handoff_rounds": faultable_rounds,
        "ideal_audit_tail_rounds": ideal_tail_rounds,
        "future_audit_tail_is_noiseless": True,
        "physical_locations": len(xloc),
        "single_fault_classes": len(paired),
        "two_fault_class_pairs": s7b._pair_denominator(xloc),
        "location_counts": xmeta["location_counts"],
        "detectors_after_zxz": BASE_DETECTORS,
        "detectors_at_actual_live_handoff": BASE_DETECTORS + faultable_rounds * DETECTORS_PER_SURFACE_ROUND,
        "future_ideal_round_detectors": ideal_tail_rounds * DETECTORS_PER_SURFACE_ROUND,
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
        "syndrome_only_correction_only_decoder_exists_le2": analysis["syndrome_only_ideal_recovery_exists_le2"],
        **selective,
        "ambiguity_witnesses": analysis["ambiguity_witnesses"],
    }

    # Strong invariants.
    if summary["xz_recovery_syndrome_mismatches"] != 0:
        raise AssertionError(summary["xz_recovery_syndrome_mismatches"])
    if summary["selective_exact_accepted_single_failures"] != 0:
        raise AssertionError(summary["selective_exact_accepted_single_failures"])
    if summary["selective_exact_accepted_pair_failures"] != 0:
        raise AssertionError(summary["selective_exact_accepted_pair_failures"])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"s7b3_faultable{faultable_rounds}_tail{ideal_tail_rounds}"
    (out_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    with (out_dir / f"{stem}_single_catalog.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "fault_id", "location_id", "tick", "gate", "qubits", "fault_class",
            "recovery_syndrome_hex", "frame",
        ])
        for r in paired:
            w.writerow([
                r.fault_id, r.location_id, r.tick, r.gate,
                " ".join(map(str, r.qubits)), r.fault_class,
                hex(r.recovery_syndrome), FRAME_LABELS[r.frame],
            ])

    # state_table is large and intentionally omitted from the persisted summary.
    return summary


def _static_self_test() -> None:
    assert BASE_LOCATIONS + SURFACE_ROUND_LOCATIONS == 900
    assert BASE_SINGLE_CLASSES + SURFACE_ROUND_SINGLE_CLASSES == 8502
    # Reuse the validated exact ambiguity engine.
    rows = [
        s7b2.RecoveryFault(0, 0, 0, "RZ", (0,), "X", 0b001, 0),
        s7b2.RecoveryFault(1, 1, 0, "RZ", (1,), "X", 0b010, 1),
    ]
    a = s7b2._analyze_exact_le2(rows)
    assert a["ambiguous_single_syndromes"] == 0
    assert a["ambiguous_recovery_syndromes_le2"] == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", required=True, type=Path)
    ap.add_argument("--faultable-rounds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ideal-tail-rounds", type=int, default=1)
    ap.add_argument("--signature-p", type=float, default=1e-6)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stim_pipeline/interface/s7b3_results"),
    )
    args = ap.parse_args()

    _static_self_test()
    if args.ideal_tail_rounds < 1:
        raise SystemExit("--ideal-tail-rounds must be >=1")

    results = []
    for k in args.faultable_rounds:
        if k < 0:
            raise SystemExit("--faultable-rounds values must be >=0")
        print(
            f"Running S7-B3 with {k} faultable handoff round(s) "
            f"+ {args.ideal_tail_rounds} noiseless audit tail round(s)..."
        )
        s = audit_depth(
            args.hirano_src,
            k,
            args.ideal_tail_rounds,
            args.signature_p,
            args.out_dir,
        )
        results.append(s)
        print(json.dumps(s, indent=2, sort_keys=True))

    (args.out_dir / "s7b3_all_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("S7-B3 PASS: faultable-handoff + noiseless-tail exact <=2 recovery audit completed")


if __name__ == "__main__":
    main()
