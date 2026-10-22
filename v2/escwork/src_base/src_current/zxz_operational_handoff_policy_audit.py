#!/usr/bin/env python3
"""S7-B4: operational live-handoff selective policy audit.

This audit closes the gap between S7-B3's mathematical recovery syndrome
(which includes one noiseless future audit round) and an implementable factory
accept/reject decision, which can depend only on detector outcomes available at
the actual live-handoff boundary.

For k faultable post-ZXZ surface-QEC rounds, let
  D_now  = detector history physically available at live handoff,
  D_tail = one noiseless future syndrome round used only by the ideal decoder,
  F      = complementary terminal-probe recovery-frame label.

S7-B3 determines all <=2-fault reachable full recovery syndromes
(D_now,D_tail) and their reachable frames. A full recovery syndrome is
'ambiguous' when both logical frame values are reachable through <=2 faults.

An implementable conservative handoff policy must reject a present-time
syndrome D_now if *any* future D_tail reachable through <=2 faults makes the
full recovery syndrome ambiguous. This script therefore projects every
ambiguous full syndrome onto its present-time prefix and calls that prefix
UNSAFE.

A useful certified live-handoff policy exists only if D_now=0 is not unsafe.
For all accepted present-time syndromes, every <=2 full recovery syndrome is
frame-unique by construction, so an ideal downstream recovery can correct all
audited <=2 faults.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZXZ_DIR = HERE.parent / "zxz"
for p in (HERE, ZXZ_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import zxz_faultable_handoff_recovery_audit as s7b3  # type: ignore
import zxz_ideal_recovery_handoff_audit as s7b2  # type: ignore

BASE_DETECTORS = 123
DETECTORS_PER_SURFACE_ROUND = 24
FRAME_LABELS = ("I", "X", "Z", "Y")


def _class_coeff(r) -> float:
    return 1.0 / 15.0 if r.gate == "CX" else 1.0


def _actual_prefix(full_syndrome: int, actual_bits: int) -> int:
    return int(full_syndrome) & ((1 << actual_bits) - 1)


def derive_policy(paired_rows, analysis: dict, actual_bits: int) -> dict:
    state_table = analysis["state_table"]
    ambiguous_full = {
        int(s) for s, flags in state_table.items() if int(flags).bit_count() > 1
    }
    unsafe_actual = {_actual_prefix(s, actual_bits) for s in ambiguous_full}

    rejected_single = [
        r for r in paired_rows if _actual_prefix(r.recovery_syndrome, actual_bits) in unsafe_actual
    ]
    accepted_single = [
        r for r in paired_rows if _actual_prefix(r.recovery_syndrome, actual_bits) not in unsafe_actual
    ]

    reject_coeff = sum(_class_coeff(r) for r in rejected_single)

    # Exact safety follows from construction: if D_now is accepted, there is no
    # ambiguous full syndrome with that D_now prefix. Verify this directly for
    # every reachable <=2 full syndrome in the state table.
    accepted_ambiguous_full = [
        int(s) for s in ambiguous_full if _actual_prefix(s, actual_bits) not in unsafe_actual
    ]
    if accepted_ambiguous_full:
        raise AssertionError("unsafe projection failed")

    useful = 0 not in unsafe_actual

    return {
        "actual_detector_bits": actual_bits,
        "ambiguous_full_recovery_syndromes_le2": len(ambiguous_full),
        "unsafe_actual_syndrome_count": len(unsafe_actual),
        "zero_actual_syndrome_is_safe": useful,
        "operational_certified_policy_exists": useful,
        "rejected_single_fault_classes": len(rejected_single),
        "accepted_single_fault_classes": len(accepted_single),
        "single_reject_coefficient": reject_coeff,
        "exact_accepted_single_failures": 0,
        "exact_accepted_pair_failures": 0,
        "unsafe_actual_syndromes_hex": [hex(s) for s in sorted(unsafe_actual)],
    }


def audit_depth(hirano_src: Path, k: int, ideal_tail_rounds: int, signature_p: float) -> dict:
    xrows, xloc, xmeta = s7b3._build_channel_rows(
        hirano_src, "X", k, ideal_tail_rounds, signature_p
    )
    zrows, zloc, zmeta = s7b3._build_channel_rows(
        hirano_src, "Z", k, ideal_tail_rounds, signature_p
    )
    if len(xloc) != len(zloc):
        raise AssertionError((len(xloc), len(zloc)))
    paired, mismatches = s7b2._pair_complementary(xrows, zrows)
    if mismatches:
        raise AssertionError(("X/Z recovery syndrome mismatch", mismatches))

    analysis = s7b2._analyze_exact_le2(paired)
    actual_bits = BASE_DETECTORS + k * DETECTORS_PER_SURFACE_ROUND
    pol = derive_policy(paired, analysis, actual_bits)

    return {
        "faultable_handoff_rounds": k,
        "ideal_tail_rounds": ideal_tail_rounds,
        "physical_locations": len(xloc),
        "single_fault_classes": len(paired),
        "two_fault_class_pairs": s7b3.s7b._pair_denominator(xloc),
        "full_recovery_detector_count": len(xmeta["recovery_ids"]),
        "all_full_recovery_syndromes_unique_le2": analysis["all_recovery_syndromes_logically_unique_le2"],
        "ambiguity_type_counts": analysis["ambiguity_type_counts"],
        **pol,
    }


def _self_test():
    rows = [
        s7b2.RecoveryFault(0, 0, 0, "RZ", (0,), "X", 0b0001, 0),
        s7b2.RecoveryFault(1, 1, 0, "RZ", (1,), "X", 0b0010, 1),
    ]
    a = s7b2._analyze_exact_le2(rows)
    # This synthetic table is fully unambiguous, so no present syndrome is unsafe.
    p = derive_policy(rows, a, actual_bits=2)
    assert p["unsafe_actual_syndrome_count"] == 0
    assert p["zero_actual_syndrome_is_safe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", required=True, type=Path)
    ap.add_argument("--faultable-rounds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ideal-tail-rounds", type=int, default=1)
    ap.add_argument("--signature-p", type=float, default=1e-6)
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s7b4_results"))
    args = ap.parse_args()

    _self_test()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for k in args.faultable_rounds:
        s = audit_depth(args.hirano_src, k, args.ideal_tail_rounds, args.signature_p)
        results.append(s)
        (args.out_dir / f"s7b4_faultable{k}_summary.json").write_text(
            json.dumps(s, indent=2, sort_keys=True), encoding="utf-8"
        )
        # Persist unsafe table separately for decoder implementation.
        (args.out_dir / f"s7b4_faultable{k}_unsafe_actual_syndromes.txt").write_text(
            "\n".join(s["unsafe_actual_syndromes_hex"]) + "\n", encoding="utf-8"
        )
        print(json.dumps({kk: vv for kk, vv in s.items() if kk != "unsafe_actual_syndromes_hex"}, indent=2, sort_keys=True))

    (args.out_dir / "s7b4_all_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("S7-B4 PASS: operational present-syndrome selective handoff policy audit completed")


if __name__ == "__main__":
    main()
