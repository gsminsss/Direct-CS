#!/usr/bin/env python3
"""S7-B5: Monte Carlo performance of the certified k=2 live-handoff policy.

This stage follows S7-B4.  S7-B4 established, under the canonical active-location
stochastic-Pauli model, that among k=0,1,2 post-ZXZ faultable surface-QEC rounds,
k=2 is the first live-handoff depth for which an implementable accept/reject
rule based ONLY on detector outcomes available at handoff can:

  * accept the fault-free present syndrome D_now=0; and
  * guarantee no accepted harmful order-1 or order-2 residual in the exact audit.

S7-B5 measures the *performance* of that operational rule.

Important semantics
===================
The Monte Carlo is sampled from the exact single-fault signature catalogue, not
from an approximate graphlike DEM.  At each physical location:

  * with probability p, the location fails;
  * a CX failure chooses one of the 15 non-II two-qubit Paulis uniformly;
  * RX/RZ/MX/MZ have the one canonical anticommuting Pauli class.

At most one fault class can occur at a physical location.  Different locations
fail independently.  Because the entire ZXZ + handoff circuit is Clifford and
we use Pauli faults, multi-fault detector and terminal-frame signatures are the
GF(2) XOR of the exact single-fault signatures.  This is therefore an exact
Monte Carlo sampler for the canonical stochastic-Pauli model represented by
S7-B3/S7-B4, not a graphlike approximation.

Acceptance is decided using ONLY the 171 detector bits physically available at
the k=2 live handoff.  A present syndrome is rejected iff S7-B4 marked it
UNSAFE, i.e. iff some <=2-fault continuation with the same present prefix has
an ambiguous ideal-recovery frame.

Scoring beyond fault order 2
============================
The exact certification only constrains <=2 faults.  For Monte Carlo shots with
higher-order faults, the one noiseless future audit round supplies a 195-bit
full recovery syndrome and the paired complementary probes supply the actual
4-valued frame I/X/Z/Y.

For each accepted shot we report:

  * known_unique: its full syndrome occurs in the exact <=2 table with exactly
    one frame.  The <=2 table then supplies a concrete recovery prediction.
  * ambiguous: its full syndrome occurs in the <=2 table with >1 frame.
  * unknown: its full syndrome is not reachable through <=2 faults.

A mismatch on a known_unique syndrome is a definite accepted logical failure
for the certified <=2 lookup recovery.  Ambiguous/unknown higher-order shots
are not silently assigned a favorable correction.  Instead S7-B5 reports:

  lower_bound = known_unique mismatches / accepted
  upper_bound = (known_unique mismatches + ambiguous + unknown) / accepted

The upper bound pessimistically treats every unresolved accepted shot as a
failure.  These bounds are performance diagnostics, not a claim about an
optimal downstream surface-code decoder.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ZXZ_DIR = HERE.parent / "zxz"
for p in (HERE, ZXZ_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import zxz_faultable_handoff_recovery_audit as s7b3  # type: ignore
import zxz_ideal_recovery_handoff_audit as s7b2  # type: ignore
import zxz_operational_handoff_policy_audit as s7b4  # type: ignore

FAULTABLE_ROUNDS = 2
IDEAL_TAIL_ROUNDS = 1
ACTUAL_BITS = 171
FULL_BITS = 195
EXPECTED_LOCATIONS = 1028
EXPECTED_SINGLE_CLASSES = 9750
EXPECTED_PAIR_CLASSES = 47_460_960
EXPECTED_UNSAFE = 14_170
EXPECTED_REJECTED_SINGLE = 1_457
EXPECTED_ACCEPTED_SINGLE = 8_293
EXPECTED_REJECT_COEFF = 157.8
FRAME_LABELS = ("I", "X", "Z", "Y")


@dataclass(frozen=True)
class SweepRow:
    p: float
    shots: int
    seed: int
    operational_accepted: int
    operational_rejected: int
    operational_acceptance: float
    zero_present_syndrome_shots: int
    zero_present_syndrome_acceptance: float
    accepted_nonzero_present_syndrome_shots: int
    accepted_known_unique_full_syndrome: int
    accepted_ambiguous_full_syndrome: int
    accepted_unknown_full_syndrome: int
    known_unique_frame_mismatches: int
    conditional_known_unique_mismatch_rate: float | None
    conditional_logical_error_lower_bound: float | None
    conditional_logical_error_upper_bound: float | None
    unresolved_fraction_among_accepted: float | None
    first_order_acceptance_approx: float
    two_block_independent_acceptance_estimate: float
    sampled_faults_total: int
    mean_faults_per_shot: float
    max_faults_in_any_shot: int


def _int_to_words(v: int, nwords: int = 4) -> np.ndarray:
    return np.array([(int(v) >> (64 * i)) & ((1 << 64) - 1) for i in range(nwords)], dtype=np.uint64)


def _words_to_ints(words: np.ndarray) -> list[int]:
    a = np.asarray(words, dtype=np.uint64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise AssertionError(a.shape)
    out: list[int] = []
    for row in a:
        out.append(
            int(row[0])
            | (int(row[1]) << 64)
            | (int(row[2]) << 128)
            | (int(row[3]) << 192)
        )
    return out


def _build_exact_catalogue(hirano_src: Path, signature_p: float):
    xrows, xloc, xmeta = s7b3._build_channel_rows(
        hirano_src, "X", FAULTABLE_ROUNDS, IDEAL_TAIL_ROUNDS, signature_p
    )
    zrows, zloc, zmeta = s7b3._build_channel_rows(
        hirano_src, "Z", FAULTABLE_ROUNDS, IDEAL_TAIL_ROUNDS, signature_p
    )
    if len(xloc) != len(zloc):
        raise AssertionError((len(xloc), len(zloc)))
    paired, mismatches = s7b2._pair_complementary(xrows, zrows)
    if mismatches:
        raise AssertionError(("X/Z recovery syndrome mismatch", mismatches))

    if len(xloc) != EXPECTED_LOCATIONS:
        raise AssertionError((len(xloc), EXPECTED_LOCATIONS))
    if len(paired) != EXPECTED_SINGLE_CLASSES:
        raise AssertionError((len(paired), EXPECTED_SINGLE_CLASSES))
    pair_count = s7b3.s7b._pair_denominator(xloc)
    if pair_count != EXPECTED_PAIR_CLASSES:
        raise AssertionError((pair_count, EXPECTED_PAIR_CLASSES))
    if len(xmeta["recovery_ids"]) != FULL_BITS:
        raise AssertionError((len(xmeta["recovery_ids"]), FULL_BITS))

    analysis = s7b2._analyze_exact_le2(paired)
    pol = s7b4.derive_policy(paired, analysis, ACTUAL_BITS)

    # Reassert the user's S7-B4 k=2 fingerprint.  These are scientific results,
    # not assumptions hidden inside the sampler.
    checks = {
        "operational_certified_policy_exists": True,
        "zero_actual_syndrome_is_safe": True,
        "unsafe_actual_syndrome_count": EXPECTED_UNSAFE,
        "rejected_single_fault_classes": EXPECTED_REJECTED_SINGLE,
        "accepted_single_fault_classes": EXPECTED_ACCEPTED_SINGLE,
        "exact_accepted_single_failures": 0,
        "exact_accepted_pair_failures": 0,
    }
    for k, v in checks.items():
        if pol[k] != v:
            raise AssertionError((k, pol[k], v))
    if abs(float(pol["single_reject_coefficient"]) - EXPECTED_REJECT_COEFF) > 1e-9:
        raise AssertionError(pol["single_reject_coefficient"])

    unsafe = {int(h, 16) for h in pol["unsafe_actual_syndromes_hex"]}
    state_table: dict[int, int] = {int(k): int(v) for k, v in analysis["state_table"].items()}

    # Group the exact paired fault rows by physical location.  Every CX location
    # must contribute 15 equiprobable classes; every other location contributes 1.
    by_loc: dict[int, list[Any]] = {}
    for r in paired:
        by_loc.setdefault(int(r.location_id), []).append(r)
    if set(by_loc) != set(range(EXPECTED_LOCATIONS)):
        raise AssertionError("location ids are not contiguous 0..1027")

    loc_syndrome_words: list[np.ndarray] = []
    loc_frames: list[np.ndarray] = []
    for lid in range(EXPECTED_LOCATIONS):
        rs = sorted(by_loc[lid], key=lambda r: int(r.fault_id))
        expected = 15 if rs[0].gate == "CX" else 1
        if len(rs) != expected or any(r.gate != rs[0].gate for r in rs):
            raise AssertionError((lid, rs[0].gate, len(rs), expected))
        loc_syndrome_words.append(np.vstack([_int_to_words(int(r.recovery_syndrome)) for r in rs]))
        loc_frames.append(np.array([int(r.frame) for r in rs], dtype=np.uint8))

    meta = {
        "faultable_handoff_rounds": FAULTABLE_ROUNDS,
        "ideal_tail_rounds": IDEAL_TAIL_ROUNDS,
        "actual_detector_bits": ACTUAL_BITS,
        "full_recovery_detector_bits": FULL_BITS,
        "physical_locations": EXPECTED_LOCATIONS,
        "single_fault_classes": EXPECTED_SINGLE_CLASSES,
        "two_fault_class_pairs": EXPECTED_PAIR_CLASSES,
        "unsafe_actual_syndrome_count": len(unsafe),
        "single_reject_coefficient": float(pol["single_reject_coefficient"]),
        "exact_accepted_single_failures": int(pol["exact_accepted_single_failures"]),
        "exact_accepted_pair_failures": int(pol["exact_accepted_pair_failures"]),
        "ambiguous_full_recovery_syndromes_le2": int(pol["ambiguous_full_recovery_syndromes_le2"]),
        "unique_full_recovery_syndromes_le2": int(len(state_table) - pol["ambiguous_full_recovery_syndromes_le2"]),
        "location_counts": xmeta["location_counts"],
    }
    return loc_syndrome_words, loc_frames, unsafe, state_table, meta


def _single_frame_from_flags(flags: int) -> int | None:
    f = int(flags)
    if f == 0 or f.bit_count() != 1:
        return None
    return f.bit_length() - 1


def _sample_p(
    *,
    p: float,
    shots: int,
    seed: int,
    loc_syndrome_words: list[np.ndarray],
    loc_frames: list[np.ndarray],
    unsafe: set[int],
    state_table: dict[int, int],
    reject_coeff: float,
) -> SweepRow:
    if not (0 < p < 1):
        raise ValueError(p)
    if shots <= 0:
        raise ValueError(shots)

    rng = np.random.default_rng(seed)
    synd = np.zeros((shots, 4), dtype=np.uint64)
    frame = np.zeros(shots, dtype=np.uint8)
    fault_count = np.zeros(shots, dtype=np.uint16)

    # Exact independent-location sampling.  For one location, drawing the total
    # number of failed shots as Binomial(shots,p), then choosing that many shot
    # indices uniformly without replacement, is exactly equivalent to iid
    # Bernoulli(p) failure indicators over shots.  Locations are sampled
    # independently.  Conditional on a CX failure, choose one of 15 classes.
    for sigs, frames in zip(loc_syndrome_words, loc_frames, strict=True):
        m = int(rng.binomial(shots, p))
        if m == 0:
            continue
        idx = rng.choice(shots, size=m, replace=False)
        if len(frames) == 1:
            cls = np.zeros(m, dtype=np.int64)
        else:
            cls = rng.integers(0, len(frames), size=m, endpoint=False)
        synd[idx] ^= sigs[cls]
        frame[idx] ^= frames[cls]
        fault_count[idx] += 1

    full_masks = _words_to_ints(synd)
    actual_mask = (1 << ACTUAL_BITS) - 1

    operational_accepted = 0
    zero_now = 0
    accepted_nonzero = 0
    known_unique = 0
    ambiguous = 0
    unknown = 0
    mismatches = 0

    for full, actual_frame in zip(full_masks, frame, strict=True):
        now = full & actual_mask
        if now == 0:
            zero_now += 1
        if now in unsafe:
            continue
        operational_accepted += 1
        if now != 0:
            accepted_nonzero += 1

        flags = state_table.get(full)
        if flags is None:
            unknown += 1
            continue
        pred = _single_frame_from_flags(flags)
        if pred is None:
            ambiguous += 1
            continue
        known_unique += 1
        if int(pred) != int(actual_frame):
            mismatches += 1

    rejected = shots - operational_accepted
    unresolved = ambiguous + unknown
    acceptance = operational_accepted / shots
    zero_acceptance = zero_now / shots

    if operational_accepted:
        lower = mismatches / operational_accepted
        upper = (mismatches + unresolved) / operational_accepted
        unresolved_frac = unresolved / operational_accepted
    else:
        lower = upper = unresolved_frac = None

    known_mismatch_rate = (mismatches / known_unique) if known_unique else None

    return SweepRow(
        p=float(p),
        shots=int(shots),
        seed=int(seed),
        operational_accepted=int(operational_accepted),
        operational_rejected=int(rejected),
        operational_acceptance=float(acceptance),
        zero_present_syndrome_shots=int(zero_now),
        zero_present_syndrome_acceptance=float(zero_acceptance),
        accepted_nonzero_present_syndrome_shots=int(accepted_nonzero),
        accepted_known_unique_full_syndrome=int(known_unique),
        accepted_ambiguous_full_syndrome=int(ambiguous),
        accepted_unknown_full_syndrome=int(unknown),
        known_unique_frame_mismatches=int(mismatches),
        conditional_known_unique_mismatch_rate=(float(known_mismatch_rate) if known_mismatch_rate is not None else None),
        conditional_logical_error_lower_bound=(float(lower) if lower is not None else None),
        conditional_logical_error_upper_bound=(float(upper) if upper is not None else None),
        unresolved_fraction_among_accepted=(float(unresolved_frac) if unresolved_frac is not None else None),
        first_order_acceptance_approx=float(max(0.0, 1.0 - reject_coeff * p)),
        two_block_independent_acceptance_estimate=float(acceptance * acceptance),
        sampled_faults_total=int(np.sum(fault_count, dtype=np.uint64)),
        mean_faults_per_shot=float(np.mean(fault_count)),
        max_faults_in_any_shot=int(np.max(fault_count)),
    )


def _synthetic_sampler_self_test() -> None:
    # One one-class location with deterministic syndrome/frame signature.
    # p=1 is forbidden by the production sampler, so directly test packing and
    # policy/lookup helpers instead.
    v = (1 << 0) | (1 << 70) | (1 << 130) | (1 << 194)
    words = _int_to_words(v)
    got = _words_to_ints(words.reshape(1, 4))[0]
    if got != v:
        raise AssertionError((hex(got), hex(v)))
    if _single_frame_from_flags(1 << 3) != 3:
        raise AssertionError("single-frame lookup")
    if _single_frame_from_flags((1 << 1) | (1 << 2)) is not None:
        raise AssertionError("ambiguous-frame lookup")


def _write_csv(path: Path, rows: list[SweepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(SweepRow.__dataclass_fields__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", required=True, type=Path)
    ap.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=[2e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    )
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--signature-p", type=float, default=1e-6)
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s7b5_results"))
    args = ap.parse_args()

    _synthetic_sampler_self_test()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Building exact k=2 S7-B4 catalogue/policy...")
    loc_sigs, loc_frames, unsafe, state_table, meta = _build_exact_catalogue(
        args.hirano_src, args.signature_p
    )
    (args.out_dir / "s7b5_catalogue_summary.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, sort_keys=True))

    rows: list[SweepRow] = []
    for i, p in enumerate(args.p_values):
        # Different deterministic sub-seed per p avoids reusing the same random
        # stream while preserving exact reproducibility.
        seed = int(args.seed + 1009 * i)
        print(f"Sampling p={p:g}, shots={args.shots}, seed={seed}...")
        r = _sample_p(
            p=float(p),
            shots=int(args.shots),
            seed=seed,
            loc_syndrome_words=loc_sigs,
            loc_frames=loc_frames,
            unsafe=unsafe,
            state_table=state_table,
            reject_coeff=float(meta["single_reject_coefficient"]),
        )
        rows.append(r)
        print(json.dumps(asdict(r), indent=2, sort_keys=True))

    _write_csv(args.out_dir / "s7b5_sweep.csv", rows)
    (args.out_dir / "s7b5_sweep.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2, sort_keys=True), encoding="utf-8"
    )

    print("S7-B5 PASS: k=2 operational live-handoff exact-catalogue Monte Carlo completed")


if __name__ == "__main__":
    main()
