#!/usr/bin/env python3
"""S5.6 ambiguity-aware lookup decoding for the Hirano d=5 full-ZXZ escape.

This program is the follow-up to S5.5.  S5.5 proved:
  * every <=1-fault syndrome is logically unambiguous;
  * genuine logical ambiguity first appears at order 2.

We therefore compare three syndrome-handling policies on exactly the same
797-location canonical noisy circuit:

  full_postselection
      Accept iff all 135 detectors are zero.  This is the already-certified
      historical Step-4 behavior.

  selective_certified
      Build the exact <=2 syndrome table.  If a syndrome has a unique logical
      observable through order 2, accept and apply that correction.  If both
      observables are reachable through order 2, reject.  Syndromes not
      reachable by <=2 faults are also rejected in this conservative version.
      Consequently every accepted fault set of size <=2 is corrected exactly.

  min_order_lookup
      Never reject syndromes present in the <=2 table.  For ambiguous syndromes,
      choose the logical branch appearing at the lower minimum fault order;
      if tied, choose observable 0.  Unknown (>2-only) syndromes are rejected.
      This corrects every single fault but can fail at order 2; it is included
      as a correction-oriented comparison, not as a d_f=3 certified mode.

The script performs exact order<=2 structural audits before Monte Carlo and then
samples the physical Stim circuit (not the DEM) over a user-specified p grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zxz_exact_fault_audit as s3  # noqa: E402
import zxz_exact_syndrome_ambiguity as s55  # noqa: E402

Channel = Literal["X", "Z"]

EXPECTED_LOCATIONS = 797
EXPECTED_SINGLE_CLASSES = 7_279
EXPECTED_PAIR_CLASSES = 26_439_666
EXPECTED_DETECTORS = 135
EXPECTED_OBSERVABLES = 1
EXPECTED_ACCEPTED_SINGLES = {"X": 450, "Z": 466}


@dataclass(frozen=True)
class LookupStats:
    channel: Channel
    states_le2: int
    ambiguous_syndromes_le2: int
    unambiguous_syndromes_le2: int
    order1_syndromes: int
    ambiguous_order1_syndromes_due_to_order2_collision: int
    full_postselection_single_reject_coefficient: float
    selective_single_reject_coefficient: float
    selective_single_rejected_fault_classes: int
    selective_exact_accepted_single_failures: int
    selective_exact_accepted_pair_failures: int
    min_order_exact_single_failures: int


@dataclass(frozen=True)
class SweepRow:
    channel: Channel
    p: float
    mode: str
    shots: int
    accepted: int
    rejected: int
    logical_errors_accepted: int
    acceptance: float
    conditional_logical_error: float | None
    unconditional_accepted_logical_error: float
    ambiguous_syndrome_shots: int
    unknown_syndrome_shots: int
    nonzero_syndrome_shots: int
    seed: int


def _build_states(rows: list[s3.FaultRow]) -> tuple[dict[int, int], list[s55.SigGroup]]:
    groups = s55._build_groups(rows)
    states: dict[int, int] = {}
    s55._set_state_bit(states, 0, 0, 0)
    for g in groups:
        s55._set_state_bit(states, g.detector_mask, g.observable, 1)
    for i, a in enumerate(groups):
        for j in range(i, len(groups)):
            b = groups[j]
            if not s55._pair_groups_legal(a, b, i == j):
                continue
            s55._set_state_bit(
                states,
                a.detector_mask ^ b.detector_mask,
                a.observable ^ b.observable,
                2,
            )
    return states, groups


def _decision_from_flags(flags: int, mode: str) -> tuple[bool, int | None, bool]:
    """Returns (accept, prediction, ambiguous_through_order2)."""
    k0 = s55._min_order(flags, 0)
    k1 = s55._min_order(flags, 1)
    ambiguous = k0 is not None and k1 is not None

    if mode == "selective_certified":
        if ambiguous:
            return False, None, True
        if k0 is not None:
            return True, 0, False
        if k1 is not None:
            return True, 1, False
        raise AssertionError("state exists but has neither logical branch")

    if mode == "min_order_lookup":
        if k0 is None and k1 is None:
            raise AssertionError("state exists but has neither logical branch")
        if k0 is None:
            return True, 1, False
        if k1 is None:
            return True, 0, False
        if k0 < k1:
            return True, 0, True
        if k1 < k0:
            return True, 1, True
        # Equal-order ambiguity.  Deterministic tie-break only; this is a baseline.
        return True, 0, True

    raise ValueError(mode)


def _single_class_alpha(row: s3.FaultRow) -> float:
    # Canonical model: DEPOLARIZE2(p) makes each of 15 CX Pauli classes p/15.
    # RX/RZ/MX/MZ each have one anticommuting Pauli mechanism with probability p.
    return 1.0 / 15.0 if row.gate == "CX" else 1.0


def _audit_lookup(
    rows: list[s3.FaultRow],
    states: dict[int, int],
    groups: list[s55.SigGroup],
    channel: Channel,
) -> LookupStats:
    ambiguous = {
        mask
        for mask, flags in states.items()
        if s55._min_order(flags, 0) is not None and s55._min_order(flags, 1) is not None
    }
    order1_masks = {g.detector_mask for g in groups}
    ambiguous_order1_masks = ambiguous & order1_masks

    # Exact single-fault policy checks.
    sel_single_fail = 0
    min_single_fail = 0
    sel_rejected_classes = 0
    full_reject_coeff = 0.0
    sel_reject_coeff = 0.0
    for r in rows:
        alpha = _single_class_alpha(r)
        if r.detector_mask != 0:
            full_reject_coeff += alpha

        flags = states[r.detector_mask]
        acc, pred, _ = _decision_from_flags(flags, "selective_certified")
        if not acc:
            sel_rejected_classes += 1
            sel_reject_coeff += alpha
        elif int(pred) != int(r.observable_flip):
            sel_single_fail += 1

        acc2, pred2, _ = _decision_from_flags(flags, "min_order_lookup")
        if not acc2:
            raise AssertionError("min_order lookup unexpectedly rejects a known single syndrome")
        if int(pred2) != int(r.observable_flip):
            min_single_fail += 1

    if sel_single_fail != 0:
        raise AssertionError(f"{channel}: selective decoder fails {sel_single_fail} single faults")
    if min_single_fail != 0:
        raise AssertionError(f"{channel}: min-order decoder fails {min_single_fail} single faults")

    # Exact <=2 existence audit for selective policy.  We only need to establish
    # that no legal pair is both accepted and logically miscorrected.  Group
    # quotienting is exact for existence, as in S5.5.
    selective_pair_failures = 0
    witness: dict[str, Any] | None = None
    for i, a in enumerate(groups):
        for j in range(i, len(groups)):
            b = groups[j]
            if not s55._pair_groups_legal(a, b, i == j):
                continue
            mask = a.detector_mask ^ b.detector_mask
            obs = a.observable ^ b.observable
            flags = states[mask]
            acc, pred, _ = _decision_from_flags(flags, "selective_certified")
            if acc and int(pred) != int(obs):
                selective_pair_failures += 1
                if witness is None:
                    witness = {
                        "mask": hex(mask),
                        "observable": int(obs),
                        "prediction": int(pred),
                        "group_a": int(a.group_id),
                        "group_b": int(b.group_id),
                    }
    if selective_pair_failures:
        raise AssertionError(
            f"{channel}: selective policy has {selective_pair_failures} accepted harmful <=2 group pairs; "
            f"first={witness}"
        )

    return LookupStats(
        channel=channel,
        states_le2=len(states),
        ambiguous_syndromes_le2=len(ambiguous),
        unambiguous_syndromes_le2=len(states) - len(ambiguous),
        order1_syndromes=len(order1_masks),
        ambiguous_order1_syndromes_due_to_order2_collision=len(ambiguous_order1_masks),
        full_postselection_single_reject_coefficient=float(full_reject_coeff),
        selective_single_reject_coefficient=float(sel_reject_coeff),
        selective_single_rejected_fault_classes=int(sel_rejected_classes),
        selective_exact_accepted_single_failures=int(sel_single_fail),
        selective_exact_accepted_pair_failures=int(selective_pair_failures),
        min_order_exact_single_failures=int(min_single_fail),
    )


def _detector_rows_to_masks(dets: np.ndarray) -> list[int]:
    a = np.asarray(dets, dtype=np.uint8)
    if a.ndim != 2 or a.shape[1] != EXPECTED_DETECTORS:
        raise AssertionError(f"unexpected detector sample shape {a.shape}")
    packed = np.packbits(a, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def _sample_channel_at_p(
    hirano_src: Path,
    *,
    channel: Channel,
    p: float,
    shots: int,
    seed: int,
    states: dict[int, int],
) -> list[SweepRow]:
    circuit, wrapper, init_range = s3.build_public_zxz_with_init_range(hirano_src, channel=channel)
    post_ids = [int(d.id) for d in wrapper.detectors_for_post_selection]
    if circuit.num_detectors != EXPECTED_DETECTORS or circuit.num_observables != EXPECTED_OBSERVABLES:
        raise AssertionError((circuit.num_detectors, circuit.num_observables))
    if len(post_ids) != EXPECTED_DETECTORS or set(post_ids) != set(range(EXPECTED_DETECTORS)):
        raise AssertionError(f"{channel}: expected all 135 detectors in historical postselection set")

    noisy, locations, _ = s3.instrument_canonical_fault_model(circuit, init_range=init_range, p=p)
    if len(locations) != EXPECTED_LOCATIONS:
        raise AssertionError(f"{channel}: {len(locations)} locations")
    if s3._pair_denominator(locations) != EXPECTED_PAIR_CLASSES:
        raise AssertionError(f"{channel}: pair denominator mismatch")

    sampler = noisy.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    if obs.ndim == 2:
        actual = obs[:, 0]
    else:
        actual = obs
    masks = _detector_rows_to_masks(dets)
    nonzero = np.any(dets.astype(bool), axis=1)

    rows_out: list[SweepRow] = []

    # Historical full postselection.
    accepted_idx = ~nonzero
    accepted = int(np.sum(accepted_idx))
    errors = int(np.sum(actual[accepted_idx])) if accepted else 0
    rows_out.append(
        SweepRow(
            channel=channel,
            p=float(p),
            mode="full_postselection",
            shots=shots,
            accepted=accepted,
            rejected=shots - accepted,
            logical_errors_accepted=errors,
            acceptance=accepted / shots,
            conditional_logical_error=(errors / accepted if accepted else None),
            unconditional_accepted_logical_error=errors / shots,
            ambiguous_syndrome_shots=0,
            unknown_syndrome_shots=0,
            nonzero_syndrome_shots=int(np.sum(nonzero)),
            seed=seed,
        )
    )

    for mode in ("selective_certified", "min_order_lookup"):
        acc_count = 0
        err_count = 0
        amb_count = 0
        unknown_count = 0
        for mask, o in zip(masks, actual, strict=True):
            flags = states.get(mask)
            if flags is None:
                # Conservative: reject syndromes not reachable by <=2 canonical faults.
                unknown_count += 1
                continue
            accept, pred, amb = _decision_from_flags(flags, mode)
            if amb:
                amb_count += 1
            if not accept:
                continue
            acc_count += 1
            if int(pred) != int(o):
                err_count += 1
        rows_out.append(
            SweepRow(
                channel=channel,
                p=float(p),
                mode=mode,
                shots=shots,
                accepted=acc_count,
                rejected=shots - acc_count,
                logical_errors_accepted=err_count,
                acceptance=acc_count / shots,
                conditional_logical_error=(err_count / acc_count if acc_count else None),
                unconditional_accepted_logical_error=err_count / shots,
                ambiguous_syndrome_shots=amb_count,
                unknown_syndrome_shots=unknown_count,
                nonzero_syndrome_shots=int(np.sum(nonzero)),
                seed=seed,
            )
        )

    return rows_out


def _write_csv(path: Path, rows: list[SweepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _synthetic_self_test() -> None:
    # flags: O0@1 and O1@2 => selective rejects; min-order chooses O0.
    flags = s55.B_O0_K1 | s55.B_O1_K2
    a = _decision_from_flags(flags, "selective_certified")
    b = _decision_from_flags(flags, "min_order_lookup")
    if a != (False, None, True):
        raise AssertionError(a)
    if b != (True, 0, True):
        raise AssertionError(b)

    # O1 only at order2 => both accept and predict 1.
    flags2 = s55.B_O1_K2
    if _decision_from_flags(flags2, "selective_certified") != (True, 1, False):
        raise AssertionError("selective unique O1 test")
    if _decision_from_flags(flags2, "min_order_lookup") != (True, 1, False):
        raise AssertionError("min-order unique O1 test")

    # Detector bit packing convention: detector ids 0,2,8 => mask 1+4+256.
    arr = np.zeros((1, EXPECTED_DETECTORS), dtype=np.uint8)
    arr[0, [0, 2, 8]] = 1
    got = _detector_rows_to_masks(arr)[0]
    if got != (1 << 0) | (1 << 2) | (1 << 8):
        raise AssertionError(f"mask packing self-test failed: {got:#x}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--channel", choices=["X", "Z", "both"], default="both")
    ap.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=[2e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    )
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/zxz/s56_results"))
    args = ap.parse_args()

    _synthetic_self_test()
    channels: list[Channel] = ["X", "Z"] if args.channel == "both" else [args.channel]  # type: ignore[list-item]

    lookup_summaries: list[LookupStats] = []
    all_sweep: list[SweepRow] = []

    for ci, ch in enumerate(channels):
        rows, locations, accepted, malignant_pairs = s55._rebuild_exact_rows(
            args.hirano_src, channel=ch, p=args.signature_p
        )
        if len(locations) != EXPECTED_LOCATIONS or len(rows) != EXPECTED_SINGLE_CLASSES:
            raise AssertionError(f"{ch}: upstream S3 inventory regression")
        if accepted != EXPECTED_ACCEPTED_SINGLES[ch] or malignant_pairs != 0:
            raise AssertionError(f"{ch}: upstream S3 regression accepted={accepted}, mal_pairs={malignant_pairs}")

        states, groups = _build_states(rows)
        audit = _audit_lookup(rows, states, groups, ch)
        lookup_summaries.append(audit)

        for pi, p in enumerate(args.p_values):
            seed = int(args.seed + ci * 100_000 + pi)
            all_sweep.extend(
                _sample_channel_at_p(
                    args.hirano_src,
                    channel=ch,
                    p=float(p),
                    shots=args.shots,
                    seed=seed,
                    states=states,
                )
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "s56_lookup_summary.json").write_text(
        json.dumps([asdict(x) for x in lookup_summaries], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "s56_sweep.json").write_text(
        json.dumps([asdict(x) for x in all_sweep], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(args.out_dir / "s56_sweep.csv", all_sweep)

    print(json.dumps([asdict(x) for x in lookup_summaries], indent=2, sort_keys=True))
    print(json.dumps([asdict(x) for x in all_sweep], indent=2, sort_keys=True))
    print("S5.6 PASS: selective certified lookup exact <=2 audit + physical-circuit MC completed")


if __name__ == "__main__":
    main()
