#!/usr/bin/env python3
"""S5.5 exact <=2-fault syndrome/observable ambiguity audit.

Purpose
-------
The generic graphlike PyMatching baseline from S5 misdecodes some *single*
physical faults even though S5 found zero ambiguous single-fault syndrome groups.
This program asks the more fundamental information-theoretic question:

    For every detector syndrome D reachable by 0, 1, or 2 canonical physical
    faults, is the logical transfer observable O uniquely determined by D?

If yes, then a syndrome-only decoder exists that can correct every canonical
fault set of size <=2; the O(p) behavior of the generic PyMatching baseline is
purely a decoder-representation artifact.  If not, this program identifies the
first ambiguity order and concrete physical witnesses.

Scope and noise model
---------------------
Exactly the already-reproduced S3/S4 Hirano d=5 full-ZXZ encoded-input scope:
  * 797 active locations;
  * 7,279 single fault classes;
  * 26,439,666 unordered two-fault class pairs at DISTINCT physical locations;
  * ideal Steane initialization retained in the circuit but excluded from fault
    locations;
  * CX: 15 non-II Pauli classes; RX/RZ/MX/MZ: one anticommuting Pauli class;
  * all 135 detectors define the syndrome; observable id 0 is the transfer bit.

Exact compression
-----------------
Every physical fault class has an exact S3 signature (D,O).  Physical pair
signatures obey Clifford/Pauli linearity:

    D(f,g) = D(f) XOR D(g)
    O(f,g) = O(f) XOR O(g)

Instead of iterating 26,439,666 physical pairs, we group single faults by
identical (D,O).  A pair of signature groups represents at least one legal
DISTINCT-location physical pair unless both groups live only at the same one
location (or, for a self-pair, unless the group occurs at fewer than two
locations).  Therefore iterating ~2k signature groups is an exact existence
audit over all 26.4M physical pair classes, not a Monte Carlo approximation.

Output semantics
----------------
For each syndrome, the script records which observables are reachable and the
minimum fault order for O=0 and O=1.  Ambiguity types are reported as e.g.
  1v2 : one observable occurs at order 1, the opposite at order 2
  2v2 : both observables occur at order 2
  0v2 : fault-free and a two-fault event disagree (this would be an accepted
        malignant pair and is already known to be absent from S3/S4).

No assertion assumes the scientific result.  The program structurally PASSes
when the exact audit is completed; it then prints either UNAMBIGUOUS<=2 or the
number/type of genuine ambiguities found.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zxz_exact_fault_audit as s3  # noqa: E402

Channel = Literal["X", "Z"]

EXPECTED_LOCATIONS = 797
EXPECTED_SINGLE_CLASSES = 7_279
EXPECTED_PAIR_CLASSES = 26_439_666
EXPECTED_DETECTORS = 135
EXPECTED_OBSERVABLES = 1
EXPECTED_ACCEPTED_SINGLES = {"X": 450, "Z": 466}

# Packed syndrome-state flags.  O=1 at order 0 is impossible by definition.
B_O0_K0 = 1 << 0
B_O0_K1 = 1 << 1
B_O1_K1 = 1 << 2
B_O0_K2 = 1 << 3
B_O1_K2 = 1 << 4


@dataclass
class SigGroup:
    group_id: int
    detector_mask: int
    observable: int
    fault_count: int
    location_count: int
    examples_by_location: dict[int, s3.FaultRow]

    @property
    def locations(self) -> set[int]:
        return set(self.examples_by_location)


@dataclass(frozen=True)
class S55Summary:
    channel: Channel
    num_detectors: int
    num_observables: int
    physical_locations: int
    single_fault_classes: int
    physical_two_fault_class_pairs: int
    accepted_single_faults: int
    malignant_single_faults: int
    malignant_two_fault_pairs: int
    signature_groups: int
    signature_group_pair_candidates: int
    signature_group_pairs_legal: int
    signature_group_pairs_skipped_same_location_only: int
    unique_syndromes_order0: int
    unique_syndromes_order1: int
    unique_syndromes_order2: int
    unique_syndromes_le2: int
    ambiguous_syndromes_le1: int
    ambiguous_syndromes_le2: int
    ambiguity_type_counts: dict[str, int]
    all_single_syndromes_logically_unambiguous: bool
    all_le2_syndromes_logically_unambiguous: bool
    syndrome_only_decoder_corrects_all_le2: bool
    unavoidable_decoder_failure_order_within_le2: int | None
    ambiguity_witnesses: list[dict[str, Any]]


def _row_dict(r: s3.FaultRow) -> dict[str, Any]:
    return {
        "fault_id": int(r.fault_id),
        "location_id": int(r.location_id),
        "tick": int(r.tick),
        "gate": r.gate,
        "qubits": list(r.qubits),
        "fault_class": r.fault_class,
        "detector_mask_hex": hex(r.detector_mask),
        "observable_flip": int(r.observable_flip),
    }


def _set_state_bit(states: dict[int, int], mask: int, observable: int, order: int) -> None:
    if order == 0:
        if observable != 0:
            raise AssertionError("fault-free observable must be 0")
        bit = B_O0_K0
    elif order == 1:
        bit = B_O1_K1 if observable else B_O0_K1
    elif order == 2:
        bit = B_O1_K2 if observable else B_O0_K2
    else:
        raise ValueError(order)
    states[mask] = states.get(mask, 0) | bit


def _min_order(flags: int, observable: int) -> int | None:
    if observable == 0:
        if flags & B_O0_K0:
            return 0
        if flags & B_O0_K1:
            return 1
        if flags & B_O0_K2:
            return 2
        return None
    if flags & B_O1_K1:
        return 1
    if flags & B_O1_K2:
        return 2
    return None


def _has_order(flags: int, order: int) -> bool:
    if order == 0:
        return bool(flags & B_O0_K0)
    if order == 1:
        return bool(flags & (B_O0_K1 | B_O1_K1))
    if order == 2:
        return bool(flags & (B_O0_K2 | B_O1_K2))
    raise ValueError(order)


def _build_groups(rows: list[s3.FaultRow]) -> list[SigGroup]:
    tmp: dict[tuple[int, int], dict[int, s3.FaultRow]] = defaultdict(dict)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for r in rows:
        key = (int(r.detector_mask), int(r.observable_flip))
        counts[key] += 1
        tmp[key].setdefault(int(r.location_id), r)
    groups: list[SigGroup] = []
    for key in sorted(tmp.keys(), key=lambda x: (x[0], x[1])):
        groups.append(
            SigGroup(
                group_id=len(groups),
                detector_mask=key[0],
                observable=key[1],
                fault_count=counts[key],
                location_count=len(tmp[key]),
                examples_by_location=tmp[key],
            )
        )
    return groups


def _pair_groups_legal(a: SigGroup, b: SigGroup, same_group: bool) -> bool:
    """Whether at least one physical pair from the groups uses distinct locations."""
    if same_group:
        return a.location_count >= 2
    if a.location_count == 1 and b.location_count == 1:
        la = next(iter(a.examples_by_location))
        lb = next(iter(b.examples_by_location))
        return la != lb
    return True


def _pick_distinct_pair(a: SigGroup, b: SigGroup, same_group: bool) -> tuple[s3.FaultRow, s3.FaultRow]:
    if same_group:
        items = list(a.examples_by_location.items())
        if len(items) < 2:
            raise AssertionError("requested illegal self-group witness")
        return items[0][1], items[1][1]

    for la, ra in a.examples_by_location.items():
        for lb, rb in b.examples_by_location.items():
            if la != lb:
                return ra, rb
    raise AssertionError("requested illegal cross-group witness")


def _single_witness(rows: list[s3.FaultRow], mask: int, observable: int) -> dict[str, Any] | None:
    for r in rows:
        if r.detector_mask == mask and r.observable_flip == observable:
            return {"order": 1, "fault": _row_dict(r)}
    return None


def _build_pair_witnesses_for_ambiguous(
    groups: list[SigGroup],
    ambiguous_masks: set[int],
    wanted: set[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Second exact group-pair pass; collect one order-2 witness per wanted (D,O)."""
    out: dict[tuple[int, int], dict[str, Any]] = {}
    if not wanted:
        return out
    n = len(groups)
    for i, a in enumerate(groups):
        for j in range(i, n):
            b = groups[j]
            same = i == j
            if not _pair_groups_legal(a, b, same):
                continue
            mask = a.detector_mask ^ b.detector_mask
            if mask not in ambiguous_masks:
                continue
            obs = a.observable ^ b.observable
            key = (mask, obs)
            if key not in wanted or key in out:
                continue
            r0, r1 = _pick_distinct_pair(a, b, same)
            out[key] = {
                "order": 2,
                "fault_a": _row_dict(r0),
                "fault_b": _row_dict(r1),
                "combined_detector_mask_hex": hex(mask),
                "combined_observable": int(obs),
            }
            if len(out) == len(wanted):
                return out
    return out


def _rebuild_exact_rows(
    hirano_src: Path,
    *,
    channel: Channel,
    p: float,
) -> tuple[list[s3.FaultRow], list[s3.PhysicalLocation], int, int]:
    circuit, wrapper, init_range = s3.build_public_zxz_with_init_range(
        hirano_src, channel=channel
    )
    post_ids = [int(d.id) for d in wrapper.detectors_for_post_selection]
    if circuit.num_detectors != EXPECTED_DETECTORS or circuit.num_observables != EXPECTED_OBSERVABLES:
        raise AssertionError((circuit.num_detectors, circuit.num_observables))
    if len(post_ids) != EXPECTED_DETECTORS or set(post_ids) != set(range(EXPECTED_DETECTORS)):
        raise AssertionError(f"{channel}: expected all 135 detectors to be postselected")

    noisy, locations, noise_map = s3.instrument_canonical_fault_model(
        circuit, init_range=init_range, p=p
    )
    rows = s3.build_fault_rows(locations)
    dem, _, _ = s3.populate_signatures_from_dem(noisy, locations, rows, noise_map)
    if dem.num_detectors != EXPECTED_DETECTORS or dem.num_observables != EXPECTED_OBSERVABLES:
        raise AssertionError((dem.num_detectors, dem.num_observables))

    if len(locations) != EXPECTED_LOCATIONS:
        raise AssertionError(f"{channel}: locations={len(locations)} != {EXPECTED_LOCATIONS}")
    if len(rows) != EXPECTED_SINGLE_CLASSES:
        raise AssertionError(f"{channel}: rows={len(rows)} != {EXPECTED_SINGLE_CLASSES}")
    pair_den = s3._pair_denominator(locations)
    if pair_den != EXPECTED_PAIR_CLASSES:
        raise AssertionError(f"{channel}: pair denominator={pair_den} != {EXPECTED_PAIR_CLASSES}")

    accepted = sum(r.detector_mask == 0 for r in rows)
    malignant_single = sum(r.detector_mask == 0 and r.observable_flip == 1 for r in rows)
    malignant_pairs, _ = s3.exact_malignant_pair_hash(rows)
    if accepted != EXPECTED_ACCEPTED_SINGLES[channel]:
        raise AssertionError(f"{channel}: accepted singles {accepted} != {EXPECTED_ACCEPTED_SINGLES[channel]}")
    if malignant_single != 0 or malignant_pairs != 0:
        raise AssertionError(
            f"{channel}: S3 regression failed: malignant_single={malignant_single}, "
            f"malignant_pairs={malignant_pairs}"
        )
    return rows, locations, accepted, malignant_pairs


def audit_channel(
    hirano_src: Path,
    *,
    channel: Channel,
    p: float,
    out_dir: Path | None,
    max_witnesses: int,
) -> S55Summary:
    rows, locations, accepted, malignant_pairs = _rebuild_exact_rows(
        hirano_src, channel=channel, p=p
    )
    groups = _build_groups(rows)

    # Exact order-0/1/2 reachability state, compressed by signature groups.
    states: dict[int, int] = {}
    _set_state_bit(states, 0, 0, 0)
    for g in groups:
        _set_state_bit(states, g.detector_mask, g.observable, 1)

    n = len(groups)
    group_pair_candidates = n * (n + 1) // 2
    legal_group_pairs = 0
    skipped = 0
    for i, a in enumerate(groups):
        for j in range(i, n):
            b = groups[j]
            same = i == j
            if not _pair_groups_legal(a, b, same):
                skipped += 1
                continue
            legal_group_pairs += 1
            _set_state_bit(
                states,
                a.detector_mask ^ b.detector_mask,
                a.observable ^ b.observable,
                2,
            )

    ambiguous_le1: set[int] = set()
    ambiguous_le2: set[int] = set()
    type_counts: dict[str, int] = defaultdict(int)
    for mask, flags in states.items():
        k0 = _min_order(flags, 0)
        k1 = _min_order(flags, 1)
        if k0 is not None and k1 is not None:
            ambiguous_le2.add(mask)
            type_counts[f"{k0}v{k1}"] += 1
            if k0 <= 1 and k1 <= 1:
                ambiguous_le1.add(mask)

    # Cross-check S5's empirical statement: no single-syndrome ambiguity.
    if ambiguous_le1:
        raise AssertionError(
            f"{channel}: found {len(ambiguous_le1)} <=1 ambiguity groups despite S5 result; "
            f"examples={[hex(x) for x in list(ambiguous_le1)[:5]]}"
        )

    order0_synd = sum(_has_order(f, 0) for f in states.values())
    order1_synd = sum(_has_order(f, 1) for f in states.values())
    order2_synd = sum(_has_order(f, 2) for f in states.values())

    # Build concrete minimal-order witnesses only after the scientific result is known.
    witness_masks = sorted(ambiguous_le2)[:max_witnesses]
    wanted_pair: set[tuple[int, int]] = set()
    partial_witness: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for mask in witness_masks:
        flags = states[mask]
        for obs in (0, 1):
            k = _min_order(flags, obs)
            if k is None:
                continue
            if k == 0:
                partial_witness[mask][obs] = {"order": 0, "fault": None}
            elif k == 1:
                w = _single_witness(rows, mask, obs)
                if w is None:
                    raise AssertionError((mask, obs, k))
                partial_witness[mask][obs] = w
            elif k == 2:
                wanted_pair.add((mask, obs))

    pair_w = _build_pair_witnesses_for_ambiguous(
        groups, set(witness_masks), wanted_pair
    )
    for key in wanted_pair:
        if key not in pair_w:
            raise AssertionError(f"failed to reconstruct pair witness for {key}")
        partial_witness[key[0]][key[1]] = pair_w[key]

    witnesses: list[dict[str, Any]] = []
    for mask in witness_masks:
        flags = states[mask]
        witnesses.append(
            {
                "detector_mask_hex": hex(mask),
                "detector_weight": int(mask.bit_count()),
                "min_order_observable_0": _min_order(flags, 0),
                "min_order_observable_1": _min_order(flags, 1),
                "observable_0_witness": partial_witness[mask].get(0),
                "observable_1_witness": partial_witness[mask].get(1),
            }
        )

    # If an ambiguity exists through order 2, no deterministic syndrome-only
    # decoder can correct every <=2 event.  Because <=1 is proven unambiguous,
    # the earliest unavoidable failure order in this audited window is 2.
    unavoidable_order = None if not ambiguous_le2 else 2

    summary = S55Summary(
        channel=channel,
        num_detectors=EXPECTED_DETECTORS,
        num_observables=EXPECTED_OBSERVABLES,
        physical_locations=len(locations),
        single_fault_classes=len(rows),
        physical_two_fault_class_pairs=s3._pair_denominator(locations),
        accepted_single_faults=accepted,
        malignant_single_faults=0,
        malignant_two_fault_pairs=malignant_pairs,
        signature_groups=len(groups),
        signature_group_pair_candidates=group_pair_candidates,
        signature_group_pairs_legal=legal_group_pairs,
        signature_group_pairs_skipped_same_location_only=skipped,
        unique_syndromes_order0=order0_synd,
        unique_syndromes_order1=order1_synd,
        unique_syndromes_order2=order2_synd,
        unique_syndromes_le2=len(states),
        ambiguous_syndromes_le1=len(ambiguous_le1),
        ambiguous_syndromes_le2=len(ambiguous_le2),
        ambiguity_type_counts=dict(sorted(type_counts.items())),
        all_single_syndromes_logically_unambiguous=(len(ambiguous_le1) == 0),
        all_le2_syndromes_logically_unambiguous=(len(ambiguous_le2) == 0),
        syndrome_only_decoder_corrects_all_le2=(len(ambiguous_le2) == 0),
        unavoidable_decoder_failure_order_within_le2=unavoidable_order,
        ambiguity_witnesses=witnesses,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"s55_{channel}_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        with (out_dir / f"s55_{channel}_ambiguities.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "detector_mask_hex",
                    "detector_weight",
                    "min_order_observable_0",
                    "min_order_observable_1",
                    "ambiguity_type",
                ]
            )
            for mask in sorted(ambiguous_le2):
                flags = states[mask]
                k0 = _min_order(flags, 0)
                k1 = _min_order(flags, 1)
                w.writerow([hex(mask), mask.bit_count(), k0, k1, f"{k0}v{k1}"])

    return summary


def _synthetic_self_test() -> None:
    """Test group-location exclusion and a known 1v2 ambiguity."""
    # Two order-1 O=0 faults at distinct locations make D=3,O=0.  One order-1
    # O=1 fault already has D=3,O=1, so D=3 must become a 1v2 ambiguity.
    rows = [
        s3.FaultRow(0, 0, 0, "RZ", (0,), "X", detector_mask=1, detector_weight=1, observable_flip=0),
        s3.FaultRow(1, 1, 0, "RZ", (1,), "X", detector_mask=2, detector_weight=1, observable_flip=0),
        s3.FaultRow(2, 2, 0, "RZ", (2,), "X", detector_mask=3, detector_weight=2, observable_flip=1),
    ]
    groups = _build_groups(rows)
    states: dict[int, int] = {0: B_O0_K0}
    for g in groups:
        _set_state_bit(states, g.detector_mask, g.observable, 1)
    for i, a in enumerate(groups):
        for j in range(i, len(groups)):
            b = groups[j]
            if _pair_groups_legal(a, b, i == j):
                _set_state_bit(states, a.detector_mask ^ b.detector_mask, a.observable ^ b.observable, 2)
    if _min_order(states[3], 0) != 2 or _min_order(states[3], 1) != 1:
        raise AssertionError(f"synthetic 1v2 ambiguity self-test failed: flags={states[3]}")

    # Same-location-only cross groups must not define a legal physical pair.
    a = SigGroup(0, 1, 0, 1, 1, {7: rows[0]})
    b = SigGroup(1, 2, 1, 1, 1, {7: rows[2]})
    if _pair_groups_legal(a, b, False):
        raise AssertionError("same-single-location group-pair exclusion failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--channel", choices=["X", "Z", "both"], default="both")
    ap.add_argument(
        "--noise-probability",
        type=float,
        default=1e-3,
        help="Only used to expose all S3 DEM mechanisms; signatures are probability-independent.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/zxz/s55_results"))
    ap.add_argument("--max-witnesses", type=int, default=20)
    args = ap.parse_args()

    _synthetic_self_test()
    s3._self_test_noise_append_fusion()
    s3._self_test_explainer_api()

    channels: list[Channel] = ["X", "Z"] if args.channel == "both" else [args.channel]
    summaries = [
        audit_channel(
            args.hirano_src,
            channel=ch,
            p=args.noise_probability,
            out_dir=args.out_dir,
            max_witnesses=args.max_witnesses,
        )
        for ch in channels
    ]

    print(json.dumps([asdict(s) for s in summaries], indent=2, sort_keys=True))
    if all(s.all_le2_syndromes_logically_unambiguous for s in summaries):
        print("S5.5 RESULT: UNAMBIGUOUS<=2 — a syndrome-only decoder can correct every audited <=2 fault set")
    else:
        print("S5.5 RESULT: GENUINE<=2 AMBIGUITY FOUND — syndrome-only perfect correction of all <=2 faults is impossible")
    print("S5.5 PASS: exact <=2-fault syndrome/observable ambiguity audit completed")


if __name__ == "__main__":
    main()
