#!/usr/bin/env python3
"""S4-B exact boundary-composability audit for the Hirano d=5 ZXZ escape.

This script closes the remaining historical Step-4 boundary checks after:
  S1: public Hirano ZXZ circuit fingerprint reproduced,
  S2: exact DETECTOR/OBSERVABLE GF(2) semantics audited,
  S3/S4-A: 7,279 single escape faults and 26,439,666 two-fault pairs audited.

For each complementary logical-transfer channel X/Z, this program checks:

1) Mixed incoming-Steane-single + escape-single faults
   21 incoming Steane Pauli classes (7 coordinates x X/Y/Z)
   x 7,279 escape fault classes
   = 152,859 exact combinations per channel.

2) Fault-free escape with two late Steane coordinate errors
   C(7,2) x 3 x 3 = 189 distinct-coordinate Pauli pairs per channel.

Boundary convention
-------------------
The incoming Steane Pauli is inserted immediately AFTER Hirano's perfect encoded
Steane initialization and BEFORE the first ZXZ lattice-surgery instruction.
This is the encoded-input boundary used by the historical Step-4 audit.

A stochastic X_ERROR/Y_ERROR/Z_ERROR mechanism is inserted at this boundary so
that the ideal/reference circuit is unchanged. Stim's detector error model then
returns the exact detector signature D and logical-transfer observable flip O
for that incoming Pauli.

Because the remaining ZXZ circuit is Clifford and all faults are Pauli,
signatures compose linearly over GF(2):
    D(a+b) = D(a) XOR D(b)
    O(a+b) = O(a) XOR O(b)

With full postselection of all 135 detectors:
    accepted iff D == 0
    harmful  iff D == 0 and O == 1

Historical regression targets from results/direct_cs_step4_audit.md:
    mixed cases/channel                      = 152,859
    malignant mixed X                       = 0
    malignant mixed Z                       = 0
    distinct-coordinate weight-2/channel    = 189
    accepted harmful weight-2 X             = 0
    accepted harmful weight-2 Z             = 0
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zxz_exact_fault_audit as s3  # noqa: E402
import hirano_zxz_generator as s1  # noqa: E402

Channel = Literal["X", "Z"]
PAULI1 = ("X", "Y", "Z")

EXPECTED_DETECTORS = 135
EXPECTED_OBSERVABLES = 1
EXPECTED_INCOMING_SINGLES = 21
EXPECTED_ESCAPE_CLASSES = 7_279
EXPECTED_MIXED_CASES = 152_859
EXPECTED_WEIGHT2_CASES = 189
EXPECTED_MALIGNANT_MIXED = {"X": 0, "Z": 0}
EXPECTED_HARMFUL_WEIGHT2 = {"X": 0, "Z": 0}


@dataclass(frozen=True)
class IncomingSignature:
    coordinate: int
    steane_position: tuple[int, int]
    qubit_id: int
    pauli: str
    detector_mask: int
    detector_weight: int
    observable_flip: int

    @property
    def accepted_fault_free_escape(self) -> bool:
        return self.detector_mask == 0

    @property
    def harmful_fault_free_escape(self) -> bool:
        return self.detector_mask == 0 and self.observable_flip == 1


@dataclass(frozen=True)
class Weight2Row:
    coord_a: int
    pauli_a: str
    qubit_a: int
    coord_b: int
    pauli_b: str
    qubit_b: int
    detector_mask: int
    detector_weight: int
    observable_flip: int
    accepted: bool
    harmful: bool


@dataclass(frozen=True)
class BoundarySummary:
    channel: Channel
    boundary_instruction_index: int
    steane_positions: list[list[int]]
    steane_qubit_ids: list[int]
    num_detectors: int
    num_observables: int
    postselection_detector_count: int
    postselection_ids_are_all_detectors: bool
    incoming_single_classes: int
    incoming_single_accepted_fault_free: int
    incoming_single_harmful_fault_free: int
    escape_single_classes: int
    upstream_s3_accepted_singles: int
    upstream_s3_malignant_singles: int
    upstream_s3_malignant_pairs: int
    mixed_cases: int
    mixed_accepted_cases: int
    malignant_mixed_cases: int
    malignant_mixed_target: int
    malignant_mixed_witnesses: list[dict[str, Any]]
    weight2_cases: int
    weight2_accepted_cases: int
    harmful_weight2_cases: int
    harmful_weight2_target: int
    harmful_weight2_witnesses: list[dict[str, Any]]


def _flag(obj: Any, name: str) -> bool:
    v = getattr(obj, name)
    return bool(v() if callable(v) else v)


def _dem_instruction_signature(inst: Any) -> tuple[int, int]:
    if inst.type != "error":
        raise AssertionError(f"expected DEM error instruction, got {inst.type!r}")
    detector_mask = 0
    observable_flip = 0
    for t in inst.targets_copy():
        if _flag(t, "is_relative_detector_id"):
            detector_mask ^= 1 << int(t.val)
        elif _flag(t, "is_logical_observable_id"):
            oid = int(t.val)
            if oid != 0:
                raise AssertionError(f"unexpected logical observable L{oid}")
            observable_flip ^= 1
        elif _flag(t, "is_separator"):
            raise AssertionError(
                "separator in undecomposed single-incoming-error DEM; refusing ambiguous signature"
            )
        else:
            raise AssertionError(f"unexpected DEM target {t!r}")
    return detector_mask, observable_flip


def _single_stochastic_error_signature(circuit: Any) -> tuple[int, int]:
    """Return (D,O) for a circuit containing exactly one stochastic mechanism.

    Zero-symptom errors are omitted from a DEM and therefore map to (0,0).
    """
    dem = circuit.detector_error_model(decompose_errors=False).flattened()
    errors = [inst for inst in dem if getattr(inst, "type", None) == "error"]
    if len(errors) == 0:
        return 0, 0
    if len(errors) != 1:
        raise AssertionError(
            f"expected <=1 DEM error from one injected boundary mechanism, got {len(errors)}: {dem}"
        )
    return _dem_instruction_signature(errors[0])


def _self_test_single_error_signature() -> None:
    import stim

    c = stim.Circuit()
    c.append("R", [0])
    c.append("X_ERROR", [0], 0.01)
    c.append("M", [0])
    c.append("DETECTOR", [stim.target_rec(-1)])
    d, o = _single_stochastic_error_signature(c)
    if d != 1 or o != 0:
        raise AssertionError(f"single-error DEM parser self-test failed: D={d:b}, O={o}")

    c2 = stim.Circuit()
    c2.append("R", [0])
    c2.append("X_ERROR", [0], 0.01)
    c2.append("M", [0])
    c2.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    d2, o2 = _single_stochastic_error_signature(c2)
    if d2 != 0 or o2 != 1:
        raise AssertionError(f"observable DEM parser self-test failed: D={d2:b}, O={o2}")


def _insert_boundary_error(
    circuit: Any,
    *,
    insert_index: int,
    qubit: int,
    pauli: str,
    p: float,
) -> Any:
    import stim

    if pauli not in PAULI1:
        raise ValueError(pauli)
    if not (0 < p < 1):
        raise ValueError("p must satisfy 0<p<1")

    out = stim.Circuit()
    out += circuit[:insert_index]
    out.append(f"{pauli}_ERROR", [qubit], p)
    out += circuit[insert_index:]

    # The injected stochastic mechanism must leave the ideal circuit unchanged.
    if out.without_noise() != circuit:
        raise AssertionError("boundary-injected circuit does not reduce to the original ideal circuit")
    return out


def _steane_boundary_qubits(hirano_src: Path) -> tuple[list[tuple[int, int]], list[int]]:
    # Ensure Hirano src is importable and obtain the exact public constants.
    _, util = s1._load_hirano(hirano_src)
    steane_code = importlib.import_module("steane_code")
    mapping = util.QubitMapping(30, 30)

    positions = [
        tuple(getattr(steane_code, f"STEANE_{i}"))
        for i in range(7)
    ]
    ids = [int(mapping.get_id(*pos)) for pos in positions]
    if len(set(positions)) != 7 or len(set(ids)) != 7:
        raise AssertionError(f"Steane coordinate/id mapping is not one-to-one: {positions}, {ids}")
    return positions, ids


def build_incoming_signatures(
    circuit: Any,
    *,
    hirano_src: Path,
    boundary_index: int,
    p: float,
) -> tuple[list[IncomingSignature], list[tuple[int, int]], list[int]]:
    positions, ids = _steane_boundary_qubits(hirano_src)
    rows: list[IncomingSignature] = []
    for coord, (pos, q) in enumerate(zip(positions, ids)):
        for pauli in PAULI1:
            c = _insert_boundary_error(
                circuit,
                insert_index=boundary_index,
                qubit=q,
                pauli=pauli,
                p=p,
            )
            d, o = _single_stochastic_error_signature(c)
            rows.append(
                IncomingSignature(
                    coordinate=coord,
                    steane_position=pos,
                    qubit_id=q,
                    pauli=pauli,
                    detector_mask=d,
                    detector_weight=d.bit_count(),
                    observable_flip=o,
                )
            )
    if len(rows) != EXPECTED_INCOMING_SINGLES:
        raise AssertionError(f"incoming singles={len(rows)} != {EXPECTED_INCOMING_SINGLES}")
    return rows, positions, ids


def _rebuild_escape_fault_catalog(
    circuit: Any,
    *,
    init_range: tuple[int, int],
    p: float,
) -> tuple[list[Any], dict[str, Any]]:
    noisy, locations, noise_map = s3.instrument_canonical_fault_model(
        circuit, init_range=init_range, p=p
    )
    rows = s3.build_fault_rows(locations)
    if len(rows) != EXPECTED_ESCAPE_CLASSES:
        raise AssertionError(f"escape classes={len(rows)} != {EXPECTED_ESCAPE_CLASSES}")
    dem, n_groups, zero_filled = s3.populate_signatures_from_dem(
        noisy, locations, rows, noise_map
    )
    accepted = sum(r.accepted for r in rows)
    malignant_single = sum(r.malignant_single for r in rows)
    malignant_pairs, witnesses = s3.exact_malignant_pair_hash(rows)
    return rows, {
        "dem_num_detectors": int(dem.num_detectors),
        "dem_num_observables": int(dem.num_observables),
        "dem_signature_groups": int(n_groups),
        "zero_symptom_faults_filled": int(zero_filled),
        "accepted_singles": int(accepted),
        "malignant_singles": int(malignant_single),
        "malignant_pairs": int(malignant_pairs),
        "malignant_pair_witnesses": witnesses,
    }


def exact_mixed_audit(
    incoming: list[IncomingSignature],
    escape_rows: list[Any],
    *,
    max_witnesses: int = 20,
) -> tuple[int, int, list[dict[str, Any]]]:
    accepted = 0
    malignant = 0
    witnesses: list[dict[str, Any]] = []
    for inc in incoming:
        for f in escape_rows:
            d = inc.detector_mask ^ int(f.detector_mask)
            o = inc.observable_flip ^ int(f.observable_flip)
            if d == 0:
                accepted += 1
                if o:
                    malignant += 1
                    if len(witnesses) < max_witnesses:
                        witnesses.append(
                            {
                                "incoming": {
                                    "coordinate": inc.coordinate,
                                    "position": list(inc.steane_position),
                                    "qubit_id": inc.qubit_id,
                                    "pauli": inc.pauli,
                                    "detector_mask_hex": hex(inc.detector_mask),
                                    "observable_flip": inc.observable_flip,
                                },
                                "escape_fault": {
                                    "fault_id": int(f.fault_id),
                                    "location_id": int(f.location_id),
                                    "tick": int(f.tick),
                                    "gate": f.gate,
                                    "qubits": list(f.qubits),
                                    "fault_class": f.fault_class,
                                    "detector_mask_hex": hex(int(f.detector_mask)),
                                    "observable_flip": int(f.observable_flip),
                                },
                            }
                        )
    return accepted, malignant, witnesses


def exact_weight2_audit(
    incoming: list[IncomingSignature],
) -> list[Weight2Row]:
    by_key = {(r.coordinate, r.pauli): r for r in incoming}
    out: list[Weight2Row] = []
    for a in range(7):
        for b in range(a + 1, 7):
            for pa in PAULI1:
                for pb in PAULI1:
                    ra = by_key[(a, pa)]
                    rb = by_key[(b, pb)]
                    d = ra.detector_mask ^ rb.detector_mask
                    o = ra.observable_flip ^ rb.observable_flip
                    accepted = d == 0
                    harmful = accepted and o == 1
                    out.append(
                        Weight2Row(
                            coord_a=a,
                            pauli_a=pa,
                            qubit_a=ra.qubit_id,
                            coord_b=b,
                            pauli_b=pb,
                            qubit_b=rb.qubit_id,
                            detector_mask=d,
                            detector_weight=d.bit_count(),
                            observable_flip=o,
                            accepted=accepted,
                            harmful=harmful,
                        )
                    )
    if len(out) != EXPECTED_WEIGHT2_CASES:
        raise AssertionError(f"weight-2 cases={len(out)} != {EXPECTED_WEIGHT2_CASES}")
    return out


def _write_incoming_csv(path: Path, rows: list[IncomingSignature]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "coordinate", "steane_position", "qubit_id", "pauli",
            "detector_mask_hex", "detector_weight", "detector_ids",
            "observable_flip", "accepted_fault_free_escape", "harmful_fault_free_escape",
        ])
        for r in rows:
            det_ids = [i for i in range(EXPECTED_DETECTORS) if (r.detector_mask >> i) & 1]
            w.writerow([
                r.coordinate,
                f"{r.steane_position[0]},{r.steane_position[1]}",
                r.qubit_id,
                r.pauli,
                hex(r.detector_mask),
                r.detector_weight,
                " ".join(map(str, det_ids)),
                r.observable_flip,
                int(r.accepted_fault_free_escape),
                int(r.harmful_fault_free_escape),
            ])


def _write_weight2_csv(path: Path, rows: list[Weight2Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "coord_a", "pauli_a", "qubit_a", "coord_b", "pauli_b", "qubit_b",
            "detector_mask_hex", "detector_weight", "detector_ids",
            "observable_flip", "accepted", "harmful",
        ])
        for r in rows:
            det_ids = [i for i in range(EXPECTED_DETECTORS) if (r.detector_mask >> i) & 1]
            w.writerow([
                r.coord_a, r.pauli_a, r.qubit_a,
                r.coord_b, r.pauli_b, r.qubit_b,
                hex(r.detector_mask), r.detector_weight,
                " ".join(map(str, det_ids)),
                r.observable_flip, int(r.accepted), int(r.harmful),
            ])


def audit_channel(
    hirano_src: Path,
    *,
    channel: Channel,
    p: float,
    out_dir: Path | None,
    do_assert: bool,
) -> BoundarySummary:
    circuit, wrapper, init_range = s3.build_public_zxz_with_init_range(
        hirano_src, channel=channel
    )
    boundary_index = init_range[1]

    post_ids = [int(d.id) for d in wrapper.detectors_for_post_selection]
    post_all = (
        len(post_ids) == circuit.num_detectors
        and set(post_ids) == set(range(circuit.num_detectors))
    )

    incoming, positions, ids = build_incoming_signatures(
        circuit,
        hirano_src=hirano_src,
        boundary_index=boundary_index,
        p=p,
    )
    escape_rows, s3_diag = _rebuild_escape_fault_catalog(
        circuit, init_range=init_range, p=p
    )

    mixed_total = len(incoming) * len(escape_rows)
    mixed_accepted, mixed_malignant, mixed_witnesses = exact_mixed_audit(
        incoming, escape_rows
    )

    weight2 = exact_weight2_audit(incoming)
    w2_accepted = sum(r.accepted for r in weight2)
    w2_harmful = sum(r.harmful for r in weight2)
    w2_witnesses = [
        {
            "coord_a": r.coord_a,
            "pauli_a": r.pauli_a,
            "qubit_a": r.qubit_a,
            "coord_b": r.coord_b,
            "pauli_b": r.pauli_b,
            "qubit_b": r.qubit_b,
            "detector_mask_hex": hex(r.detector_mask),
            "observable_flip": r.observable_flip,
        }
        for r in weight2 if r.harmful
    ][:20]

    summary = BoundarySummary(
        channel=channel,
        boundary_instruction_index=boundary_index,
        steane_positions=[list(p) for p in positions],
        steane_qubit_ids=ids,
        num_detectors=int(circuit.num_detectors),
        num_observables=int(circuit.num_observables),
        postselection_detector_count=len(post_ids),
        postselection_ids_are_all_detectors=post_all,
        incoming_single_classes=len(incoming),
        incoming_single_accepted_fault_free=sum(r.accepted_fault_free_escape for r in incoming),
        incoming_single_harmful_fault_free=sum(r.harmful_fault_free_escape for r in incoming),
        escape_single_classes=len(escape_rows),
        upstream_s3_accepted_singles=s3_diag["accepted_singles"],
        upstream_s3_malignant_singles=s3_diag["malignant_singles"],
        upstream_s3_malignant_pairs=s3_diag["malignant_pairs"],
        mixed_cases=mixed_total,
        mixed_accepted_cases=mixed_accepted,
        malignant_mixed_cases=mixed_malignant,
        malignant_mixed_target=EXPECTED_MALIGNANT_MIXED[channel],
        malignant_mixed_witnesses=mixed_witnesses,
        weight2_cases=len(weight2),
        weight2_accepted_cases=w2_accepted,
        harmful_weight2_cases=w2_harmful,
        harmful_weight2_target=EXPECTED_HARMFUL_WEIGHT2[channel],
        harmful_weight2_witnesses=w2_witnesses,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_incoming_csv(out_dir / f"s4b_{channel}_incoming_single.csv", incoming)
        _write_weight2_csv(out_dir / f"s4b_{channel}_weight2.csv", weight2)
        (out_dir / f"s4b_{channel}_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        (out_dir / f"s4b_{channel}_upstream_s3_diag.json").write_text(
            json.dumps(s3_diag, indent=2, sort_keys=True), encoding="utf-8"
        )

    if do_assert:
        if circuit.num_detectors != EXPECTED_DETECTORS:
            raise AssertionError(f"{channel}: detectors={circuit.num_detectors} != {EXPECTED_DETECTORS}")
        if circuit.num_observables != EXPECTED_OBSERVABLES:
            raise AssertionError(f"{channel}: observables={circuit.num_observables} != {EXPECTED_OBSERVABLES}")
        if not post_all:
            raise AssertionError(f"{channel}: not all detectors are postselected")
        if len(incoming) != EXPECTED_INCOMING_SINGLES:
            raise AssertionError(f"{channel}: incoming={len(incoming)} != {EXPECTED_INCOMING_SINGLES}")
        if len(escape_rows) != EXPECTED_ESCAPE_CLASSES:
            raise AssertionError(f"{channel}: escape={len(escape_rows)} != {EXPECTED_ESCAPE_CLASSES}")
        if mixed_total != EXPECTED_MIXED_CASES:
            raise AssertionError(f"{channel}: mixed={mixed_total} != {EXPECTED_MIXED_CASES}")
        if len(weight2) != EXPECTED_WEIGHT2_CASES:
            raise AssertionError(f"{channel}: weight2={len(weight2)} != {EXPECTED_WEIGHT2_CASES}")

        # Reconfirm S3 closure before relying on its signatures.
        if s3_diag["accepted_singles"] != s3.CANONICAL_ACCEPTED_SINGLES[channel]:
            raise AssertionError(
                f"{channel}: upstream accepted singles changed: {s3_diag['accepted_singles']}"
            )
        if s3_diag["malignant_singles"] != 0 or s3_diag["malignant_pairs"] != 0:
            raise AssertionError(f"{channel}: upstream S3 is no longer closed: {s3_diag}")

        if mixed_malignant != EXPECTED_MALIGNANT_MIXED[channel]:
            raise AssertionError(
                f"{channel}: malignant mixed={mixed_malignant}, target=0; "
                f"witnesses={mixed_witnesses[:5]}"
            )
        if w2_harmful != EXPECTED_HARMFUL_WEIGHT2[channel]:
            raise AssertionError(
                f"{channel}: harmful weight2={w2_harmful}, target=0; "
                f"witnesses={w2_witnesses[:5]}"
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
        help="Arbitrary nonzero p used only to expose exact DEM signatures; results are p-independent.",
    )
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    _self_test_single_error_signature()
    # Reuse S3's already-tested fusion/explainer sanity checks.
    s3._self_test_noise_append_fusion()
    s3._self_test_explainer_api()

    channels: list[Channel] = ["X", "Z"] if args.channel == "both" else [args.channel]
    results = []
    for ch in channels:
        r = audit_channel(
            args.hirano_src,
            channel=ch,
            p=args.noise_probability,
            out_dir=args.out_dir,
            do_assert=not args.no_assert,
        )
        results.append(asdict(r))

    print(json.dumps(results, indent=2, sort_keys=True))
    if not args.no_assert:
        print(
            "S4-B PASS: 152,859 mixed cases/channel + 189 weight-2 boundary cases/channel"
        )


if __name__ == "__main__":
    main()
