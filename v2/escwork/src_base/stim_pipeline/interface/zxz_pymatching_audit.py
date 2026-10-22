#!/usr/bin/env python3
"""S5 PyMatching decoder audit for the reproduced Hirano d=5 full-ZXZ escape.

Prerequisites already closed by S1--S4B
--------------------------------------
* S1: public Hirano ZXZ generator reproduced canonical fingerprint.
* S2: all 135 detector annotations and observable 0 audited exactly over GF(2).
* S3/S4-A: 7,279 canonical single-fault classes and 26,439,666 distinct-location
  two-fault pairs reproduced, with zero malignant single/pair faults under full
  postselection.
* S4-B: 21x7,279 mixed incoming-single/escape cases and 189 incoming weight-2
  cases reproduced with zero harmful accepted cases.

This stage does NOT replace the certification protocol. It adds a decoder-mode
performance baseline while keeping certification statistics separate.

For each X/Z logical-transfer channel this program:
1. Rebuilds the exact public d=5 ZXZ circuit and applies the canonical active-
   location stochastic-Pauli model at physical probability p, excluding the
   perfect Steane input initialization from the fault budget exactly as S3 did.
2. Creates both an undecomposed and a graphlike/decomposed Stim DEM.
3. Audits every decomposed DEM component to ensure it is graphlike (<=2
   detectors per component), then constructs a PyMatching MWPM graph.
4. Reuses S3's exact 7,279 single-fault (D,O) catalogue and decodes ALL single
   faults. This is diagnostic: it reports decoder failures and syndrome-label
   ambiguities but does not assume a priori that MWPM must correct every class.
5. Monte-Carlo samples the same noisy Stim circuit and reports, separately:
     certification mode: accept iff all 135 detectors are zero;
     MWPM mode: decode all detector patterns using PyMatching.
6. Writes DEMs, exact-single diagnostics, and a JSON summary.

Important semantics
-------------------
Certification mode:
    accepted = (D == 0)
    logical failure = accepted and O_actual == 1

MWPM mode:
    O_pred = matching.decode_batch(D)
    logical failure = (O_pred XOR O_actual) == 1

These two modes answer different questions and MUST NOT be conflated.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
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

Channel = Literal["X", "Z"]
EXPECTED_DETECTORS = 135
EXPECTED_OBSERVABLES = 1
EXPECTED_SINGLE_CLASSES = 7_279
EXPECTED_LOCATIONS = 797
EXPECTED_PAIR_CLASSES = 26_439_666


@dataclass(frozen=True)
class SingleDecoderRow:
    fault_id: int
    location_id: int
    gate: str
    fault_class: str
    detector_weight: int
    detector_mask_hex: str
    actual_observable: int
    predicted_observable: int
    decoder_success: bool
    same_syndrome_has_both_observables: bool


@dataclass(frozen=True)
class S5Summary:
    channel: Channel
    p: float
    shots: int
    seed: int
    stim_version: str
    pymatching_version: str
    numpy_version: str
    num_detectors: int
    num_observables: int
    postselection_detector_count: int
    postselection_ids_are_all_detectors: bool
    physical_locations: int
    single_fault_classes: int
    two_fault_class_pairs: int
    raw_dem_error_instructions: int
    raw_dem_hyperedge_instructions: int
    raw_dem_max_detector_weight: int
    graphlike_dem_error_instructions: int
    graphlike_dem_components: int
    graphlike_dem_max_component_detector_weight: int
    graphlike_dem_bad_components: int
    matching_num_detectors: int
    matching_num_nodes: int
    matching_num_edges: int
    matching_num_fault_ids: int
    exact_single_decoder_failures: int
    exact_single_decoder_successes: int
    exact_single_zero_syndrome_classes: int
    exact_single_ambiguous_syndrome_groups: int
    exact_single_ambiguous_fault_classes: int
    exact_single_failure_witnesses: list[dict[str, Any]]
    sampled_nonzero_syndrome_shots: int
    sampled_zero_syndrome_shots: int
    certification_accepted_shots: int
    certification_rejected_shots: int
    certification_logical_errors: int
    certification_acceptance: float
    certification_conditional_logical_error: float | None
    mwpm_logical_errors: int
    mwpm_logical_error_rate: float
    mwpm_nonzero_syndrome_logical_errors: int
    mwpm_nonzero_syndrome_logical_error_rate: float | None
    mwpm_zero_syndrome_logical_errors: int


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _flag(obj: Any, name: str) -> bool:
    v = getattr(obj, name)
    return bool(v() if callable(v) else v)


def _dem_component_detector_weight(component: list[Any]) -> int:
    return sum(1 for t in component if _flag(t, "is_relative_detector_id"))


def audit_dem_structure(dem: Any) -> dict[str, int]:
    """Audit DEM error instructions and '^'-separated graphlike components."""
    raw_error_instructions = 0
    hyperedge_instructions = 0
    max_detector_weight = 0
    component_count = 0
    max_component_weight = 0
    bad_components = 0

    flat = dem.flattened()
    for inst in flat:
        if getattr(inst, "type", None) != "error":
            continue
        raw_error_instructions += 1
        targets = inst.targets_copy()
        total_dw = sum(1 for t in targets if _flag(t, "is_relative_detector_id"))
        max_detector_weight = max(max_detector_weight, total_dw)
        if total_dw > 2:
            hyperedge_instructions += 1

        comp: list[Any] = []
        for t in targets:
            if _flag(t, "is_separator"):
                w = _dem_component_detector_weight(comp)
                component_count += 1
                max_component_weight = max(max_component_weight, w)
                if w > 2:
                    bad_components += 1
                comp = []
            else:
                comp.append(t)
        w = _dem_component_detector_weight(comp)
        component_count += 1
        max_component_weight = max(max_component_weight, w)
        if w > 2:
            bad_components += 1

    return {
        "error_instructions": raw_error_instructions,
        "hyperedge_instructions": hyperedge_instructions,
        "max_detector_weight": max_detector_weight,
        "components": component_count,
        "max_component_detector_weight": max_component_weight,
        "bad_components": bad_components,
    }


def _mask_to_row(mask: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.uint8)
    while mask:
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        if i >= n:
            raise AssertionError(f"detector mask contains D{i} but n={n}")
        out[i] = 1
        mask ^= lsb
    return out


def _build_single_fault_matrix(rows: list[s3.FaultRow], n_detectors: int) -> np.ndarray:
    m = np.zeros((len(rows), n_detectors), dtype=np.uint8)
    for k, r in enumerate(rows):
        m[k, :] = _mask_to_row(r.detector_mask, n_detectors)
    return m


def _single_syndrome_ambiguity(rows: list[s3.FaultRow]) -> tuple[int, int, set[int]]:
    by_mask: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)
    for r in rows:
        by_mask[r.detector_mask].add(int(r.observable_flip))
        counts[r.detector_mask] += 1
    ambiguous_masks = {m for m, obs in by_mask.items() if len(obs) > 1}
    ambiguous_faults = sum(counts[m] for m in ambiguous_masks)
    return len(ambiguous_masks), ambiguous_faults, ambiguous_masks


def audit_exact_single_decoder(
    matching: Any,
    rows: list[s3.FaultRow],
    *,
    n_detectors: int,
    max_witnesses: int = 30,
) -> tuple[list[SingleDecoderRow], dict[str, Any]]:
    if len(rows) != EXPECTED_SINGLE_CLASSES:
        raise AssertionError(f"single row count {len(rows)} != {EXPECTED_SINGLE_CLASSES}")
    synd = _build_single_fault_matrix(rows, n_detectors)
    predictions = matching.decode_batch(synd)
    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if predictions.shape[0] != len(rows):
        raise AssertionError(f"prediction rows {predictions.shape} vs faults {len(rows)}")
    if predictions.shape[1] < 1:
        raise AssertionError("PyMatching returned no fault-id predictions")

    ambiguous_groups, ambiguous_faults, ambiguous_masks = _single_syndrome_ambiguity(rows)
    out: list[SingleDecoderRow] = []
    witnesses: list[dict[str, Any]] = []
    failures = 0
    zero_syndrome = 0
    for i, r in enumerate(rows):
        actual = int(r.observable_flip)
        pred = int(predictions[i, 0])
        success = actual == pred
        if r.detector_mask == 0:
            zero_syndrome += 1
        if not success:
            failures += 1
            if len(witnesses) < max_witnesses:
                witnesses.append(
                    {
                        "fault_id": r.fault_id,
                        "location_id": r.location_id,
                        "tick": r.tick,
                        "gate": r.gate,
                        "qubits": list(r.qubits),
                        "fault_class": r.fault_class,
                        "detector_mask_hex": hex(r.detector_mask),
                        "detector_weight": r.detector_weight,
                        "actual_observable": actual,
                        "predicted_observable": pred,
                        "ambiguous_single_syndrome": r.detector_mask in ambiguous_masks,
                    }
                )
        out.append(
            SingleDecoderRow(
                fault_id=r.fault_id,
                location_id=r.location_id,
                gate=r.gate,
                fault_class=r.fault_class,
                detector_weight=r.detector_weight,
                detector_mask_hex=hex(r.detector_mask),
                actual_observable=actual,
                predicted_observable=pred,
                decoder_success=success,
                same_syndrome_has_both_observables=r.detector_mask in ambiguous_masks,
            )
        )

    return out, {
        "failures": failures,
        "successes": len(rows) - failures,
        "zero_syndrome_classes": zero_syndrome,
        "ambiguous_syndrome_groups": ambiguous_groups,
        "ambiguous_fault_classes": ambiguous_faults,
        "failure_witnesses": witnesses,
    }


def _self_test_pymatching() -> None:
    import pymatching
    import stim

    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    m = pymatching.Matching.from_detector_error_model(dem)
    m.ensure_num_fault_ids(1)
    pred = np.asarray(m.decode_batch(np.array([[0], [1]], dtype=np.uint8)), dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.shape != (2, 1) or pred[:, 0].tolist() != [0, 1]:
        raise AssertionError(f"PyMatching fault-id self-test failed: shape={pred.shape}, pred={pred}")


def _rate(k: int, n: int) -> float | None:
    return None if n == 0 else float(k / n)


def _write_single_csv(path: Path, rows: list[SingleDecoderRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "fault_id", "location_id", "gate", "fault_class", "detector_weight",
            "detector_mask_hex", "actual_observable", "predicted_observable",
            "decoder_success", "same_syndrome_has_both_observables",
        ])
        for r in rows:
            w.writerow([
                r.fault_id, r.location_id, r.gate, r.fault_class, r.detector_weight,
                r.detector_mask_hex, r.actual_observable, r.predicted_observable,
                int(r.decoder_success), int(r.same_syndrome_has_both_observables),
            ])


def audit_channel(
    hirano_src: Path,
    *,
    channel: Channel,
    p: float,
    shots: int,
    seed: int,
    out_dir: Path | None,
    do_assert: bool,
) -> S5Summary:
    import pymatching
    import stim

    # Rebuild exact circuit and canonical noisy model using the already-closed S3 code.
    circuit, wrapper, init_range = s3.build_public_zxz_with_init_range(hirano_src, channel=channel)
    post_ids = [int(d.id) for d in wrapper.detectors_for_post_selection]
    post_all = len(post_ids) == circuit.num_detectors and set(post_ids) == set(range(circuit.num_detectors))
    if do_assert and not post_all:
        raise AssertionError(f"{channel}: postselection detector set is not all detectors")

    noisy, locations, noise_map = s3.instrument_canonical_fault_model(circuit, init_range=init_range, p=p)
    rows = s3.build_fault_rows(locations)
    if do_assert:
        if len(locations) != EXPECTED_LOCATIONS:
            raise AssertionError(f"{channel}: locations={len(locations)} != {EXPECTED_LOCATIONS}")
        if len(rows) != EXPECTED_SINGLE_CLASSES:
            raise AssertionError(f"{channel}: faults={len(rows)} != {EXPECTED_SINGLE_CLASSES}")
        if s3._pair_denominator(locations) != EXPECTED_PAIR_CLASSES:
            raise AssertionError(f"{channel}: pair denominator mismatch")

    # Populate exact physical single-fault (D,O) labels from the undecomposed DEM.
    _, _, _ = s3.populate_signatures_from_dem(noisy, locations, rows, noise_map)

    raw_dem = noisy.detector_error_model(decompose_errors=False)
    raw_stats = audit_dem_structure(raw_dem)

    # PyMatching requires graphlike error mechanisms. Refuse silent dropping: Stim must
    # successfully decompose and every '^'-separated component must have <=2 detectors.
    graph_dem = noisy.detector_error_model(decompose_errors=True)
    graph_stats = audit_dem_structure(graph_dem)
    if do_assert and graph_stats["bad_components"] != 0:
        raise AssertionError(
            f"{channel}: decomposed DEM still has {graph_stats['bad_components']} non-graphlike components"
        )

    matching = pymatching.Matching.from_detector_error_model(graph_dem)
    matching.ensure_num_fault_ids(EXPECTED_OBSERVABLES)
    if do_assert:
        if matching.num_detectors != EXPECTED_DETECTORS:
            raise AssertionError(f"{channel}: matching.num_detectors={matching.num_detectors}")
        if matching.num_fault_ids != EXPECTED_OBSERVABLES:
            raise AssertionError(f"{channel}: matching.num_fault_ids={matching.num_fault_ids}")

    single_decoder_rows, single_stats = audit_exact_single_decoder(
        matching, rows, n_detectors=EXPECTED_DETECTORS
    )

    # Monte Carlo: sample the physical circuit, not the DEM, so the actual circuit-level
    # correlations and measurement semantics are preserved in the source samples.
    sampler = noisy.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    if obs.ndim == 1:
        obs = obs[:, None]
    if dets.shape != (shots, EXPECTED_DETECTORS):
        raise AssertionError(f"{channel}: detector sample shape {dets.shape}")
    if obs.shape != (shots, EXPECTED_OBSERVABLES):
        raise AssertionError(f"{channel}: observable sample shape {obs.shape}")

    zero = ~np.any(dets.astype(bool), axis=1)
    n_zero = int(np.sum(zero))
    n_nonzero = shots - n_zero
    cert_errors = int(np.sum(obs[zero, 0])) if n_zero else 0

    pred = np.asarray(matching.decode_batch(dets), dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.shape != obs.shape:
        raise AssertionError(f"{channel}: prediction shape {pred.shape} vs obs {obs.shape}")
    failures = np.any(pred != obs, axis=1)
    mwpm_errors = int(np.sum(failures))
    mwpm_zero_errors = int(np.sum(failures[zero])) if n_zero else 0
    mwpm_nonzero_errors = int(np.sum(failures[~zero])) if n_nonzero else 0

    summary = S5Summary(
        channel=channel,
        p=p,
        shots=shots,
        seed=seed,
        stim_version=_version("stim"),
        pymatching_version=_version("pymatching"),
        numpy_version=np.__version__,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
        postselection_detector_count=len(post_ids),
        postselection_ids_are_all_detectors=post_all,
        physical_locations=len(locations),
        single_fault_classes=len(rows),
        two_fault_class_pairs=s3._pair_denominator(locations),
        raw_dem_error_instructions=raw_stats["error_instructions"],
        raw_dem_hyperedge_instructions=raw_stats["hyperedge_instructions"],
        raw_dem_max_detector_weight=raw_stats["max_detector_weight"],
        graphlike_dem_error_instructions=graph_stats["error_instructions"],
        graphlike_dem_components=graph_stats["components"],
        graphlike_dem_max_component_detector_weight=graph_stats["max_component_detector_weight"],
        graphlike_dem_bad_components=graph_stats["bad_components"],
        matching_num_detectors=int(matching.num_detectors),
        matching_num_nodes=int(matching.num_nodes),
        matching_num_edges=int(matching.num_edges),
        matching_num_fault_ids=int(matching.num_fault_ids),
        exact_single_decoder_failures=int(single_stats["failures"]),
        exact_single_decoder_successes=int(single_stats["successes"]),
        exact_single_zero_syndrome_classes=int(single_stats["zero_syndrome_classes"]),
        exact_single_ambiguous_syndrome_groups=int(single_stats["ambiguous_syndrome_groups"]),
        exact_single_ambiguous_fault_classes=int(single_stats["ambiguous_fault_classes"]),
        exact_single_failure_witnesses=single_stats["failure_witnesses"],
        sampled_nonzero_syndrome_shots=n_nonzero,
        sampled_zero_syndrome_shots=n_zero,
        certification_accepted_shots=n_zero,
        certification_rejected_shots=n_nonzero,
        certification_logical_errors=cert_errors,
        certification_acceptance=float(n_zero / shots) if shots else math.nan,
        certification_conditional_logical_error=_rate(cert_errors, n_zero),
        mwpm_logical_errors=mwpm_errors,
        mwpm_logical_error_rate=float(mwpm_errors / shots) if shots else math.nan,
        mwpm_nonzero_syndrome_logical_errors=mwpm_nonzero_errors,
        mwpm_nonzero_syndrome_logical_error_rate=_rate(mwpm_nonzero_errors, n_nonzero),
        mwpm_zero_syndrome_logical_errors=mwpm_zero_errors,
    )

    if do_assert:
        if summary.num_detectors != EXPECTED_DETECTORS or summary.num_observables != EXPECTED_OBSERVABLES:
            raise AssertionError(f"{channel}: detector/observable count drift")
        if summary.graphlike_dem_bad_components != 0:
            raise AssertionError(f"{channel}: non-graphlike decomposed components remain")
        # Zero-syndrome matching must predict no logical correction. Therefore MWPM
        # errors on zero syndrome must exactly equal certification logical errors.
        if summary.mwpm_zero_syndrome_logical_errors != summary.certification_logical_errors:
            raise AssertionError(
                f"{channel}: zero-syndrome MWPM/certification mismatch: "
                f"{summary.mwpm_zero_syndrome_logical_errors} vs {summary.certification_logical_errors}"
            )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"s5_{channel}_raw.dem").write_text(str(raw_dem), encoding="utf-8")
        (out_dir / f"s5_{channel}_graphlike.dem").write_text(str(graph_dem), encoding="utf-8")
        _write_single_csv(out_dir / f"s5_{channel}_exact_single_decoder.csv", single_decoder_rows)
        (out_dir / f"s5_{channel}_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
        )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--p", type=float, default=1e-3, help="uniform active-location error probability")
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--channel", choices=["X", "Z", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/zxz/s5_results"))
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    if not (0 < args.p < 0.5):
        raise SystemExit("--p must satisfy 0 < p < 0.5")
    if args.shots <= 0:
        raise SystemExit("--shots must be positive")

    _self_test_pymatching()
    s3._self_test_noise_append_fusion()

    channels: list[Channel] = ["X", "Z"] if args.channel == "both" else [args.channel]  # type: ignore[list-item]
    results = [
        audit_channel(
            args.hirano_src,
            channel=ch,
            p=args.p,
            shots=args.shots,
            seed=args.seed + (0 if ch == "X" else 1),
            out_dir=args.out_dir,
            do_assert=not args.no_assert,
        )
        for ch in channels
    ]
    print(json.dumps([asdict(x) for x in results], indent=2, sort_keys=True))
    print("S5 PASS: graphlike DEM + PyMatching construction + exact-single decoder audit + direct MC")


if __name__ == "__main__":
    main()
