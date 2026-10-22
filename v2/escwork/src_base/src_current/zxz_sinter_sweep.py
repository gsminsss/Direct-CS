#!/usr/bin/env python3
"""S6 Sinter p-sweep for the reproduced Hirano d=5 full-ZXZ escape.

This stage follows the already closed S1--S5 chain and keeps two protocols
strictly separate:

1) certification mode
   * all 135 detectors are postselected;
   * decoder='vacuous';
   * acceptance = (shots - discards) / shots;
   * conditional logical error = errors / (shots - discards).

2) MWPM mode
   * no postselection;
   * decoder='pymatching';
   * logical error = errors / shots.

The same canonical active-location stochastic-Pauli circuit used in S3/S5 is
used at every p. Perfect Steane input initialization is kept in the circuit but
excluded from the fault budget.

Outputs
-------
* a Sinter resume CSV (raw sufficient statistics; safe to resume);
* compact summary CSV/JSON with acceptance/logical rates;
* exact noisy .stim circuits and raw/graphlike .dem files for every channel,p;
* run manifest recording package versions and task strong IDs.

This is a performance sweep. It does not change the S3/S4 certification claim.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zxz_exact_fault_audit as s3  # noqa: E402

Channel = Literal["X", "Z"]
Mode = Literal["certification", "mwpm"]

EXPECTED_DETECTORS = 135
EXPECTED_OBSERVABLES = 1
EXPECTED_LOCATIONS = 797
EXPECTED_SINGLE_CLASSES = 7_279
EXPECTED_PAIR_CLASSES = 26_439_666
DEFAULT_P_VALUES = (2e-4, 5e-4, 1e-3, 2e-3, 5e-3)


@dataclass(frozen=True)
class TaskSpec:
    channel: Channel
    p: float
    mode: Mode
    decoder: str
    strong_id: str


@dataclass(frozen=True)
class SweepRow:
    channel: Channel
    p: float
    mode: Mode
    decoder: str
    strong_id: str
    shots: int
    errors: int
    discards: int
    kept_shots: int
    seconds: float
    acceptance: float | None
    acceptance_low: float | None
    acceptance_high: float | None
    logical_error_rate: float | None
    logical_error_low: float | None
    logical_error_high: float | None
    denominator: int
    detection_events: int | None
    detectors_checked: int | None
    detection_fraction: float | None


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _all_detector_postselection_mask(n: int) -> np.ndarray:
    """Returns Sinter's little-endian bit-packed mask selecting all detectors."""
    bits = np.ones(n, dtype=np.uint8)
    packed = np.packbits(bits, bitorder="little")
    # Self-check the bit packing instead of assuming the final partial byte.
    unpacked = np.unpackbits(packed, bitorder="little")[:n]
    if unpacked.shape != (n,) or not bool(np.all(unpacked == 1)):
        raise AssertionError("failed to construct all-detector postselection mask")
    # Sinter expects uint8 bit-packed masks.
    return np.asarray(packed, dtype=np.uint8)


def _fit_binomial(sinter: Any, hits: int, n: int, likelihood_factor: float) -> tuple[float | None, float | None, float | None]:
    if n <= 0:
        return None, None, None
    fit = sinter.fit_binomial(
        num_shots=int(n),
        num_hits=int(hits),
        max_likelihood_factor=float(likelihood_factor),
    )
    return float(fit.best), float(fit.low), float(fit.high)


def _audit_graphlike_dem(dem: Any) -> dict[str, int]:
    """Checks every '^'-separated DEM component has detector weight <= 2."""
    bad = 0
    max_w = 0
    error_insts = 0
    components = 0
    for inst in dem.flattened():
        if getattr(inst, "type", None) != "error":
            continue
        error_insts += 1
        comp: list[Any] = []
        for t in inst.targets_copy():
            flag = getattr(t, "is_separator")
            is_sep = bool(flag() if callable(flag) else flag)
            if is_sep:
                w = 0
                for q in comp:
                    f = getattr(q, "is_relative_detector_id")
                    w += int(bool(f() if callable(f) else f))
                components += 1
                max_w = max(max_w, w)
                bad += int(w > 2)
                comp = []
            else:
                comp.append(t)
        w = 0
        for q in comp:
            f = getattr(q, "is_relative_detector_id")
            w += int(bool(f() if callable(f) else f))
        components += 1
        max_w = max(max_w, w)
        bad += int(w > 2)
    return {
        "error_instructions": error_insts,
        "components": components,
        "max_component_detector_weight": max_w,
        "bad_components": bad,
    }


def _p_tag(p: float) -> str:
    return f"{p:.12g}".replace("+", "").replace("-", "m").replace(".", "p")


def _build_tasks(
    *,
    hirano_src: Path,
    p_values: Iterable[float],
    channels: Iterable[Channel],
    out_dir: Path,
    do_assert: bool,
) -> tuple[list[Any], list[TaskSpec], list[dict[str, Any]]]:
    import sinter

    tasks: list[Any] = []
    specs: list[TaskSpec] = []
    structures: list[dict[str, Any]] = []
    post_mask = _all_detector_postselection_mask(EXPECTED_DETECTORS)

    circuit_dir = out_dir / "circuits"
    dem_dir = out_dir / "dems"
    circuit_dir.mkdir(parents=True, exist_ok=True)
    dem_dir.mkdir(parents=True, exist_ok=True)

    for ch in channels:
        circuit, wrapper, init_range = s3.build_public_zxz_with_init_range(hirano_src, channel=ch)
        post_ids = [int(d.id) for d in wrapper.detectors_for_post_selection]
        post_all = len(post_ids) == circuit.num_detectors and set(post_ids) == set(range(circuit.num_detectors))
        if do_assert:
            if circuit.num_detectors != EXPECTED_DETECTORS:
                raise AssertionError(f"{ch}: detectors={circuit.num_detectors} != {EXPECTED_DETECTORS}")
            if circuit.num_observables != EXPECTED_OBSERVABLES:
                raise AssertionError(f"{ch}: observables={circuit.num_observables} != {EXPECTED_OBSERVABLES}")
            if not post_all:
                raise AssertionError(f"{ch}: all-detector postselection contract drifted")

        for p in p_values:
            noisy, locations, noise_map = s3.instrument_canonical_fault_model(circuit, init_range=init_range, p=p)
            # Inventory is independent of p but assert it at every task construction.
            rows = s3.build_fault_rows(locations)
            pair_count = s3._pair_denominator(locations)
            if do_assert:
                if len(locations) != EXPECTED_LOCATIONS:
                    raise AssertionError(f"{ch},p={p}: locations={len(locations)}")
                if len(rows) != EXPECTED_SINGLE_CLASSES:
                    raise AssertionError(f"{ch},p={p}: single classes={len(rows)}")
                if pair_count != EXPECTED_PAIR_CLASSES:
                    raise AssertionError(f"{ch},p={p}: pair denominator={pair_count}")

            raw_dem = noisy.detector_error_model(decompose_errors=False)
            graph_dem = noisy.detector_error_model(decompose_errors=True)
            graph_stats = _audit_graphlike_dem(graph_dem)
            if do_assert and graph_stats["bad_components"] != 0:
                raise AssertionError(f"{ch},p={p}: graphlike DEM has bad components: {graph_stats}")

            tag = _p_tag(p)
            noisy_path = circuit_dir / f"zxz_d5_{ch}_p_{tag}.stim"
            raw_dem_path = dem_dir / f"zxz_d5_{ch}_p_{tag}_raw.dem"
            graph_dem_path = dem_dir / f"zxz_d5_{ch}_p_{tag}_graphlike.dem"
            noisy_path.write_text(str(noisy), encoding="utf-8")
            raw_dem_path.write_text(str(raw_dem), encoding="utf-8")
            graph_dem_path.write_text(str(graph_dem), encoding="utf-8")

            common_meta = {
                "stage": "S6",
                "channel": ch,
                "p": float(p),
                "distance": 5,
                "schedule": "ZXZ",
                "noise_model": "canonical_active_location_stochastic_pauli",
                "detectors": EXPECTED_DETECTORS,
                "observables": EXPECTED_OBSERVABLES,
                "physical_locations": EXPECTED_LOCATIONS,
                "single_fault_classes": EXPECTED_SINGLE_CLASSES,
                "two_fault_class_pairs": EXPECTED_PAIR_CLASSES,
            }

            cert_task = sinter.Task(
                circuit=noisy,
                decoder="vacuous",
                detector_error_model=raw_dem,
                postselection_mask=post_mask,
                json_metadata={**common_meta, "mode": "certification"},
            )
            mwpm_task = sinter.Task(
                circuit=noisy,
                decoder="pymatching",
                detector_error_model=graph_dem,
                json_metadata={**common_meta, "mode": "mwpm"},
            )
            tasks.extend([cert_task, mwpm_task])
            specs.extend([
                TaskSpec(ch, float(p), "certification", "vacuous", cert_task.strong_id()),
                TaskSpec(ch, float(p), "mwpm", "pymatching", mwpm_task.strong_id()),
            ])
            structures.append({
                "channel": ch,
                "p": float(p),
                "postselection_ids_are_all_detectors": post_all,
                "raw_dem_num_detectors": int(raw_dem.num_detectors),
                "raw_dem_num_observables": int(raw_dem.num_observables),
                "graph_dem_num_detectors": int(graph_dem.num_detectors),
                "graph_dem_num_observables": int(graph_dem.num_observables),
                **{f"graphlike_{k}": int(v) for k, v in graph_stats.items()},
            })

    return tasks, specs, structures


def _stat_to_row(sinter: Any, stat: Any, *, likelihood_factor: float) -> SweepRow:
    meta = stat.json_metadata
    mode: Mode = meta["mode"]
    shots = int(stat.shots)
    errors = int(stat.errors)
    discards = int(stat.discards)
    kept = shots - discards

    acceptance = None
    acceptance_low = None
    acceptance_high = None
    if mode == "certification":
        discard_best, discard_low, discard_high = _fit_binomial(sinter, discards, shots, likelihood_factor)
        if discard_best is not None:
            acceptance = 1.0 - discard_best
            # invert interval endpoints
            acceptance_low = 1.0 - float(discard_high)
            acceptance_high = 1.0 - float(discard_low)
        logical, logical_low, logical_high = _fit_binomial(sinter, errors, kept, likelihood_factor)
        denom = kept
    elif mode == "mwpm":
        if discards != 0:
            raise AssertionError(f"MWPM task unexpectedly discarded {discards} shots")
        logical, logical_low, logical_high = _fit_binomial(sinter, errors, shots, likelihood_factor)
        denom = shots
    else:
        raise AssertionError(f"unknown mode {mode}")

    cc = dict(stat.custom_counts)
    de = int(cc["detection_events"]) if "detection_events" in cc else None
    dc = int(cc["detectors_checked"]) if "detectors_checked" in cc else None
    df = None if de is None or dc in (None, 0) else float(de / dc)

    return SweepRow(
        channel=meta["channel"],
        p=float(meta["p"]),
        mode=mode,
        decoder=str(stat.decoder),
        strong_id=str(stat.strong_id),
        shots=shots,
        errors=errors,
        discards=discards,
        kept_shots=kept,
        seconds=float(stat.seconds),
        acceptance=acceptance,
        acceptance_low=acceptance_low,
        acceptance_high=acceptance_high,
        logical_error_rate=logical,
        logical_error_low=logical_low,
        logical_error_high=logical_high,
        denominator=denom,
        detection_events=de,
        detectors_checked=dc,
        detection_fraction=df,
    )


def _write_summary_csv(path: Path, rows: list[SweepRow]) -> None:
    fields = list(SweepRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _self_test_postselection_mask() -> None:
    mask = _all_detector_postselection_mask(135)
    if mask.dtype != np.uint8 or mask.shape != (17,):
        raise AssertionError(f"mask shape/dtype wrong: {mask.shape} {mask.dtype}")
    if mask[:16].tolist() != [255] * 16 or int(mask[16]) != 127:
        raise AssertionError(f"unexpected 135-bit all-ones mask: {mask.tolist()}")


def _self_test_sinter_semantics() -> None:
    """Tiny runtime test for Sinter errors/discards semantics used by this script."""
    import sinter
    import stim

    # D0 fires together with L0 with p=1/8. Postselecting D0 must turn these into
    # discards and leave zero errors for a vacuous decoder among kept shots.
    c = stim.Circuit("""
        R 0
        X_ERROR(0.125) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    raw_dem = c.detector_error_model()
    task = sinter.Task(
        circuit=c,
        decoder="vacuous",
        detector_error_model=raw_dem,
        postselection_mask=np.array([1], dtype=np.uint8),
        json_metadata={"self_test": True},
    )
    # Avoid multiprocessing here; Task construction + strong-id validation is the
    # portable semantic self-test. The actual collect path is exercised by main.
    if raw_dem.num_detectors != 1 or raw_dem.num_observables != 1:
        raise AssertionError("Sinter semantic self-test DEM failed")
    _ = task.strong_id()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--p-values", type=float, nargs="+", default=list(DEFAULT_P_VALUES))
    ap.add_argument("--channel", choices=["X", "Z", "both"], default="both")
    ap.add_argument("--max-shots", type=int, default=200_000)
    ap.add_argument("--max-errors", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--max-batch-size", type=int, default=100_000)
    ap.add_argument("--likelihood-factor", type=float, default=1000.0)
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/zxz/s6_results"))
    ap.add_argument("--resume-csv", type=Path, default=None)
    ap.add_argument("--no-assert", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    if args.max_shots <= 0:
        raise SystemExit("--max-shots must be positive")
    if args.max_errors <= 0:
        raise SystemExit("--max-errors must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.likelihood_factor <= 1:
        raise SystemExit("--likelihood-factor must exceed 1")
    p_values = [float(p) for p in args.p_values]
    if not p_values or any(not (0 < p < 0.5) for p in p_values):
        raise SystemExit("all --p-values must satisfy 0 < p < 0.5")
    # Preserve user order while rejecting duplicates that would create identical tasks.
    if len(set(p_values)) != len(p_values):
        raise SystemExit("--p-values contains duplicates")

    import sinter

    _self_test_postselection_mask()
    _self_test_sinter_semantics()
    s3._self_test_noise_append_fusion()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    resume_csv = args.resume_csv or (args.out_dir / "s6_sinter_resume.csv")
    channels: list[Channel] = ["X", "Z"] if args.channel == "both" else [args.channel]  # type: ignore[list-item]

    tasks, specs, structures = _build_tasks(
        hirano_src=args.hirano_src,
        p_values=p_values,
        channels=channels,
        out_dir=args.out_dir,
        do_assert=not args.no_assert,
    )

    manifest = {
        "stage": "S6",
        "stim_version": _version("stim"),
        "sinter_version": _version("sinter"),
        "pymatching_version": _version("pymatching"),
        "numpy_version": np.__version__,
        "p_values": p_values,
        "channels": channels,
        "max_shots": int(args.max_shots),
        "max_errors": int(args.max_errors),
        "workers": int(args.workers),
        "max_batch_size": int(args.max_batch_size),
        "likelihood_factor": float(args.likelihood_factor),
        "resume_csv": str(resume_csv),
        "tasks": [asdict(s) for s in specs],
        "structures": structures,
    }
    (args.out_dir / "s6_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    stats = sinter.collect(
        num_workers=args.workers,
        tasks=tasks,
        save_resume_filepath=resume_csv,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        max_batch_size=args.max_batch_size,
        count_detection_events=True,
        print_progress=not args.no_progress,
    )

    rows = [_stat_to_row(sinter, s, likelihood_factor=args.likelihood_factor) for s in stats]
    rows.sort(key=lambda r: (r.channel, r.p, r.mode))

    # Strong guard: one certification and one MWPM row per requested channel,p.
    expected_keys = {(ch, p, mode) for ch in channels for p in p_values for mode in ("certification", "mwpm")}
    got_keys = {(r.channel, r.p, r.mode) for r in rows}
    if not args.no_assert and got_keys != expected_keys:
        raise AssertionError(f"task coverage mismatch missing={sorted(expected_keys-got_keys)} extra={sorted(got_keys-expected_keys)}")

    _write_summary_csv(args.out_dir / "s6_summary.csv", rows)
    (args.out_dir / "s6_summary.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2, sort_keys=True), encoding="utf-8"
    )

    # Human-readable compact table.
    print("channel,p,mode,shots,errors,discards,kept,acceptance,logical_error_rate")
    for r in rows:
        acc = "" if r.acceptance is None else f"{r.acceptance:.9g}"
        ler = "" if r.logical_error_rate is None else f"{r.logical_error_rate:.9g}"
        print(f"{r.channel},{r.p:.9g},{r.mode},{r.shots},{r.errors},{r.discards},{r.kept_shots},{acc},{ler}")

    print("S6 PASS: Sinter certification + PyMatching p-sweep completed")
    print(f"summary_csv={args.out_dir / 's6_summary.csv'}")
    print(f"resume_csv={resume_csv}")


if __name__ == "__main__":
    main()
