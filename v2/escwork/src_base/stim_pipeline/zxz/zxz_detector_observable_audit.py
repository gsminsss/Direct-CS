#!/usr/bin/env python3
"""S2 v6 parser-fixed audit for Hirano d=5 full-ZXZ Steane->surface escape.

This audit is deliberately annotation-first and reference-frame safe.
It does NOT monkey-patch the physical circuit. Instead it verifies the exact
GF(2) relationship between the public Hirano DETECTOR annotations and
OBSERVABLE_INCLUDE(0) at the terminal surface-code readout boundary.

Checks performed for each X/Z transfer channel:
  1) Re-run the S1 canonical fingerprint regression.
  2) Construct a zero-noise Stim detector error model (determinism check).
  3) Resolve every DETECTOR / OBSERVABLE_INCLUDE rec[-k] into absolute
     measurement IDs and build the binary annotation matrices H and L.
  4) Identify the final d^2=25 destructive surface measurements from the
     actual Stim measurement stream, verify their basis and 5x5 geometry from
     QUBIT_COORDS, and derive the intended logical strings geometrically.
  5) Verify the terminal part of observable 0 is exactly:
       X channel: first column X-readout parity,
       Z channel: top row Z-readout parity.
  6) Apply only the *measurement-record action* of the complementary logical
     Pauli at the terminal boundary:
       X channel: top-row Z_L flips the five terminal MX records,
       Z channel: first-column X_L flips the five terminal MZ records.
     Verify exactly H e = 0 and L e = 1 over GF(2).
  7) Independently confirm, on noiseless raw samples, that XOR by e leaves all
     detector parities unchanged and flips observable 0 on every shot.
  8) Independently confirm Stim's detector sampler produces no zero-noise
     detection events / observable flips relative to its reference sample.

This is the precise semantic condition needed by the later fault catalogue:
    undetected logical transfer error  <=>  detector signature = 0,
                                            observable flip = 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hirano_zxz_generator as s1

Channel = Literal["X", "Z"]
D = 5
EXPECTED_TOTAL_DETECTORS = 135
EXPECTED_PRE_READOUT_DETECTORS = 123
EXPECTED_TERMINAL_DETECTORS = 12
EXPECTED_OBSERVABLES = 1
EXPECTED_TERMINAL_MEASUREMENTS = D * D

_SIMPLE_MEAS_GATES = {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"}
_ALL_MEAS_GATES = _SIMPLE_MEAS_GATES | {"MPP", "MXX", "MYY", "MZZ", "MPAD"}


@dataclass(frozen=True)
class ChannelAudit:
    channel: Channel
    num_detectors: int
    num_observables: int
    num_measurements: int
    postselection_detector_count: int
    postselection_ids_are_all_detectors: bool
    dem_constructed: bool
    dem_num_detectors: int
    dem_num_observables: int
    detector_sampler_nonzero_event_shots: int
    detector_sampler_nonzero_observable_shots: int
    raw_unique_detector_rows: int
    raw_unique_observable_rows: int
    raw_baseline_observable_value: int
    terminal_measurement_ids: list[int]
    terminal_measurement_basis: str
    terminal_detector_ids: list[int]
    pre_readout_detector_ids: list[int]
    terminal_grid_x_values: list[float]
    terminal_grid_y_values: list[float]
    expected_terminal_observable_ids: list[int]
    actual_terminal_observable_ids: list[int]
    terminal_observable_support_exact_match: bool
    anticommuting_boundary_flip_ids: list[int]
    anticommuting_detector_syndrome_weight: int
    anticommuting_pre_detector_syndrome_weight: int
    anticommuting_terminal_detector_syndrome_weight: int
    anticommuting_observable_flip: int
    synthetic_detector_mismatch_shots: int
    synthetic_observable_flip_shots: int


def _rec_offset(target: Any) -> int | None:
    """Return the negative lookback of a Stim rec target using the official API.

    Do not parse ``str(target)``.  Stim's GateTarget textual representation is
    not the API contract for identifying measurement-record targets.  The
    supported contract is ``is_measurement_record_target`` plus ``value``.
    """
    flag = getattr(target, "is_measurement_record_target", False)
    if callable(flag):
        flag = flag()
    if not bool(flag):
        return None
    off = int(target.value)
    if off >= 0:
        raise AssertionError(f"measurement-record target has non-negative value: {target!r}")
    return off


def _count_measurements_in_instruction(inst: Any) -> int:
    """Return the number of measurement bits produced by one Stim instruction."""
    # Modern Stim exposes this directly on CircuitInstruction.
    n = getattr(inst, "num_measurements", None)
    if n is not None:
        return int(n() if callable(n) else n)

    # Compatibility fallback for older Stim versions.
    import stim
    if inst.name not in _ALL_MEAS_GATES:
        return 0
    one = stim.Circuit()
    one.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
    return int(one.num_measurements)


def _self_test_annotation_parser() -> None:
    """Fail fast if the local Stim GateTarget API is interpreted incorrectly."""
    import stim

    c = stim.Circuit("""
        R 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    deps = annotation_dependency_map(c)
    if deps["num_measurements"] != 2:
        raise AssertionError(f"parser self-test: measurements={deps['num_measurements']} != 2")
    if len(deps["detectors"]) != 1 or deps["detectors"][0]["measurement_ids"] != [1, 0]:
        raise AssertionError(f"parser self-test detector deps failed: {deps['detectors']}")
    if len(deps["observables"]) != 1 or deps["observables"][0]["measurement_ids"] != [1]:
        raise AssertionError(f"parser self-test observable deps failed: {deps['observables']}")
    H, L = _annotation_matrices(deps, 1)
    if H.shape != (1, 2) or H.tolist() != [[1, 1]]:
        raise AssertionError(f"parser self-test H failed: {H.tolist()}")
    if L.shape != (1, 2) or L.tolist() != [[0, 1]]:
        raise AssertionError(f"parser self-test L failed: {L.tolist()}")


def annotation_dependency_map(circuit: Any) -> dict[str, Any]:
    """Resolve all rec[-k] DETECTOR/OBSERVABLE references to absolute IDs."""
    measurement_count = 0
    detector_id = 0
    detectors: list[dict[str, Any]] = []
    observables: list[dict[str, Any]] = []

    for instruction_index, inst in enumerate(circuit.flattened()):
        name = inst.name
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            mids: list[int] = []
            for t in inst.targets_copy():
                off = _rec_offset(t)
                if off is None:
                    continue
                mid = measurement_count + off
                if not (0 <= mid < measurement_count):
                    raise AssertionError(
                        f"bad record target {t} at instruction {instruction_index}; "
                        f"measurement_count={measurement_count}"
                    )
                mids.append(mid)
            if name == "DETECTOR":
                detectors.append({
                    "detector_id": detector_id,
                    "instruction_index": instruction_index,
                    "measurement_ids": mids,
                })
                detector_id += 1
            else:
                args = list(inst.gate_args_copy())
                obs_id = int(args[0]) if args else 0
                observables.append({
                    "observable_id": obs_id,
                    "instruction_index": instruction_index,
                    "measurement_ids": mids,
                })

        measurement_count += _count_measurements_in_instruction(inst)

    if detector_id != circuit.num_detectors:
        raise AssertionError(
            f"parsed detector count {detector_id} != Stim {circuit.num_detectors}"
        )
    if measurement_count != circuit.num_measurements:
        raise AssertionError(
            f"parsed measurement count {measurement_count} != Stim {circuit.num_measurements}"
        )
    return {
        "num_measurements": measurement_count,
        "detectors": detectors,
        "observables": observables,
    }


def _annotation_matrices(deps: dict[str, Any], nobs_expected: int) -> tuple[np.ndarray, np.ndarray]:
    """Return detector matrix H and observable matrix L over GF(2).

    Multiple OBSERVABLE_INCLUDE instructions with the same observable id are
    XOR-aggregated, matching Stim semantics.
    """
    nm = deps["num_measurements"]
    H = np.zeros((len(deps["detectors"]), nm), dtype=np.uint8)
    for d in deps["detectors"]:
        for mid in d["measurement_ids"]:
            H[d["detector_id"], mid] ^= 1

    L = np.zeros((nobs_expected, nm), dtype=np.uint8)
    for o in deps["observables"]:
        oid = int(o["observable_id"])
        if not (0 <= oid < nobs_expected):
            raise AssertionError(f"observable id {oid} outside expected range")
        for mid in o["measurement_ids"]:
            L[oid, mid] ^= 1
    return H, L


def _qubit_coords(circuit: Any) -> dict[int, tuple[float, ...]]:
    coords: dict[int, tuple[float, ...]] = {}
    for inst in circuit.flattened():
        if inst.name != "QUBIT_COORDS":
            continue
        args = tuple(float(x) for x in inst.gate_args_copy())
        for t in inst.targets_copy():
            q = int(t.value)
            coords[q] = args
    return coords


def _measurement_metadata(circuit: Any) -> list[dict[str, Any]]:
    """Metadata for each absolute measurement id.

    Simple one-qubit measurements are mapped to their measured qubit. Other
    measurement-producing instructions are still counted, but have qubit=None.
    """
    out: list[dict[str, Any]] = []
    mid = 0
    for instruction_index, inst in enumerate(circuit.flattened()):
        name = inst.name
        n = _count_measurements_in_instruction(inst)
        if n == 0:
            continue
        if name in _SIMPLE_MEAS_GATES:
            targets = inst.targets_copy()
            if len(targets) != n:
                raise AssertionError(
                    f"{name}: {len(targets)} targets but {n} measurement results"
                )
            for t in targets:
                out.append({
                    "measurement_id": mid,
                    "instruction_index": instruction_index,
                    "gate": name,
                    "qubit": int(t.value),
                })
                mid += 1
        else:
            for _ in range(n):
                out.append({
                    "measurement_id": mid,
                    "instruction_index": instruction_index,
                    "gate": name,
                    "qubit": None,
                })
                mid += 1
    if mid != circuit.num_measurements or len(out) != circuit.num_measurements:
        raise AssertionError((mid, len(out), circuit.num_measurements))
    return out


def _terminal_geometry(circuit: Any, channel: Channel) -> dict[str, Any]:
    """Identify the actual final 25 surface measurements and logical strings."""
    meta = _measurement_metadata(circuit)
    coords = _qubit_coords(circuit)
    terminal = meta[-EXPECTED_TERMINAL_MEASUREMENTS:]
    terminal_ids = [m["measurement_id"] for m in terminal]

    expected_basis = "MX" if channel == "X" else "Z"
    for m in terminal:
        g = m["gate"]
        if channel == "X":
            ok = g == "MX"
        else:
            ok = g in {"M", "MZ"}
        if not ok:
            raise AssertionError(
                f"{channel}: final 25 measurements are not all terminal {expected_basis}; "
                f"found gate {g} at measurement {m['measurement_id']}"
            )
        if m["qubit"] is None or m["qubit"] not in coords:
            raise AssertionError(
                f"{channel}: terminal measurement {m['measurement_id']} has no QUBIT_COORDS"
            )

    points: dict[int, tuple[float, float]] = {}
    for m in terminal:
        c = coords[m["qubit"]]
        if len(c) < 2:
            raise AssertionError(f"qubit {m['qubit']} has <2 coordinates: {c}")
        points[m["measurement_id"]] = (float(c[0]), float(c[1]))

    xs = sorted({p[0] for p in points.values()})
    ys = sorted({p[1] for p in points.values()})
    if len(xs) != D or len(ys) != D:
        raise AssertionError(f"{channel}: terminal geometry is not {D}x{D}: xs={xs}, ys={ys}")
    if len(set(points.values())) != D * D:
        raise AssertionError(f"{channel}: terminal coordinates are not 25 distinct grid points")

    min_x = xs[0]
    min_y = ys[0]
    first_col = sorted(mid for mid, (x, _y) in points.items() if x == min_x)
    top_row = sorted(mid for mid, (_x, y) in points.items() if y == min_y)
    if len(first_col) != D or len(top_row) != D:
        raise AssertionError((first_col, top_row))

    if channel == "X":
        expected_obs = first_col
        anti_flip = top_row
    else:
        expected_obs = top_row
        anti_flip = first_col

    return {
        "terminal_measurement_ids": terminal_ids,
        "terminal_measurement_basis": expected_basis,
        "x_values": xs,
        "y_values": ys,
        "expected_observable_terminal_ids": expected_obs,
        "anticommuting_boundary_flip_ids": anti_flip,
        "measurement_points": {str(k): list(v) for k, v in points.items()},
    }


def _parities(measurements: np.ndarray, H: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = measurements.astype(np.uint8, copy=False)
    d = (m @ H.T) & 1
    o = (m @ L.T) & 1
    return d.astype(np.uint8), o.astype(np.uint8)


def audit_channel(
    hirano_src: Path,
    *,
    channel: Channel,
    shots: int,
    out_dir: Path | None,
    do_assert: bool,
) -> ChannelAudit:
    circuit, wrapper = s1.build_public_zxz(hirano_src, channel=channel)

    # S1 regression must stay closed while S2 is audited.
    fp = s1.fingerprint(circuit, wrapper, channel=channel)
    if do_assert:
        s1.assert_canonical(fp)

    if circuit.num_detectors != EXPECTED_TOTAL_DETECTORS:
        raise AssertionError(circuit.num_detectors)
    if circuit.num_observables != EXPECTED_OBSERVABLES:
        raise AssertionError(circuit.num_observables)

    # Full-postselection should contain exactly all detector ids. Compare sets,
    # not list order, to avoid an unnecessary implementation assumption.
    post_ids = [int(d.id) for d in wrapper.detectors_for_post_selection]
    all_ids = set(range(EXPECTED_TOTAL_DETECTORS))
    post_all = len(post_ids) == EXPECTED_TOTAL_DETECTORS and set(post_ids) == all_ids
    if do_assert and not post_all:
        raise AssertionError(
            f"postselection detector set mismatch: n={len(post_ids)}, "
            f"missing={sorted(all_ids - set(post_ids))}, "
            f"extra={sorted(set(post_ids) - all_ids)}"
        )

    # This throws if a declared detector is non-deterministic (unless gauge
    # detectors are explicitly allowed, which we do not do here).
    dem = circuit.detector_error_model(decompose_errors=False)

    deps = annotation_dependency_map(circuit)
    H, L = _annotation_matrices(deps, circuit.num_observables)
    geom = _terminal_geometry(circuit, channel)
    terminal_ids = geom["terminal_measurement_ids"]
    terminal_set = set(terminal_ids)

    # Derive which detectors are terminal from actual dependency on the final
    # 25 surface measurements; do not assume detector-id ordering.
    terminal_detector_ids = [
        i for i in range(H.shape[0]) if np.any(H[i, terminal_ids])
    ]
    terminal_detector_set = set(terminal_detector_ids)
    pre_detector_ids = [i for i in range(H.shape[0]) if i not in terminal_detector_set]

    if do_assert:
        if len(terminal_detector_ids) != EXPECTED_TERMINAL_DETECTORS:
            raise AssertionError(
                f"{channel}: terminal detector count {len(terminal_detector_ids)} "
                f"!= {EXPECTED_TERMINAL_DETECTORS}; ids={terminal_detector_ids}"
            )
        if len(pre_detector_ids) != EXPECTED_PRE_READOUT_DETECTORS:
            raise AssertionError(
                f"{channel}: pre-readout detector count {len(pre_detector_ids)} "
                f"!= {EXPECTED_PRE_READOUT_DETECTORS}"
            )

    actual_obs_terminal = sorted(
        mid for mid in terminal_ids if int(L[0, mid]) == 1
    )
    expected_obs_terminal = list(geom["expected_observable_terminal_ids"])
    support_match = actual_obs_terminal == expected_obs_terminal

    anti_flip_ids = list(geom["anticommuting_boundary_flip_ids"])
    flip = np.zeros(deps["num_measurements"], dtype=np.uint8)
    flip[anti_flip_ids] = 1
    d_syn = (H @ flip) & 1
    o_syn = (L @ flip) & 1

    total_w = int(np.count_nonzero(d_syn))
    pre_w = int(np.count_nonzero(d_syn[pre_detector_ids]))
    terminal_w = int(np.count_nonzero(d_syn[terminal_detector_ids]))
    obs_flip = int(o_syn[0])

    # Raw measurement parity audit. This compares the same physical sample
    # before/after the abstract terminal boundary record action, so Stim's
    # reference-sample normalization cannot hide the observable change.
    raw = np.asarray(circuit.compile_sampler().sample(shots=shots), dtype=np.uint8)
    d0, o0 = _parities(raw, H, L)
    raw2 = raw ^ flip[None, :]
    d1, o1 = _parities(raw2, H, L)
    detector_mismatch_shots = int(np.count_nonzero(np.any(d1 ^ d0, axis=1)))
    observable_flip_shots = int(np.count_nonzero((o1[:, 0] ^ o0[:, 0]) == 1))
    unique_d = int(np.unique(d0, axis=0).shape[0])
    unique_o = int(np.unique(o0, axis=0).shape[0])
    raw_obs_value = int(o0[0, 0]) if unique_o == 1 else -1

    # Independent Stim-native zero-noise sanity check. These are detection-event
    # and logical-flip values relative to Stim's own noiseless reference sample.
    ds, os = circuit.compile_detector_sampler().sample(
        shots=shots, separate_observables=True
    )
    detector_sampler_nonzero_event_shots = int(
        np.count_nonzero(np.any(np.asarray(ds, dtype=np.uint8), axis=1))
    )
    detector_sampler_nonzero_observable_shots = int(
        np.count_nonzero(np.any(np.asarray(os, dtype=np.uint8), axis=1))
    )

    if do_assert:
        if not support_match:
            raise AssertionError(
                f"{channel}: terminal observable support mismatch; "
                f"actual={actual_obs_terminal}, expected={expected_obs_terminal}"
            )
        if total_w != 0:
            raise AssertionError(
                f"{channel}: complementary boundary logical record triggers detectors; "
                f"total={total_w}, pre={pre_w}, terminal={terminal_w}, "
                f"detector_ids={np.flatnonzero(d_syn).tolist()}"
            )
        if obs_flip != 1:
            raise AssertionError(
                f"{channel}: complementary boundary logical record gives "
                f"observable flip {obs_flip}, expected 1"
            )
        if detector_mismatch_shots != 0:
            raise AssertionError(
                f"{channel}: synthetic boundary action changed detector parity "
                f"in {detector_mismatch_shots}/{shots} raw samples"
            )
        if observable_flip_shots != shots:
            raise AssertionError(
                f"{channel}: synthetic boundary action flipped observable in "
                f"{observable_flip_shots}/{shots} samples, expected all"
            )
        if unique_d != 1:
            raise AssertionError(
                f"{channel}: raw zero-noise detector parity is not deterministic: "
                f"{unique_d} unique rows"
            )
        if unique_o != 1:
            raise AssertionError(
                f"{channel}: raw zero-noise observable parity is not deterministic: "
                f"{unique_o} unique rows"
            )
        if detector_sampler_nonzero_event_shots != 0:
            raise AssertionError(
                f"{channel}: Stim detector sampler saw zero-noise events in "
                f"{detector_sampler_nonzero_event_shots}/{shots} shots"
            )
        if detector_sampler_nonzero_observable_shots != 0:
            raise AssertionError(
                f"{channel}: Stim detector sampler saw zero-noise observable flips in "
                f"{detector_sampler_nonzero_observable_shots}/{shots} shots"
            )

    result = ChannelAudit(
        channel=channel,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
        num_measurements=deps["num_measurements"],
        postselection_detector_count=len(post_ids),
        postselection_ids_are_all_detectors=post_all,
        dem_constructed=True,
        dem_num_detectors=dem.num_detectors,
        dem_num_observables=dem.num_observables,
        detector_sampler_nonzero_event_shots=detector_sampler_nonzero_event_shots,
        detector_sampler_nonzero_observable_shots=detector_sampler_nonzero_observable_shots,
        raw_unique_detector_rows=unique_d,
        raw_unique_observable_rows=unique_o,
        raw_baseline_observable_value=raw_obs_value,
        terminal_measurement_ids=terminal_ids,
        terminal_measurement_basis=geom["terminal_measurement_basis"],
        terminal_detector_ids=terminal_detector_ids,
        pre_readout_detector_ids=pre_detector_ids,
        terminal_grid_x_values=list(geom["x_values"]),
        terminal_grid_y_values=list(geom["y_values"]),
        expected_terminal_observable_ids=expected_obs_terminal,
        actual_terminal_observable_ids=actual_obs_terminal,
        terminal_observable_support_exact_match=support_match,
        anticommuting_boundary_flip_ids=anti_flip_ids,
        anticommuting_detector_syndrome_weight=total_w,
        anticommuting_pre_detector_syndrome_weight=pre_w,
        anticommuting_terminal_detector_syndrome_weight=terminal_w,
        anticommuting_observable_flip=obs_flip,
        synthetic_detector_mismatch_shots=detector_mismatch_shots,
        synthetic_observable_flip_shots=observable_flip_shots,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"s2_{channel}_annotation_dependencies.json").write_text(
            json.dumps(deps, indent=2, sort_keys=True), encoding="utf-8"
        )
        (out_dir / f"s2_{channel}_terminal_geometry.json").write_text(
            json.dumps(geom, indent=2, sort_keys=True), encoding="utf-8"
        )
        (out_dir / f"s2_{channel}_audit.json").write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8"
        )
        (out_dir / f"s2_{channel}_baseline.stim").write_text(
            str(circuit), encoding="utf-8"
        )

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--channel", choices=["X", "Z", "both"], default="both")
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be positive")

    # Validate our interpretation of Stim rec targets before touching the
    # Hirano circuit.  This specifically guards against the parser bug that
    # caused v5 to see zero terminal detector dependencies.
    _self_test_annotation_parser()

    channels: list[Channel] = ["X", "Z"] if args.channel == "both" else [args.channel]
    results = [
        asdict(
            audit_channel(
                args.hirano_src,
                channel=ch,
                shots=args.shots,
                out_dir=args.out_dir,
                do_assert=not args.no_assert,
            )
        )
        for ch in channels
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    if not args.no_assert:
        print("S2 PASS: exact detector/observable GF(2) boundary audit")


if __name__ == "__main__":
    main()
