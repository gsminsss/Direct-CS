#!/usr/bin/env python3
"""Rebuild Hirano's full ZXZ Steane->d=5 rotated-surface-code escape.

This is a thin reproducibility adapter around the authors' public `msc-ls`
implementation (branch `experiment-2025-sep`).  It intentionally does *not*
copy or reinterpret the lattice-surgery schedule.  The purpose of S1 is to
freeze the exact public generator invocation and audit it against the
canonical Step-4 fingerprint of the direct-|CS> project.

Canonical encoded-input Step-4 target (pre terminal readout):
    CX=463, RX=87, RZ=77, MX=70, MZ=75,
    postselection detectors=123.
Terminal complementary readout adds d^2=25 measurements.

The public Hirano driver prepares an ideal Steane logical input internally.
Those ideal input-preparation gates are *not* part of the encoded-input escape
fault-location count, so this adapter reports both raw-public and
encoded-input-normalized fingerprints.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Channel = Literal["X", "Z"]

CANONICAL_PRE_READOUT = {
    "CX": 463,
    "RX": 87,
    "RZ": 77,
    "MX": 70,
    "MZ": 75,
}
CANONICAL_PRE_READOUT_POSTSELECT_DETECTORS = 123
CANONICAL_TERMINAL_MEASUREMENTS = 25  # d=5 surface data qubits
CANONICAL_TERMINAL_STABILIZER_DETECTORS = 12  # (d^2-1)/2 for one basis
CANONICAL_ACTIVE_LOCATIONS = 797
CANONICAL_SINGLE_FAULT_CLASSES = 7279
CANONICAL_TWO_FAULT_CLASS_PAIRS = 26_439_666

# Direct stim.Circuit initialization used by Hirano's public wrapper.  These
# gates instantiate the *input state* and therefore are intentionally removed
# from the encoded-input escape location inventory.
IDEAL_STEANE_INIT = {
    "X": {"CX": 11, "RX": 4, "RZ": 3, "MX": 0, "MZ": 0},  # |+>_L
    "Z": {"CX": 11, "RX": 3, "RZ": 4, "MX": 0, "MZ": 0},  # |0>_L
}


@dataclass(frozen=True)
class Fingerprint:
    channel: Channel
    raw_counts: dict[str, int]
    encoded_input_terminal_counts: dict[str, int]
    encoded_input_pre_readout_counts: dict[str, int]
    total_detectors: int
    postselection_detectors: int
    pre_readout_postselection_detectors: int
    num_observables: int
    active_locations: int
    single_fault_classes: int
    two_fault_class_pairs: int


def _install_hirano_src(path: Path) -> None:
    path = path.resolve()
    if not (path / "lattice_surgery_error_detection.py").exists():
        raise FileNotFoundError(
            f"{path} does not look like msc-ls/src: missing "
            "lattice_surgery_error_detection.py"
        )
    sys.path.insert(0, str(path))


def _load_hirano(path: Path) -> tuple[Any, Any]:
    _install_hirano_src(path)
    try:
        lsed = importlib.import_module("lattice_surgery_error_detection")
        util = importlib.import_module("util")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import Hirano dependencies. Install the branch-compatible "
            "Stim/Sinter, PyMatching, and NumPy environment first."
        ) from exc
    return lsed, util


def build_public_zxz(
    hirano_src: Path,
    *,
    channel: Channel,
    surface_distance: int = 5,
    error_probability: float = 0.0,
    full_post_selection: bool = True,
) -> tuple[Any, Any]:
    """Return (stim.Circuit, Hirano Circuit wrapper) for the official ZXZ path."""
    if surface_distance != 5:
        raise ValueError("S1 canonical reproduction is pinned to surface_distance=5")
    lsed, util = _load_hirano(hirano_src)
    mapping = util.QubitMapping(30, 30)
    initial_value = (
        lsed.InitialValue.Plus if channel == "X" else lsed.InitialValue.Zero
    )
    runner = lsed.SteanePlusSurfaceCode(
        mapping,
        surface_distance,
        initial_value,
        lsed.SteaneSyndromeExtractionPattern.ZXZ,
        error_probability,
        full_post_selection,
    )
    runner.run()
    return runner.circuit.circuit, runner.circuit


def _count_targets(circuit: Any) -> dict[str, int]:
    """Count physical gate *targets/pairs*, not Stim instruction lines."""
    counts = {"CX": 0, "RX": 0, "RZ": 0, "MX": 0, "MZ": 0}
    flat = circuit.flattened()
    for inst in flat:
        name = inst.name
        targets = inst.targets_copy()
        if name in ("CX", "CNOT"):
            if len(targets) % 2:
                raise AssertionError(f"odd CX target count: {len(targets)}")
            counts["CX"] += len(targets) // 2
        elif name == "RX":
            counts["RX"] += len(targets)
        elif name in ("R", "RZ"):
            counts["RZ"] += len(targets)
        elif name == "MX":
            counts["MX"] += len(targets)
        elif name in ("M", "MZ"):
            counts["MZ"] += len(targets)
    return counts


def _sub(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a[k] - b.get(k, 0) for k in a}


def _fault_inventory(active_counts_terminal: dict[str, int]) -> tuple[int, int, int]:
    """Canonical Step-4 active-location Pauli-class inventory.

    CX has 15 non-identity two-qubit Pauli classes.  Every reset or
    measurement location has one basis-flipping/outcome-flip class.
    Pairs are restricted to distinct physical locations.
    """
    n_cx = active_counts_terminal["CX"]
    n_other = sum(active_counts_terminal.values()) - n_cx
    locations = n_cx + n_other
    singles = 15 * n_cx + n_other
    sum_w_sq = n_cx * 15**2 + n_other
    pairs = (singles**2 - sum_w_sq) // 2
    return locations, singles, pairs


def fingerprint(circuit: Any, wrapper: Any, *, channel: Channel) -> Fingerprint:
    raw = _count_targets(circuit)
    active_terminal = _sub(raw, IDEAL_STEANE_INIT[channel])
    pre = dict(active_terminal)
    if channel == "X":
        pre["MX"] -= CANONICAL_TERMINAL_MEASUREMENTS
    else:
        pre["MZ"] -= CANONICAL_TERMINAL_MEASUREMENTS

    locations, singles, pairs = _fault_inventory(active_terminal)
    post = len(wrapper.detectors_for_post_selection)
    pre_post = post - CANONICAL_TERMINAL_STABILIZER_DETECTORS
    return Fingerprint(
        channel=channel,
        raw_counts=raw,
        encoded_input_terminal_counts=active_terminal,
        encoded_input_pre_readout_counts=pre,
        total_detectors=circuit.num_detectors,
        postselection_detectors=post,
        pre_readout_postselection_detectors=pre_post,
        num_observables=circuit.num_observables,
        active_locations=locations,
        single_fault_classes=singles,
        two_fault_class_pairs=pairs,
    )


def assert_canonical(fp: Fingerprint) -> None:
    assert fp.encoded_input_pre_readout_counts == CANONICAL_PRE_READOUT, (
        fp.encoded_input_pre_readout_counts,
        CANONICAL_PRE_READOUT,
    )
    assert (
        fp.pre_readout_postselection_detectors
        == CANONICAL_PRE_READOUT_POSTSELECT_DETECTORS
    ), fp.pre_readout_postselection_detectors
    assert fp.num_observables == 1, fp.num_observables
    assert fp.active_locations == CANONICAL_ACTIVE_LOCATIONS, fp.active_locations
    assert fp.single_fault_classes == CANONICAL_SINGLE_FAULT_CLASSES, fp.single_fault_classes
    assert fp.two_fault_class_pairs == CANONICAL_TWO_FAULT_CLASS_PAIRS, fp.two_fault_class_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hirano-src",
        type=Path,
        required=True,
        help="Path to msc-ls/src from branch experiment-2025-sep",
    )
    ap.add_argument("--channel", choices=["X", "Z", "both"], default="both")
    ap.add_argument("--write-stim-dir", type=Path)
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    channels: list[Channel] = ["X", "Z"] if args.channel == "both" else [args.channel]
    out: list[dict[str, Any]] = []
    if args.write_stim_dir:
        args.write_stim_dir.mkdir(parents=True, exist_ok=True)

    for ch in channels:
        circuit, wrapper = build_public_zxz(args.hirano_src, channel=ch)
        fp = fingerprint(circuit, wrapper, channel=ch)
        if not args.no_assert:
            assert_canonical(fp)
        out.append(asdict(fp))
        if args.write_stim_dir:
            (args.write_stim_dir / f"hirano_zxz_d5_{ch}.stim").write_text(
                str(circuit), encoding="utf-8"
            )

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
