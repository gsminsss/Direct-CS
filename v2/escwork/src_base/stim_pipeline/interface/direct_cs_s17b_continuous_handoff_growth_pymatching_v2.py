#!/usr/bin/env python3
"""S17-B: continuous-decoder Hirano d=5 -> Sahay d=13 Stim/PyMatching audit.

Why S17-B exists
================
S17-A deliberately asked a very strong *closed handoff* question: can the
logical Pauli frame be inferred at a live d=5 boundary using only detector bits
available before an ideal destructive probe?  The user-run S17-A result showed
that this is not possible even after a third d=5 syndrome round.  This is the
standard terminal-boundary effect of a live QEC state: a fault late in the last
round can leave a correctable data error that is only revealed by later QEC.

Therefore S17-B changes the boundary semantics instead of adding more finite
'dead-end' rounds.  The decoder state is carried forward through the *actual
next Clifford operation*, namely the Sahay d=5->7->13 growth.  The final d=13
state is then closed by an IDEAL scoring QEC/readout boundary.  The ideal close
is not a physical factory location and is used only to define the logical
observable for simulation, exactly as in standard finite-window surface-code
memory experiments.

Circuit per logical output patch
================================
  Hirano ZXZ escape
    + 2 faultable d=5 surface syndrome rounds       1028 active locations
  verified d=5 code isomorphism
  Sahay d=5 -> d=7 growth                            971 active locations
  Sahay d=7 -> d=13 unitary growth                   408 active locations
  ------------------------------------------------ ------------------------
  faultable Clifford interface total                2407 active locations
  + ideal d=13 syndrome close                       scoring only
  + ideal destructive logical X or Z close          truth scoring only

The 25 old d=5 Sahay data qubits are relabelled onto the 25 Hirano output data
qubits using the S16-certified reflection

    normalized Hirano (i,j) -> Sahay (i,4-j).

All other Sahay data/ancilla qubits are allocated fresh physical ids.

Noise convention
================
The faultable part uses the same project active-location stochastic-Pauli
convention:
  * Hirano part: existing zxz_online_handoff_audit instrumentation.
  * Sahay H: DEPOLARIZE1(p) after the H location.
  * Sahay CX: DEPOLARIZE2(p) after the CX location.
  * Sahay R: X_ERROR(p) after reset.
  * Sahay M: X_ERROR(p) immediately before Z-basis measurement.
  * source-file idle DEPOLARIZE1 terms are NOT imported.
  * no idle, leakage, coherent/calibration, or multi-location correlated noise.

Decoder interpretation
======================
Two complementary stabilizer reference circuits are built:
  channel X: initial logical |+>, final logical-X destructive readout; detects
             a residual logical-Z component.
  channel Z: initial logical |0>, final logical-Z destructive readout; detects
             a residual logical-X component.

The decoder detector model contains only information available before destructive
readout: Hirano live history + Sahay growth detectors + an ideal final d=13
syndrome close.  The Sahay code-deformation boundary contains gauge detectors: some
first growth-round check outcomes are intrinsically random because newly-added data
qubits are reset in a basis that anti-commutes with those checks.  Therefore Stim DEM
extraction uses allow_gauge_detectors=True, which represents these gauge freedoms by
50% gauge mechanisms instead of treating them as invalid detectors.  The final destructive X/Z data measurement contributes ONLY the
truth OBSERVABLE and contributes no decoder DETECTOR bits.  This is NOT a claim
that the factory can know a Pauli frame at the earlier live d=5 boundary.  It is
a continuous-decoder / live-handoff claim with an explicit ideal-final-syndrome
boundary.

S16 remains the exact physical order<=2 composability certificate.  S17-B adds
an operational high-order PyMatching baseline for the Clifford handoff+growth
module.  It does not yet include the non-Clifford direct-CS core finite-p
trajectory; that is the next hybrid E2E step after this gate passes.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent if HERE.name == "interface" else HERE.parent
INTERFACE = ROOT / "stim_pipeline" / "interface"
ZXZ_DIR = ROOT / "stim_pipeline" / "zxz"
SRC_CURRENT = ROOT / "src_current"
for pth in reversed((INTERFACE, ZXZ_DIR, SRC_CURRENT, HERE)):
    if pth.exists() and str(pth) not in sys.path:
        sys.path.insert(0, str(pth))

try:
    import zxz_online_handoff_audit as s7b  # type: ignore
    import zxz_pymatching_audit as s5  # type: ignore
except Exception as ex:  # pragma: no cover
    raise RuntimeError(
        "Could not import existing Direct-CS ZXZ pipeline modules. Place this "
        "script in stim_pipeline/interface/ of the provided source tree."
    ) from ex

WAIT_ROUNDS = 2
EXPECTED_HIRANO_LOCATIONS = 1028
EXPECTED_HIRANO_LIVE_DETECTORS = 171
EXPECTED_G57_LOCATIONS = 971
EXPECTED_G713_LOCATIONS = 408
EXPECTED_GROWTH_LOCATIONS = EXPECTED_G57_LOCATIONS + EXPECTED_G713_LOCATIONS
EXPECTED_TOTAL_LOCATIONS = EXPECTED_HIRANO_LOCATIONS + EXPECTED_GROWTH_LOCATIONS
EXPECTED_GROWTH_DETECTORS = 131
EXPECTED_D13_CLOSE_DETECTORS = 168
EXPECTED_TERMINAL_BOUNDARY_DETECTORS = 0
EXPECTED_TOTAL_DETECTORS = (
    EXPECTED_HIRANO_LIVE_DETECTORS
    + EXPECTED_GROWTH_DETECTORS
    + EXPECTED_D13_CLOSE_DETECTORS
)
EXPECTED_OBSERVABLES = 1

G57_START, G57_END = 603, 832
G713_START, G713_END = 833, 855
CLOSE_START, CLOSE_END = 856, 1038

SOURCE_NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}
INT_RE = re.compile(r"^-?\d+$")
COORD_RE = re.compile(r"QUBIT_COORDS\(([^)]*)\)\s+(\d+)")

# Preserved S16 Hirano d=5 data patch.  The coordinates are re-verified at run time.
HIRANO_D5_DATA = [
    233, 248, 263, 278, 293,
    234, 249, 264, 279, 294,
    235, 250, 265, 280, 295,
    236, 251, 266, 281, 296,
    237, 252, 267, 282, 297,
]


def _version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def _bool_attr(x: Any, name: str) -> bool:
    v = getattr(x, name, False)
    return bool(v() if callable(v) else v)


def _target_value(t: Any) -> int:
    v = getattr(t, "value")
    return int(v() if callable(v) else v)


def _opname_text(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    return s.split()[0].split("(")[0].split("[")[0]


def _ints_after_op(s: str) -> list[int]:
    return [int(x) for x in s.split()[1:] if INT_RE.fullmatch(x)]


def _read_source(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < CLOSE_END:
        raise ValueError(f"Sahay source too short: {len(lines)} lines")
    return lines


def _source_geometry(lines: list[str]) -> dict[str, Any]:
    coords: dict[int, tuple[float, ...]] = {}
    for s in lines:
        m = COORD_RE.match(s.strip())
        if m:
            coords[int(m.group(2))] = tuple(map(float, m.group(1).split(",")))

    # Same preserved public-circuit geometry used by S15/S16.
    d7_data = set(range(0, 97, 2))
    g57_new = set(_ints_after_op(lines[G57_START - 1]))
    old_d5 = sorted(d7_data - g57_new)
    if len(old_d5) != 25:
        raise AssertionError(("old d5", len(old_d5), old_d5))

    stab_support: dict[int, set[int]] = {}
    for ln in (858, 860, 862, 864):
        s = lines[ln - 1]
        if _opname_text(s) != "CX":
            raise AssertionError((ln, s))
        q = _ints_after_op(s)
        if len(q) % 2:
            raise AssertionError((ln, len(q)))
        for a, b in zip(q[::2], q[1::2]):
            if a >= 300 and b < 300:
                anc, dat = a, b
            elif b >= 300 and a < 300:
                anc, dat = b, a
            else:
                raise AssertionError(("final support pair", ln, a, b))
            stab_support.setdefault(anc, set()).add(dat)

    x_stabs = {a: stab_support[a] for a in sorted(stab_support) if 300 <= a <= 383}
    z_stabs = {a: stab_support[a] for a in sorted(stab_support) if 384 <= a <= 467}
    if len(x_stabs) != 84 or len(z_stabs) != 84:
        raise AssertionError((len(x_stabs), len(z_stabs)))
    data: set[int] = set()
    for ss in stab_support.values():
        data |= ss
    if len(data) != 169:
        raise AssertionError(("d13 data", len(data)))
    xlog = sorted(q for q in data if abs(coords[q][0] - 0.0) < 1e-9)
    zlog = sorted(q for q in data if abs(coords[q][1] - 0.0) < 1e-9)
    if len(xlog) != 13 or len(zlog) != 13:
        raise AssertionError((len(xlog), len(zlog)))

    return {
        "coords": coords,
        "old_d5": old_d5,
        "data": sorted(data),
        "x_stabs": x_stabs,
        "z_stabs": z_stabs,
        "xlog": xlog,
        "zlog": zlog,
    }


def _hirano_coords(circuit: Any) -> dict[int, tuple[float, ...]]:
    out: dict[int, tuple[float, ...]] = {}
    for inst in circuit:
        if not hasattr(inst, "name") or inst.name != "QUBIT_COORDS":
            continue
        ts = inst.targets_copy()
        if len(ts) != 1 or not _bool_attr(ts[0], "is_qubit_target"):
            raise AssertionError(inst)
        out[_target_value(ts[0])] = tuple(float(x) for x in inst.gate_args_copy())
    return out


def _build_qmap(pre_clean: Any, geo: dict[str, Any]) -> tuple[dict[int, int], dict[str, Any]]:
    hc = _hirano_coords(pre_clean)
    missing = [q for q in HIRANO_D5_DATA if q not in hc]
    if missing:
        raise AssertionError(("missing Hirano d5 coordinates", missing))

    def hpos(q: int) -> tuple[int, int]:
        x, y = hc[q][:2]
        return int(round((x - 1.0) / 2.0)), int(round((y - 17.0) / 2.0))

    hpositions = {q: hpos(q) for q in HIRANO_D5_DATA}
    if set(hpositions.values()) != {(i, j) for i in range(5) for j in range(5)}:
        raise AssertionError(("Hirano d5 coordinate grid drift", hpositions))

    sc = geo["coords"]
    old = list(geo["old_d5"])
    spos = {q: (int(round(sc[q][0])), int(round(sc[q][1]))) for q in old}
    bypos = {v: q for q, v in spos.items()}
    if len(bypos) != 25:
        raise AssertionError("Sahay old d5 coordinates are not unique")

    # S16-certified reflection: Hirano (i,j) -> Sahay (i,4-j).
    h_to_s = {hq: bypos[(i, 4 - j)] for hq, (i, j) in hpositions.items()}
    s_to_h = {sq: hq for hq, sq in h_to_s.items()}
    if set(s_to_h) != set(old):
        raise AssertionError(("old d5 map incomplete", len(s_to_h)))

    # Determine all qubit ids touched by Sahay growth/closure from the source geometry.
    # Allocate everything except the old d5 data as fresh ids beyond the Hirano circuit.
    fresh = int(pre_clean.num_qubits) + 100
    qmap: dict[int, int] = dict(s_to_h)
    used_source: set[int] = set()
    for q in geo["data"]:
        used_source.add(int(q))
    used_source |= set(range(300, 468))
    # Growth also uses d7 ancillas/data not present in the final d13 support dictionaries.
    # Add every source qubit that has coordinates and is plausibly used in the relevant range.
    used_source |= {q for q in geo["coords"] if 0 <= q <= 467}
    for q in sorted(used_source):
        if q not in qmap:
            qmap[q] = fresh
            fresh += 1

    if len({qmap[q] for q in old}) != 25:
        raise AssertionError("old d5 mapped qubits collide")
    return qmap, {
        "mapping_rule": "normalized Hirano (i,j) -> Sahay (i,4-j)",
        "old_d5_pairs_sahay_to_hirano": {str(s): int(h) for s, h in sorted(s_to_h.items())},
        "fresh_global_id_start": int(pre_clean.num_qubits) + 100,
        "fresh_global_id_end_exclusive": fresh,
    }


def _clean_source_range(lines: list[str], start: int, end: int) -> Any:
    """Parse source range, remove source noise, and strip measurement flip args."""
    import stim

    out = stim.Circuit()
    for ln in range(start, end + 1):
        raw = lines[ln - 1].strip()
        if not raw:
            continue
        op = _opname_text(raw)
        if op in SOURCE_NOISE_OPS:
            continue
        one = stim.Circuit(raw)
        if len(one) != 1:
            raise AssertionError((ln, raw, len(one)))
        inst = one[0]
        args = list(inst.gate_args_copy())
        # Sahay embeds measurement flip probability as M(0.001); our project model
        # instruments measurement as an explicit active location instead.
        if inst.name in {"M", "MZ", "MX", "MY", "MR"}:
            args = []
        out.append(inst.name, inst.targets_copy(), args)
    return out


def _remap_circuit(circuit: Any, qmap: dict[int, int]) -> Any:
    import stim

    out = stim.Circuit()
    for inst in circuit:
        if not hasattr(inst, "name"):
            raise AssertionError("repeat blocks are not expected in Sahay fragment")
        ts = []
        for t in inst.targets_copy():
            if _bool_attr(t, "is_qubit_target"):
                q = _target_value(t)
                if q not in qmap:
                    raise AssertionError(("missing Sahay qubit map", q, inst))
                ts.append(qmap[q])
            else:
                ts.append(t)
        out.append(inst.name, ts, inst.gate_args_copy())
    return out


def _instrument_growth(clean: Any, p: float) -> tuple[Any, dict[str, int]]:
    """Instrument only active operations, exactly matching S15 971+408 convention."""
    import stim

    if not (0 < p < 1):
        raise ValueError(p)
    out = stim.Circuit()
    counts = {"R": 0, "H": 0, "CX": 0, "M": 0}
    for inst in clean:
        if not hasattr(inst, "name"):
            raise AssertionError("repeat block")
        name = inst.name
        ts = inst.targets_copy()
        if name == "R":
            q = [_target_value(t) for t in ts]
            out.append(inst)
            out.append("X_ERROR", q, p)
            counts["R"] += len(q)
        elif name == "H":
            q = [_target_value(t) for t in ts]
            out.append(inst)
            out.append("DEPOLARIZE1", q, p)
            counts["H"] += len(q)
        elif name in {"CX", "CNOT"}:
            q = [_target_value(t) for t in ts]
            if len(q) % 2:
                raise AssertionError((name, q))
            out.append(inst)
            out.append("DEPOLARIZE2", q, p)
            counts["CX"] += len(q) // 2
        elif name in {"M", "MZ"}:
            q = [_target_value(t) for t in ts]
            out.append("X_ERROR", q, p)
            out.append(inst)
            counts["M"] += len(q)
        elif name == "MX":
            q = [_target_value(t) for t in ts]
            out.append("Z_ERROR", q, p)
            out.append(inst)
            counts["M"] += len(q)
        elif name == "MY":
            q = [_target_value(t) for t in ts]
            out.append("X_ERROR", q, p)  # X anticommutes with Y; one flip class.
            out.append(inst)
            counts["M"] += len(q)
        else:
            out.append(inst)
    counts["total"] = counts["R"] + counts["H"] + counts["CX"] + counts["M"]
    return out, counts


def _append_terminal_boundary(circuit: Any, channel: str, geo: dict[str, Any], qmap: dict[int, int]) -> dict[str, Any]:
    """Append ideal destructive d13 X/Z boundary and stabilizer-closing detectors."""
    import stim

    if channel not in {"X", "Z"}:
        raise ValueError(channel)
    # The ideal Sahay close ends with one MR over source ancillas 300..467 and no
    # later measurements.  Therefore the final 168 records before terminal readout
    # are exactly these ancilla measurements in ascending source-id order.
    pre_terminal = int(circuit.num_measurements)
    if pre_terminal < 168:
        raise AssertionError(pre_terminal)
    mr_start = pre_terminal - 168

    data_src = list(geo["data"])
    data_global = [qmap[q] for q in data_src]
    circuit.append("TICK")
    terminal_start = int(circuit.num_measurements)
    circuit.append("MX" if channel == "X" else "M", data_global)
    current = int(circuit.num_measurements)
    if current - terminal_start != 169:
        raise AssertionError((current, terminal_start))
    data_mid = {q: terminal_start + i for i, q in enumerate(data_src)}

    def rec(abs_mid: int):
        off = int(abs_mid) - current
        if off >= 0:
            raise AssertionError((abs_mid, current))
        return stim.target_rec(off)

    # IMPORTANT: do NOT append stabilizer-closing DETECTORs that involve the
    # destructive data measurements.  Those 84 bits would be future information
    # unavailable for a live delivered state.  The preceding ideal d13 syndrome
    # close already supplied the 168 final X/Z stabilizer bits used by the decoder.
    logical = geo["xlog"] if channel == "X" else geo["zlog"]
    circuit.append("OBSERVABLE_INCLUDE", [rec(data_mid[q]) for q in logical], 0)
    return {
        "terminal_basis": channel,
        "terminal_data_measurements": 169,
        "terminal_boundary_detectors": 0,
        "logical_support_weight": len(logical),
        "ideal_terminal_scoring_only": True,
        "terminal_measurements_used_by_decoder": False,
    }


def _build_channel(
    hirano_src: Path,
    sahay_stim: Path,
    channel: str,
    p: float,
) -> tuple[Any, dict[str, Any]]:
    lines = _read_source(sahay_stim)
    geo = _source_geometry(lines)

    full, _wrapper, init_range, probe_start, online_ids, terminal_ids = s7b._build_with_wait_rounds(
        hirano_src, channel, WAIT_ROUNDS
    )
    pre_clean = full[:probe_start]
    if len(online_ids) != EXPECTED_HIRANO_LIVE_DETECTORS:
        raise AssertionError((len(online_ids), EXPECTED_HIRANO_LIVE_DETECTORS))
    if len(terminal_ids) != 12:
        raise AssertionError(("historical terminal ids", terminal_ids))
    if int(pre_clean.num_detectors) != EXPECTED_HIRANO_LIVE_DETECTORS:
        raise AssertionError((pre_clean.num_detectors, EXPECTED_HIRANO_LIVE_DETECTORS))

    noisy_pre, hloc, _noise_map = s7b._instrument_before_probe(
        pre_clean, init_range, len(pre_clean), p
    )
    if len(hloc) != EXPECTED_HIRANO_LOCATIONS:
        raise AssertionError((len(hloc), EXPECTED_HIRANO_LOCATIONS))

    qmap, map_meta = _build_qmap(pre_clean, geo)

    g57_clean_src = _clean_source_range(lines, G57_START, G57_END)
    g713_clean_src = _clean_source_range(lines, G713_START, G713_END)
    close_clean_src = _clean_source_range(lines, CLOSE_START, CLOSE_END)
    g57_clean = _remap_circuit(g57_clean_src, qmap)
    g713_clean = _remap_circuit(g713_clean_src, qmap)
    close_clean = _remap_circuit(close_clean_src, qmap)

    g57_noisy, c57 = _instrument_growth(g57_clean, p)
    g713_noisy, c713 = _instrument_growth(g713_clean, p)
    if c57["total"] != EXPECTED_G57_LOCATIONS:
        raise AssertionError(("g57 active locations", c57))
    if c713["total"] != EXPECTED_G713_LOCATIONS:
        raise AssertionError(("g713 active locations", c713))

    c = noisy_pre + g57_noisy + g713_noisy + close_clean
    before_terminal_det = int(c.num_detectors)
    expected_before_terminal = (
        EXPECTED_HIRANO_LIVE_DETECTORS + EXPECTED_GROWTH_DETECTORS + EXPECTED_D13_CLOSE_DETECTORS
    )
    if before_terminal_det != expected_before_terminal:
        raise AssertionError(("detectors before terminal", before_terminal_det, expected_before_terminal))
    if int(c.num_observables) != 0:
        raise AssertionError(("unexpected preterminal observable", c.num_observables))
    terminal_meta = _append_terminal_boundary(c, channel, geo, qmap)

    if int(c.num_detectors) != EXPECTED_TOTAL_DETECTORS:
        raise AssertionError((c.num_detectors, EXPECTED_TOTAL_DETECTORS))
    if int(c.num_observables) != EXPECTED_OBSERVABLES:
        raise AssertionError((c.num_observables, EXPECTED_OBSERVABLES))

    meta = {
        "channel": channel,
        "p": float(p),
        "wait_rounds": WAIT_ROUNDS,
        "hirano_live_detectors": EXPECTED_HIRANO_LIVE_DETECTORS,
        "growth_detectors": EXPECTED_GROWTH_DETECTORS,
        "ideal_d13_close_detectors": EXPECTED_D13_CLOSE_DETECTORS,
        "ideal_terminal_boundary_detectors": 0,
        "total_detectors": int(c.num_detectors),
        "observables": int(c.num_observables),
        "active_locations": {
            "hirano_escape_plus_two_d5_rounds": len(hloc),
            "sahay_d5_to_d7": c57["total"],
            "sahay_d7_to_d13": c713["total"],
            "total_faultable_clifford_interface": len(hloc) + c57["total"] + c713["total"],
            "ideal_d13_close": 0,
            "ideal_terminal_boundary": 0,
        },
        "growth_location_types": {"d5_to_d7": c57, "d7_to_d13": c713},
        "mapping": map_meta,
        "terminal": terminal_meta,
        "boundary_semantics": "continuous decoder through growth; ideal d13 close/readout used only for scoring",
    }
    if meta["active_locations"]["total_faultable_clifford_interface"] != EXPECTED_TOTAL_LOCATIONS:
        raise AssertionError(meta["active_locations"])
    return c, meta


def _dem_signature_audit(raw_dem: Any, matching: Any, n_detectors: int, max_examples: int = 20) -> dict[str, Any]:
    """Audit physical first-order DEM signatures separately from gauge freedoms.

    With allow_gauge_detectors=True Stim inserts exact p=0.5 mechanisms representing
    intrinsically random gauge-detector degrees of freedom at code-deformation
    boundaries.  These are *not* physical fault mechanisms and must not be counted as
    first-order active-location faults.  Physical mechanisms are audited separately.
    """
    rows: list[tuple[list[int], int, float | None, bool]] = []
    for inst in raw_dem.flattened():
        if getattr(inst, "type", None) != "error":
            continue
        dets: list[int] = []
        obs = 0
        for t in inst.targets_copy():
            if _bool_attr(t, "is_relative_detector_id"):
                dets.append(_target_value(t))
            elif _bool_attr(t, "is_logical_observable_id"):
                if _target_value(t) != 0:
                    raise AssertionError(t)
                obs ^= 1
            elif _bool_attr(t, "is_separator"):
                raise AssertionError("raw decompose_errors=False DEM unexpectedly contains separator")
            else:
                raise AssertionError(t)
        args = list(inst.args_copy()) if hasattr(inst, "args_copy") else []
        prob = float(args[0]) if args else None
        is_gauge = bool(prob is not None and abs(prob - 0.5) <= 1e-12)
        rows.append((dets, obs, prob, is_gauge))

    mat = np.zeros((len(rows), n_detectors), dtype=np.uint8)
    actual = np.zeros((len(rows), 1), dtype=np.uint8)
    gauge = np.zeros(len(rows), dtype=bool)
    for i, (ds, o, _p, is_gauge) in enumerate(rows):
        for d in ds:
            if not (0 <= d < n_detectors):
                raise AssertionError((d, n_detectors))
            mat[i, d] ^= 1
        actual[i, 0] = o
        gauge[i] = is_gauge
    pred = np.asarray(matching.decode_batch(mat), dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.shape != actual.shape:
        raise AssertionError((pred.shape, actual.shape))
    bad = np.any(pred != actual, axis=1)
    physical = ~gauge
    bad_physical = np.logical_and(bad, physical)
    bad_gauge = np.logical_and(bad, gauge)
    zero_nontrivial = np.logical_and(~np.any(mat.astype(bool), axis=1), actual[:, 0].astype(bool))
    zero_nontrivial_physical = np.logical_and(zero_nontrivial, physical)

    examples = []
    for i in np.flatnonzero(bad)[:max_examples]:
        ds, o, pr, is_gauge = rows[int(i)]
        examples.append({
            "raw_dem_error_index": int(i),
            "detectors": [int(x) for x in ds],
            "detector_weight": len(ds),
            "actual_observable": int(o),
            "predicted_observable": int(pred[int(i), 0]),
            "probability": pr,
            "is_gauge_p_half_mechanism": bool(is_gauge),
        })
    return {
        "raw_dem_error_instructions": len(rows),
        "gauge_p_half_error_instructions": int(np.sum(gauge)),
        "physical_error_instructions": int(np.sum(physical)),
        "pymatching_physical_signature_failures": int(np.sum(bad_physical)),
        "pymatching_physical_signature_successes": int(np.sum(physical) - np.sum(bad_physical)),
        "all_first_order_physical_dem_signatures_corrected": bool(np.sum(bad_physical) == 0),
        "pymatching_gauge_signature_failures": int(np.sum(bad_gauge)),
        "zero_detector_nontrivial_observable_instructions_physical": int(np.sum(zero_nontrivial_physical)),
        "zero_detector_nontrivial_observable_instructions_all": int(np.sum(zero_nontrivial)),
        "failure_examples": examples,
        "scope_note": "Stim p=0.5 gauge mechanisms are excluded from the active-location first-order fault count; S16 remains the exact physical <=2 certificate.",
    }


def _make_matching(circuit: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    import pymatching

    # The first stabilizer measurements introduced by d=5->7 code deformation
    # contain gauge degrees of freedom.  Stim's documented treatment is to allow
    # these non-deterministic detectors; it inserts p=0.5 gauge mechanisms and
    # Gaussian-eliminates the redundant gauge freedom.  Logical observables remain
    # required to be deterministic.
    raw = circuit.detector_error_model(
        decompose_errors=False,
        allow_gauge_detectors=True,
    )
    raw_stats = s5.audit_dem_structure(raw)
    graph = circuit.detector_error_model(
        decompose_errors=True,
        allow_gauge_detectors=True,
    )
    graph_stats = s5.audit_dem_structure(graph)
    if graph_stats["bad_components"] != 0:
        raise AssertionError(
            f"Stim decomposition left {graph_stats['bad_components']} non-graphlike components"
        )
    m = pymatching.Matching.from_detector_error_model(graph)
    m.ensure_num_fault_ids(EXPECTED_OBSERVABLES)
    if int(m.num_detectors) != EXPECTED_TOTAL_DETECTORS:
        raise AssertionError((m.num_detectors, EXPECTED_TOTAL_DETECTORS))
    return raw, graph, m, {"raw_dem": raw_stats, "graphlike_dem": graph_stats,
                            "allow_gauge_detectors": True,
                            "matching_num_detectors": int(m.num_detectors),
                            "matching_num_nodes": int(m.num_nodes),
                            "matching_num_edges": int(m.num_edges),
                            "matching_num_fault_ids": int(m.num_fault_ids)}


def _run_mc(circuit: Any, matching: Any, shots: int, seed: int) -> dict[str, Any]:
    sampler = circuit.compile_detector_sampler(seed=int(seed))
    dets, obs = sampler.sample(shots=int(shots), separate_observables=True)
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    if obs.ndim == 1:
        obs = obs[:, None]
    if dets.shape != (shots, EXPECTED_TOTAL_DETECTORS):
        raise AssertionError(dets.shape)
    if obs.shape != (shots, EXPECTED_OBSERVABLES):
        raise AssertionError(obs.shape)
    pred = np.asarray(matching.decode_batch(dets), dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.shape != obs.shape:
        raise AssertionError((pred.shape, obs.shape))
    fail = np.any(pred != obs, axis=1)
    k = int(np.sum(fail))
    return {
        "shots": int(shots),
        "logical_failures": k,
        "logical_failure_rate": k / shots if shots else None,
        "wilson95": _wilson(k, shots),
        "mean_detector_weight": float(np.mean(np.sum(dets, axis=1))) if shots else None,
        "zero_detector_fraction": float(np.mean(~np.any(dets.astype(bool), axis=1))) if shots else None,
    }


def _channel_precheck(hirano_src: Path, sahay_stim: Path, channel: str, p: float, out_dir: Path) -> dict[str, Any]:
    c, meta = _build_channel(hirano_src, sahay_stim, channel, p)
    raw, graph, matching, dem_meta = _make_matching(c)
    sig = _dem_signature_audit(raw, matching, EXPECTED_TOTAL_DETECTORS)
    (out_dir / f"s17b_{channel}_signature_circuit.stim").write_text(str(c), encoding="utf-8")
    (out_dir / f"s17b_{channel}_raw.dem").write_text(str(raw), encoding="utf-8")
    (out_dir / f"s17b_{channel}_graphlike.dem").write_text(str(graph), encoding="utf-8")
    result = {"meta": meta, "dem_matching": dem_meta, "first_order_signature_gate": sig}
    _dump(out_dir / f"s17b_{channel}_signature_summary.json", result)
    return result


def _static_source_counts(lines: list[str], start: int, end: int) -> dict[str, int]:
    c = {"R": 0, "H": 0, "CX": 0, "M": 0}
    for ln in range(start, end + 1):
        s = lines[ln - 1].strip()
        if not s:
            continue
        op = _opname_text(s)
        if op == "R":
            c["R"] += len(_ints_after_op(s))
        elif op == "H":
            c["H"] += len(_ints_after_op(s))
        elif op in {"CX", "CNOT"}:
            c["CX"] += len(_ints_after_op(s)) // 2
        elif op in {"M", "MZ", "MX", "MY"}:
            c["M"] += len(_ints_after_op(s))
    c["total"] = sum(c.values())
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--sahay-stim", type=Path, default=ROOT / "reference" / "fd3_baseline_ckt.stim")
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--p-values", type=float, nargs="+", default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=[20000, 30000, 50000, 100000, 50000])
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE / "s17b_results")
    args = ap.parse_args()
    if len(args.p_values) != len(args.shots_per_p):
        raise SystemExit("--p-values and --shots-per-p must have equal length")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hirano_src = args.hirano_src.resolve()
    sahay_stim = args.sahay_stim.resolve()
    if not sahay_stim.is_file():
        raise SystemExit(f"Sahay Stim source not found: {sahay_stim}")

    lines = _read_source(sahay_stim)
    static = {
        "d5_to_d7": _static_source_counts(lines, G57_START, G57_END),
        "d7_to_d13": _static_source_counts(lines, G713_START, G713_END),
    }
    if static["d5_to_d7"]["total"] != EXPECTED_G57_LOCATIONS:
        raise AssertionError(static["d5_to_d7"])
    if static["d7_to_d13"]["total"] != EXPECTED_G713_LOCATIONS:
        raise AssertionError(static["d7_to_d13"])

    t0 = time.time()
    print("S17-B: building X continuous-decoder circuit...", flush=True)
    xpre = _channel_precheck(hirano_src, sahay_stim, "X", float(args.signature_p), args.out_dir)
    print("S17-B: building Z continuous-decoder circuit...", flush=True)
    zpre = _channel_precheck(hirano_src, sahay_stim, "Z", float(args.signature_p), args.out_dir)

    precheck_pass = bool(
        xpre["first_order_signature_gate"]["all_first_order_physical_dem_signatures_corrected"]
        and zpre["first_order_signature_gate"]["all_first_order_physical_dem_signatures_corrected"]
        and xpre["first_order_signature_gate"]["zero_detector_nontrivial_observable_instructions_physical"] == 0
        and zpre["first_order_signature_gate"]["zero_detector_nontrivial_observable_instructions_physical"] == 0
    )

    rows = []
    if not args.skip_mc:
        for i, (p, shots) in enumerate(zip(args.p_values, args.shots_per_p)):
            print(f"S17-B p={p:g}: X circuit/MC...", flush=True)
            xc, _ = _build_channel(hirano_src, sahay_stim, "X", float(p))
            _xr, _xg, xm, _ = _make_matching(xc)
            xr = _run_mc(xc, xm, int(shots), int(args.seed) + 1000 * i)
            print(f"S17-B p={p:g}: Z circuit/MC...", flush=True)
            zc, _ = _build_channel(hirano_src, sahay_stim, "Z", float(p))
            _zr, _zg, zm, _ = _make_matching(zc)
            zr = _run_mc(zc, zm, int(shots), int(args.seed) + 1000 * i + 1)
            px = float(xr["logical_failure_rate"])
            pz = float(zr["logical_failure_rate"])
            lo = max(px, pz)
            hi = min(1.0, px + pz)
            row = {
                "p": float(p),
                "shots_per_channel": int(shots),
                "X_channel_logical_Z_component": xr,
                "Z_channel_logical_X_component": zr,
                "single_patch_any_pauli_lower_max_component": lo,
                "single_patch_any_pauli_upper_union_bound": hi,
                "two_patch_any_pauli_lower_independent_from_lower": 1 - (1 - lo) ** 2,
                "two_patch_any_pauli_upper_independent_from_upper": 1 - (1 - hi) ** 2,
            }
            rows.append(row)
            print(
                f"  X={px:.6g} Z={pz:.6g} any-patch bracket=[{lo:.6g},{hi:.6g}]",
                flush=True,
            )
        _dump(args.out_dir / "s17b_sweep.json", rows)

    summary = {
        "stage": "S17-B continuous-decoder Hirano d5 -> Sahay d13 PyMatching",
        "stim_version": _version("stim"),
        "pymatching_version": _version("pymatching"),
        "numpy_version": np.__version__,
        "source_hirano": str(hirano_src),
        "source_sahay_stim": str(sahay_stim),
        "signature_p": float(args.signature_p),
        "static_growth_location_counts": static,
        "faultable_locations_per_patch": EXPECTED_TOTAL_LOCATIONS,
        "detectors_per_channel_with_ideal_close": EXPECTED_TOTAL_DETECTORS,
        "ideal_close_locations_counted_in_resource": 0,
        "continuous_decoder_semantics": True,
        "allow_gauge_detectors": True,
        "gauge_detector_semantics": "Stim p=0.5 gauge mechanisms at code-deformation boundary are decoder degrees of freedom, not active-location physical faults",
        "s17a_closed_live195_frame_gate_is_not_reused": True,
        "s16_exact_physical_order_le2_certificate_remains_required": True,
        "first_order_pymatching_signature_gate_pass": precheck_pass,
        "X_precheck": xpre,
        "Z_precheck": zpre,
        "mc_was_run": not args.skip_mc,
        "p_values": [float(x) for x in args.p_values],
        "shots_per_p": [int(x) for x in args.shots_per_p],
        "seed": int(args.seed),
        "sweep": rows,
        "important_scope": [
            "The ideal d13 syndrome close is an explicit scoring/decoder boundary, not a counted factory operation; a later stage must physicalize it if certification-before-delivery is required.",
            "The destructive terminal X/Z readout supplies truth OBSERVABLE only; none of its measurement outcomes are exposed to PyMatching as detector bits.",
            "This stage measures the Clifford handoff+growth contribution; it does not yet sample the non-Clifford direct-CS core trajectory.",
            "Separate X/Z reference circuits bound, but do not reconstruct, the joint X/Z/Y logical-Pauli distribution.",
            "Sahay d5->d7 contains gauge detectors at the code-deformation boundary. DEM extraction therefore uses allow_gauge_detectors=True; p=0.5 gauge mechanisms are excluded from physical first-order signature counts.",
            "Do not quote the S16 9037.8 locations/output yield proxy as an error-matched cost until the later hybrid E2E stage combines this decoder with S12 core trajectories.",
        ],
        "elapsed_seconds": time.time() - t0,
    }
    _dump(args.out_dir / "s17b_summary.json", summary)
    print(json.dumps({
        "first_order_pymatching_signature_gate_pass": precheck_pass,
        "faultable_locations_per_patch": EXPECTED_TOTAL_LOCATIONS,
        "detectors_per_channel_with_ideal_close": EXPECTED_TOTAL_DETECTORS,
        "mc_was_run": not args.skip_mc,
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)

    if not precheck_pass:
        print(
            "S17-B PRECHECK FAIL: continuous stitched circuit is valid, but the current "
            "graphlike PyMatching baseline does not correct every first-order raw DEM signature. "
            "Do not run/interpret the full MC as the final decoder until failure witnesses are audited.",
            flush=True,
        )
    else:
        print("S17-B PRECHECK PASS: run finite-p sweep / proceed to hybrid S17-C after reviewing MC.", flush=True)


if __name__ == "__main__":
    main()
