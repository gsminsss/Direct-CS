#!/usr/bin/env python3
"""S21-OPT3B: live Hirano-d5 -> direct-d13 splice + exact <=2 structural certificate.

Scientific question
===================
Can we replace the current
    Hirano d5 -> Sahay d5->d7 -> Sahay d7->d13 -> noisy first d13 close
interface by a *direct* live d5->d13 expansion, while preserving the low-order
postselected fault-safety contract?

This script does NOT run finite-p Monte Carlo.  It first constructs the live
splice and then performs an exact all-zero-record <=2 audit using concrete
Stim fault mechanisms.

Key design choices
==================
* UPWARD expansion only.  Its initial d=5 data coordinates are exactly the
  Hirano live d=5 patch coordinates:
      x = 1,3,5,7,9 ; y = 17,19,21,23,25
* The public expansion's artificial d=5 reset/two-round initialization is
  BYPASSED.  Instead, the actual Hirano `surface_syndrome_measurements`
  objects (including their last_measurement history) are handed directly to
  `SurfaceCodePatch._perform_code_expansion()`.
* One direct d=5->d=13 syndrome round is faultable under the project's active
  stochastic-Pauli location model.
* One additional stationary d=13 syndrome round is ideal scoring/future
  syndrome only, exactly analogous in role to the ideal follow-up in S18-M.
* Terminal logical scoring is ideal. The original Hirano frame is retained,
  and any additional direct-expansion byproduct parity is derived only from
  measurements in the faultable expansion round, then exact-certified by Stim.
* No idle noise is introduced.
* Structural acceptance for this theorem gate is all operational detectors=0.

The exact gate checks:
  1. clean X/Z splice is deterministic and has identical detector definitions;
  2. X/Z stochastic schedules are identical;
  3. no single physical fault has zero detector record + non-I final Pauli;
  4. no two DISTINCT physical locations can cancel to zero detector record
     while leaving non-I final Pauli;
  5. canonical incoming Steane weight-1/2 sectors are safe alone;
  6. incoming canonical weight-1 sector + one physical fault is safe.

Only if all gates PASS should a finite-p practical decoder be built in OPT3C.

Scope warning
=============
This proves a conservative all-zero-detector structural contract for the
spliced interface under the same active-location stochastic-Pauli model.
It is not yet a paper-final practical-acceptance resource result.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAULI_NAMES = ("I", "X", "Z", "Y")
PAULI1 = ("X", "Y", "Z")
EXPECTED_HIRANO_LIVE_DETECTORS = 171
EXPECTED_HIRANO_LOCATIONS = 1028
WAIT_ROUNDS = 2
D_ESC = 5
D_FINAL = 13

def _make_mapping(util):
    m = util.QubitMapping(30, 30)
    if D_ESC == 5:
        return m
    # Append extra coordinates above the grid (negative y) without changing existing ids.
    nid = len(m.mapping)
    ylo = 17 - 2 * (D_FINAL - D_ESC) - 2
    for x in range(30):
        for y in range(ylo, 0):
            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 != 0):
                m.mapping.append((nid, (x, y))); nid += 1
    return m


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return mod


def _find(root: Path, names: list[str], *, want_dir: bool = False) -> Path:
    for name in names:
        p = (root / name).resolve()
        if p.exists() and (p.is_dir() if want_dir else p.is_file()):
            return p
    basenames = {Path(x).name for x in names}
    hits = []
    for p in root.rglob("*"):
        if p.name not in basenames:
            continue
        if want_dir and not p.is_dir():
            continue
        if not want_dir and not p.is_file():
            continue
        hits.append(p)
    if not hits:
        raise FileNotFoundError(names)
    hits.sort(key=lambda p: (len(p.parts), len(str(p))))
    return hits[0].resolve()


def _bool_attr(obj: Any, name: str) -> bool:
    v = getattr(obj, name)
    return bool(v() if callable(v) else v)


def _target_value(t: Any) -> int:
    if hasattr(t, "val"):
        v = getattr(t, "val")
        return int(v() if callable(v) else v)
    if hasattr(t, "value"):
        v = getattr(t, "value")
        return int(v() if callable(v) else v)
    s = str(t)
    if s and s[0] in "DL":
        return int(s[1:])
    return int(s)


def _detector_lines(c: Any) -> list[str]:
    return [x.strip() for x in str(c).splitlines() if x.strip().startswith("DETECTOR")]


def _count_ticks(c: Any) -> int:
    return sum(1 for inst in c if hasattr(inst, "name") and inst.name == "TICK")


def _build_hirano_boundary_runner(s7b: Any, hirano_src: Path, channel: str):
    """Run canonical Hirano path and stop *before* destructive d5 scoring probe.

    Unlike s7b._build_with_wait_rounds, this helper retains the runner and its
    live SurfaceSyndromeMeasurement objects.
    """
    lsed, util = s7b.s1._load_hirano(hirano_src)
    steane_code = importlib.import_module("steane_code")
    mapping = _make_mapping(util)
    initial_value = lsed.InitialValue.Plus if channel == "X" else lsed.InitialValue.Zero
    runner = lsed.SteanePlusSurfaceCode(
        mapping, D_ESC, initial_value, lsed.SteaneSyndromeExtractionPattern.ZXZ, 0.0, True
    )

    init_name = (
        "perform_perfect_steane_plus_initialization"
        if channel == "X"
        else "perform_perfect_steane_zero_initialization"
    )
    init_original = getattr(steane_code, init_name)
    meth_name = "_perform_destructive_x_measurement" if channel == "X" else "_perform_destructive_z_measurement"
    probe_original = getattr(runner, meth_name)
    rec: dict[str, Any] = {}

    class _StopAtLiveBoundary(Exception):
        pass

    def init_wrapped(c, m):
        a = len(c)
        init_original(c, m)
        rec["init_range"] = (a, len(c))

    def stop_probe():
        rec["pre_wait_len"] = len(runner.circuit.circuit)
        rec["pre_wait_meas"] = int(runner.circuit.circuit.num_measurements)
        # Match S18/S17: append two non-destructive full d5 syndrome rounds.
        for _ in range(WAIT_ROUNDS):
            for _stage in range(6):
                for sm in runner.surface_syndrome_measurements.values():
                    sm.run()
                runner.circuit.place_tick()
        for sm in runner.surface_syndrome_measurements.values():
            if not sm.is_complete():
                raise AssertionError("Hirano boundary not on a syndrome-round boundary")
            if sm.last_measurement is None:
                raise AssertionError(("Hirano syndrome lacks history", sm.ancilla_position))
        rec["probe_start"] = len(runner.circuit.circuit)
        raise _StopAtLiveBoundary()

    setattr(steane_code, init_name, init_wrapped)
    setattr(runner, meth_name, stop_probe)
    try:
        runner.run()
        raise AssertionError("expected live-boundary stop")
    except _StopAtLiveBoundary:
        pass
    finally:
        setattr(steane_code, init_name, init_original)
        setattr(runner, meth_name, probe_original)

    if "init_range" not in rec or "probe_start" not in rec:
        raise AssertionError(rec)

    # IMPORTANT: take an immutable snapshot of the live Hirano prefix.
    # `runner.circuit.circuit` is mutated in-place when the direct expansion is
    # appended later.  Returning the same object here would make `pre` grow as
    # well, causing the expansion/follow-up gates to be instrumented a second
    # time as if they belonged to the Hirano prefix.
    pre = runner.circuit.circuit[:]
    if len(pre) != int(rec["probe_start"]):
        raise AssertionError((len(pre), rec["probe_start"]))
    if int(pre.num_detectors) != EXPECTED_HIRANO_LIVE_DETECTORS:
        raise AssertionError(("Hirano live detectors", pre.num_detectors))

    if D_ESC == 5:
        full, _wrapper, init_range2, probe_start2, online_ids, terminal_ids = s7b._build_with_wait_rounds(
            hirano_src, channel, WAIT_ROUNDS
        )
        canonical_pre = full[:probe_start2]
        if str(canonical_pre) != str(pre):
            raise RuntimeError("retained-runner prefix differs from canonical S17/S18 Hirano prefix")
        if tuple(rec["init_range"]) != tuple(init_range2):
            raise RuntimeError(("init range drift", rec["init_range"], init_range2))
        if len(online_ids) != EXPECTED_HIRANO_LIVE_DETECTORS or len(terminal_ids) != 12:
            raise AssertionError((len(online_ids), len(terminal_ids)))
    else:
        r2 = lsed.SteanePlusSurfaceCode(
            _make_mapping(util), D_ESC, initial_value, lsed.SteaneSyndromeExtractionPattern.ZXZ, 0.0, True
        )
        r2.run()
        full = r2.circuit.circuit
        cut = int(rec["pre_wait_len"])
        if str(full[:cut]) != str(pre[:cut]):
            raise RuntimeError("full independent run prefix differs from stopped live prefix")

    # Original measurement-dependent logical frame, excluding terminal d5 probe.
    _d, obs_deps = s7b._annotation_dependencies(full)
    if len(obs_deps) != 1:
        raise AssertionError(("Hirano observable count", len(obs_deps)))
    pre_m = int(pre.num_measurements) if D_ESC == 5 else int(rec["pre_wait_meas"])
    obs_all = [int(x) for x in obs_deps[0]]
    hirano_pre = [m for m in obs_all if m < pre_m]
    hirano_term = [m for m in obs_all if m >= pre_m]
    if not hirano_pre or len(hirano_term) != D_ESC:
        raise AssertionError((channel, hirano_pre, hirano_term))

    return runner, pre, tuple(rec["init_range"]), hirano_pre


def _syndrome_signature(sm: Any) -> tuple[str, str]:
    cls = type(sm).__name__
    patt = getattr(sm.pattern, "name", str(sm.pattern))
    return cls, patt


def _validate_upward_splice_contract(sce: Any, runner: Any) -> dict[str, Any]:
    """Compare live Hirano d5 syndrome layout with expansion's expected d5 layout."""
    util = importlib.import_module("util")
    mapping = _make_mapping(util)
    refc = util.Circuit(mapping, 0.0)
    ref = sce.SurfaceCodePatch(refc, D_ESC, D_FINAL, 1, sce.ExpansionPattern.UPWARD)
    ref.offset = (1, 17)
    ref._setup_initial_state()

    live = runner.surface_syndrome_measurements
    if len(live) != D_ESC * D_ESC - 1 or len(ref.syndrome_measurements) != D_ESC * D_ESC - 1:
        raise AssertionError((len(live), len(ref.syndrome_measurements)))
    if set(live) != set(ref.syndrome_measurements):
        raise RuntimeError(
            "UPWARD expansion d5 syndrome coordinates do not exactly match live Hirano coordinates"
        )

    mismatches = []
    for pos in sorted(live):
        a = _syndrome_signature(live[pos])
        b = _syndrome_signature(ref.syndrome_measurements[pos])
        if a != b:
            mismatches.append({"position": list(pos), "hirano": a, "expansion": b})
    if mismatches:
        raise RuntimeError(("syndrome type/pattern mismatch", mismatches[:20]))

    # Verify the 25 data coordinates themselves are the exact live d5 patch.
    expected_data = {(1 + 2*j, 17 + 2*i) for i in range(D_ESC) for j in range(D_ESC)}
    qcoords = {}
    for q, pos in runner.mapping.mapping:
        qcoords[tuple(pos)] = int(q)
    missing = sorted(expected_data - set(qcoords))
    if missing:
        raise RuntimeError(("mapping lacks d5 data coordinates", missing))

    return {
        "upward_initial_offset": [1, 17],
        "live_syndrome_count": len(live),
        "syndrome_coordinate_sets_identical": True,
        "syndrome_class_and_pattern_identical": True,
        "expected_live_data_coordinates": D_ESC * D_ESC,
        "mapping_contains_all_live_data_coordinates": True,
    }


def _solve_gf2_equations(rows: list[int], nvars: int) -> int | None:
    """Solve A x=b where each packed row stores lhs bits then rhs at bit nvars."""
    if nvars == 0:
        return 0 if all(((r >> nvars) & 1) == 0 for r in rows) else None
    lhs_mask = (1 << nvars) - 1
    basis: dict[int, int] = {}
    for packed in rows:
        eq = int(packed)
        while True:
            lhs = eq & lhs_mask
            if lhs == 0:
                if ((eq >> nvars) & 1) != 0:
                    return None
                break
            low = lhs & -lhs
            pivot = low.bit_length() - 1
            if pivot in basis:
                eq ^= basis[pivot]
            else:
                basis[pivot] = eq
                break

    x = 0
    for pivot in sorted(basis, reverse=True):
        eq = basis[pivot]
        rhs = (eq >> nvars) & 1
        other = (eq & lhs_mask) & ~(1 << pivot)
        parity = (other & x).bit_count() & 1
        if rhs ^ parity:
            x |= 1 << pivot
    return x


def _append_observable_from_absolute_measurements(clean_unscored: Any, ids: list[int]):
    """Append OBSERVABLE_INCLUDE using absolute measurement ids, mod-2 deduped."""
    import stim
    out = clean_unscored[:]
    n = int(out.num_measurements)
    parity: dict[int, int] = {}
    for mid in ids:
        i = int(mid)
        if not (0 <= i < n):
            raise AssertionError(("measurement id out of range", i, n))
        parity[i] = parity.get(i, 0) ^ 1
    kept = sorted(i for i, bit in parity.items() if bit)
    targets = [stim.target_rec(i - n) for i in kept]
    out.append("OBSERVABLE_INCLUDE", targets, 0)
    return out, kept


def _solve_expansion_byproduct_frame(
    clean_unscored: Any,
    *,
    fixed_hirano_ids: list[int],
    variable_expansion_ids: list[int],
    final_measurement_id: int,
    channel: str,
):
    """Find an operational expansion-measurement parity transporting the old frame.

    The old Hirano frame is held fixed. Only measurements generated by the
    *faultable direct-expansion round* are allowed as new byproduct-frame bits.
    Candidate relations are learned from noiseless stabilizer samples and then
    accepted only if Stim's exact detector_error_model proves the resulting
    observable deterministic.
    """
    nmeas = int(clean_unscored.num_measurements)
    fixed = [int(x) for x in fixed_hirano_ids]
    variables = [int(x) for x in variable_expansion_ids]
    final_id = int(final_measurement_id)
    if len(set(variables)) != len(variables):
        raise AssertionError("duplicate variable expansion measurement ids")
    if any(v in fixed for v in variables):
        raise AssertionError("expansion-frame variables overlap fixed Hirano frame")
    if final_id in fixed or final_id in variables:
        raise AssertionError("final logical measurement overlaps frame inputs")
    if not (0 <= final_id < nmeas):
        raise AssertionError((final_id, nmeas))

    sampler = clean_unscored.compile_sampler(seed=2026090403)
    nvars = len(variables)
    attempts = [
        max(512, 4 * max(1, nvars) + 128),
        max(1024, 8 * max(1, nvars) + 256),
        max(2048, 16 * max(1, nvars) + 512),
    ]
    last_error = None
    for shots in attempts:
        samples = sampler.sample(shots)
        rows: list[int] = []
        for rec in samples:
            mask = 0
            for j, mid in enumerate(variables):
                if bool(rec[mid]):
                    mask |= 1 << j
            rhs = int(bool(rec[final_id]))
            for mid in fixed:
                rhs ^= int(bool(rec[mid]))
            rows.append(mask | (rhs << nvars))

        sol = _solve_gf2_equations(rows, nvars)
        if sol is None:
            last_error = f"inconsistent sampled GF(2) system at {shots} shots"
            continue

        selected = [variables[j] for j in range(nvars) if (sol >> j) & 1]
        scored, obs_ids = _append_observable_from_absolute_measurements(
            clean_unscored, fixed + selected + [final_id]
        )
        try:
            # This is the scientific gate. Sampling only proposes a parity;
            # Stim's stabilizer analysis must prove it exactly deterministic.
            _ = scored.detector_error_model(
                decompose_errors=False, allow_gauge_detectors=False
            )
        except ValueError as ex:
            last_error = f"exact determinism failed after {shots} shots: {ex}"
            continue

        # Independent clean sampling sanity check.
        ds = scored.compile_detector_sampler(seed=2026090404)
        dets, obs = ds.sample(64, separate_observables=True)
        if bool(dets.any()) or bool(obs.any()):
            last_error = (
                f"exact DEM passed but clean sampling was nonzero: "
                f"dets={bool(dets.any())}, obs={bool(obs.any())}"
            )
            continue
        return scored, {
            "channel": channel,
            "solver": "sample-propose + exact-Stim-determinism-certify",
            "fixed_hirano_frame_measurements": len(fixed),
            "candidate_operational_expansion_measurements": nvars,
            "selected_expansion_byproduct_measurements": len(selected),
            "selected_expansion_measurement_ids": selected,
            "final_logical_measurement_id": final_id,
            "observable_measurement_ids_mod2": obs_ids,
            "proposal_shots": shots,
            "exact_determinism_pass": True,
            "uses_ideal_followup_measurements_in_frame": False,
        }

    raise RuntimeError(
        (
            "Could not derive an exact transported logical frame using only "
            "the old Hirano frame plus measurements from the single faultable "
            "direct-expansion round. This is evidence that one operational "
            "expansion round may be insufficient; do not silently use the ideal "
            "follow-up for feed-forward.",
            channel,
            last_error,
            {
                "fixed": len(fixed),
                "variables": nvars,
                "final_id": final_id,
                "total_measurements": nmeas,
            },
        )
    )


def _append_direct_expansion_clean(sce: Any, runner: Any, channel: str, hirano_pre: list[int]):
    """Append live direct expansion and derive the transported operational frame."""
    import stim

    patch = sce.SurfaceCodePatch(
        runner.circuit, D_ESC, D_FINAL, 1, sce.ExpansionPattern.UPWARD
    )
    patch.offset = (1, 17)

    patch.syndrome_measurements = runner.surface_syndrome_measurements

    exp_start = len(runner.circuit.circuit)
    exp_measurement_start = int(runner.circuit.circuit.num_measurements)
    patch._perform_code_expansion()
    fy = 17 - 2 * (D_FINAL - D_ESC)
    if tuple(patch.offset) != (1, fy):
        raise AssertionError(("final offset", patch.offset))
    if len(patch.syndrome_measurements) != 168:
        raise AssertionError(("d13 stabilizers", len(patch.syndrome_measurements)))

    # One faultable direct expansion syndrome round.
    for _stage in range(6):
        for sm in patch.syndrome_measurements.values():
            sm.run()
        runner.circuit.place_tick()
    for sm in patch.syndrome_measurements.values():
        if not sm.is_complete() or sm.last_measurement is None:
            raise AssertionError(("incomplete expansion syndrome", sm.ancilla_position))
    exp_end = len(runner.circuit.circuit)
    exp_measurement_end = int(runner.circuit.circuit.num_measurements)

    # One ideal stationary follow-up is truth/scoring only. It is NOT allowed
    # to participate in the operational Pauli-frame solver below.
    for _stage in range(6):
        for sm in patch.syndrome_measurements.values():
            sm.run()
        runner.circuit.place_tick()
    follow_end = len(runner.circuit.circuit)
    follow_measurement_end = int(runner.circuit.circuit.num_measurements)

    # Final ideal logical MPP. Geometry follows the vendored expansion
    # generator: UPWARD logical Z is the bottom boundary. X uses the
    # complementary left boundary convention already clean-tested in HOTFIX-1.
    ps = stim.PauliString()
    if channel == "X":
        for i in range(13):
            q = runner.circuit.mapping.get_id(1, fy + 2*i)
            ps *= stim.PauliString(f"X{q}")
        logical_boundary = "left-X"
    elif channel == "Z":
        for j in range(13):
            q = runner.circuit.mapping.get_id(1 + 2*j, fy + 24)
            ps *= stim.PauliString(f"Z{q}")
        logical_boundary = "bottom-Z"
    else:
        raise ValueError(channel)

    final_measurement_id = int(runner.circuit.circuit.num_measurements)
    _final_m = runner.circuit.place_mpp(ps)
    if int(runner.circuit.circuit.num_measurements) != final_measurement_id + 1:
        raise AssertionError("final MPP did not add exactly one measurement")

    clean_unscored = runner.circuit.circuit[:]
    if int(clean_unscored.num_observables) != 0:
        raise AssertionError(("unexpected pre-existing observable", clean_unscored.num_observables))

    # The new frame may depend on reset/measurement outcomes created during
    # _perform_code_expansion as well as the first six-stage syndrome round.
    variable_ids = list(range(exp_measurement_start, exp_measurement_end))
    clean, frame_meta = _solve_expansion_byproduct_frame(
        clean_unscored,
        fixed_hirano_ids=hirano_pre,
        variable_expansion_ids=variable_ids,
        final_measurement_id=final_measurement_id,
        channel=channel,
    )

    if int(clean.num_observables) != 1:
        raise AssertionError(clean.num_observables)
    return clean, {
        "expansion_instruction_start": int(exp_start),
        "expansion_faultable_instruction_end": int(exp_end),
        "ideal_followup_instruction_end": int(follow_end),
        "expansion_measurement_start": exp_measurement_start,
        "expansion_faultable_measurement_end": exp_measurement_end,
        "ideal_followup_measurement_end": follow_measurement_end,
        "final_detector_count": int(clean.num_detectors),
        "final_measurement_count": int(clean.num_measurements),
        "final_qubits": int(clean.num_qubits),
        "logical_boundary": logical_boundary,
        "transport_frame": frame_meta,
    }


def _instrument_direct_expansion(clean_segment: Any, p: float):
    """Instrument only active locations in the first direct expansion round."""
    import stim
    if not (0 < p < 1):
        raise ValueError(p)
    out = stim.Circuit()
    counts = Counter()

    for inst in clean_segment:
        if not hasattr(inst, "name"):
            raise AssertionError("unexpected repeat block")
        name = inst.name
        ts = inst.targets_copy()
        qs = [_target_value(t) for t in ts if _bool_attr(t, "is_qubit_target")]

        if name == "CX":
            if len(qs) % 2:
                raise AssertionError((name, qs))
            out.append(inst)
            out.append("DEPOLARIZE2", qs, p)
            counts["CX"] += len(qs) // 2
        elif name == "R":
            out.append(inst)
            if qs:
                out.append("X_ERROR", qs, p)
            counts["R"] += len(qs)
        elif name == "RX":
            out.append(inst)
            if qs:
                out.append("Z_ERROR", qs, p)
            counts["RX"] += len(qs)
        elif name in {"M", "MZ"}:
            if qs:
                out.append("X_ERROR", qs, p)
            out.append(inst)
            counts["M"] += len(qs)
        elif name == "MX":
            if qs:
                out.append("Z_ERROR", qs, p)
            out.append(inst)
            counts["MX"] += len(qs)
        elif name in {"DETECTOR", "SHIFT_COORDS", "TICK"}:
            out.append(inst)
        else:
            # The live expansion segment should not contain MPP/observable or
            # non-Clifford operations.  QUBIT_COORDS are all declared in the
            # pre-existing Hirano wrapper and therefore absent here.
            raise RuntimeError(("unexpected operation in direct expansion faultable segment", name, str(inst)))

    counts["total"] = sum(counts.values())
    return out, dict(counts)


def _build_channel(s7b: Any, sce: Any, hirano_src: Path, channel: str, p: float):
    """Return noisy scored, clean scored, and splice metadata."""
    runner, pre_clean, init_range, hirano_pre = _build_hirano_boundary_runner(
        s7b, hirano_src, channel
    )
    contract = _validate_upward_splice_contract(sce, runner)

    boundary_instruction_index = int(init_range[1])
    pre_instruction_count = len(pre_clean)
    pre_tick_count = _count_ticks(pre_clean)

    clean_scored, smeta = _append_direct_expansion_clean(
        sce, runner, channel, hirano_pre
    )
    # Regression guard for the aliasing bug fixed in HOTFIX-1.
    if len(pre_clean) != pre_instruction_count:
        raise RuntimeError(
            ("Hirano prefix snapshot mutated after expansion append",
             pre_instruction_count, len(pre_clean))
        )
    exp_start = int(smeta["expansion_instruction_start"])
    exp_end = int(smeta["expansion_faultable_instruction_end"])
    if exp_start != pre_instruction_count:
        raise AssertionError((exp_start, pre_instruction_count))

    # Clean determinism check.  DEM construction catches non-deterministic detectors.
    _ = clean_scored.detector_error_model(
        decompose_errors=False, allow_gauge_detectors=False
    )
    # Sample clean observable to ensure the transported logical frame is deterministic.
    sampler = clean_scored.compile_detector_sampler(seed=12345)
    dets, obs = sampler.sample(32, separate_observables=True)
    if bool(dets.any()):
        raise RuntimeError(("clean circuit produced detection event", channel))
    if bool(obs.any()):
        raise RuntimeError(("clean transported observable is not deterministic zero", channel))

    # Existing Hirano active-location instrumentation.
    noisy_pre, hloc, _ = s7b._instrument_before_probe(
        pre_clean, init_range, len(pre_clean), p
    )
    if len(hloc) != EXPECTED_HIRANO_LOCATIONS:
        raise AssertionError(("Hirano active locations", len(hloc)))

    exp_clean = clean_scored[exp_start:exp_end]
    ideal_tail = clean_scored[exp_end:]
    noisy_exp, ec = _instrument_direct_expansion(exp_clean, p)
    noisy_scored = noisy_pre + noisy_exp + ideal_tail

    if noisy_scored.without_noise() != clean_scored:
        raise RuntimeError("direct-splice instrumentation changed clean circuit")
    if int(noisy_scored.num_detectors) != int(clean_scored.num_detectors):
        raise AssertionError((noisy_scored.num_detectors, clean_scored.num_detectors))
    if int(noisy_scored.num_observables) != 1:
        raise AssertionError(noisy_scored.num_observables)

    meta = {
        "channel": channel,
        "signature_p": float(p),
        "boundary_instruction_index": boundary_instruction_index,
        "pre_tick_count": int(pre_tick_count),
        "splice_contract": contract,
        "splice": smeta,
        "location_counts": {
            "hirano_escape_plus_two_d5_rounds": len(hloc),
            "direct_d5_to_d13_faultable_round": int(ec["total"]),
            "total": len(hloc) + int(ec["total"]),
            "direct_types": ec,
            "ideal_stationary_followup": 0,
            "ideal_terminal_scoring": 0,
        },
        "detectors": int(noisy_scored.num_detectors),
    }
    return noisy_scored, clean_scored, meta


@dataclass
class Effect:
    syndrome: int
    pauli: int
    locations: set[int]
    example: tuple[Any, ...]


def _group_effects(cert: Any, xmap: dict, zmap: dict, universe: set[tuple[Any, ...]]):
    grouped: dict[tuple[int, int], Effect] = {}
    loc_weight_sum: dict[int, float] = defaultdict(float)
    for m in universe:
        sx, lx = xmap[m]
        sz, lz = zmap[m]
        if int(sx) != int(sz):
            raise RuntimeError(("X/Z syndrome mismatch", cert._mechanism_json(m), hex(int(sx)), hex(int(sz))))
        pauli = int(lz) | (int(lx) << 1)
        loc = int(m[0])
        loc_weight_sum[loc] += float(cert._mechanism_weight(cert._location_gate(loc)))
        key = (int(sx), int(pauli))
        if key not in grouped:
            grouped[key] = Effect(int(sx), int(pauli), set(), m)
        grouped[key].locations.add(loc)

    bad = [(k, v) for k, v in sorted(loc_weight_sum.items()) if abs(v - 1.0) > 1e-10]
    if bad:
        raise RuntimeError(("mechanism weights per location do not sum to one", bad[:20]))
    return list(grouped.values())


def _distinct_pair_exists(a: Effect, b: Effect) -> bool:
    return any(x != y for x in a.locations for y in b.locations)


def _pair_witness(a: Effect, b: Effect):
    for x in sorted(a.locations):
        for y in sorted(b.locations):
            if x != y:
                return x, y
    return None


def _physical_zero_record_audit(cert: Any, effects: list[Effect], max_examples: int):
    by_s: dict[int, list[Effect]] = defaultdict(list)
    for e in effects:
        by_s[e.syndrome].append(e)

    harmful_single = []
    for e in by_s.get(0, []):
        if e.pauli != 0:
            harmful_single.append(e)

    pair_count = 0
    pair_examples = []
    for s, es in by_s.items():
        # Same syndrome is necessary and sufficient for XOR cancellation.
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                a, b = es[i], es[j]
                if (a.pauli ^ b.pauli) == 0:
                    continue
                if not _distinct_pair_exists(a, b):
                    continue
                pair_count += 1
                if len(pair_examples) < max_examples:
                    w = _pair_witness(a, b)
                    pair_examples.append({
                        "syndrome_hex": hex(int(s)),
                        "pauli_A": PAULI_NAMES[a.pauli],
                        "pauli_B": PAULI_NAMES[b.pauli],
                        "final_pauli": PAULI_NAMES[a.pauli ^ b.pauli],
                        "distinct_location_witness": [int(w[0]), int(w[1])] if w else None,
                        "mechanism_A": cert._mechanism_json(a.example),
                        "mechanism_B": cert._mechanism_json(b.example),
                    })

    return {
        "zero_record_harmful_single_effect_classes": len(harmful_single),
        "zero_record_harmful_distinct_location_pair_effect_class_pairs": int(pair_count),
        "single_gate_pass": len(harmful_single) == 0,
        "pair_gate_pass": pair_count == 0,
        "single_examples": [
            {
                "pauli": PAULI_NAMES[e.pauli],
                "mechanism": cert._mechanism_json(e.example),
            }
            for e in harmful_single[:max_examples]
        ],
        "pair_examples": pair_examples,
    }


def _build_sector_map(s18m: Any, ca: Any, s8b: Any,
                      xclean: Any, zclean: Any, xm: dict, zm: dict,
                      hirano_src: Path, p: float):
    # Reuse the already-audited S18 sector insertion machinery.  The helper is
    # detector-count agnostic despite its historical detector_638 field name.
    return s18m._build_sector_map_638(
        ca, s8b, xclean, zclean, xm, zm, hirano_src, p
    )


def _sector_audit(sectors: list[dict], effects: list[Effect], max_examples: int):
    by_s: dict[int, list[Effect]] = defaultdict(list)
    for e in effects:
        by_s[e.syndrome].append(e)

    alone_harmful = Counter()
    alone_examples = []
    w1_plus_phys_harmful = 0
    w1_plus_phys_examples = []

    for r in sectors:
        bw = int(r["canonical_weight"])
        s = int(r["detector_638_hex"], 16)
        ps = int(r["raw_terminal_pauli_int"])

        if bw in (1, 2) and s == 0 and ps != 0:
            alone_harmful[bw] += 1
            if len(alone_examples) < max_examples:
                alone_examples.append({
                    "boundary_weight": bw,
                    "steane_syndrome": int(r["steane_syndrome"]),
                    "boundary_pauli": r["raw_terminal_pauli"],
                })

        if bw != 1:
            continue
        for e in by_s.get(s, []):
            if (ps ^ e.pauli) == 0:
                continue
            w1_plus_phys_harmful += 1
            if len(w1_plus_phys_examples) < max_examples:
                w1_plus_phys_examples.append({
                    "steane_syndrome": int(r["steane_syndrome"]),
                    "boundary_record_hex": r["detector_638_hex"],
                    "boundary_pauli": r["raw_terminal_pauli"],
                    "physical_effect_pauli": PAULI_NAMES[e.pauli],
                    "final_pauli": PAULI_NAMES[ps ^ e.pauli],
                    "physical_mechanism": e.example,
                })

    return {
        "harmful_zero_record_boundary_weight1_alone": int(alone_harmful[1]),
        "harmful_zero_record_boundary_weight2_alone": int(alone_harmful[2]),
        "harmful_zero_record_boundary_weight1_plus_physical1_effect_classes": int(w1_plus_phys_harmful),
        "boundary_alone_gate_pass": int(alone_harmful[1] + alone_harmful[2]) == 0,
        "weight1_plus_physical_gate_pass": w1_plus_phys_harmful == 0,
        "boundary_alone_examples": alone_examples,
        "weight1_plus_physical_examples": w1_plus_phys_examples,
        "note": "Weight-2 boundary + physical1 is total order 3 and is intentionally non-gating.",
    }


def _self_test():
    a = Effect(7, 1, {0}, (0, ()))
    b = Effect(7, 2, {1}, (1, ()))
    c = Effect(7, 1, {2}, (2, ()))
    assert _distinct_pair_exists(a, b)
    assert (a.pauli ^ b.pauli) != 0
    assert (a.pauli ^ c.pauli) == 0
    return {
        "passed": True,
        "identity": "two detector records XOR to zero iff their syndrome masks are equal",
        "distinct_location_requirement_enforced": True,
        "truth_bits_decoder_visible": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--signature-p", type=float, default=0.001)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("stim_pipeline/interface/s21opt3b_results"))
    ap.add_argument("--max-examples", type=int, default=20)
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.self_test_only:
        print(json.dumps(_self_test(), indent=2))
        return

    root = args.search_root.resolve()
    outdir = args.out_dir
    if not outdir.is_absolute():
        outdir = (root / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Dependencies from the frozen S18 stack.
    s7bp = _find(root, ["stim_pipeline/interface/zxz_online_handoff_audit.py"])
    certp = _find(root, ["stim_pipeline/interface/direct_cs_s17b_v13d_joint_postselection_certificate.py"])
    cap = _find(root, ["stim_pipeline/interface/direct_cs_s17c_a_sector_to_470_map.py"])
    s8bp = _find(root, ["stim_pipeline/interface/zxz_core_boundary_signature_map.py"])
    s18mp = _find(root, ["stim_pipeline/interface/direct_cs_s18m_modular_noisy_d13_certificate.py"])
    scepath = _find(root, ["msc-ls/src/surface_code_expansion.py"])
    hirano_src = _find(root, ["msc-ls/src"], want_dir=True)

    deps = {
        "s7b": str(s7bp),
        "certificate": str(certp),
        "sector_map": str(cap),
        "core_boundary_map": str(s8bp),
        "s18m": str(s18mp),
        "surface_code_expansion": str(scepath),
        "hirano_src": str(hirano_src),
    }

    preflight = {
        "stage": "S21-OPT3B preflight",
        "dependencies": deps,
        "signature_p": float(args.signature_p),
        "expansion_pattern": "UPWARD",
        "wait_rounds": WAIT_ROUNDS,
        "passed": True,
    }
    _dump(outdir / "s21opt3b_preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        print("S21-OPT3B preflight PASS")
        return

    s7b = _load_module(s7bp, "s21opt3b_s7b")
    cert = _load_module(certp, "s21opt3b_cert")
    ca = _load_module(cap, "s21opt3b_ca")
    s8b = _load_module(s8bp, "s21opt3b_s8b")
    s18m = _load_module(s18mp, "s21opt3b_s18m")

    # Ensure msc-ls imports resolve to the same vendored source tree.
    if str(hirano_src) not in sys.path:
        sys.path.insert(0, str(hirano_src))
    sce = _load_module(scepath, "s21opt3b_sce")

    print("S21-OPT3B: building clean/noisy X splice...", flush=True)
    xn, xc, xm = _build_channel(s7b, sce, hirano_src, "X", float(args.signature_p))
    print("S21-OPT3B: building clean/noisy Z splice...", flush=True)
    zn, zc, zm = _build_channel(s7b, sce, hirano_src, "Z", float(args.signature_p))

    if int(xn.num_detectors) != int(zn.num_detectors):
        raise RuntimeError(("X/Z detector count mismatch", xn.num_detectors, zn.num_detectors))
    if _detector_lines(xn) != _detector_lines(zn):
        raise RuntimeError("X/Z detector definitions differ")
    if xm["location_counts"] != zm["location_counts"]:
        raise RuntimeError(("X/Z location-count drift", xm["location_counts"], zm["location_counts"]))

    xpath = outdir / "s21opt3b_X_direct_d5_d13_noisy.stim"
    zpath = outdir / "s21opt3b_Z_direct_d5_d13_noisy.stim"
    xpath.write_text(str(xn), encoding="utf-8")
    zpath.write_text(str(zn), encoding="utf-8")
    (outdir / "s21opt3b_X_direct_d5_d13_clean.stim").write_text(str(xc), encoding="utf-8")
    (outdir / "s21opt3b_Z_direct_d5_d13_clean.stim").write_text(str(zc), encoding="utf-8")

    total_locations = int(xm["location_counts"]["total"])
    cert.EXPECTED_LOCATIONS = total_locations
    cert.EXPECTED_DETECTORS = int(xn.num_detectors)

    exp_tick = int(xm["pre_tick_count"])
    def _stage(tick: int) -> str:
        return "hirano_escape_plus_two_d5_rounds" if tick < exp_tick else "direct_d5_to_d13"
    cert._stage_from_tick = _stage

    xd = cert._detector_lines(xpath)
    zd = cert._detector_lines(zpath)
    if xd != zd:
        raise RuntimeError("textual X/Z detector basis mismatch")

    xl, xnative, xmeta = cert._noise_locations_from_text(xpath)
    zl, znative, zmeta = cert._noise_locations_from_text(zpath)
    if len(xl) != total_locations or len(zl) != total_locations:
        raise RuntimeError((len(xl), len(zl), total_locations))
    xphys = [(xmeta[i]["tick"], xmeta[i]["gate"], tuple(xmeta[i]["qubits"])) for i in xl]
    zphys = [(zmeta[i]["tick"], zmeta[i]["gate"], tuple(zmeta[i]["qubits"])) for i in zl]
    if xphys != zphys:
        raise RuntimeError("X/Z physical stochastic schedules differ")
    cert._LOCATION_META.clear()
    cert._LOCATION_META.update(xmeta)

    print(f"S21-OPT3B: recovering X mechanisms across {total_locations} locations...", flush=True)
    xp, xpmeta = cert._recover_mechanism_map(xn, cert._raw_dem(xn), "X", xnative)
    print(f"S21-OPT3B: recovering Z mechanisms across {total_locations} locations...", flush=True)
    zp, zpmeta = cert._recover_mechanism_map(zn, cert._raw_dem(zn), "Z", znative)

    universe = cert._all_mechanism_keys(xl)
    xmap = {m: xp.get(m, (0, 0)) for m in universe}
    zmap = {m: zp.get(m, (0, 0)) for m in universe}
    effects = _group_effects(cert, xmap, zmap, universe)

    print(f"S21-OPT3B: exact <=2 zero-record joins over {len(effects)} effect classes...", flush=True)
    paudit = _physical_zero_record_audit(cert, effects, int(args.max_examples))

    print("S21-OPT3B: propagating canonical incoming Steane sectors...", flush=True)
    sectors, sector_meta = _build_sector_map(
        s18m, ca, s8b, xc, zc, xm, zm, hirano_src, float(args.signature_p)
    )
    saudit = _sector_audit(sectors, effects, int(args.max_examples))

    main_pass = bool(
        paudit["single_gate_pass"]
        and paudit["pair_gate_pass"]
        and saudit["boundary_alone_gate_pass"]
        and saudit["weight1_plus_physical_gate_pass"]
    )

    summary = {
        "stage": "S21-OPT3B live direct d5->d13 exact structural certificate",
        "status": "PASS" if main_pass else "FAIL",
        "signature_p": float(args.signature_p),
        "scientific_policy": {
            "structural_acceptance": "all operational detectors zero",
            "finite_p_practical_decoder": False,
            "idle_noise": False,
            "expansion_pattern": "UPWARD only, exactly aligned to live Hirano d5 coordinates",
            "transported_logicals": "left-X / bottom-Z with exact-derived operational expansion byproduct frame",
            "fault_model": "active-location stochastic Pauli; distinct physical locations for order-2",
            "terminal_scoring": "ideal follow-up syndrome round + ideal logical MPP; scoring only",
        },
        "schedule": {
            "detectors": int(xn.num_detectors),
            "physical_locations": total_locations,
            "X_Z_detector_definitions_identical": True,
            "X_Z_physical_schedules_identical": True,
            "X": xm,
            "Z": zm,
        },
        "mechanisms": {
            "concrete_universe": len(universe),
            "unique_joint_effect_classes": len(effects),
            "X_explained": len(xp),
            "Z_explained": len(zp),
            "X_meta": xpmeta,
            "Z_meta": zpmeta,
        },
        "physical_fault_audit": paudit,
        "sector_map": {
            "meta": sector_meta,
            "rows": sectors,
        },
        "incoming_sector_audit": saudit,
        "main_gate": {
            "physical_single_zero_record_identity_only": paudit["single_gate_pass"],
            "physical_distinct_pair_zero_record_identity_only": paudit["pair_gate_pass"],
            "canonical_boundary_weight1_2_alone_safe": saudit["boundary_alone_gate_pass"],
            "canonical_boundary_weight1_plus_physical1_safe": saudit["weight1_plus_physical_gate_pass"],
            "passed": main_pass,
        },
        "next_if_pass": (
            "S21-OPT3C: build practical direct-expansion operational decoder and finite-p paired validation; "
            "only then update S20 resource numbers."
        ),
        "stop_if_fail": (
            "Do not run finite-p MC. Inspect exact harmful witnesses and either add one more faultable/ideal "
            "syndrome round or abandon direct expansion optimization."
        ),
    }
    _dump(outdir / "s21opt3b_summary.json", summary)
    _dump(outdir / "s21opt3b_self_test.json", _self_test())

    print(json.dumps({
        "status": summary["status"],
        "detectors": summary["schedule"]["detectors"],
        "locations": summary["schedule"]["physical_locations"],
        "effect_classes": summary["mechanisms"]["unique_joint_effect_classes"],
        "main_gate": summary["main_gate"],
    }, indent=2))
    print("S21-OPT3B", summary["status"])
    if not main_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
