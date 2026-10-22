#!/usr/bin/env python3
"""S18-A: physicalize the first d=13 QEC round and re-certify the continuous boundary.

Scientific purpose
==================
S17-C/D2V validated a practical higher-order decoder on the 470-bit continuous
handoff history, but the history ended in an IDEAL d=13 syndrome close.  S18-A
turns that close into a real noisy d=13 QEC round and then appends one additional
IDEAL d=13 syndrome round only as a scoring/decoder closure.  This is the same
finite-window convention used for a live QEC state: the delivered state has
undergone a noisy first d=13 QEC round; the following ideal round supplies future
syndrome information for simulation truth and is not charged as a factory
operation.

Boundary per output patch
=========================
  Hirano escape + 2 noisy d5 rounds            1028 locations
  Sahay d5->7 growth                             971 locations
  Sahay d7->13 growth                            408 locations
  first d13 QEC round (NOISY)                   1128 locations [conservative M+R split]
  ideal follow-up d13 syndrome round               0 locations [scoring only]
  ideal destructive logical boundary               0 locations [truth only]
  -----------------------------------------------------------------------------
  charged faultable Clifford interface          3535 locations / patch

The 1128-location d13 round counts H=168, CX=624, measurement=168, reset=168.
For each MR ancilla operation we insert an X_ERROR(p) before MR (measurement
flip opportunity) and an independent X_ERROR(p) after MR (reset error carried
into the next round).  This deliberately conservative split preserves the
project convention that preparation/reset and measurement are separate active
locations.  No idle/correlated/coherent noise is added.

Detector semantics
==================
The first d13 round retains the 168 growth-to-d13 closing checks.  The ideal
follow-up round adds 168 stationary detectors comparing each check outcome to
the same check in the preceding noisy d13 round.  The 35 invalid fresh-input
first-growth detector annotations remain removed and Stim missing_detectors()
is rerun on the *extended clean circuit* to synthesize the complete deterministic
complement.  Expected operational detector count is therefore

    171 + (131-35) + 168 + 168 + 35 = 638.

The terminal destructive X/Z data measurements remain scoring-only and are not
decoder bits.

Low-order certificate
=====================
After writing matched X/Z reference circuits, this script reuses the S17-B v13d
physical-mechanism machinery with updated dimensions (3535 locations, 638
operational detector bits).  It requires identical X/Z stochastic schedules,
explicitly joins the same physical Pauli mechanism across probes, and by default
enumerates every location-compatible fault set of weight <=2.  Ambiguous nonzero
records form the new S18 rejection set.  PASS means every accepted <=2-fault
record has a unique final joint Pauli and zero syndrome has identity only.

This stage does NOT yet include the non-Clifford core finite-p trajectory.  If
S18-A passes, S18-B must redo the 64-sector closure and finite-p D2 evaluation on
the new 638-bit boundary; do not reuse the 470-bit reject set unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

EXPECTED_HIRANO = 1028
EXPECTED_G57 = 971
EXPECTED_G713 = 408
EXPECTED_D13_H = 168
EXPECTED_D13_CX = 624
EXPECTED_D13_MEAS = 168
EXPECTED_D13_RESET = 168
EXPECTED_D13_ROUND = EXPECTED_D13_H + EXPECTED_D13_CX + EXPECTED_D13_MEAS + EXPECTED_D13_RESET
EXPECTED_LOCATIONS = EXPECTED_HIRANO + EXPECTED_G57 + EXPECTED_G713 + EXPECTED_D13_ROUND
EXPECTED_DETECTORS = 638
EXPECTED_STEADY_DETECTORS = 168
EXPECTED_SYNTH_MISSING = 35


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _bool_attr(x: Any, name: str) -> bool:
    v = getattr(x, name, False)
    return bool(v() if callable(v) else v)


def _target_value(t: Any) -> int:
    if hasattr(t, "val"):
        v = getattr(t, "val")
        return int(v() if callable(v) else v)
    if hasattr(t, "value"):
        v = getattr(t, "value")
        return int(v() if callable(v) else v)
    s = str(t)
    if s.startswith("rec[") and s.endswith("]"):
        return int(s[4:-1])
    raise TypeError(t)


def _quantum_round_only(close_clean: Any) -> Any:
    """Drop close DETECTOR/SHIFT annotations, preserving H/CX/MR/TICK operations."""
    import stim
    out = stim.Circuit()
    for inst in close_clean:
        if not hasattr(inst, "name"):
            raise AssertionError("repeat block in d13 close")
        if inst.name in {"DETECTOR", "SHIFT_COORDS"}:
            continue
        out.append(inst)
    if int(out.num_measurements) != 168:
        raise AssertionError(("steady round measurements", out.num_measurements))
    return out


def _close_detector_args(close_clean: Any) -> list[list[float]]:
    rows = []
    for inst in close_clean:
        if hasattr(inst, "name") and inst.name == "DETECTOR":
            rows.append([float(x) for x in inst.gate_args_copy()])
    if len(rows) != 168:
        raise AssertionError(("first close detector templates", len(rows)))
    return rows


def _make_ideal_stationary_close(close_clean: Any) -> Any:
    """One ideal d13 syndrome round + same-check temporal detector differences."""
    import stim
    out = _quantum_round_only(close_clean)
    args = _close_detector_args(close_clean)
    # Immediately after this round's 168 MR outcomes:
    # current check j  = rec[-168+j]
    # previous check j = rec[-336+j]
    for j, a in enumerate(args):
        out.append(
            "DETECTOR",
            [stim.target_rec(-168 + j), stim.target_rec(-336 + j)],
            a,
        )
    out.append("SHIFT_COORDS", [], [0, 0, 1])
    if int(out.num_detectors) != 168:
        raise AssertionError(out.num_detectors)
    return out


def _instrument_noisy_d13_close(close_clean: Any, p: float) -> tuple[Any, dict[str, int]]:
    """Conservative physical d13 round: separate measurement and reset faults for MR."""
    import stim
    if not (0 < p < 1):
        raise ValueError(p)
    out = stim.Circuit()
    c = {"H": 0, "CX": 0, "MR_measurement": 0, "MR_reset": 0}
    for inst in close_clean:
        if not hasattr(inst, "name"):
            raise AssertionError("repeat block in close")
        name = inst.name
        ts = inst.targets_copy()
        if name == "H":
            qs = [_target_value(t) for t in ts]
            out.append(inst)
            out.append("DEPOLARIZE1", qs, p)
            c["H"] += len(qs)
        elif name in {"CX", "CNOT"}:
            qs = [_target_value(t) for t in ts]
            if len(qs) % 2:
                raise AssertionError((name, qs))
            out.append(inst)
            out.append("DEPOLARIZE2", qs, p)
            c["CX"] += len(qs) // 2
        elif name in {"MR", "MRZ"}:
            qs = [_target_value(t) for t in ts]
            # measurement flip before the Z-basis measure/reset primitive
            out.append("X_ERROR", qs, p)
            out.append(inst)
            # reset/preparation error that persists into the next QEC round
            out.append("X_ERROR", qs, p)
            c["MR_measurement"] += len(qs)
            c["MR_reset"] += len(qs)
        else:
            out.append(inst)
    c["total"] = sum(c.values())
    exp = {
        "H": EXPECTED_D13_H,
        "CX": EXPECTED_D13_CX,
        "MR_measurement": EXPECTED_D13_MEAS,
        "MR_reset": EXPECTED_D13_RESET,
        "total": EXPECTED_D13_ROUND,
    }
    if c != exp:
        raise AssertionError(("d13 active-location count drift", c, exp))
    if out.without_noise() != close_clean:
        raise AssertionError("d13 instrumentation changed clean circuit")
    return out, c


def _make_pauli(v7: Any, n: int, support: list[int], basis: str) -> Any:
    return v7._make_pauli(n, support, basis)


def _solve_transport_flow_s18(v7: Any, lines: list[str], geo: dict[str, Any], channel: str) -> dict[str, Any]:
    """Re-solve d5->d13 logical transport through close1 + ideal follow-up round."""
    import stim
    if channel not in {"X", "Z"}:
        raise ValueError(channel)
    g57 = v7._clean_source_range(lines, v7.G57_START, v7.G57_END)
    g713 = v7._clean_source_range(lines, v7.G713_START, v7.G713_END)
    close1 = v7._clean_source_range(lines, v7.CLOSE_START, v7.CLOSE_END)
    close2q = _quantum_round_only(close1)
    transport = g57 + g713 + close1 + close2q

    coords = geo["coords"]
    old = list(geo["old_d5"])
    if channel == "X":
        in_support = sorted(q for q in old if abs(coords[q][0] - 0.0) < 1e-9)
        out_support = list(geo["xlog"])
        basis = "X"
    else:
        in_support = sorted(q for q in old if abs(coords[q][1] - 4.0) < 1e-9)
        out_support = list(geo["zlog"])
        basis = "Z"
    if len(in_support) != 5 or len(out_support) != 13:
        raise AssertionError((channel, in_support, out_support))

    n = max(int(transport.num_qubits), max(out_support) + 1, max(in_support) + 1)
    pin = _make_pauli(v7, n, in_support, basis)
    pout = _make_pauli(v7, n, out_support, basis)
    query = stim.Flow(input=pin, output=pout)
    sol = transport.solve_flow_measurements([query])
    if len(sol) != 1 or sol[0] is None:
        raise AssertionError(f"No S18 {channel}_L flow through noisy-round boundary scaffold")
    mcount = int(transport.num_measurements)
    parity: dict[int, int] = {}
    for kk in sol[0]:
        k = int(kk)
        if k < 0:
            k += mcount
        if not 0 <= k < mcount:
            raise AssertionError((k, mcount))
        parity[k] = parity.get(k, 0) ^ 1
    mids = sorted(k for k, v in parity.items() if v)
    solved = stim.Flow(input=pin, output=pout, measurements=mids)
    if not transport.has_flow(solved, unsigned=True):
        raise AssertionError(("extended flow self-check", channel, solved))

    m57 = int(g57.num_measurements)
    m713 = int(g713.num_measurements)
    m1 = int(close1.num_measurements)
    m2 = int(close2q.num_measurements)
    b1 = m57 + m713
    b2 = b1 + m1
    return {
        "channel": channel,
        "input_basis": basis,
        "input_logical_support": [int(x) for x in in_support],
        "output_logical_support": [int(x) for x in out_support],
        "transport_measurements_total": mcount,
        "g57_measurements": m57,
        "g713_measurements": m713,
        "close_measurements": m1 + m2,
        "first_d13_close_measurements": m1,
        "ideal_followup_d13_measurements": m2,
        "mediating_measurement_indices": [int(x) for x in mids],
        "mediating_measurement_count": len(mids),
        "mediating_from_g57": sum(k < m57 for k in mids),
        "mediating_from_g713": sum(m57 <= k < b1 for k in mids),
        "mediating_from_first_d13_close": sum(b1 <= k < b2 for k in mids),
        "mediating_from_ideal_followup_d13": sum(k >= b2 for k in mids),
        # compatibility with v7 terminal helper
        "mediating_from_d13_close": sum(k >= b1 for k in mids),
        "flow_string": str(solved),
        "unsigned_flow_verified": True,
    }


def _build_channel(v7: Any, v11: Any, hirano_src: Path, sahay_stim: Path, channel: str, p: float) -> tuple[Any, dict[str, Any]]:
    lines = v7._read_source(sahay_stim)
    geo = v7._source_geometry(lines)
    full, _wrapper, init_range, probe_start, online_ids, terminal_ids = v7.s7b._build_with_wait_rounds(
        hirano_src, channel, 2
    )
    pre_clean = full[:probe_start]
    _d, obs_deps = v7.s7b._annotation_dependencies(full)
    if len(obs_deps) != 1:
        raise AssertionError(("Hirano observables", len(obs_deps)))
    pre_m = int(pre_clean.num_measurements)
    obs_all = [int(x) for x in obs_deps[0]]
    hirano_pre = [m for m in obs_all if m < pre_m]
    hirano_term = [m for m in obs_all if m >= pre_m]
    if len(hirano_term) != 5 or not hirano_pre:
        raise AssertionError((channel, hirano_pre, hirano_term))
    if len(online_ids) != 171 or int(pre_clean.num_detectors) != 171 or len(terminal_ids) != 12:
        raise AssertionError((len(online_ids), pre_clean.num_detectors, len(terminal_ids)))

    noisy_pre, hloc, _ = v7.s7b._instrument_before_probe(pre_clean, init_range, len(pre_clean), p)
    if len(hloc) != EXPECTED_HIRANO:
        raise AssertionError(len(hloc))
    qmap, map_meta = v7._build_qmap(pre_clean, geo)
    flow = _solve_transport_flow_s18(v7, lines, geo, channel)

    g57_src = v7._clean_source_range(lines, v7.G57_START, v7.G57_END)
    g57_src, strip_meta = v11._strip_first_growth_detectors(g57_src)
    g713_src = v7._clean_source_range(lines, v7.G713_START, v7.G713_END)
    close_src = v7._clean_source_range(lines, v7.CLOSE_START, v7.CLOSE_END)

    g57 = v7._remap_circuit(g57_src, qmap)
    g713 = v7._remap_circuit(g713_src, qmap)
    close1 = v7._remap_circuit(close_src, qmap)
    close2 = _make_ideal_stationary_close(close1)
    ng57, c57 = v7._instrument_growth(g57, p)
    ng713, c713 = v7._instrument_growth(g713, p)
    nclose1, c13 = _instrument_noisy_d13_close(close1, p)
    if c57["total"] != EXPECTED_G57 or c713["total"] != EXPECTED_G713:
        raise AssertionError((c57, c713))

    clean_preterminal = pre_clean + g57 + g713 + close1 + close2
    noisy_preterminal = noisy_pre + ng57 + ng713 + nclose1 + close2
    if int(clean_preterminal.num_measurements) != int(noisy_preterminal.num_measurements):
        raise AssertionError("noise changed measurements")

    missing, method = v11._missing_detectors(clean_preterminal)
    transition, trans_meta = v11._complete_missing_detector_basis(
        missing,
        total_measurements=int(clean_preterminal.num_measurements),
        pre_growth_measurements=pre_m,
        first_growth_count=48,
    )
    if int(transition.num_detectors) != EXPECTED_SYNTH_MISSING:
        raise AssertionError(("extended missing basis rank drift", transition.num_detectors, trans_meta))
    clean_preterminal += transition
    noisy_preterminal += transition

    # No gauge freedom is allowed on the operational detector basis.
    clean_dem_error = None
    try:
        clean_preterminal.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    except Exception as ex:
        clean_dem_error = repr(ex)

    c = noisy_preterminal
    term = v7._append_terminal_boundary(
        c,
        channel,
        geo,
        qmap,
        transport_measurement_offset=int(noisy_pre.num_measurements),
        transport_flow=flow,
        hirano_preprobe_observable_measurements=hirano_pre,
    )
    if int(c.num_detectors) != EXPECTED_DETECTORS:
        raise AssertionError(("detectors", c.num_detectors, EXPECTED_DETECTORS))
    total_loc = len(hloc) + c57["total"] + c713["total"] + c13["total"]
    if total_loc != EXPECTED_LOCATIONS:
        raise AssertionError((total_loc, EXPECTED_LOCATIONS))

    return c, {
        "channel": channel,
        "p": float(p),
        "boundary_convention": "one noisy d13 QEC round + one ideal follow-up d13 syndrome round + ideal destructive scorer",
        "charged_locations_per_patch": total_loc,
        "location_counts": {
            "hirano_escape_plus_two_d5_rounds": len(hloc),
            "sahay_d5_to_d7": c57["total"],
            "sahay_d7_to_d13": c713["total"],
            "noisy_first_d13_qec_round": c13["total"],
            "ideal_followup_d13_scoring_round": 0,
            "total": total_loc,
        },
        "noisy_d13_location_types": c13,
        "detector_accounting": {
            "hirano": 171,
            "growth_after_invalid35_removed": 96,
            "first_d13_close": 168,
            "ideal_followup_stationary": 168,
            "synthesized_missing_basis": int(transition.num_detectors),
            "total": int(c.num_detectors),
        },
        "removed_first_growth_detectors": strip_meta["removed_first_growth_detectors"],
        "missing_detector_method": method,
        "missing_detector_meta": trans_meta,
        "clean_preterminal_determinism_error": clean_dem_error,
        "mapping": map_meta,
        "transport_flow": flow,
        "terminal": term,
        "hirano_preprobe_frame_measurement_ids": hirano_pre,
        "terminal_d5_boundary_measurement_ids_removed": hirano_term,
    }


def _configure_certificate_module(cert: Any) -> None:
    cert.EXPECTED_LOCATIONS = EXPECTED_LOCATIONS
    cert.EXPECTED_DETECTORS = EXPECTED_DETECTORS
    # Existing tick boundaries remain correct through the end of d7->13.
    def stage(tick: int) -> str:
        if tick < 44:
            return "hirano_escape_plus_d5_rounds"
        if tick < 71:
            return "sahay_d5_to_d7"
        if tick < 77:
            return "sahay_d7_to_d13"
        return "noisy_first_d13_qec_round"
    cert._stage_from_tick = stage


def _run_certificate(cert: Any, x_path: Path, z_path: Path, p_values: list[float], skip_order2: bool) -> dict[str, Any]:
    import stim
    _configure_certificate_module(cert)
    xd = cert._detector_lines(x_path)
    zd = cert._detector_lines(z_path)
    if xd != zd or len(xd) != EXPECTED_DETECTORS:
        raise RuntimeError(("X/Z detector schedule mismatch", len(xd), len(zd)))
    xl, xn, xm = cert._noise_locations_from_text(x_path)
    zl, zn, zm = cert._noise_locations_from_text(z_path)
    if len(xl) != EXPECTED_LOCATIONS or len(zl) != EXPECTED_LOCATIONS:
        raise RuntimeError((len(xl), len(zl), EXPECTED_LOCATIONS))
    xphys = [(xm[i]["tick"], xm[i]["gate"], tuple(xm[i]["qubits"])) for i in xl]
    zphys = [(zm[i]["tick"], zm[i]["gate"], tuple(zm[i]["qubits"])) for i in zl]
    if xphys != zphys:
        raise RuntimeError("X/Z physical stochastic schedules differ")
    cert._LOCATION_META.clear()
    cert._LOCATION_META.update(xm)

    xc = stim.Circuit(x_path.read_text(encoding="utf-8"))
    zc = stim.Circuit(z_path.read_text(encoding="utf-8"))
    if int(xc.num_detectors) != EXPECTED_DETECTORS or int(zc.num_detectors) != EXPECTED_DETECTORS:
        raise AssertionError((xc.num_detectors, zc.num_detectors))

    print("S18-A: recovering X-probe concrete mechanisms...", flush=True)
    xp, xmeta = cert._recover_mechanism_map(xc, cert._raw_dem(xc), "X", xn)
    print("S18-A: recovering Z-probe concrete mechanisms...", flush=True)
    zp, zmeta = cert._recover_mechanism_map(zc, cert._raw_dem(zc), "Z", zn)
    universe = cert._all_mechanism_keys(xl)
    if (set(xp) - universe) or (set(zp) - universe):
        raise RuntimeError("explained mechanism outside physical universe")
    xmap = {m: xp.get(m, (0, 0)) for m in universe}
    zmap = {m: zp.get(m, (0, 0)) for m in universe}
    xmeta["complete_physical_mechanism_universe"] = len(universe)
    zmeta["complete_physical_mechanism_universe"] = len(universe)
    xmeta["exactly_trivial_mechanisms_omitted_by_raw_dem"] = len(universe) - len(xp)
    zmeta["exactly_trivial_mechanisms_omitted_by_raw_dem"] = len(universe) - len(zp)

    effects, joint = cert._build_joint_effects(xmap, zmap)
    states1, witnesses1, first = cert._joint_first_order(effects)

    # S18 is a postselected factory certificate, not a correction-only decoder
    # certificate.  Ambiguous nonzero single-fault syndromes are allowed provided
    # they are rejected.  The non-negotiable first-order safety condition is that
    # zero syndrome remains uniquely identity.  The full order<=2 pass below then
    # constructs a rejection set containing every ambiguous syndrome.
    first_postselection_safe = (
        not first["zero_syndrome_nonidentity_first_order"]
        and first["zero_syndrome_pauli_labels"] == ["I"]
    )
    if not first_postselection_safe:
        raise RuntimeError(("S18 zero-syndrome first-order safety failed", first))

    first_amb_examples = []
    for s, bits in sorted(states1.items()):
        if bits.bit_count() <= 1:
            continue
        labels = [q for q in range(4) if bits & (1 << q)]
        first_amb_examples.append({
            "syndrome_hex": hex(s),
            "possible_paulis": [cert.PAULI_NAMES[q] for q in labels],
            "witnesses": {cert.PAULI_NAMES[q]: witnesses1.get((s, q)) for q in labels},
        })

    out: dict[str, Any] = {
        "schedule_guard": {
            "detectors": EXPECTED_DETECTORS,
            "physical_locations": EXPECTED_LOCATIONS,
            "identical_XZ_detector_lines": True,
            "identical_XZ_physical_schedule": True,
        },
        "mechanism_mapping": {"X": xmeta, "Z": zmeta, "joint": joint},
        "joint_first_order": {
            **first,
            "postselection_safe_zero_syndrome": bool(first_postselection_safe),
            "ambiguous_nonzero_syndromes_will_be_rejected": int(first["ambiguous_first_order_syndromes"]),
            "ambiguity_examples": first_amb_examples[:30],
        },
        "joint_unique_effects": len(effects),
        "order2_skipped": bool(skip_order2),
    }
    if skip_order2:
        out["main_gate"] = {
            "first_order_correction_only_pass": bool(first["joint_first_order_information_gate_pass"]),
            "first_order_postselection_safe": bool(first_postselection_safe),
            "full_order_le2_certificate_pass": None,
        }
        return out

    print(f"S18-A: {len(effects)} joint effects; enumerating all location-compatible order<=2 records...", flush=True)
    states, witnesses, order2 = cert._enumerate_joint_order2(effects)
    reject, policy = cert._build_rejection_policy(states)
    rej = cert._rejection_enumerators(effects, reject, p_values)
    reject_examples = []
    for s in sorted(reject)[:50]:
        labels = [q for q in range(4) if states[s] & (1 << q)]
        reject_examples.append({
            "syndrome_hex": hex(s),
            "possible_paulis": [cert.PAULI_NAMES[q] for q in labels],
            "witnesses": {cert.PAULI_NAMES[q]: witnesses.get((s, q)) for q in labels},
        })
    gate = (
        not order2["zero_syndrome_nonidentity_order_le2"]
        and order2["zero_syndrome_pauli_labels"] == ["I"]
        and policy["all_accepted_order_le2_syndromes_have_unique_joint_pauli"]
        and policy["postselected_detect_or_correct_df_ge_3_low_order_gate_pass"]
    )
    out.update({
        "joint_order_le2": order2,
        "selective_rejection_policy": policy,
        "rejection_enumerators": rej,
        "reject_set": {
            "detector_bits": EXPECTED_DETECTORS,
            "rejected_syndrome_count": len(reject),
            "rejected_syndromes_hex": [hex(s) for s in sorted(reject)],
            "examples": reject_examples,
        },
        "main_gate": {
            "zero_syndrome_identity_only_through_order2": not order2["zero_syndrome_nonidentity_order_le2"] and order2["zero_syndrome_pauli_labels"] == ["I"],
            "all_accepted_order_le2_records_unique_joint_pauli": policy["all_accepted_order_le2_syndromes_have_unique_joint_pauli"],
            "postselected_detect_or_correct_df_ge_3_low_order_gate_pass": bool(gate),
            "full_order_le2_certificate_pass": bool(gate),
        },
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--v7-script", type=Path, default=None)
    ap.add_argument("--v11-script", type=Path, default=None)
    ap.add_argument("--v13d-script", type=Path, default=None)
    ap.add_argument("--hirano-src", type=Path, default=None)
    ap.add_argument("--sahay-stim", type=Path, default=None)
    ap.add_argument("--signature-p", type=float, default=0.001)
    ap.add_argument("--p-values", type=float, nargs="*", default=[0.0002, 0.0005, 0.001])
    ap.add_argument("--skip-order2", action="store_true", help="Smoke test only; not sufficient to advance to S18-B.")
    ap.add_argument("--out-dir", type=Path, default=Path("s18a_noisy_d13_boundary_results"))
    args = ap.parse_args()

    root = args.search_root.resolve()
    def find(name: str, hints: tuple[str, ...] = ()) -> Path:
        for h in hints:
            q = root / h / name
            if q.is_file():
                return q
        hits = list(root.rglob(name))
        if not hits:
            raise FileNotFoundError(name)
        return hits[0]

    v7p = (args.v7_script or find("direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py", ("stim_pipeline/interface",))).resolve()
    v11p = (args.v11_script or find("direct_cs_s17b_v11_complete_missing_basis.py", ("stim_pipeline/interface",))).resolve()
    v13p = (args.v13d_script or find("direct_cs_s17b_v13d_joint_postselection_certificate.py", ("stim_pipeline/interface",))).resolve()
    if args.hirano_src is not None:
        hp = args.hirano_src.resolve()
    else:
        hp = (root / "msc-ls" / "src").resolve()
        if not (hp / "lattice_surgery_error_detection.py").is_file():
            hits = list(root.rglob("lattice_surgery_error_detection.py"))
            if not hits:
                raise FileNotFoundError("Could not locate Hirano msc-ls/src directory")
            hp = hits[0].parent.resolve()
    sp = (args.sahay_stim or find("fd3_baseline_ckt.stim", ("reference",))).resolve()
    outd = args.out_dir.resolve()
    outd.mkdir(parents=True, exist_ok=True)

    v7 = _load(v7p, "s18_v7")
    v11 = _load(v11p, "s18_v11")
    cert = _load(v13p, "s18_v13d")

    summary: dict[str, Any] = {
        "stage": "S18-A noisy first d13 QEC round + ideal follow-up close low-order certificate",
        "boundary_semantics": {
            "charged": "one noisy d13 QEC round after d7->13 growth",
            "scoring_only": "one ideal follow-up d13 syndrome round + ideal destructive logical probe",
            "why_followup_is_needed": "a live state needs future syndrome information to score/correct faults late in the final noisy round; a dead-end noisy round would reproduce the S17-A boundary pathology",
            "MR_noise_convention": "two active opportunities per ancilla: pre-MR measurement flip and post-MR reset error",
        },
        "expected": {
            "detectors": EXPECTED_DETECTORS,
            "locations_per_patch": EXPECTED_LOCATIONS,
            "new_noisy_d13_locations_per_patch": EXPECTED_D13_ROUND,
            "two_patch_new_noisy_d13_locations": 2 * EXPECTED_D13_ROUND,
        },
        "inputs": {"v7": str(v7p), "v11": str(v11p), "v13d": str(v13p), "hirano": str(hp), "sahay": str(sp)},
        "channels": {},
    }

    for ch in ("X", "Z"):
        print(f"S18-A: building {ch} extended boundary circuit...", flush=True)
        c, meta = _build_channel(v7, v11, hp, sp, ch, float(args.signature_p))
        cpath = outd / f"s18a_{ch}_noisy_d13_boundary.stim"
        # Avoid Stim C++ path-opening issues on Windows paths containing non-ASCII characters.
        cpath.write_text(str(c), encoding="utf-8")
        _dump(outd / f"s18a_{ch}_build_meta.json", meta)
        summary["channels"][ch] = {"circuit": str(cpath), "build": meta}
        if meta["clean_preterminal_determinism_error"] is not None:
            raise RuntimeError((ch, meta["clean_preterminal_determinism_error"]))

    certout = _run_certificate(
        cert,
        outd / "s18a_X_noisy_d13_boundary.stim",
        outd / "s18a_Z_noisy_d13_boundary.stim",
        [float(x) for x in args.p_values],
        bool(args.skip_order2),
    )
    summary["certificate"] = certout
    full_pass = certout["main_gate"].get("full_order_le2_certificate_pass")
    summary["main_gate"] = {
        "deterministic_extended_detector_basis": True,
        "first_order_correction_only_pass": bool(certout["joint_first_order"]["joint_first_order_information_gate_pass"]),
        "first_order_postselection_safe": bool(certout["joint_first_order"]["postselection_safe_zero_syndrome"]),
        "first_order_ambiguous_nonzero_syndromes_rejected": int(certout["joint_first_order"]["ambiguous_first_order_syndromes"]),
        "full_order_le2_certificate_pass": full_pass,
        "passed": bool(full_pass) if not args.skip_order2 else None,
    }
    summary["next"] = (
        "If full order<=2 PASS: build S18-B 64-sector closure + D2 finite-p evaluation on the 638-bit boundary. "
        "If FAIL: inspect only the new d13-round ambiguity witnesses; do not tune the D2 gap yet."
    )
    _dump(outd / "s18a_summary.json", summary)
    if not args.skip_order2 and "reject_set" in certout:
        _dump(outd / "s18a_reject_policy.json", certout["reject_set"])
    print(json.dumps({
        "detectors": EXPECTED_DETECTORS,
        "locations_per_patch": EXPECTED_LOCATIONS,
        "new_d13_locations_per_patch": EXPECTED_D13_ROUND,
        "full_order_le2_certificate_pass": full_pass,
        "out_dir": str(outd),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
