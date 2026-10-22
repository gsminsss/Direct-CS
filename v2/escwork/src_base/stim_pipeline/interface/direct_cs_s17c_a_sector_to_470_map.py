#!/usr/bin/env python3
"""S17-C/A: exact Steane-sector -> 470-bit continuous-handoff boundary map.

Purpose
=======
The non-Clifford direct-|CS> core (S10/S12) exits in a coherent/classical mixture
of Steane syndrome sectors after the physical detector measurement.  Historical
S8-B/S8-C mapped the 64 canonical sectors into the older 171/195-bit ZXZ
recovery context.  S17-B v11/v13d replaced that recovery context with the actual
470-bit continuous detector history carried through d5->d7->d13.

Before finite-p E2E Monte Carlo, this stage rebuilds the boundary map in the NEW
operational basis:

    canonical Steane syndrome s
      -> R_470(s) detector offset
      -> raw terminal logical Pauli P_L(s) in {I,X,Z,Y}

The raw Pauli label is obtained from the transported complementary X/Z terminal
observables of the v11 circuit, NOT from historical S12 ``SectorRow.frame``.

Exact construction
==================
* Rebuild the clean v11 X and Z continuous-handoff circuits.
* Insert each of the 21 single-qubit Steane Pauli representatives immediately
  after Hirano's ideal encoded-Steane initialization and before the first ZXZ
  instruction.
* Extract exact 470-bit detector + terminal observable signatures with Stim.
* Build all 64 canonical Steane representatives by GF(2) XOR, using the same
  Steane recovery table as S8-B.
* Require X/Z detector signatures to agree for every physical incoming Pauli.
* Cross-check that the low 171 bits reproduce the frozen S8-B actual signatures.

This is an interface/semantics certificate.  It does NOT yet run finite-p E2E
Monte Carlo and it does NOT yet rebuild the sector-aware selective-rejection
policy; those are the next S17-C stages after this map passes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PAULI1 = ("X", "Y", "Z")
PAULI_NAMES = ("I", "X", "Z", "Y")
EXPECTED_SECTORS = 64
EXPECTED_WEIGHT_COUNTS = {0: 1, 1: 21, 2: 42}
EXPECTED_DETECTORS = 470
EXPECTED_HIRANO_PREFIX = 171


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
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
        return int(t.val)
    v = getattr(t, "value")
    return int(v() if callable(v) else v)


def _find_one(root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = (root / rel).resolve()
        if p.is_file() or p.is_dir():
            return p
    names = [Path(x).name for x in candidates]
    hits: list[Path] = []
    for n in names:
        hits.extend(root.rglob(n))
    hits = [p for p in hits if p.exists()]
    if not hits:
        raise FileNotFoundError(f"Could not find any of {candidates} under {root.resolve()}")
    # Prefer shallow paths and then newest copy.
    hits.sort(key=lambda p: (-len(p.parts), p.stat().st_mtime), reverse=True)
    return hits[0].resolve()


def _find_s8b_map(root: Path) -> Path | None:
    hits = list(root.rglob("s8b_canonical_input_map.json"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0].resolve()


def _annotation_observable_dependencies(circuit: Any, v7: Any) -> list[int]:
    _det, obs = v7.s7b._annotation_dependencies(circuit)
    if len(obs) != 1:
        raise AssertionError(("observables", len(obs), circuit.num_observables))
    return [int(x) for x in obs[0]]


def _build_clean_v11(v7: Any, v11: Any, hirano_src: Path, sahay_stim: Path, channel: str) -> tuple[Any, dict[str, Any]]:
    """Rebuild v11 with no stochastic noise, retaining the 470-bit basis + scorer."""
    lines = v7._read_source(sahay_stim)
    geo = v7._source_geometry(lines)
    wait_rounds = 2

    full, _wrapper, init_range, probe_start, online_ids, terminal_ids = v7.s7b._build_with_wait_rounds(
        hirano_src, channel, wait_rounds
    )
    pre_clean = full[:probe_start]
    pre_m = int(pre_clean.num_measurements)

    obs_all = _annotation_observable_dependencies(full, v7)
    hirano_preprobe_obs = [m for m in obs_all if m < pre_m]
    hirano_terminal_obs = [m for m in obs_all if m >= pre_m]
    if len(hirano_terminal_obs) != 5 or not hirano_preprobe_obs:
        raise AssertionError((channel, hirano_preprobe_obs, hirano_terminal_obs, pre_m))
    if len(online_ids) != 171 or int(pre_clean.num_detectors) != 171 or len(terminal_ids) != 12:
        raise AssertionError((channel, len(online_ids), pre_clean.num_detectors, len(terminal_ids)))

    boundary_instruction_index = int(init_range[1])
    if not (0 <= boundary_instruction_index <= len(pre_clean)):
        raise AssertionError(("bad Steane boundary", init_range, len(pre_clean)))

    qmap, map_meta = v7._build_qmap(pre_clean, geo)
    transport_flow = v7._solve_transport_flow(lines, geo, channel)

    g57_src = v7._clean_source_range(lines, v7.G57_START, v7.G57_END)
    g57_src, strip_meta = v11._strip_first_growth_detectors(g57_src)
    g713_src = v7._clean_source_range(lines, v7.G713_START, v7.G713_END)
    close_src = v7._clean_source_range(lines, v7.CLOSE_START, v7.CLOSE_END)

    g57_clean = v7._remap_circuit(g57_src, qmap)
    g713_clean = v7._remap_circuit(g713_src, qmap)
    close_clean = v7._remap_circuit(close_src, qmap)

    c = pre_clean + g57_clean + g713_clean + close_clean
    missing, method = v11._missing_detectors(c)
    transition, transition_meta = v11._complete_missing_detector_basis(
        missing,
        total_measurements=int(c.num_measurements),
        pre_growth_measurements=pre_m,
        first_growth_count=48,
    )
    c += transition

    if int(c.num_detectors) != EXPECTED_DETECTORS:
        raise AssertionError((channel, "preterminal detector count", c.num_detectors))
    # Require fully deterministic detector basis before terminal scorer.
    c.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)

    terminal_meta = v7._append_terminal_boundary(
        c,
        channel,
        geo,
        qmap,
        transport_measurement_offset=pre_m,
        transport_flow=transport_flow,
        hirano_preprobe_observable_measurements=hirano_preprobe_obs,
    )
    if int(c.num_detectors) != EXPECTED_DETECTORS or int(c.num_observables) != 1:
        raise AssertionError((channel, c.num_detectors, c.num_observables))

    return c, {
        "channel": channel,
        "boundary_instruction_index": boundary_instruction_index,
        "init_range": [int(init_range[0]), int(init_range[1])],
        "probe_start": int(probe_start),
        "measurements_before_growth": pre_m,
        "detectors": int(c.num_detectors),
        "observables": int(c.num_observables),
        "removed_first_growth_detectors": int(strip_meta["removed_first_growth_detectors"]),
        "synthesized_complete_basis_detectors": int(transition_meta["selected_complete_basis_detectors"]),
        "missing_detector_method": method,
        "mapping": map_meta,
        "transport_flow": transport_flow,
        "terminal": terminal_meta,
        "hirano_preprobe_observable_measurements": hirano_preprobe_obs,
    }


def _steane_qubits(s8b: Any, hirano_src: Path) -> tuple[list[tuple[int, int]], list[int]]:
    pos, ids = s8b._steane_qubits(hirano_src)
    return [tuple(int(x) for x in p) for p in pos], [int(x) for x in ids]


def _insert_single_error(circuit: Any, boundary: int, q: int, pauli: str, prob: float) -> Any:
    import stim
    if pauli not in PAULI1:
        raise ValueError(pauli)
    out = stim.Circuit()
    out += circuit[:boundary]
    out.append(f"{pauli}_ERROR", [int(q)], float(prob))
    out += circuit[boundary:]
    # The inserted instruction has no measurements, so detector/observable count must stay fixed.
    if int(out.num_detectors) != int(circuit.num_detectors) or int(out.num_observables) != int(circuit.num_observables):
        raise AssertionError("incoming Pauli insertion changed annotation inventory")
    return out


def _signature(circuit: Any) -> tuple[int, int]:
    dem = circuit.detector_error_model(decompose_errors=False, allow_gauge_detectors=False).flattened()
    errs = [inst for inst in dem if getattr(inst, "type", None) == "error"]
    if not errs:
        return 0, 0
    if len(errs) != 1:
        raise AssertionError(("expected <=1 incoming boundary DEM mechanism", len(errs), str(dem)[:2000]))
    dmask = 0
    obs = 0
    for t in errs[0].targets_copy():
        if _bool_attr(t, "is_relative_detector_id"):
            dmask ^= 1 << _target_value(t)
        elif _bool_attr(t, "is_logical_observable_id"):
            if _target_value(t) != 0:
                raise AssertionError(t)
            obs ^= 1
        elif _bool_attr(t, "is_separator"):
            raise AssertionError(("unexpected separator", errs[0]))
        else:
            raise AssertionError(t)
    return int(dmask), int(obs)


@dataclass(frozen=True)
class IncomingSingle:
    coordinate: int
    position: tuple[int, int]
    qubit_id: int
    pauli: str
    detector_470: int
    x_probe_observable_flip: int
    z_probe_observable_flip: int
    joint_pauli: int


def _build_new_single_map(xc: Any, zc: Any, xmeta: dict[str, Any], zmeta: dict[str, Any], s8b: Any, hirano_src: Path, prob: float) -> tuple[dict[tuple[int, str], IncomingSingle], dict[str, Any]]:
    pos, ids = _steane_qubits(s8b, hirano_src)
    if len(ids) != 7 or len(set(ids)) != 7:
        raise AssertionError(ids)
    out: dict[tuple[int, str], IncomingSingle] = {}
    mismatches = []

    for coord, q in enumerate(ids):
        for p in PAULI1:
            dx, ox = _signature(_insert_single_error(xc, int(xmeta["boundary_instruction_index"]), q, p, prob))
            dz, oz = _signature(_insert_single_error(zc, int(zmeta["boundary_instruction_index"]), q, p, prob))
            if dx != dz:
                mismatches.append({
                    "coordinate": coord, "qubit": q, "pauli": p,
                    "X_detector_hex": hex(dx), "Z_detector_hex": hex(dz),
                })
                continue
            # X-probe flip = logical Z component; Z-probe flip = logical X component.
            joint = int(oz) | (int(ox) << 1)
            out[(coord, p)] = IncomingSingle(
                coordinate=coord,
                position=pos[coord],
                qubit_id=q,
                pauli=p,
                detector_470=dx,
                x_probe_observable_flip=ox,
                z_probe_observable_flip=oz,
                joint_pauli=joint,
            )
    if mismatches:
        raise RuntimeError(f"X/Z detector mismatch for incoming Steane Pauli mechanisms: {mismatches[:10]}")
    if len(out) != 21:
        raise AssertionError(len(out))
    return out, {
        "incoming_single_pauli_classes": len(out),
        "xz_detector_mismatches": 0,
        "steane_positions": [list(x) for x in pos],
        "steane_qubit_ids": ids,
    }


def _load_old_s8b_rows(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["steane_syndrome"]): r for r in rows}


def _build_sector_rows(singles: dict[tuple[int, str], IncomingSingle], s8b: Any, old_rows: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix_mask = (1 << EXPECTED_HIRANO_PREFIX) - 1
    prefix_mismatches = []
    old_frame_mismatches = []

    for syn in range(EXPECTED_SECTORS):
        rep = s8b.s8live.RECOVERY_TABLE[syn]
        det = 0
        pauli = 0
        constituents = []
        for coord, p in enumerate(rep.pauli_string):
            if p == "I":
                continue
            r = singles[(coord, p)]
            det ^= int(r.detector_470)
            pauli ^= int(r.joint_pauli)
            constituents.append({"coordinate": coord, "pauli": p, "qubit_id": r.qubit_id})

        old = old_rows.get(syn)
        old_actual = int(old["actual_171_syndrome_hex"], 16) if old is not None else None
        old_frame = int(old["logical_frame_int"]) if old is not None else None
        if old_actual is not None and (det & prefix_mask) != old_actual:
            prefix_mismatches.append({
                "syndrome": syn,
                "new_prefix_hex": hex(det & prefix_mask),
                "old_actual_hex": hex(old_actual),
            })
        if old_frame is not None and int(pauli) != old_frame:
            old_frame_mismatches.append({
                "syndrome": syn,
                "old_recovery_context_frame": PAULI_NAMES[old_frame],
                "new_raw_terminal_pauli": PAULI_NAMES[pauli],
            })

        rows.append({
            "steane_syndrome": syn,
            "steane_syndrome_bits": format(syn, "06b"),
            "canonical_pauli": str(rep.pauli_string),
            "canonical_weight": int(rep.weight),
            "constituents": constituents,
            "detector_470_hex": hex(det),
            "detector_470_weight": int(det.bit_count()),
            "hirano_prefix_171_hex": hex(det & prefix_mask),
            "hirano_prefix_171_weight": int((det & prefix_mask).bit_count()),
            "raw_terminal_pauli": PAULI_NAMES[pauli],
            "raw_terminal_pauli_int": int(pauli),
            "historical_s8b_recovery_context_frame": (PAULI_NAMES[old_frame] if old_frame is not None else None),
            "historical_s8b_recovery_context_frame_int": old_frame,
        })

    sig_groups: dict[str, list[int]] = {}
    for r in rows:
        sig_groups.setdefault(r["detector_470_hex"], []).append(int(r["steane_syndrome"]))
    collisions = {k: v for k, v in sig_groups.items() if len(v) > 1}
    dmin = min(
        (int(a["detector_470_hex"], 16) ^ int(b["detector_470_hex"], 16)).bit_count()
        for i, a in enumerate(rows) for b in rows[i + 1:]
    )
    wc = Counter(int(r["canonical_weight"]) for r in rows)
    frame_counts = Counter(str(r["raw_terminal_pauli"]) for r in rows)
    if dict(sorted(wc.items())) != EXPECTED_WEIGHT_COUNTS:
        raise AssertionError((wc, EXPECTED_WEIGHT_COUNTS))

    z = rows[0]
    zero_ok = int(z["detector_470_hex"], 16) == 0 and z["raw_terminal_pauli"] == "I"
    summary = {
        "canonical_input_sectors": len(rows),
        "canonical_weight_counts": {str(k): int(v) for k, v in sorted(wc.items())},
        "detector_bits": EXPECTED_DETECTORS,
        "signature_unique_count": len(sig_groups),
        "signature_injective": not collisions and len(sig_groups) == EXPECTED_SECTORS,
        "signature_collision_groups": collisions,
        "minimum_470_signature_hamming_distance": int(dmin),
        "nontrivial_sectors_with_zero_470_signature": sum(
            int(r["steane_syndrome"]) != 0 and int(r["detector_470_hex"], 16) == 0 for r in rows
        ),
        "zero_sector_signature_is_zero_and_raw_pauli_I": bool(zero_ok),
        "raw_terminal_pauli_distribution": dict(sorted(frame_counts.items())),
        "historical_s8b_map_loaded": bool(old_rows),
        "historical_171_prefix_mismatch_count": len(prefix_mismatches),
        "historical_171_prefix_mismatch_examples": prefix_mismatches[:20],
        "historical_recovery_context_frame_vs_new_raw_terminal_pauli_mismatch_count": len(old_frame_mismatches),
        "historical_recovery_context_frame_vs_new_raw_terminal_pauli_mismatch_examples": old_frame_mismatches[:30],
    }
    return rows, summary


def _selftest() -> dict[str, Any]:
    # Bit convention guard: X probe -> Z component, Z probe -> X component.
    labels = {}
    for ox in (0, 1):
        for oz in (0, 1):
            p = oz | (ox << 1)
            labels[f"xprobe{ox}_zprobe{oz}"] = PAULI_NAMES[p]
    assert labels == {
        "xprobe0_zprobe0": "I",
        "xprobe0_zprobe1": "X",
        "xprobe1_zprobe0": "Z",
        "xprobe1_zprobe1": "Y",
    }
    return {"passed": True, "joint_probe_label_guard": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--v7-script", type=Path, default=None)
    ap.add_argument("--v11-script", type=Path, default=None)
    ap.add_argument("--s8b-script", type=Path, default=None)
    ap.add_argument("--hirano-src", type=Path, default=None)
    ap.add_argument("--sahay-stim", type=Path, default=None)
    ap.add_argument("--historical-s8b-map", type=Path, default=None)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    st = _selftest()
    _dump(args.out_dir / "s17c_a_self_test.json", st)
    if args.self_test:
        print(json.dumps(st, indent=2))
        print("S17-C/A SELF-TEST PASS")
        return

    root = args.search_root.resolve()
    v7_path = (args.v7_script.resolve() if args.v7_script else _find_one(root, [
        "stim_pipeline/interface/direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py",
        "direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py",
    ]))
    v11_path = (args.v11_script.resolve() if args.v11_script else _find_one(root, [
        "stim_pipeline/interface/direct_cs_s17b_v11_complete_missing_basis.py",
        "direct_cs_s17b_v11_complete_missing_basis.py",
    ]))
    s8b_path = (args.s8b_script.resolve() if args.s8b_script else _find_one(root, [
        "stim_pipeline/interface/zxz_core_boundary_signature_map.py",
    ]))
    hirano_src = (args.hirano_src.resolve() if args.hirano_src else _find_one(root, ["msc-ls/src"]))
    sahay_stim = (args.sahay_stim.resolve() if args.sahay_stim else _find_one(root, ["reference/fd3_baseline_ckt.stim"]))
    old_map_path = args.historical_s8b_map.resolve() if args.historical_s8b_map else _find_s8b_map(root)

    print(f"S17-C/A v7: {v7_path}", flush=True)
    print(f"S17-C/A v11: {v11_path}", flush=True)
    print(f"S17-C/A S8-B: {s8b_path}", flush=True)
    print(f"S17-C/A Hirano src: {hirano_src}", flush=True)
    print(f"S17-C/A Sahay source: {sahay_stim}", flush=True)
    print(f"S17-C/A historical S8-B map: {old_map_path}", flush=True)

    v7 = _load("s17c_a_v7", v7_path)
    v11 = _load("s17c_a_v11", v11_path)
    s8b = _load("s17c_a_s8b", s8b_path)

    print("S17-C/A: rebuilding clean X continuous-handoff circuit...", flush=True)
    xc, xm = _build_clean_v11(v7, v11, hirano_src, sahay_stim, "X")
    print("S17-C/A: rebuilding clean Z continuous-handoff circuit...", flush=True)
    zc, zm = _build_clean_v11(v7, v11, hirano_src, sahay_stim, "Z")

    # The detector basis itself must be channel-independent.
    x_det_lines = [str(i) for i in xc if hasattr(i, "name") and i.name == "DETECTOR"]
    z_det_lines = [str(i) for i in zc if hasattr(i, "name") and i.name == "DETECTOR"]
    if x_det_lines != z_det_lines or len(x_det_lines) != EXPECTED_DETECTORS:
        raise RuntimeError(("X/Z 470 detector definitions differ", len(x_det_lines), len(z_det_lines)))

    print("S17-C/A: extracting 21 incoming Steane Pauli signatures in complementary probes...", flush=True)
    singles, single_meta = _build_new_single_map(xc, zc, xm, zm, s8b, hirano_src, float(args.signature_p))
    old_rows = _load_old_s8b_rows(old_map_path)
    rows, sector_meta = _build_sector_rows(singles, s8b, old_rows)

    gate = {
        "xz_incoming_detector_signature_match": single_meta["xz_detector_mismatches"] == 0,
        "all_64_sector_signatures_injective": bool(sector_meta["signature_injective"]),
        "zero_sector_is_zero_record_and_identity": bool(sector_meta["zero_sector_signature_is_zero_and_raw_pauli_I"]),
        "no_nontrivial_zero_record_sector": int(sector_meta["nontrivial_sectors_with_zero_470_signature"]) == 0,
        "historical_171_prefix_compatibility": (
            not old_rows or int(sector_meta["historical_171_prefix_mismatch_count"]) == 0
        ),
    }
    gate["s17c_a_sector_to_470_map_gate_pass"] = all(bool(v) for v in gate.values())

    single_rows = []
    for key in sorted(singles):
        r = singles[key]
        d = asdict(r)
        d["position"] = list(r.position)
        d["detector_470_hex"] = hex(int(r.detector_470))
        d["joint_pauli_label"] = PAULI_NAMES[int(r.joint_pauli)]
        single_rows.append(d)

    summary = {
        "stage": "S17-C/A exact Steane-sector to 470-bit continuous-handoff map",
        "inputs": {
            "v7_script": str(v7_path),
            "v11_script": str(v11_path),
            "s8b_script": str(s8b_path),
            "hirano_src": str(hirano_src),
            "sahay_stim": str(sahay_stim),
            "historical_s8b_map": str(old_map_path) if old_map_path else None,
            "signature_p": float(args.signature_p),
        },
        "channel_build": {"X": xm, "Z": zm},
        "incoming_single_map": single_meta,
        "sector_map": sector_meta,
        "main_gate": gate,
        "semantics": {
            "detector_record": "470-bit v11/v13d continuous operational history: 171 Hirano/live bits + growth/close deterministic basis",
            "raw_terminal_pauli": "paired transported terminal observables; X-probe flip is Z component and Z-probe flip is X component",
            "historical_frame_warning": "S12/S8-C SectorRow.frame is retained only as a diagnostic comparison and is not used to define the new raw terminal Pauli",
            "finite_p_status": "not run in S17-C/A",
        },
        "next_if_pass": "Build S17-C/B sector-aware 470-bit selective-rejection policy for boundary_weight + downstream_fault_locations <= 2, then integrate finite-p non-Clifford core trajectories.",
    }

    _dump(args.out_dir / "s17c_a_incoming_single_map.json", single_rows)
    _dump(args.out_dir / "s17c_a_sector_map.json", rows)
    _dump(args.out_dir / "s17c_a_summary.json", summary)

    print(json.dumps({
        "gate_pass": gate["s17c_a_sector_to_470_map_gate_pass"],
        "unique_sector_signatures": sector_meta["signature_unique_count"],
        "minimum_hamming_distance": sector_meta["minimum_470_signature_hamming_distance"],
        "prefix171_mismatches": sector_meta["historical_171_prefix_mismatch_count"],
        "old_frame_vs_raw_pauli_mismatches": sector_meta["historical_recovery_context_frame_vs_new_raw_terminal_pauli_mismatch_count"],
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)
    if gate["s17c_a_sector_to_470_map_gate_pass"]:
        print("S17-C/A PASS: exact 64-sector -> 470-bit operational boundary map established.", flush=True)
    else:
        print("S17-C/A FAIL: inspect s17c_a_summary.json before any finite-p E2E simulation.", flush=True)


if __name__ == "__main__":
    main()
