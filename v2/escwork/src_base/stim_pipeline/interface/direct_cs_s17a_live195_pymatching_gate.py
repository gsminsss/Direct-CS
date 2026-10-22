#!/usr/bin/env python3
"""S17-A: physicalize the missing 24-bit handoff tail and audit live-195 decoding.

Scientific motivation
=====================
S12 used 171 *live* detector bits for accept/reject, but its frame decoder used
195 bits = 171 live bits + a noiseless 24-bit future-audit tail.  A later S16
experiment showed that simply carrying the 171-bit record through d=5->13 code
growth is not enough to infer the *S12 target Pauli frame*: target-frame-aware
single-fault collisions already exist.  Therefore the 24 future bits must be
physicalized (or otherwise replaced) before a paper-final operational decoder
claim is made.

This script tests the minimal physicalization:

    Hirano ZXZ
      -> 3 FAULTABLE full d=5 surface-syndrome rounds
      -> LIVE-195 boundary
      -> ideal destructive complementary X/Z probe (SCORING ONLY)

The third real round adds exactly 24 live detector bits relative to S12's
2-round live-171 boundary.  The terminal destructive probe is never in the
fault budget.  Its 12 terminal DETECTOR annotations are removed before the
PyMatching audit; OBSERVABLE_INCLUDE is retained only to provide the ground-
truth logical bit.  Thus PyMatching receives exactly the 195 live detector
bits and no ideal future syndrome information.

Outputs
=======
1. Exact active-location <=2 Pauli-frame ambiguity audit at the LIVE-195
   boundary, using paired complementary X/Z scoring circuits.
2. Per-channel graphlike DEM audit and exact-single PyMatching audit.
3. Finite-p physical-circuit Monte Carlo for PyMatching at user-selected p.
4. The existing exact-lookup/selective decoder Monte Carlo from S7-B, using
   the same paired 4-valued Pauli-frame instrument.  This is reported
   separately from PyMatching and must not be conflated with it.

Interpretation gate
===================
A strong PASS requires:
  * no X/Z online-syndrome mismatch;
  * no ambiguous single-fault LIVE-195 syndrome;
  * zero syndrome has only identity through <=2;
  * no LIVE-195 syndrome has multiple Pauli frames through <=2.

If the last condition fails, the script still reports a selective exact <=2
policy (reject ambiguous keys) and PyMatching diagnostics.  It does not hide
or silently repair a structural ambiguity.

Noise model
===========
Exactly the existing project canonical active-location local-Pauli model used
by zxz_online_handoff_audit.py:
  * CX location: DEPOLARIZE2(p), 15 equiprobable nonidentity Pauli classes;
  * RX/RZ/reset-like 1q active locations: canonical single-Pauli insertion;
  * MX/MZ readout: canonical pre-measurement flip;
  * perfect Steane initialization excluded from the fault budget;
  * terminal destructive scorer excluded from the fault budget;
  * no idles/leakage/coherent/calibration/correlated-multilocation mechanisms.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
# Expected install location: <repo>/stim_pipeline/interface/this_file.py
REPO = HERE.parent.parent if HERE.name == "interface" else HERE.parent
ZXZ_DIR = REPO / "stim_pipeline" / "zxz"
SRC_CURRENT = REPO / "src_current"
INTERFACE = REPO / "stim_pipeline" / "interface"
for pth in reversed((INTERFACE, ZXZ_DIR, SRC_CURRENT, HERE)):
    if pth.exists() and str(pth) not in sys.path:
        sys.path.insert(0, str(pth))

try:
    import zxz_online_handoff_audit as s7b  # type: ignore
    import zxz_exact_fault_audit as s3  # type: ignore
    import zxz_pymatching_audit as s5  # type: ignore
except Exception as ex:  # pragma: no cover - actionable runtime error
    raise RuntimeError(
        "Could not import the existing Direct-CS pipeline modules. Place this "
        "file in stim_pipeline/interface/ inside the provided source tree."
    ) from ex

PAULI_LABELS = tuple(getattr(s7b, "PAULI_LABELS", ("I", "X", "Z", "Y")))
EXPECTED_WAIT_ROUNDS = 3
EXPECTED_ONLINE_DETECTORS = 123 + 3 * 24  # 195
EXPECTED_TERMINAL_DETECTORS = 12
EXPECTED_LOCATIONS = 772 + 3 * 128  # 1156
EXPECTED_SINGLE_CLASSES = 7254 + 3 * 1248  # 10998
EXPECTED_OBSERVABLES = 1


def _version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in x]
    return x


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2, sort_keys=True), encoding="utf-8")


def _wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def _mask_to_row(mask: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.uint8)
    x = int(mask)
    while x:
        lsb = x & -x
        i = lsb.bit_length() - 1
        if i >= n:
            raise AssertionError(("detector bit out of range", i, n, hex(mask)))
        out[i] = 1
        x ^= lsb
    return out


def _drop_terminal_detectors(circuit: Any, terminal_ids: list[int]) -> Any:
    """Remove only terminal DETECTOR annotations; keep scorer measurements/observable."""
    import stim

    terminal = {int(x) for x in terminal_ids}
    out = stim.Circuit()
    did = 0
    dropped = 0
    for inst in circuit:
        if not hasattr(inst, "name"):
            raise AssertionError("unexpected repeat block")
        if inst.name == "DETECTOR":
            if did in terminal:
                dropped += 1
                did += 1
                continue
            did += 1
        out.append(inst)
    if did != int(circuit.num_detectors):
        raise AssertionError(("detector annotation count", did, circuit.num_detectors))
    if dropped != len(terminal):
        raise AssertionError(("terminal detector drop count", dropped, len(terminal)))
    if out.num_detectors != EXPECTED_ONLINE_DETECTORS:
        raise AssertionError(("operational detector count", out.num_detectors, EXPECTED_ONLINE_DETECTORS))
    if out.num_observables != EXPECTED_OBSERVABLES:
        raise AssertionError(("observable count", out.num_observables, EXPECTED_OBSERVABLES))
    return out


def _single_ambiguity(rows: list[Any]) -> tuple[int, int, set[int]]:
    by_syn: dict[int, set[int]] = defaultdict(set)
    cnt: dict[int, int] = defaultdict(int)
    for r in rows:
        by_syn[int(r.detector_mask)].add(int(r.observable_flip))
        cnt[int(r.detector_mask)] += 1
    amb = {s for s, obs in by_syn.items() if len(obs) > 1}
    return len(amb), sum(cnt[s] for s in amb), amb


def _pymatching_exact_single(matching: Any, rows: list[Any], n_detectors: int, max_examples: int = 30) -> dict[str, Any]:
    synd = np.zeros((len(rows), n_detectors), dtype=np.uint8)
    for i, r in enumerate(rows):
        synd[i, :] = _mask_to_row(int(r.detector_mask), n_detectors)
    pred = np.asarray(matching.decode_batch(synd), dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.shape[0] != len(rows) or pred.shape[1] < 1:
        raise AssertionError(("PyMatching prediction shape", pred.shape, len(rows)))

    amb_groups, amb_faults, amb_masks = _single_ambiguity(rows)
    failures = 0
    zero_nontrivial = 0
    examples: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        actual = int(r.observable_flip)
        guess = int(pred[i, 0])
        if int(r.detector_mask) == 0 and actual:
            zero_nontrivial += 1
        if actual != guess:
            failures += 1
            if len(examples) < max_examples:
                examples.append({
                    "fault_id": int(r.fault_id),
                    "location_id": int(r.location_id),
                    "tick": int(r.tick),
                    "gate": str(r.gate),
                    "qubits": [int(q) for q in r.qubits],
                    "fault_class": str(r.fault_class),
                    "detector_mask_hex": hex(int(r.detector_mask)),
                    "detector_weight": int(r.detector_weight),
                    "actual_observable": actual,
                    "predicted_observable": guess,
                    "inherently_ambiguous_single_syndrome": int(r.detector_mask) in amb_masks,
                })
    return {
        "single_fault_classes": len(rows),
        "failures": failures,
        "successes": len(rows) - failures,
        "failure_fraction": failures / len(rows) if rows else None,
        "inherent_ambiguous_single_syndrome_groups": amb_groups,
        "inherent_ambiguous_single_fault_classes": amb_faults,
        "zero_syndrome_nontrivial_single_fault_classes": zero_nontrivial,
        "failure_examples": examples,
    }


def _build_live195_exact(hirano_src: Path, signature_p: float) -> tuple[list[Any], list[Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    xrows, xloc, xmeta = s7b._build_channel_rows(hirano_src, "X", EXPECTED_WAIT_ROUNDS, signature_p)
    zrows, zloc, zmeta = s7b._build_channel_rows(hirano_src, "Z", EXPECTED_WAIT_ROUNDS, signature_p)

    if len(xloc) != EXPECTED_LOCATIONS or len(zloc) != EXPECTED_LOCATIONS:
        raise AssertionError(("physical locations", len(xloc), len(zloc), EXPECTED_LOCATIONS))
    if len(xrows) != EXPECTED_SINGLE_CLASSES or len(zrows) != EXPECTED_SINGLE_CLASSES:
        raise AssertionError(("single classes", len(xrows), len(zrows), EXPECTED_SINGLE_CLASSES))
    if len(xmeta["online_ids"]) != EXPECTED_ONLINE_DETECTORS or len(zmeta["online_ids"]) != EXPECTED_ONLINE_DETECTORS:
        raise AssertionError("LIVE detector count is not 195")
    if len(xmeta["terminal_ids"]) != EXPECTED_TERMINAL_DETECTORS or len(zmeta["terminal_ids"]) != EXPECTED_TERMINAL_DETECTORS:
        raise AssertionError("terminal scoring detector count drift")

    for a, b in zip(xloc, zloc):
        ma = (a.location_id, a.tick, a.gate, a.qubits, a.class_count)
        mb = (b.location_id, b.tick, b.gate, b.qubits, b.class_count)
        if ma != mb:
            raise AssertionError(("X/Z physical-location mismatch", ma, mb))

    paired, mismatches = s7b._pair_complementary(xrows, zrows)
    analysis = s7b._analyze_le2(paired, xloc)
    exact = {
        "wait_rounds": EXPECTED_WAIT_ROUNDS,
        "live_detector_count": EXPECTED_ONLINE_DETECTORS,
        "terminal_scoring_detector_count": EXPECTED_TERMINAL_DETECTORS,
        "physical_locations_per_patch": len(xloc),
        "single_fault_classes": len(paired),
        "two_fault_class_pairs": int(s7b._pair_denominator(xloc)),
        "location_counts": xmeta["location_counts"],
        "xz_online_syndrome_mismatches": int(mismatches),
        "single_syndromes": int(analysis["single_syndromes"]),
        "ambiguous_single_syndromes": int(analysis["ambiguous_single_syndromes"]),
        "ambiguous_single_fault_classes": int(analysis["ambiguous_single_fault_classes"]),
        "zero_syndrome_single_nontrivial_paulis": int(analysis["zero_syndrome_single_nontrivial_paulis"]),
        "unique_syndromes_le2": int(analysis["unique_syndromes_le2"]),
        "ambiguous_syndromes_le2": int(analysis["ambiguous_syndromes_le2"]),
        "zero_syndrome_pauli_set_le2": analysis["zero_syndrome_pauli_set_le2"],
        "selective_rejected_single_fault_classes": int(analysis["selective_rejected_single_fault_classes"]),
        "selective_single_reject_coefficient": float(analysis["selective_single_reject_coefficient"]),
        "selective_exact_accepted_single_failures": int(analysis["selective_exact_accepted_single_failures"]),
        "selective_exact_accepted_pair_failures": int(analysis["selective_exact_accepted_pair_failures"]),
        "ambiguity_witnesses": analysis["ambiguity_witnesses"],
    }
    exact["strong_live195_unique_le2_gate_pass"] = bool(
        exact["xz_online_syndrome_mismatches"] == 0
        and exact["ambiguous_single_syndromes"] == 0
        and exact["zero_syndrome_single_nontrivial_paulis"] == 0
        and exact["ambiguous_syndromes_le2"] == 0
        and exact["zero_syndrome_pauli_set_le2"] == ["I"]
    )
    exact["selective_live195_df3_gate_pass"] = bool(
        exact["xz_online_syndrome_mismatches"] == 0
        and exact["zero_syndrome_single_nontrivial_paulis"] == 0
        and exact["selective_exact_accepted_single_failures"] == 0
        and exact["selective_exact_accepted_pair_failures"] == 0
        and exact["zero_syndrome_pauli_set_le2"] == ["I"]
    )
    return paired, xloc, xmeta, zmeta, {"exact": exact, "analysis": analysis, "xrows": xrows, "zrows": zrows}


def _matching_from_noisy_operational(noisy_with_scorer_detectors: Any, terminal_ids: list[int]) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import pymatching

    operational = _drop_terminal_detectors(noisy_with_scorer_detectors, terminal_ids)
    raw_dem = operational.detector_error_model(decompose_errors=False)
    raw_stats = s5.audit_dem_structure(raw_dem)
    graph_dem = operational.detector_error_model(decompose_errors=True)
    graph_stats = s5.audit_dem_structure(graph_dem)
    if graph_stats["bad_components"] != 0:
        raise AssertionError(("non-graphlike components remain", graph_stats))
    matching = pymatching.Matching.from_detector_error_model(graph_dem)
    matching.ensure_num_fault_ids(EXPECTED_OBSERVABLES)
    if int(matching.num_detectors) != EXPECTED_ONLINE_DETECTORS:
        raise AssertionError(("matching detectors", matching.num_detectors, EXPECTED_ONLINE_DETECTORS))
    if int(matching.num_fault_ids) != EXPECTED_OBSERVABLES:
        raise AssertionError(("matching fault ids", matching.num_fault_ids, EXPECTED_OBSERVABLES))
    return operational, matching, raw_stats, graph_stats


def _pymatching_signature_audit(xmeta: dict[str, Any], zmeta: dict[str, Any], xrows: list[Any], zrows: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for channel, meta, rows in (("X", xmeta, xrows), ("Z", zmeta, zrows)):
        operational, matching, raw_stats, graph_stats = _matching_from_noisy_operational(
            meta["noisy"], list(meta["terminal_ids"])
        )
        exact_single = _pymatching_exact_single(matching, rows, EXPECTED_ONLINE_DETECTORS)
        out[channel] = {
            "operational_detectors": int(operational.num_detectors),
            "operational_observables": int(operational.num_observables),
            "raw_dem": raw_stats,
            "graphlike_dem": graph_stats,
            "matching_num_detectors": int(matching.num_detectors),
            "matching_num_nodes": int(matching.num_nodes),
            "matching_num_edges": int(matching.num_edges),
            "matching_num_fault_ids": int(matching.num_fault_ids),
            "exact_single": exact_single,
        }
    out["all_single_faults_corrected_by_mwpm"] = bool(
        out["X"]["exact_single"]["failures"] == 0
        and out["Z"]["exact_single"]["failures"] == 0
    )
    return out


def _build_operational_noisy(hirano_src: Path, channel: str, p: float) -> tuple[Any, list[Any]]:
    clean, _wrapper, init_range, probe_start, online_ids, terminal_ids = s7b._build_with_wait_rounds(
        hirano_src, channel, EXPECTED_WAIT_ROUNDS
    )
    if len(online_ids) != EXPECTED_ONLINE_DETECTORS:
        raise AssertionError((channel, len(online_ids)))
    noisy, locations, _noise_map = s7b._instrument_before_probe(clean, init_range, probe_start, p)
    if len(locations) != EXPECTED_LOCATIONS:
        raise AssertionError((channel, "locations", len(locations)))
    operational = _drop_terminal_detectors(noisy, terminal_ids)
    return operational, locations


def _mc_one_channel(hirano_src: Path, channel: str, p: float, shots: int, seed: int) -> dict[str, Any]:
    operational, locations = _build_operational_noisy(hirano_src, channel, p)

    # Build directly because terminal scoring DETECTOR annotations were already removed.
    import pymatching
    raw_dem = operational.detector_error_model(decompose_errors=False)
    raw_stats = s5.audit_dem_structure(raw_dem)
    graph_dem = operational.detector_error_model(decompose_errors=True)
    graph_stats = s5.audit_dem_structure(graph_dem)
    if graph_stats["bad_components"] != 0:
        raise AssertionError((channel, p, "non-graphlike decomposed DEM", graph_stats))
    matching = pymatching.Matching.from_detector_error_model(graph_dem)
    matching.ensure_num_fault_ids(1)

    sampler = operational.compile_detector_sampler(seed=int(seed))
    dets, obs = sampler.sample(shots=int(shots), separate_observables=True)
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    if obs.ndim == 1:
        obs = obs[:, None]
    if dets.shape != (shots, EXPECTED_ONLINE_DETECTORS):
        raise AssertionError((channel, p, "detector sample shape", dets.shape))
    if obs.shape != (shots, 1):
        raise AssertionError((channel, p, "observable sample shape", obs.shape))

    pred = np.asarray(matching.decode_batch(dets), dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.shape != obs.shape:
        raise AssertionError((channel, p, "prediction shape", pred.shape, obs.shape))
    failures = np.any(pred != obs, axis=1)
    k = int(np.sum(failures))
    zero = ~np.any(dets.astype(bool), axis=1)
    nzero = int(np.sum(zero))
    cert_err = int(np.sum(obs[zero, 0])) if nzero else 0
    return {
        "channel": channel,
        "p": float(p),
        "shots": int(shots),
        "seed": int(seed),
        "physical_locations": len(locations),
        "detectors": int(operational.num_detectors),
        "observables": int(operational.num_observables),
        "raw_dem": raw_stats,
        "graphlike_dem": graph_stats,
        "matching_num_nodes": int(matching.num_nodes),
        "matching_num_edges": int(matching.num_edges),
        "mwpm_logical_errors": k,
        "mwpm_logical_error_rate": k / shots if shots else None,
        "mwpm_logical_error_wilson95": _wilson(k, shots),
        "zero_live195_shots": nzero,
        "zero_live195_fraction": nzero / shots if shots else None,
        "zero_live195_logical_errors": cert_err,
        "zero_live195_conditional_logical_error": cert_err / nzero if nzero else None,
        "zero_live195_conditional_logical_error_wilson95": _wilson(cert_err, nzero),
    }


def _write_exact_catalog(path: Path, paired: list[Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "fault_id", "location_id", "tick", "gate", "qubits", "fault_class",
            "live195_syndrome_hex", "target_pauli_frame",
        ])
        for r in paired:
            w.writerow([
                int(r.fault_id), int(r.location_id), int(r.tick), str(r.gate),
                " ".join(map(str, r.qubits)), str(r.fault_class), hex(int(r.syndrome)),
                PAULI_LABELS[int(r.residual_pauli)],
            ])


def _self_test() -> None:
    # Static project fingerprints.
    assert EXPECTED_ONLINE_DETECTORS == 195
    assert EXPECTED_LOCATIONS == 1156
    assert EXPECTED_SINGLE_CLASSES == 10998
    s7b._self_test_pauli_packing()
    # Reuse established Stim/PyMatching smoke tests.
    s3._self_test_noise_append_fusion()
    s3._self_test_explainer_api()
    s5._self_test_pymatching()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", required=True, type=Path)
    ap.add_argument("--signature-p", type=float, default=0.001)
    ap.add_argument("--p-values", type=float, nargs="+", default=[0.0001, 0.0002, 0.0005, 0.001, 0.002])
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=[20000, 30000, 50000, 100000, 50000])
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--skip-mc", action="store_true", help="run exact + PyMatching signature gate only")
    ap.add_argument("--require-strong-le2-unique", action="store_true", help="exit nonzero if LIVE-195 is not fully frame-unique through <=2")
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s17a_results"))
    args = ap.parse_args()

    if len(args.p_values) != len(args.shots_per_p):
        raise SystemExit("--p-values and --shots-per-p must have the same length")
    if any(s <= 0 for s in args.shots_per_p):
        raise SystemExit("all --shots-per-p values must be >0")
    if not (0 < args.signature_p < 1):
        raise SystemExit("--signature-p must be in (0,1)")

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _self_test()

    print("[S17-A] Building exact LIVE-195 paired X/Z fault catalogue...")
    paired, locations, xmeta, zmeta, payload = _build_live195_exact(args.hirano_src, args.signature_p)
    exact = payload["exact"]
    _dump(args.out_dir / "s17a_live195_exact_le2_summary.json", exact)
    _write_exact_catalog(args.out_dir / "s17a_live195_single_fault_catalog.csv", paired)
    print(json.dumps(_json_safe(exact), indent=2, sort_keys=True))

    print("[S17-A] Building graphlike DEMs and auditing exact-single PyMatching...")
    pymatch_available = True
    try:
        pymatch_sig = _pymatching_signature_audit(
            xmeta, zmeta, payload["xrows"], payload["zrows"]
        )
    except Exception as ex:
        pymatch_available = False
        pymatch_sig = {
            "available": False,
            "error_type": type(ex).__name__,
            "error": str(ex),
            "interpretation": (
                "LIVE-195 exact ambiguity results remain valid, but a graphlike "
                "PyMatching model could not be constructed. Inspect this error before MC."
            ),
            "all_single_faults_corrected_by_mwpm": False,
        }
    _dump(args.out_dir / "s17a_pymatching_signature_summary.json", pymatch_sig)
    print(json.dumps(_json_safe(pymatch_sig), indent=2, sort_keys=True))

    exact_lookup_mc = []
    pymatching_mc = []
    if not args.skip_mc:
        print("[S17-A] Running finite-p selective exact-lookup + physical-circuit PyMatching MC...")
        for i, (p, shots) in enumerate(zip(args.p_values, args.shots_per_p)):
            p = float(p); shots = int(shots)
            # Paired 4-valued exact/selective decoder MC using the exact physical fault catalogue.
            emc = s7b._mc(
                paired, locations, payload["analysis"]["states"], p, shots,
                int(args.seed + 10000 + i),
            )
            exact_lookup_mc.append({"p": p, "shots": shots, **emc})

            # Operational MWPM channels: scorer terminal DETECTORs removed, live195 only.
            if pymatch_available:
                for ci, channel in enumerate(("X", "Z")):
                    print(f"  p={p:g}, shots={shots}, channel={channel}")
                    row = _mc_one_channel(
                        args.hirano_src, channel, p, shots,
                        int(args.seed + i * 1000 + ci * 100000),
                    )
                    pymatching_mc.append(row)

        _dump(args.out_dir / "s17a_selective_exact_lookup_mc.json", exact_lookup_mc)
        _dump(args.out_dir / "s17a_pymatching_mc.json", pymatching_mc)

    summary = {
        "stage": "S17-A LIVE-195 physical-tail + PyMatching gate",
        "stim_version": _version("stim"),
        "pymatching_version": _version("pymatching"),
        "numpy_version": np.__version__,
        "source_hirano": str(args.hirano_src),
        "faultable_surface_rounds_after_zxz": 3,
        "live_detector_bits": 195,
        "ideal_future_detector_bits_used_by_decoder": 0,
        "terminal_probe_role": "ideal scoring only; terminal DETECTOR annotations removed; OBSERVABLE_INCLUDE retained as ground truth",
        "physical_locations_per_patch_at_live195": EXPECTED_LOCATIONS,
        "increment_vs_s12_live171_per_patch": 128,
        "increment_vs_s12_live171_two_patches": 256,
        "exact_gate": exact,
        "pymatching_signature_gate": pymatch_sig,
        "strong_live195_unique_le2_gate_pass": bool(exact["strong_live195_unique_le2_gate_pass"]),
        "selective_live195_df3_gate_pass": bool(exact["selective_live195_df3_gate_pass"]),
        "pymatching_graphlike_model_available": bool(pymatch_available),
        "pymatching_all_single_faults_corrected": bool(pymatch_sig.get("all_single_faults_corrected_by_mwpm", False)),
        "mc_was_run": not args.skip_mc,
        "pymatching_mc_was_run": bool((not args.skip_mc) and pymatch_available),
        "p_values": [float(x) for x in args.p_values],
        "shots_per_p": [int(x) for x in args.shots_per_p],
        "seed": int(args.seed),
        "elapsed_seconds": time.time() - t0,
        "important_scope": [
            "This is an operational d=5 handoff decoder gate, not yet the complete S12 factory + d=5->13 growth finite-p result.",
            "Do not reuse the S16 9037.8 no-live-rejection proxy unless a later target-frame-aware end-to-end stage revalidates it.",
            "If strong uniqueness fails but selective_live195_df3_gate_pass is true, ambiguous LIVE-195 keys must still be rejected.",
        ],
    }
    _dump(args.out_dir / "s17a_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))

    if args.require_strong_le2_unique and not exact["strong_live195_unique_le2_gate_pass"]:
        raise SystemExit(2)
    print("S17-A COMPLETE: LIVE-195 exact/PyMatching gate finished; inspect s17a_summary.json before S17-B.")


if __name__ == "__main__":
    main()
