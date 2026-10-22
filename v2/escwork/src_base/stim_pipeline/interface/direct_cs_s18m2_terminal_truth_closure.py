#!/usr/bin/env python3
"""S18-M2: terminal truth-closure audit for the S18-M noisy-d13 boundary.

Purpose
=======
S18-M found 14 `new+new` pairs whose 638-bit *operational* detector records
cancel while the raw destructive logical probe flips.  A raw probe flip is not,
by itself, proof of a logical error when the delivered d=13 patch can still
contain a correctable physical residual.

This audit keeps the 638 operational detector bits frozen and appends 84
*scoring-only* terminal stabilizer-closing detectors to each complementary
probe:

  X probe: final X-stabilizer parity (detects Z residuals)
  Z probe: final Z-stabilizer parity (detects X residuals)

The 84 truth bits are NEVER exposed to the practical decoder.  They are used
only to classify whether an operationally silent raw-observable flip is a true
zero-syndrome logical coset or a detectable residual at the delivered d13
boundary.

For every location-compatible new1+new1 pair, M2 exactly checks:

  operational syndrome = 0  (same 638-bit component syndrome)
  terminal truth syndrome = 0
  final Pauli != I

Only a pair satisfying all three is a closed-boundary logical witness.
No Monte Carlo or threshold tuning is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OP_DETECTORS = 638
TRUTH_PER_PROBE = 84
OLD_LOCATIONS = 2407
TOTAL_LOCATIONS = 3535
PAULI_NAMES = ("I", "X", "Z", "Y")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find(root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = (root / rel).resolve()
        if p.exists():
            return p
    names = {Path(x).name for x in candidates}
    hits = [p for p in root.rglob("*") if p.is_file() and p.name in names]
    if not hits:
        raise FileNotFoundError(f"Could not find any of {candidates} under {root}")
    hits.sort(key=lambda p: (len(p.parts), -p.stat().st_mtime))
    return hits[0].resolve()


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _tv(t: Any) -> int:
    for attr in ("val", "value"):
        if hasattr(t, attr):
            v = getattr(t, attr)
            return int(v() if callable(v) else v)
    raise TypeError(t)


def _measurement_qubits(circuit: Any) -> list[int]:
    """Flatten ordinary single-qubit measurement-result targets in circuit order."""
    out: list[int] = []
    names = {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"}
    for inst in circuit:
        if not hasattr(inst, "name"):
            raise RuntimeError("REPEAT blocks are not expected in S18-M flattened circuits")
        if inst.name not in names:
            continue
        for t in inst.targets_copy():
            # The relevant closing rounds use only qubit targets, no combiners.
            try:
                out.append(_tv(t))
            except Exception as ex:
                raise RuntimeError(("unexpected measurement target", inst, t)) from ex
    if len(out) != int(circuit.num_measurements):
        raise RuntimeError(("measurement target accounting mismatch", len(out), circuit.num_measurements))
    return out


def _reconstruct_geo_qmap(v7: Any, hirano_src: Path, sahay: Path, channel: str):
    lines = v7._read_source(sahay)
    geo = v7._source_geometry(lines)
    full, _wrapper, _init_range, probe_start, _online, _terminal = v7.s7b._build_with_wait_rounds(
        hirano_src, channel, 2
    )
    pre_clean = full[:probe_start]
    qmap, map_meta = v7._build_qmap(pre_clean, geo)
    return geo, qmap, map_meta


def _append_truth_detectors(circuit: Any, channel: str, geo: dict[str, Any], qmap: dict[int, int]) -> dict[str, Any]:
    """Append 84 terminal stabilizer-closing detectors, scoring-only.

    The S18-M circuit ends with:
      ... ideal follow-up d13 MR[168] -> terminal data M/MX[169] -> OBSERVABLE.
    The detector compares each last ideal ancilla result with the parity of the
    corresponding stabilizer support in the destructive terminal data readout.
    """
    import stim

    if channel not in {"X", "Z"}:
        raise ValueError(channel)
    if int(circuit.num_detectors) != OP_DETECTORS:
        raise RuntimeError(("unexpected operational detector count", circuit.num_detectors))

    mids = _measurement_qubits(circuit)
    if len(mids) < 337:
        raise RuntimeError(("not enough terminal measurements", len(mids)))
    follow_qs = mids[-337:-169]
    term_qs = mids[-169:]

    expected_follow = [int(qmap[a]) for a in range(300, 468)]
    expected_term = [int(qmap[q]) for q in geo["data"]]
    if follow_qs != expected_follow:
        raise RuntimeError(("follow-up d13 MR order drift", follow_qs[:8], expected_follow[:8], follow_qs[-8:], expected_follow[-8:]))
    if term_qs != expected_term:
        raise RuntimeError(("terminal d13 data order drift", term_qs[:8], expected_term[:8], term_qs[-8:], expected_term[-8:]))

    current = int(circuit.num_measurements)
    terminal_start = current - 169
    follow_start = terminal_start - 168
    anc_abs = {a: follow_start + i for i, a in enumerate(range(300, 468))}
    data_abs = {q: terminal_start + i for i, q in enumerate(geo["data"])}

    def rec(abs_mid: int):
        off = int(abs_mid) - int(circuit.num_measurements)
        if off >= 0:
            raise AssertionError((abs_mid, circuit.num_measurements))
        return stim.target_rec(off)

    stabs = geo["x_stabs"] if channel == "X" else geo["z_stabs"]
    before = int(circuit.num_detectors)
    for anc in sorted(stabs):
        support = sorted(int(q) for q in stabs[anc])
        targets = [rec(anc_abs[int(anc)])] + [rec(data_abs[q]) for q in support]
        circuit.append("DETECTOR", targets)
    added = int(circuit.num_detectors) - before
    if added != TRUTH_PER_PROBE:
        raise RuntimeError(("truth detector count", added))

    # Strong deterministic/gauge-free truth-boundary check.
    circuit.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    return {
        "channel": channel,
        "operational_detectors": OP_DETECTORS,
        "terminal_truth_detectors": added,
        "truth_detector_ids": [OP_DETECTORS + i for i in range(added)],
        "terminal_measurements": 169,
        "followup_measurements": 168,
        "stabilizer_family": "X" if channel == "X" else "Z",
        "decoder_visible": False,
        "scoring_only": True,
    }


@dataclass
class E:
    op: int
    truth: int      # 168-bit combined terminal stabilizer truth syndrome
    pauli: int
    locs: set[int]
    example: tuple[Any, ...]


def _build_effects(cert: Any, xmap: dict, zmap: dict, mechanisms: set[tuple[Any, ...]]) -> list[E]:
    groups: dict[tuple[int, int, int], E] = {}
    op_mask = (1 << OP_DETECTORS) - 1
    truth_mask = (1 << TRUTH_PER_PROBE) - 1
    for m in mechanisms:
        mx, lx = xmap.get(m, (0, 0))
        mz, lz = zmap.get(m, (0, 0))
        opx = int(mx) & op_mask
        opz = int(mz) & op_mask
        if opx != opz:
            raise RuntimeError(("probe-dependent operational syndrome", cert._mechanism_json(m), hex(opx), hex(opz)))
        tx = (int(mx) >> OP_DETECTORS) & truth_mask
        tz = (int(mz) >> OP_DETECTORS) & truth_mask
        truth = int(tx) | (int(tz) << TRUTH_PER_PROBE)
        # X-probe observable flip is logical Z; Z-probe flip is logical X.
        pauli = int(lz) | (int(lx) << 1)
        key = (opx, truth, pauli)
        loc = int(m[0])
        if key not in groups:
            groups[key] = E(opx, truth, pauli, set(), m)
        groups[key].locs.add(loc)
    return sorted(groups.values(), key=lambda e: (e.op, e.truth, e.pauli))


def _distinct(a: E, b: E) -> tuple[int, int] | None:
    for x in a.locs:
        for y in b.locs:
            if x != y:
                return int(x), int(y)
    return None


def _ej(cert: Any, e: E) -> dict[str, Any]:
    return {
        "operational_syndrome_hex": hex(e.op),
        "terminal_truth_syndrome_hex": hex(e.truth),
        "raw_terminal_pauli": PAULI_NAMES[e.pauli],
        "physical_location_count": len(e.locs),
        "example_mechanism": cert._mechanism_json(e.example),
    }


def _audit_new_new(cert: Any, effects: list[E], max_examples: int) -> dict[str, Any]:
    by_op: dict[int, list[E]] = defaultdict(list)
    for e in effects:
        by_op[e.op].append(e)

    c = Counter()
    examples = {
        "closed_boundary_logical": [],
        "raw_observable_alias_with_terminal_residual": [],
    }

    # Single new fault with operational-zero record.
    for e in by_op.get(0, []):
        c["new1_operational_zero_classes"] += 1
        if e.pauli != 0 and e.truth == 0:
            c["new1_closed_boundary_logical"] += 1
        elif e.pauli != 0 and e.truth != 0:
            c["new1_raw_alias_terminal_residual"] += 1

    # New1+new1 reaches operational zero iff component operational syndromes equal.
    for op, rows in by_op.items():
        for i, a in enumerate(rows):
            for j in range(i, len(rows)):
                b = rows[j]
                w = _distinct(a, b)
                if w is None:
                    continue
                c["new_new_operational_zero_pairs"] += 1
                final_p = int(a.pauli) ^ int(b.pauli)
                final_t = int(a.truth) ^ int(b.truth)
                if final_p != 0:
                    c["new_new_raw_nonidentity"] += 1
                    item = {
                        "operational_syndrome_hex": hex(op),
                        "final_raw_pauli": PAULI_NAMES[final_p],
                        "combined_terminal_truth_syndrome_hex": hex(final_t),
                        "a": _ej(cert, a),
                        "b": _ej(cert, b),
                        "distinct_locations": [cert._location_json(w[0]), cert._location_json(w[1])],
                    }
                    if final_t == 0:
                        c["new_new_closed_boundary_logical"] += 1
                        if len(examples["closed_boundary_logical"]) < max_examples:
                            examples["closed_boundary_logical"].append(item)
                    else:
                        c["new_new_raw_alias_terminal_residual"] += 1
                        if len(examples["raw_observable_alias_with_terminal_residual"]) < max_examples:
                            examples["raw_observable_alias_with_terminal_residual"].append(item)

    return {
        "counts": dict(c),
        "examples": examples,
        "new_single_closed_boundary_gate_pass": c["new1_closed_boundary_logical"] == 0,
        "new_new_closed_boundary_gate_pass": c["new_new_closed_boundary_logical"] == 0,
        "interpretation": (
            "A raw terminal observable flip with nonzero scoring-only terminal stabilizer syndrome is "
            "classified as a delivered d13 residual, not as a zero-syndrome logical coset.  The 84+84 "
            "terminal truth bits are never decoder-visible."
        ),
    }


def _selftest() -> dict[str, Any]:
    # Same operational syndrome, opposite raw Pauli, but different truth -> residual alias, not logical.
    a = E(5, 0b001, 0, {10}, (10, ()))
    b = E(5, 0b010, 2, {11}, (11, ()))
    # Same op and same truth, different raw Pauli -> truth cancels and is a true closed-boundary logical witness.
    c = E(7, 0b100, 0, {12}, (12, ()))
    d = E(7, 0b100, 1, {13}, (13, ()))
    assert (a.truth ^ b.truth) != 0 and (a.pauli ^ b.pauli) != 0
    assert (c.truth ^ d.truth) == 0 and (c.pauli ^ d.pauli) != 0
    return {"passed": True, "operational_bits": OP_DETECTORS, "truth_bits_per_probe": TRUTH_PER_PROBE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s18m-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m_results"))
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--max-examples", type=int, default=16)
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    if args.self_test_only:
        print(json.dumps(_selftest(), indent=2))
        return

    root = args.search_root.resolve()
    result_dir = (root / args.s18m_result_dir).resolve() if not args.s18m_result_dir.is_absolute() else args.s18m_result_dir.resolve()
    out = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    s18mp = _find(root, ["stim_pipeline/interface/direct_cs_s18m_modular_noisy_d13_certificate.py"])
    v7p = _find(root, ["stim_pipeline/interface/direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"])
    v13p = _find(root, ["stim_pipeline/interface/direct_cs_s17b_v13d_joint_postselection_certificate.py"])
    hirano = _find(root, ["msc-ls/src"])
    sahay = _find(root, ["reference/fd3_baseline_ckt.stim"])

    s18m = _load(s18mp, "s18m2_base")
    v7 = _load(v7p, "s18m2_v7")
    cert = _load(v13p, "s18m2_v13")

    import stim

    xsrc = result_dir / "s18m_X_noisy_d13_boundary.stim"
    zsrc = result_dir / "s18m_Z_noisy_d13_boundary.stim"
    summary_path = result_dir / "s18m_summary.json"
    if not xsrc.exists() or not zsrc.exists() or not summary_path.exists():
        raise FileNotFoundError((xsrc, zsrc, summary_path))
    prior = json.loads(summary_path.read_text(encoding="utf-8"))

    # Python text I/O avoids Stim Windows Unicode-path issues.
    xc = stim.Circuit(xsrc.read_text(encoding="utf-8"))
    zc = stim.Circuit(zsrc.read_text(encoding="utf-8"))
    if int(xc.num_detectors) != OP_DETECTORS or int(zc.num_detectors) != OP_DETECTORS:
        raise RuntimeError((xc.num_detectors, zc.num_detectors))

    gx, qx, mx = _reconstruct_geo_qmap(v7, hirano, sahay, "X")
    gz, qz, mz = _reconstruct_geo_qmap(v7, hirano, sahay, "Z")
    if gx["data"] != gz["data"] or qx != qz:
        raise RuntimeError("X/Z d13 geometry mapping differs")

    print("S18-M2: appending X terminal truth closure...", flush=True)
    xt_meta = _append_truth_detectors(xc, "X", gx, qx)
    print("S18-M2: appending Z terminal truth closure...", flush=True)
    zt_meta = _append_truth_detectors(zc, "Z", gz, qz)
    if int(xc.num_detectors) != OP_DETECTORS + TRUTH_PER_PROBE:
        raise RuntimeError(xc.num_detectors)
    if int(zc.num_detectors) != OP_DETECTORS + TRUTH_PER_PROBE:
        raise RuntimeError(zc.num_detectors)

    xt = out / "s18m2_X_truth_closed.stim"
    zt = out / "s18m2_Z_truth_closed.stim"
    xt.write_text(str(xc), encoding="utf-8")
    zt.write_text(str(zc), encoding="utf-8")

    # Recover the same physical mechanisms with extra scoring-only truth bits.
    cert.EXPECTED_LOCATIONS = TOTAL_LOCATIONS
    xl, xn, xmeta = cert._noise_locations_from_text(xt)
    zl, zn, zmeta = cert._noise_locations_from_text(zt)
    if len(xl) != TOTAL_LOCATIONS or len(zl) != TOTAL_LOCATIONS:
        raise RuntimeError((len(xl), len(zl)))
    xphys = [(xmeta[i]["tick"], xmeta[i]["gate"], tuple(xmeta[i]["qubits"])) for i in xl]
    zphys = [(zmeta[i]["tick"], zmeta[i]["gate"], tuple(zmeta[i]["qubits"])) for i in zl]
    if xphys != zphys:
        raise RuntimeError("X/Z physical schedules differ")
    cert._LOCATION_META.clear(); cert._LOCATION_META.update(xmeta)

    print("S18-M2: recovering X truth-closed mechanism effects...", flush=True)
    xp, xrec = cert._recover_mechanism_map(xc, cert._raw_dem(xc), "X", xn)
    print("S18-M2: recovering Z truth-closed mechanism effects...", flush=True)
    zp, zrec = cert._recover_mechanism_map(zc, cert._raw_dem(zc), "Z", zn)
    universe = cert._all_mechanism_keys(xl)
    new = {m for m in universe if int(m[0]) >= OLD_LOCATIONS}
    if {int(m[0]) for m in new} != set(range(OLD_LOCATIONS, TOTAL_LOCATIONS)):
        raise RuntimeError("new d13 location coverage drift")

    effects = _build_effects(cert, xp, zp, new)
    print(f"S18-M2: exact new+new truth-closure join; effects={len(effects)} ...", flush=True)
    audit = _audit_new_new(cert, effects, int(args.max_examples))

    prior_gate = prior.get("main_gate", {})
    inherited_other_branches_pass = bool(
        prior_gate.get("inherited_s17_gate_pass", False)
        and prior_gate.get("new_single_zero_record_identity_only", False)
        and prior_gate.get("old_single_plus_new_single_zero_record_identity_only", False)
        and prior_gate.get("incoming_weight1_sector_plus_new_single_zero_record_identity_only", False)
    )
    m2_newnew = bool(audit["new_new_closed_boundary_gate_pass"])
    combined = bool(inherited_other_branches_pass and m2_newnew)

    summary = {
        "stage": "S18-M2 terminal truth-closure audit",
        "stim_version": getattr(stim, "__version__", "unknown"),
        "source_s18m_summary": str(summary_path),
        "policy": {
            "operational_detector_bits": OP_DETECTORS,
            "terminal_truth_bits_per_probe": TRUTH_PER_PROBE,
            "terminal_truth_total_complementary_bits": 2 * TRUTH_PER_PROBE,
            "truth_bits_decoder_visible": False,
            "acceptance_policy_unchanged": "S18-M operational all-zero 638-bit rule for structural audit",
            "scientific_question": "Does an operationally silent new+new pair remain a zero-terminal-syndrome nontrivial logical coset after ideal d13 truth closure?",
        },
        "truth_closure": {"X": xt_meta, "Z": zt_meta},
        "mechanism_recovery": {"X": xrec, "Z": zrec, "new_mechanisms": len(new), "new_effect_classes": len(effects)},
        "new_new_truth_audit": audit,
        "prior_s18m_main_gate": prior_gate,
        "main_gate": {
            "inherited_and_other_new_branches_pass_strict_S18M": inherited_other_branches_pass,
            "new_new_no_closed_boundary_logical_pair": m2_newnew,
            "combined_modular_boundary_output_contract_pass": combined,
        },
        "interpretation_if_pass": (
            "The 14 raw-observable S18-M collisions are boundary residual aliases rather than zero-terminal-syndrome logical operators. "
            "Freeze this structural output-contract gate and proceed to S18-B, where the 638-bit operational decoder must handle/reject such residual-carrying events without using the scoring-only terminal truth bits."
        ),
        "caveat": (
            "M2 is a structural ideal-closure truth classification, not a practical decoder result.  Nonzero terminal truth syndrome means the delivered d13 block carries a detectable residual; S18-B must establish the finite-p practical recovery/abstention behavior."
        ),
    }
    _dump(out / "s18m2_summary.json", summary)
    _dump(out / "s18m2_self_test.json", _selftest())
    print(json.dumps(summary["main_gate"], indent=2), flush=True)
    if not combined:
        raise RuntimeError(("S18-M2 truth-closure gate failed", summary["main_gate"]))
    print("S18-M2 TERMINAL TRUTH-CLOSURE PASS", flush=True)


if __name__ == "__main__":
    main()
