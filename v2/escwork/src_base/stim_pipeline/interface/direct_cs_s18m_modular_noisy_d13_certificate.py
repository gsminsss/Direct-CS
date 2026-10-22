#!/usr/bin/env python3
"""S18-M: modular exact noisy-d13 boundary certificate for direct-|CS>.

Why this stage exists
=====================
S17-B/C already certified the pre-existing continuous Clifford handoff through
surface-code growth to d=13 under a total low-order budget.  Re-enumerating all
old-old fault pairs after adding one noisy d=13 QEC round is redundant and very
memory intensive.

S18-M instead preserves the exact S17 470-detector basis as a literal prefix,
physicalizes the previously ideal first d=13 close (1128 active locations per
patch), appends one ideal stationary d=13 round only for future-syndrome scoring,
and checks only NEW fault-budget branches that were not present in S17:

  new1
  new1 + new1
  old_downstream1 + new1
  incoming_Steane_sector(weight 1) + new1

The inherited no-new branches remain covered by S17.  For a conservative
structural certificate, the acceptance rule is all-zero 638-bit detector record.
Because Pauli detector/observable effects compose by XOR, a two-component branch
can reach zero detector record iff the component syndromes are identical.  The
code therefore performs exact syndrome-key joins instead of a Cartesian pair
scan.  No Monte-Carlo or hash approximation is used for the scientific gate.

PASS of the NEW-boundary gate means no newly introduced total-budget<=2 branch
can produce zero detector record with a non-identity final logical Pauli.  When
combined with the inherited S17 no-new certificate, this gives a modular
postselected d_f >= 3 handoff certificate for the newly physicalized d13 close.

Important scope
===============
This is a conservative structural certificate, not the paper-final finite-p
acceptance policy.  It intentionally uses an all-zero detector acceptance rule.
S18-B should restore practical acceptance with the frozen D2-style decoder and
must separately verify that its accepted <=2 events remain harmless.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

OLD_LOCATIONS = 2407
NEW_D13_LOCATIONS = 1128
TOTAL_LOCATIONS = OLD_LOCATIONS + NEW_D13_LOCATIONS
OLD_DETECTORS = 470
FOLLOWUP_DETECTORS = 168
TOTAL_DETECTORS = OLD_DETECTORS + FOLLOWUP_DETECTORS
EXPECTED_HIRANO = 1028
EXPECTED_G57 = 971
EXPECTED_G713 = 408
EXPECTED_D13_H = 168
EXPECTED_D13_CX = 624
EXPECTED_D13_MEAS = 168
EXPECTED_D13_RESET = 168
PAULI_NAMES = ("I", "X", "Z", "Y")
PAULI1 = ("X", "Y", "Z")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_one(root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = (root / rel).resolve()
        if p.exists():
            return p
    names = {Path(x).name for x in candidates}
    hits = [p for p in root.rglob("*") if p.name in names]
    if not hits:
        raise FileNotFoundError(f"Could not find any of {candidates} under {root}")
    hits.sort(key=lambda p: (len(p.parts), -p.stat().st_mtime))
    return hits[0].resolve()


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


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
    """Ideal follow-up syndrome round; detects late/reset faults in noisy close1."""
    import stim
    out = _quantum_round_only(close_clean)
    args = _close_detector_args(close_clean)
    for j, a in enumerate(args):
        out.append(
            "DETECTOR",
            [stim.target_rec(-168 + j), stim.target_rec(-336 + j)],
            a,
        )
    out.append("SHIFT_COORDS", [], [0, 0, 1])
    if int(out.num_detectors) != FOLLOWUP_DETECTORS:
        raise AssertionError(out.num_detectors)
    return out


def _instrument_noisy_d13_close(close_clean: Any, p: float) -> tuple[Any, dict[str, int]]:
    """Conservative M/R split: H 168 + CX 624 + M 168 + reset 168 = 1128."""
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
            out.append("X_ERROR", qs, p)  # measurement flip opportunity
            out.append(inst)
            out.append("X_ERROR", qs, p)  # reset/preparation error into next round
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
        "total": NEW_D13_LOCATIONS,
    }
    if c != exp:
        raise AssertionError(("d13 active-location count drift", c, exp))
    if out.without_noise() != close_clean:
        raise AssertionError("d13 instrumentation changed clean close")
    return out, c


def _detector_lines_from_circuit(c: Any) -> list[str]:
    return [x.strip() for x in str(c).splitlines() if x.strip().startswith("DETECTOR")]


def _build_extended_channel(
    v7: Any,
    v11: Any,
    s18a: Any,
    hirano_src: Path,
    sahay_stim: Path,
    channel: str,
    p: float,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Build clean/noisy 638-detector channel with exact S17 470 prefix."""
    import stim
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

    boundary_instruction_index = int(init_range[1])
    noisy_pre, hloc, _ = v7.s7b._instrument_before_probe(pre_clean, init_range, len(pre_clean), p)
    if len(hloc) != EXPECTED_HIRANO:
        raise AssertionError(len(hloc))
    qmap, map_meta = v7._build_qmap(pre_clean, geo)
    flow = s18a._solve_transport_flow_s18(v7, lines, geo, channel)

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

    # IMPORTANT: synthesize and append the same 35 detector rows as S17 BEFORE
    # adding the extra follow-up round.  This makes detector IDs 0..469 a literal
    # S17 basis prefix, enabling a clean modular inheritance argument.
    old_clean_preterminal = pre_clean + g57 + g713 + close1
    old_noisy_preterminal = noisy_pre + ng57 + ng713 + nclose1
    missing, method = v11._missing_detectors(old_clean_preterminal)
    transition, trans_meta = v11._complete_missing_detector_basis(
        missing,
        total_measurements=int(old_clean_preterminal.num_measurements),
        pre_growth_measurements=pre_m,
        first_growth_count=48,
    )
    if int(transition.num_detectors) != 35:
        raise AssertionError(("transition detector count", transition.num_detectors, trans_meta))
    old_clean_preterminal += transition
    old_noisy_preterminal += transition
    if int(old_clean_preterminal.num_detectors) != OLD_DETECTORS:
        raise AssertionError(old_clean_preterminal.num_detectors)

    clean_preterminal = old_clean_preterminal + close2
    noisy_preterminal = old_noisy_preterminal + close2
    if int(clean_preterminal.num_detectors) != TOTAL_DETECTORS:
        raise AssertionError(clean_preterminal.num_detectors)
    clean_preterminal.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)

    # Rebuild the historical S17 v11 circuit and require exact detector-prefix identity.
    old_ref, old_ref_meta = v11._build_channel(v7, hirano_src, sahay_stim, channel, p)
    old_det = _detector_lines_from_circuit(old_ref)
    ext_det = _detector_lines_from_circuit(noisy_preterminal)
    if len(old_det) != OLD_DETECTORS or ext_det[:OLD_DETECTORS] != old_det:
        raise RuntimeError(("S17 detector prefix identity failed", len(old_det), len(ext_det)))

    # Append scorer only after the ideal follow-up round.
    clean_scored = stim.Circuit(str(clean_preterminal))
    noisy_scored = stim.Circuit(str(noisy_preterminal))
    term_clean = v7._append_terminal_boundary(
        clean_scored,
        channel,
        geo,
        qmap,
        transport_measurement_offset=int(pre_clean.num_measurements),
        transport_flow=flow,
        hirano_preprobe_observable_measurements=hirano_pre,
    )
    term_noisy = v7._append_terminal_boundary(
        noisy_scored,
        channel,
        geo,
        qmap,
        transport_measurement_offset=int(noisy_pre.num_measurements),
        transport_flow=flow,
        hirano_preprobe_observable_measurements=hirano_pre,
    )
    if int(noisy_scored.num_detectors) != TOTAL_DETECTORS or int(noisy_scored.num_observables) != 1:
        raise AssertionError((noisy_scored.num_detectors, noisy_scored.num_observables))

    total_loc = len(hloc) + c57["total"] + c713["total"] + c13["total"]
    if total_loc != TOTAL_LOCATIONS:
        raise AssertionError(total_loc)

    return noisy_scored, clean_scored, old_ref, {
        "channel": channel,
        "signature_p": float(p),
        "boundary_instruction_index": boundary_instruction_index,
        "s17_detector_prefix_is_literal": True,
        "detector_accounting": {
            "inherited_s17_prefix": OLD_DETECTORS,
            "ideal_followup_stationary": FOLLOWUP_DETECTORS,
            "total": TOTAL_DETECTORS,
        },
        "location_counts": {
            "inherited_old_faultable": OLD_LOCATIONS,
            "new_noisy_first_d13_close": NEW_D13_LOCATIONS,
            "total": TOTAL_LOCATIONS,
            "new_types": c13,
        },
        "removed_first_growth_detectors": strip_meta["removed_first_growth_detectors"],
        "missing_detector_method": method,
        "missing_detector_meta": trans_meta,
        "mapping": map_meta,
        "transport_flow": flow,
        "terminal_clean": term_clean,
        "terminal_noisy": term_noisy,
        "old_ref_meta": {
            "detectors": int(old_ref.num_detectors),
            "faultable_locations_per_patch": old_ref_meta.get("faultable_locations_per_patch"),
        },
    }


@dataclass
class Effect:
    syndrome: int
    pauli: int
    loc_weight: dict[int, float]
    loc_count: dict[int, int]
    example_mechanism: tuple[Any, ...]

    @property
    def total_weight(self) -> float:
        return float(sum(self.loc_weight.values()))


def _group_effects(cert: Any, xmap: dict, zmap: dict, mechanisms: Iterable[tuple[Any, ...]]) -> list[Effect]:
    grouped: dict[tuple[int, int], Effect] = {}
    loc_weight_sum: dict[int, float] = defaultdict(float)
    for m in mechanisms:
        sx, lx = xmap[m]
        sz, lz = zmap[m]
        if int(sx) != int(sz):
            raise RuntimeError(("probe-dependent syndrome", cert._mechanism_json(m), hex(int(sx)), hex(int(sz))))
        pauli = int(lz) | (int(lx) << 1)
        loc = int(m[0])
        w = float(cert._mechanism_weight(cert._location_gate(loc)))
        loc_weight_sum[loc] += w
        key = (int(sx), pauli)
        if key not in grouped:
            grouped[key] = Effect(int(sx), pauli, {}, {}, m)
        e = grouped[key]
        e.loc_weight[loc] = e.loc_weight.get(loc, 0.0) + w
        e.loc_count[loc] = e.loc_count.get(loc, 0) + 1
    bad = [(k, v) for k, v in loc_weight_sum.items() if abs(v - 1.0) > 1e-10]
    if bad:
        raise RuntimeError(("selected stage mechanism weights do not sum to one", bad[:10]))
    return sorted(grouped.values(), key=lambda e: (e.syndrome, e.pauli))


def _pair_exists_distinct(a: Effect, b: Effect, same: bool) -> bool:
    if not same:
        if set(a.loc_weight) != set(b.loc_weight):
            # Different effect classes can overlap but need just one distinct pair.
            pass
    la = set(a.loc_weight)
    lb = set(b.loc_weight)
    if not la or not lb:
        return False
    if any(x != y for x in la for y in lb):
        return True
    return False


def _distinct_witness(a: Effect, b: Effect) -> tuple[int, int] | None:
    for x in a.loc_weight:
        for y in b.loc_weight:
            if x != y:
                return int(x), int(y)
    return None


def _effect_json(cert: Any, e: Effect) -> dict[str, Any]:
    return {
        "syndrome_hex": hex(int(e.syndrome)),
        "pauli": PAULI_NAMES[int(e.pauli)],
        "physical_location_count": len(e.loc_weight),
        "example_mechanism": cert._mechanism_json(e.example_mechanism),
    }


def _zero_new_audit(cert: Any, old_effects: list[Effect], new_effects: list[Effect], max_examples: int) -> dict[str, Any]:
    old_by_s: dict[int, list[Effect]] = defaultdict(list)
    new_by_s: dict[int, list[Effect]] = defaultdict(list)
    for e in old_effects:
        old_by_s[int(e.syndrome)].append(e)
    for e in new_effects:
        new_by_s[int(e.syndrome)].append(e)

    examples: dict[str, list[dict[str, Any]]] = {
        "new_single_zero_nonidentity": [],
        "new_new_zero_nonidentity": [],
        "old_new_zero_nonidentity": [],
    }
    counts = Counter()

    # new1
    for e in new_by_s.get(0, []):
        counts["new_single_zero_effect_classes"] += 1
        if e.pauli != 0:
            counts["new_single_harmful"] += 1
            if len(examples["new_single_zero_nonidentity"]) < max_examples:
                examples["new_single_zero_nonidentity"].append(_effect_json(cert, e))

    # new1 + new1: zero syndrome iff equal syndrome; only <=4 Pauli classes per syndrome.
    for s, rows in new_by_s.items():
        for i, a in enumerate(rows):
            for j in range(i, len(rows)):
                b = rows[j]
                if not _pair_exists_distinct(a, b, i == j):
                    continue
                counts["new_new_zero_structural_pairs"] += 1
                if (a.pauli ^ b.pauli) != 0:
                    counts["new_new_harmful"] += 1
                    if len(examples["new_new_zero_nonidentity"]) < max_examples:
                        locs = _distinct_witness(a, b)
                        examples["new_new_zero_nonidentity"].append({
                            "syndrome_hex": hex(s),
                            "final_pauli": PAULI_NAMES[a.pauli ^ b.pauli],
                            "a": _effect_json(cert, a),
                            "b": _effect_json(cert, b),
                            "distinct_locations": [cert._location_json(x) for x in locs] if locs else None,
                        })

    # old1 + new1: stages are disjoint, so any matching syndrome classes are location-compatible.
    for s in set(old_by_s) & set(new_by_s):
        for a in old_by_s[s]:
            for b in new_by_s[s]:
                counts["old_new_zero_structural_pairs"] += 1
                if (a.pauli ^ b.pauli) != 0:
                    counts["old_new_harmful"] += 1
                    if len(examples["old_new_zero_nonidentity"]) < max_examples:
                        examples["old_new_zero_nonidentity"].append({
                            "syndrome_hex": hex(s),
                            "final_pauli": PAULI_NAMES[a.pauli ^ b.pauli],
                            "old": _effect_json(cert, a),
                            "new": _effect_json(cert, b),
                        })

    return {
        "counts": dict(counts),
        "examples": examples,
        "new_single_gate_pass": counts["new_single_harmful"] == 0,
        "new_new_gate_pass": counts["new_new_harmful"] == 0,
        "old_new_gate_pass": counts["old_new_harmful"] == 0,
    }


def _build_sector_map_638(ca: Any, s8b: Any, xclean: Any, zclean: Any, xm: dict[str, Any], zm: dict[str, Any], hirano_src: Path, prob: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pos, ids = s8b._steane_qubits(hirano_src)
    if len(ids) != 7:
        raise AssertionError(ids)
    singles: dict[tuple[int, str], tuple[int, int]] = {}
    mismatch = []
    for coord, q in enumerate(ids):
        for p in PAULI1:
            dx, ox = ca._signature(ca._insert_single_error(xclean, int(xm["boundary_instruction_index"]), int(q), p, prob))
            dz, oz = ca._signature(ca._insert_single_error(zclean, int(zm["boundary_instruction_index"]), int(q), p, prob))
            if dx != dz:
                mismatch.append((coord, int(q), p, hex(dx), hex(dz)))
                continue
            joint = int(oz) | (int(ox) << 1)  # same convention as S17-C/A
            singles[(coord, p)] = (int(dx), int(joint))
    if mismatch:
        raise RuntimeError(("sector X/Z detector mismatch", mismatch[:10]))
    if len(singles) != 21:
        raise AssertionError(len(singles))

    rows = []
    hist = Counter()
    for syn in range(64):
        rep = s8b.s8live.RECOVERY_TABLE[syn]
        d = 0
        q = 0
        for coord, p in enumerate(rep.pauli_string):
            if p == "I":
                continue
            ds, ps = singles[(coord, p)]
            d ^= ds
            q ^= ps
        hist[int(rep.weight)] += 1
        rows.append({
            "steane_syndrome": syn,
            "canonical_pauli": str(rep.pauli_string),
            "canonical_weight": int(rep.weight),
            "detector_638_hex": hex(d),
            "detector_638_weight": int(d.bit_count()),
            "raw_terminal_pauli_int": int(q),
            "raw_terminal_pauli": PAULI_NAMES[int(q)],
        })
    exp = {0: 1, 1: 21, 2: 42}
    if dict(hist) != exp:
        raise RuntimeError(("sector weight histogram drift", dict(hist), exp))
    zero_labels = sorted({r["raw_terminal_pauli"] for r in rows if int(r["detector_638_hex"], 16) == 0})
    return rows, {
        "sector_count": len(rows),
        "canonical_weight_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "unique_detector_signatures": len({r["detector_638_hex"] for r in rows}),
        "zero_signature_paulis": zero_labels,
    }


def _sector_new_zero_audit(cert: Any, sectors: list[dict[str, Any]], new_effects: list[Effect], max_examples: int) -> dict[str, Any]:
    new_by_s: dict[int, list[Effect]] = defaultdict(list)
    for e in new_effects:
        new_by_s[int(e.syndrome)].append(e)
    required = []
    diagnostic = []
    matched_required = 0
    matched_all = 0
    for r in sectors:
        bw = int(r["canonical_weight"])
        if bw == 0:
            continue
        s = int(r["detector_638_hex"], 16)
        ps = int(r["raw_terminal_pauli_int"])
        for e in new_by_s.get(s, []):
            outp = ps ^ int(e.pauli)
            matched_all += 1
            if bw == 1:
                matched_required += 1
            if outp == 0:
                continue
            obj = {
                "boundary_weight": bw,
                "steane_syndrome": int(r["steane_syndrome"]),
                "boundary_record_hex": r["detector_638_hex"],
                "boundary_pauli": r["raw_terminal_pauli"],
                "new_effect": _effect_json(cert, e),
                "final_pauli": PAULI_NAMES[outp],
            }
            if bw == 1:
                if len(required) < max_examples:
                    required.append(obj)
            else:
                if len(diagnostic) < max_examples:
                    diagnostic.append(obj)
    # Counts above only save examples, so recompute exact harmful counts cheaply.
    harmful_w1 = 0
    harmful_all = 0
    for r in sectors:
        bw = int(r["canonical_weight"])
        if bw == 0:
            continue
        s = int(r["detector_638_hex"], 16)
        ps = int(r["raw_terminal_pauli_int"])
        for e in new_by_s.get(s, []):
            if (ps ^ int(e.pauli)) != 0:
                harmful_all += 1
                if bw == 1:
                    harmful_w1 += 1
    return {
        "matching_sector_new_effect_classes_weight1": matched_required,
        "matching_sector_new_effect_classes_all_nonzero_sectors": matched_all,
        "harmful_zero_record_weight1_plus_new1": harmful_w1,
        "harmful_zero_record_all_sector_plus_new1_diagnostic": harmful_all,
        "required_weight1_gate_pass": harmful_w1 == 0,
        "required_examples": required,
        "stronger_weight2_plus_new_diagnostic_examples": diagnostic,
        "note": "Only canonical boundary weight-1 + new1 is part of the inherited total-budget<=2 gate. Weight-2 + new1 is a non-gating weight-3 diagnostic.",
    }


def _pure_selftest() -> dict[str, Any]:
    # Exact zero-XOR join logic: only equal syndrome keys can cancel.
    a = Effect(5, 1, {0: 1.0}, {0: 1}, (0, ()))
    b = Effect(5, 1, {1: 1.0}, {1: 1}, (1, ()))
    c = Effect(5, 2, {2: 1.0}, {2: 1}, (2, ()))
    d = Effect(7, 2, {3: 1.0}, {3: 1}, (3, ()))
    assert (a.syndrome ^ b.syndrome) == 0 and (a.pauli ^ b.pauli) == 0
    assert (a.syndrome ^ c.syndrome) == 0 and (a.pauli ^ c.pauli) != 0
    assert (a.syndrome ^ d.syndrome) != 0
    assert _pair_exists_distinct(a, b, False)
    return {
        "passed": True,
        "scientific_identity": "s1 XOR s2 == 0 iff s1 == s2; exact syndrome-key joins are sufficient for zero-record two-component searches",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--signature-p", type=float, default=0.001)
    ap.add_argument("--out-dir", type=Path, default=Path("s18m_results"))
    ap.add_argument("--max-examples", type=int, default=20)
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    if args.self_test_only:
        print(json.dumps(_pure_selftest(), indent=2))
        return

    root = args.search_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    v7p = _find_one(root, ["stim_pipeline/interface/direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"])
    v11p = _find_one(root, ["stim_pipeline/interface/direct_cs_s17b_v11_complete_missing_basis.py"])
    v13p = _find_one(root, ["stim_pipeline/interface/direct_cs_s17b_v13d_joint_postselection_certificate.py"])
    s18ap = _find_one(root, ["stim_pipeline/interface/direct_cs_s18a_noisy_d13_boundary_certificate.py"])
    cap = _find_one(root, ["stim_pipeline/interface/direct_cs_s17c_a_sector_to_470_map.py"])
    s8bp = _find_one(root, ["stim_pipeline/interface/zxz_core_boundary_signature_map.py"])
    hirano = _find_one(root, ["msc-ls/src"])
    sahay = _find_one(root, ["reference/fd3_baseline_ckt.stim"])
    s17sum = _find_one(root, ["reference_results/s17b_v13d/s17b_v13d_summary.json", "s17b_v13d_summary.json"])

    v7 = _load_module(v7p, "s18m_v7")
    v11 = _load_module(v11p, "s18m_v11")
    cert = _load_module(v13p, "s18m_v13d")
    s18a = _load_module(s18ap, "s18m_s18a")
    ca = _load_module(cap, "s18m_ca")
    s8b = _load_module(s8bp, "s18m_s8b")

    inherited = json.loads(s17sum.read_text(encoding="utf-8"))
    inherited_gate = bool(inherited.get("main_gate", {}).get("postselected_detect_or_correct_df_ge_3_low_order_gate_pass", False))
    if not inherited_gate:
        raise RuntimeError("Inherited S17-B v13d low-order gate is not PASS")

    print("S18-M: building X stable-prefix extended boundary...", flush=True)
    xc, xclean, xold, xm = _build_extended_channel(v7, v11, s18a, hirano, sahay, "X", float(args.signature_p))
    print("S18-M: building Z stable-prefix extended boundary...", flush=True)
    zc, zclean, zold, zm = _build_extended_channel(v7, v11, s18a, hirano, sahay, "Z", float(args.signature_p))

    xpath = out / "s18m_X_noisy_d13_boundary.stim"
    zpath = out / "s18m_Z_noisy_d13_boundary.stim"
    # Python file IO avoids Stim's Windows Unicode-path problem.
    xpath.write_text(str(xc), encoding="utf-8")
    zpath.write_text(str(zc), encoding="utf-8")
    (out / "s18m_X_inherited_s17_reference.stim").write_text(str(xold), encoding="utf-8")
    (out / "s18m_Z_inherited_s17_reference.stim").write_text(str(zold), encoding="utf-8")

    cert.EXPECTED_LOCATIONS = TOTAL_LOCATIONS
    cert.EXPECTED_DETECTORS = TOTAL_DETECTORS
    def stage(tick: int) -> str:
        if tick < 44: return "hirano_escape_plus_d5_rounds"
        if tick < 71: return "sahay_d5_to_d7"
        if tick < 77: return "sahay_d7_to_d13"
        return "new_noisy_first_d13_close"
    cert._stage_from_tick = stage

    xd = cert._detector_lines(xpath)
    zd = cert._detector_lines(zpath)
    if xd != zd or len(xd) != TOTAL_DETECTORS:
        raise RuntimeError(("X/Z detector mismatch", len(xd), len(zd)))
    xl, xn, xlocmeta = cert._noise_locations_from_text(xpath)
    zl, zn, zlocmeta = cert._noise_locations_from_text(zpath)
    if len(xl) != TOTAL_LOCATIONS or len(zl) != TOTAL_LOCATIONS:
        raise RuntimeError((len(xl), len(zl), TOTAL_LOCATIONS))
    xphys = [(xlocmeta[i]["tick"], xlocmeta[i]["gate"], tuple(xlocmeta[i]["qubits"])) for i in xl]
    zphys = [(zlocmeta[i]["tick"], zlocmeta[i]["gate"], tuple(zlocmeta[i]["qubits"])) for i in zl]
    if xphys != zphys:
        raise RuntimeError("X/Z physical schedules differ")
    cert._LOCATION_META.clear(); cert._LOCATION_META.update(xlocmeta)

    import stim
    print("S18-M: recovering X-probe concrete mechanisms...", flush=True)
    xp, xmeta = cert._recover_mechanism_map(xc, cert._raw_dem(xc), "X", xn)
    print("S18-M: recovering Z-probe concrete mechanisms...", flush=True)
    zp, zmeta = cert._recover_mechanism_map(zc, cert._raw_dem(zc), "Z", zn)
    universe = cert._all_mechanism_keys(xl)
    xmap = {m: xp.get(m, (0, 0)) for m in universe}
    zmap = {m: zp.get(m, (0, 0)) for m in universe}

    old_mechs = [m for m in universe if int(m[0]) < OLD_LOCATIONS]
    new_mechs = [m for m in universe if int(m[0]) >= OLD_LOCATIONS]
    if {int(m[0]) for m in old_mechs} != set(range(OLD_LOCATIONS)):
        raise RuntimeError("old mechanism location coverage drift")
    if {int(m[0]) for m in new_mechs} != set(range(OLD_LOCATIONS, TOTAL_LOCATIONS)):
        raise RuntimeError("new mechanism location coverage drift")
    old_effects = _group_effects(cert, xmap, zmap, old_mechs)
    new_effects = _group_effects(cert, xmap, zmap, new_mechs)

    print(
        f"S18-M: exact zero-record joins; old_effects={len(old_effects)}, new_effects={len(new_effects)} ...",
        flush=True,
    )
    zaudit = _zero_new_audit(cert, old_effects, new_effects, int(args.max_examples))

    print("S18-M: constructing exact 638-bit Steane sector signatures...", flush=True)
    sectors, sector_meta = _build_sector_map_638(ca, s8b, xclean, zclean, xm, zm, hirano, float(args.signature_p))
    saudit = _sector_new_zero_audit(cert, sectors, new_effects, int(args.max_examples))

    new_gate = (
        zaudit["new_single_gate_pass"]
        and zaudit["new_new_gate_pass"]
        and zaudit["old_new_gate_pass"]
        and saudit["required_weight1_gate_pass"]
    )

    summary = {
        "stage": "S18-M modular noisy-d13 zero-record exact certificate",
        "stim_version": getattr(stim, "__version__", "unknown"),
        "signature_p": float(args.signature_p),
        "scientific_policy": {
            "structural_acceptance_rule": "accept only the all-zero 638-bit detector record for the theorem gate",
            "finite_p_warning": "This conservative rule is not the final resource/performance policy; S18-B must use a practical decoder.",
            "why_no_global_order2_cartesian_scan": "All no-new <=2 branches are inherited from S17. For new-involved two-component branches, zero detector record requires equal component syndrome keys, so exact equality joins suffice.",
        },
        "inherited_s17": {
            "summary": str(s17sum),
            "postselected_df_ge_3_low_order_gate_pass": inherited_gate,
            "scope": "No-new old-old and sector/old branches are inherited; S18-M audits only branches containing the newly physicalized d13-close faults.",
        },
        "schedule_guard": {
            "old_detector_prefix": OLD_DETECTORS,
            "followup_detectors": FOLLOWUP_DETECTORS,
            "total_detectors": TOTAL_DETECTORS,
            "old_locations": OLD_LOCATIONS,
            "new_d13_locations": NEW_D13_LOCATIONS,
            "total_locations": TOTAL_LOCATIONS,
            "X_Z_detector_definitions_identical": True,
            "X_Z_physical_schedule_identical": True,
            "S17_470_detector_basis_literal_prefix_both_channels": bool(xm["s17_detector_prefix_is_literal"] and zm["s17_detector_prefix_is_literal"]),
        },
        "mechanism_mapping": {
            "complete_mechanisms": len(universe),
            "old_mechanisms": len(old_mechs),
            "new_mechanisms": len(new_mechs),
            "old_unique_effects": len(old_effects),
            "new_unique_effects": len(new_effects),
            "X_explained": len(xp),
            "Z_explained": len(zp),
            "X_meta": xmeta,
            "Z_meta": zmeta,
        },
        "new_fault_zero_record_audit": zaudit,
        "sector_map_638_meta": sector_meta,
        "sector_plus_new_zero_record_audit": saudit,
        "main_gate": {
            "inherited_s17_gate_pass": inherited_gate,
            "new_single_zero_record_identity_only": zaudit["new_single_gate_pass"],
            "new_new_zero_record_identity_only": zaudit["new_new_gate_pass"],
            "old_single_plus_new_single_zero_record_identity_only": zaudit["old_new_gate_pass"],
            "incoming_weight1_sector_plus_new_single_zero_record_identity_only": saudit["required_weight1_gate_pass"],
            "modular_new_boundary_extension_pass": bool(new_gate),
            "combined_with_inherited_s17_structural_df_ge_3_pass": bool(inherited_gate and new_gate),
        },
        "build_meta": {"X": xm, "Z": zm},
        "next_step_if_pass": "Freeze the structural S18-M theorem gate and proceed to S18-B practical finite-p decoding on the physicalized 638-bit boundary; do not return to global old-old pair enumeration.",
    }
    _dump(out / "s18m_sector_map_638.json", sectors)
    _dump(out / "s18m_summary.json", summary)
    _dump(out / "s18m_self_test.json", _pure_selftest())

    print(json.dumps(summary["main_gate"], indent=2), flush=True)
    if not (inherited_gate and new_gate):
        raise RuntimeError(("S18-M modular new-boundary certificate failed", summary["main_gate"]))
    print("S18-M MODULAR CERTIFICATE PASS", flush=True)


if __name__ == "__main__":
    main()
