#!/usr/bin/env python3
"""S8-E: global <=2-fault certificate for the preserved integrated model.

Scope
-----
This script certifies the *preserved S8-D integrated model*:

  six-check direct-|CS> core (322 check-gate locations)
  + aggregated CAT7 record-parity faults
  + final coherent Steane syndrome-sector boundary
  + two independent Hirano ZXZ escapes
  + two faultable surface-QEC handoff rounds
  + the 171-bit S7-B4/S8-C operational selective policy.

The midpoint ED remains an ideal code-space projection. Preparation, verified
CAT preparation/verification, and the physical midpoint-ED circuit are not
expanded here. Therefore PASS establishes d_f >= 3 only for this preserved
integrated model, not yet for the complete physical factory.

Exact <=2 decomposition
-----------------------
Every total physical-fault pattern of order <=2 in this model belongs to:

  (A) core0 + escape<=2
      inherited from the exact S7-B4/S8-C escape certificate;

  (B) core1 + escape1
      exhaustively enumerate every effective single core fault and every
      nonzero accepted Steane sector, then compose every reachable sector pair
      with every single ZXZ Pauli fault class on block A or B;

  (C) core2 + escape0.
      Fault-free ZXZ rejects all 63 nonzero Steane syndrome sectors, so only
      the (s_A,s_B)=(0,0) code-space component can survive.  The core2 proof is
      reduced exactly as follows:

        * two faults before midpoint: arbitrary midpoint code-space state is
          projected by the ideal second three-check record onto the rank-1
          target branch;
        * one before + one after midpoint: exhaustive first-half-single audit
          verifies the midpoint state is exactly the target branch, reducing
          to the exhaustive second-half-single audit;
        * two faults after midpoint: exhaustively enumerate every effective
          fault pair on distinct physical gate locations (plus record-parity
          events) and all four valid 6-bit records, and inspect the exact
          zero-zero code-space amplitude.

CZZ Pauli classes with identical action on the 14 data qubits and identical
record-flip bit are collapsed only for computation.  The script tracks their
multiplicity, so the collapse is an exact equivalence reduction, not sampling.

CAT readout model
-----------------
The preserved core uses q=[1-(1-2p)^7]/2 per check rather than seven explicit
measurement locations. For fault-order existence, one parity-flipped record
bit is represented by one effective measurement-fault event with multiplicity
7. Two measurement faults in the same check cancel parity and are trivially
equivalent to the zero-readout-fault action; these pairs are counted as
trivially safe but do not need state-vector replay.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
D2_PATH = HERE / "direct_cs_s8d2_failure_witness_replay.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


d2 = _load("s8e_d2", D2_PATH)
s8d = d2.s8d
m = d2.m
s8live = d2.s8live
ACTUAL_BITS = d2.ACTUAL_BITS
ACTUAL_MASK = d2.ACTUAL_MASK
FRAME_LABELS = d2.FRAME_LABELS
BRANCHES = ((0, 0), (0, 1), (1, 0), (1, 1))
TOL = 1e-10


@dataclass(frozen=True)
class EffEvent:
    kind: str                  # 'gate' or 'record'
    check_index: int
    location_id: int | None
    option_index: int | None
    multiplicity: int
    label: str

    @property
    def is_first_half(self) -> bool:
        return self.check_index < 3

    @property
    def is_second_half(self) -> bool:
        return self.check_index >= 3


@dataclass
class SingleAudit:
    original_single_fault_classes_represented: int = 0
    effective_single_events: int = 0
    nonzero_valid_record_branches: int = 0
    nonzero_sector_branches: int = 0
    reachable_sector_pairs: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    first_half_nonzero_midpoint_branches: int = 0
    first_half_midpoint_target_violations: int = 0
    logical_target_violations: int = 0
    accepted_faultfree_escape_violations: int = 0
    max_single_sector_infidelity: float = 0.0
    examples: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def _matrix_key(F: np.ndarray, flip: int) -> bytes:
    a = np.asarray(F, dtype=np.complex128)
    return np.round(a.real, 13).tobytes() + np.round(a.imag, 13).tobytes() + bytes([int(flip)])


def _build_effective_gate_events() -> tuple[list[EffEvent], dict[str, Any]]:
    events: list[EffEvent] = []
    by_gate = collections.Counter()
    original = 0
    effective = 0
    duplicate_collapses = 0
    for lid, (ci, ck, _j, gi) in enumerate(m.locations):
        groups: dict[bytes, list[int]] = collections.defaultdict(list)
        for oi, (F, flip, _lab) in enumerate(m.opt_cache[(ck, gi)]):
            groups[_matrix_key(F, int(flip))].append(int(oi))
        original += sum(len(v) for v in groups.values())
        effective += len(groups)
        duplicate_collapses += sum(len(v) - 1 for v in groups.values())
        for opts in groups.values():
            oi = opts[0]
            typ = m.seqs[ck][gi][0]
            lab = m.opt_cache[(ck, gi)][oi][2]
            events.append(EffEvent("gate", int(ci), int(lid), int(oi), len(opts), f"G{lid}:{lab}"))
            by_gate[typ] += len(opts)
    meta = {
        "physical_gate_locations": len(m.locations),
        "original_gate_pauli_classes": original,
        "effective_gate_events": effective,
        "collapsed_duplicate_classes": duplicate_collapses,
        "original_classes_by_gate_type": dict(by_gate),
    }
    return events, meta


def _build_record_events() -> list[EffEvent]:
    # One parity-flip effect per check, representing any one of seven CAT7
    # destructive-readout measurement faults.
    return [EffEvent("record", ci, None, None, 7, f"M{ci}") for ci in range(6)]


GATE_EVENTS, GATE_EVENT_META = _build_effective_gate_events()
RECORD_EVENTS = _build_record_events()
ALL_EVENTS = GATE_EVENTS + RECORD_EVENTS
FIRST_EVENTS = [e for e in ALL_EVENTS if e.is_first_half]
SECOND_EVENTS = [e for e in ALL_EVENTS if e.is_second_half]


def _target(record: list[int]) -> np.ndarray:
    return s8d._target(record)


def _record(a: int, b: int) -> list[int]:
    c = a ^ b
    return [a, b, c, a, b, c]


def _phase_infidelity(v: np.ndarray, t: np.ndarray) -> float:
    v = np.asarray(v, dtype=complex)
    t = np.asarray(t, dtype=complex)
    nv = np.linalg.norm(v)
    nt = np.linalg.norm(t)
    if nv <= 1e-15 or nt <= 1e-15:
        return 1.0
    f = float(abs(np.vdot(t / nt, v / nv)) ** 2)
    return max(0.0, min(1.0, 1.0 - f))


def _event_faultbook(events: Iterable[EffEvent]):
    fb = [collections.defaultdict(list) for _ in m.sequence]
    parity: set[int] = set()
    for e in events:
        if e.kind == "record":
            if e.check_index in parity:
                # This can only occur if explicit same-check parity events are
                # supplied. Two flips cancel.
                parity.remove(e.check_index)
            else:
                parity.add(e.check_index)
            continue
        assert e.location_id is not None and e.option_index is not None
        ci, _ck, j, gi = m.locations[e.location_id]
        if int(ci) != e.check_index:
            raise AssertionError((e, m.locations[e.location_id]))
        fb[ci][j].append((gi, e.option_index))
    return fb, parity


def _record_flip_for_check(fbc: dict[int, list[tuple[int, int]]], ck: str) -> int:
    r = 0
    for _j, fs in fbc.items():
        for gi, oi in fs:
            r ^= int(m.opt_cache[(ck, gi)][oi][1])
    return r


def _simulate_core(events: list[EffEvent], record: list[int], stop_after_midpoint: bool = False):
    fb, parity = _event_faultbook(events)
    st = m.psi0.copy()
    qprob = 1.0
    midpoint_pr = None
    last = 2 if stop_after_midpoint else 5
    for ci in range(last + 1):
        ck = m.sequence[ci]
        rf = _record_flip_for_check(fb[ci], ck)
        raw = int(record[ci]) ^ int(rf) ^ int(ci in parity)
        try:
            st, rf2, pr, _ = d2._forced_check(st, ck, fb[ci], raw)
        except AssertionError as ex:
            if "zero-probability" in str(ex) or "zero branch norm" in str(ex):
                return None
            raise
        if int(rf2) != int(rf):
            raise AssertionError((ci, rf, rf2))
        qprob *= float(pr)
        if ci == 2:
            st, midpoint_pr = m.proj(st)
            midpoint_pr = float(midpoint_pr)
            if midpoint_pr <= 1e-15:
                return None
            qprob *= midpoint_pr
    return {"state": st, "probability": qprob, "midpoint_projection_probability": midpoint_pr}


def _unnormalized_ideal_check(st: np.ndarray, ck: str, raw: int) -> np.ndarray:
    m0 = [m.ideal_u[ck][0] for _ in range(7)]
    m1 = [m.ideal_u[ck][1] for _ in range(7)]
    b0 = m.prod(st, m0)
    b1 = m.prod(st, m1)
    return (b0 + b1) / 2 if int(raw) == 0 else (b0 - b1) / 2


def _ideal_second_half_rank1_certificate() -> dict[str, Any]:
    rows = []
    max_off_target = 0.0
    min_ratio = float("inf")
    for a, b in BRANCHES:
        rec = _record(a, b)
        cols = []
        for k in range(4):
            st = m.B[:, k].copy()
            for ci in range(3, 6):
                st = _unnormalized_ideal_check(st, m.sequence[ci], rec[ci])
            cols.append(m.B.conj().T @ st)
        K = np.stack(cols, axis=1)
        svals = np.linalg.svd(K, compute_uv=False)
        rank = int(np.count_nonzero(svals > 1e-11))
        t = _target(rec)
        # Project all output columns onto the orthogonal complement of target.
        Pperp = np.eye(4, dtype=complex) - np.outer(t, t.conj())
        off = float(np.linalg.norm(Pperp @ K))
        max_off_target = max(max_off_target, off)
        if len(svals) > 1 and svals[0] > 0:
            min_ratio = min(min_ratio, float(svals[1] / svals[0]))
        rows.append({"branch": [a, b], "logical_map_rank": rank, "singular_values": [float(x) for x in svals], "off_target_frobenius_norm": off})
    passed = all(r["logical_map_rank"] == 1 and r["off_target_frobenius_norm"] < 1e-10 for r in rows)
    return {"passed": passed, "branches": rows, "max_off_target_norm": max_off_target, "max_second_to_first_singular_ratio": (0.0 if min_ratio == float("inf") else min_ratio)}


def _flatten_escape_classes(b: dict[str, Any]):
    out = []
    for lid, choices in enumerate(b["sigs"]):
        for ci, full in enumerate(choices):
            out.append((int(lid), int(ci), int(full), int(b["frames"][lid][ci])))
    return out


def _faultfree_sector_acceptance_assertions(b: dict[str, Any]) -> dict[str, Any]:
    bad = []
    zero_ok = True
    for s in range(64):
        sr = b["sector"][s]
        actual = int(sr.full) & ACTUAL_MASK
        rejected = actual in b["unsafe"]
        if s == 0:
            zero_ok = (not rejected) and s8d._unique(b["table"].get(int(sr.full))) == int(sr.frame)
        elif not rejected:
            bad.append(s)
    if not zero_ok or bad:
        raise AssertionError(("fault-free sector reduction invalid", zero_ok, bad))
    return {"zero_sector_accepted_and_correct": zero_ok, "nonzero_sectors_rejected": 63, "unexpected_accepted_nonzero_sectors": bad}


def _audit_first_half_single_midpoint(tol: float) -> dict[str, Any]:
    violations = []
    nonzero = 0
    max_inf = 0.0
    original_mult = 0
    for e in FIRST_EVENTS:
        original_mult += e.multiplicity
        for a, b in BRANCHES:
            rec = _record(a, b)
            z = _simulate_core([e], rec, stop_after_midpoint=True)
            if z is None:
                continue
            nonzero += 1
            amps = m.B.conj().T @ z["state"]
            inf = _phase_infidelity(amps, _target(rec))
            max_inf = max(max_inf, inf)
            if inf > tol:
                violations.append({"event": dataclasses.asdict(e), "record": rec, "infidelity": inf, "branch_probability": z["probability"]})
                if len(violations) >= 20:
                    break
        if len(violations) >= 20:
            break
    return {
        "effective_first_half_single_events": len(FIRST_EVENTS),
        "original_single_class_multiplicity_represented": original_mult,
        "nonzero_midpoint_branches": nonzero,
        "target_violations": len(violations),
        "max_midpoint_logical_infidelity": max_inf,
        "examples": violations,
        "passed": not violations,
    }


def _audit_all_core_singles(b: dict[str, Any], tol: float) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    violations = []
    escape0_violations = []
    reachable: set[tuple[int, int]] = set()
    nonzero_records = 0
    sector_branches = 0
    max_inf = 0.0
    original_mult = sum(e.multiplicity for e in ALL_EVENTS)

    for ei, e in enumerate(ALL_EVENTS):
        for a, bb in BRANCHES:
            rec = _record(a, bb)
            z = _simulate_core([e], rec, stop_after_midpoint=False)
            if z is None:
                continue
            nonzero_records += 1
            C, psum, recon, _ = s8live.exact_sector_tensor(z["state"])
            if abs(float(psum) - 1.0) > 1e-9 or float(recon) > 1e-9:
                raise AssertionError((psum, recon, e))
            probs = np.sum(np.abs(C) ** 2, axis=(1, 3))
            target = _target(rec)
            for sa, sb in zip(*np.nonzero(probs > 1e-14)):
                spr = float(probs[sa, sb])
                lv = C[sa, :, sb, :].reshape(4) / math.sqrt(spr)
                inf = _phase_infidelity(lv, target)
                max_inf = max(max_inf, inf)
                sector_branches += 1
                reachable.add((int(sa), int(sb)))
                if inf > tol:
                    violations.append({"event": dataclasses.asdict(e), "record": rec, "sector_pair": [int(sa), int(sb)], "sector_probability": spr, "logical_infidelity_after_canonical_sector_recovery": inf})
                    if len(violations) >= 20:
                        break

                # Exact fault-free escape check for every single-core sector.
                sra = b["sector"][int(sa)]
                srb = b["sector"][int(sb)]
                fa, fb = int(sra.full), int(srb.full)
                if (fa & ACTUAL_MASK) in b["unsafe"] or (fb & ACTUAL_MASK) in b["unsafe"]:
                    continue
                pa = s8d._unique(b["table"].get(fa))
                pb = s8d._unique(b["table"].get(fb))
                if pa is None or pb is None or int(pa) != int(sra.frame) or int(pb) != int(srb.frame) or inf > tol:
                    escape0_violations.append({"event": dataclasses.asdict(e), "record": rec, "sector_pair": [int(sa), int(sb)], "predicted_frames": [None if pa is None else FRAME_LABELS[int(pa)], None if pb is None else FRAME_LABELS[int(pb)]], "actual_frames": [FRAME_LABELS[int(sra.frame)], FRAME_LABELS[int(srb.frame)]], "logical_infidelity": inf})
                    if len(escape0_violations) >= 20:
                        break
            if violations or escape0_violations:
                if len(violations) >= 20 or len(escape0_violations) >= 20:
                    break
        if len(violations) >= 20 or len(escape0_violations) >= 20:
            break
        if (ei + 1) % 250 == 0:
            print(f"S8-E single audit progress {ei+1}/{len(ALL_EVENTS)}", flush=True)

    summary = {
        "effective_single_events": len(ALL_EVENTS),
        "effective_gate_events": len(GATE_EVENTS),
        "effective_record_parity_events": len(RECORD_EVENTS),
        "original_single_fault_class_multiplicity_represented": original_mult,
        "nonzero_valid_record_branches": nonzero_records,
        "nonzero_sector_branches": sector_branches,
        "reachable_sector_pair_count": len(reachable),
        "reachable_sector_pairs": [list(x) for x in sorted(reachable)],
        "max_single_sector_logical_infidelity": max_inf,
        "single_sector_target_violations": len(violations),
        "single_core_faultfree_escape_violations": len(escape0_violations),
        "violations": violations,
        "faultfree_escape_violations": escape0_violations,
        "passed": not violations and not escape0_violations,
    }
    return summary, reachable


def _add_frame(table: dict[int, int], syn: int, frame: int) -> None:
    table[int(syn)] = int(table.get(int(syn), 0)) | (1 << int(frame))


def _audit_core1_escape1(b: dict[str, Any], reachable: set[tuple[int, int]]):
    """Extend the local recovery table with the *actual physical-order* boundary.

    S8-C used canonical representative weight.  A single physical core fault can
    coherently reach a sector whose canonical recovery weight is two, so
    core1+escape1 is not automatically contained in that table.  We therefore
    add every local full-syndrome/frame state reachable from the exhaustively
    audited single-core sectors plus every one-escape-fault class, then rebuild
    the unsafe 171-bit prefixes.  This is exact for the actual physical budget
    core1+escape1.
    """
    esc = _flatten_escape_classes(b)
    table = {int(k): int(v) for k, v in b["table"].items()}
    current_unsafe = set(int(x) for x in b["unsafe"])
    local_sectors = sorted({s for ab in reachable for s in ab})

    added_states = 0
    for s in local_sectors:
        sr = b["sector"][s]
        before = table.get(int(sr.full), 0)
        _add_frame(table, int(sr.full), int(sr.frame))
        added_states += int(table[int(sr.full)] != before)
        for _lid, _ci, fsig, fframe in esc:
            full = int(sr.full) ^ int(fsig)
            frame = int(sr.frame) ^ int(fframe)
            before = table.get(full, 0)
            _add_frame(table, full, frame)
            added_states += int(table[full] != before)

    ambiguous = {syn: flags for syn, flags in table.items() if int(flags).bit_count() > 1}
    expanded_unsafe = {int(syn) & ACTUAL_MASK for syn in ambiguous}
    additional_unsafe = expanded_unsafe - current_unsafe
    removed_unsafe = current_unsafe - expanded_unsafe

    bad = []
    accepted = 0
    resolved = 0
    tests = 0
    # Global core1+escape1 combinations.  Single-core sector vectors were
    # already proven equal to the target; safety now reduces to exact frame
    # recovery under the rebuilt actual-budget policy.
    for sa, sb in sorted(reachable):
        sra = b["sector"][sa]
        srb = b["sector"][sb]
        for block in (0, 1):
            for lid, ci, fsig, fframe in esc:
                tests += 1
                if block == 0:
                    fa = int(sra.full) ^ fsig
                    fb = int(srb.full)
                    aa = int(sra.frame) ^ fframe
                    ab = int(srb.frame)
                else:
                    fa = int(sra.full)
                    fb = int(srb.full) ^ fsig
                    aa = int(sra.frame)
                    ab = int(srb.frame) ^ fframe
                if (fa & ACTUAL_MASK) in expanded_unsafe or (fb & ACTUAL_MASK) in expanded_unsafe:
                    continue
                accepted += 1
                pa = s8d._unique(table.get(fa))
                pb = s8d._unique(table.get(fb))
                if pa is None or pb is None:
                    bad.append({"type": "accepted_unresolved", "sector_pair": [sa, sb], "escape_block": "A" if block == 0 else "B", "escape_location": lid, "escape_class_index": ci, "full_A_hex": hex(fa), "full_B_hex": hex(fb)})
                else:
                    resolved += 1
                    if int(pa) != aa or int(pb) != ab:
                        bad.append({"type": "accepted_frame_mismatch", "sector_pair": [sa, sb], "escape_block": "A" if block == 0 else "B", "escape_location": lid, "escape_class_index": ci, "predicted": [FRAME_LABELS[int(pa)], FRAME_LABELS[int(pb)]], "actual": [FRAME_LABELS[aa], FRAME_LABELS[ab]], "full_A_hex": hex(fa), "full_B_hex": hex(fb)})
                if len(bad) >= 30:
                    break
            if len(bad) >= 30:
                break
        if len(bad) >= 30:
            break
    summary = {
        "reachable_core_single_sector_pairs": len(reachable),
        "reachable_local_sector_count": len(local_sectors),
        "reachable_local_sectors": local_sectors,
        "escape_single_fault_classes_per_block": len(esc),
        "actual_budget_table_state_count": len(table),
        "actual_budget_ambiguous_full_syndromes": len(ambiguous),
        "current_unsafe_prefix_count": len(current_unsafe),
        "expanded_actual_budget_unsafe_prefix_count": len(expanded_unsafe),
        "additional_unsafe_prefixes_vs_S8C": len(additional_unsafe),
        "removed_unsafe_prefixes_vs_S8C": len(removed_unsafe),
        "additional_unsafe_prefixes_hex": [hex(x) for x in sorted(additional_unsafe)],
        "zero_actual_syndrome_is_safe_after_actual_budget_rebuild": 0 not in expanded_unsafe,
        "unique_sector_escape_class_tests": tests,
        "accepted_tests": accepted,
        "resolved_accepted_tests": resolved,
        "accepted_unresolved_or_frame_mismatch": len(bad),
        "examples": bad,
        "existing_S8C_policy_remains_certified_for_actual_core1_escape1": len(additional_unsafe) == 0 and not bad,
        "expanded_actual_budget_policy_certified": (0 not in expanded_unsafe) and not bad,
        "passed": (0 not in expanded_unsafe) and not bad,
    }
    return summary, table, expanded_unsafe

def _pair_valid(e1: EffEvent, e2: EffEvent) -> bool:
    # Two different Pauli classes cannot occur simultaneously at the same
    # stochastic gate location. Record events represent measurement locations
    # and are always distinct from gate locations.
    if e1.kind == "gate" and e2.kind == "gate" and e1.location_id == e2.location_id:
        return False
    return True


def _simulate_second_half_pair(e1: EffEvent, e2: EffEvent, rec: list[int]):
    fb, parity = _event_faultbook([e1, e2])
    st = m.target(int(rec[0]), int(rec[1])).copy()
    qprob = 1.0
    for ci in range(3, 6):
        ck = m.sequence[ci]
        rf = _record_flip_for_check(fb[ci], ck)
        raw = int(rec[ci]) ^ int(rf) ^ int(ci in parity)
        try:
            st, rf2, pr, _ = d2._forced_check(st, ck, fb[ci], raw)
        except AssertionError as ex:
            if "zero-probability" in str(ex) or "zero branch norm" in str(ex):
                return None
            raise
        if int(rf2) != int(rf):
            raise AssertionError((ci, rf, rf2))
        qprob *= float(pr)
    return st, qprob


def _pair_worker(args):
    i0, i1, tol = args
    n = len(SECOND_EVENTS)
    effective_pairs = 0
    original_mult = 0
    nonzero_records = 0
    zero_sector_nonzero = 0
    malignant = []
    max_inf = 0.0
    for i in range(i0, i1):
        e1 = SECOND_EVENTS[i]
        for j in range(i + 1, n):
            e2 = SECOND_EVENTS[j]
            if not _pair_valid(e1, e2):
                continue
            effective_pairs += 1
            original_mult += int(e1.multiplicity) * int(e2.multiplicity)
            pair_bad = False
            for a, b in BRANCHES:
                rec = _record(a, b)
                z = _simulate_second_half_pair(e1, e2, rec)
                if z is None:
                    continue
                st, qprob = z
                nonzero_records += 1
                amps = m.B.conj().T @ st
                cp = float(np.vdot(amps, amps).real)
                if cp <= 1e-14:
                    continue
                zero_sector_nonzero += 1
                inf = _phase_infidelity(amps, _target(rec))
                max_inf = max(max_inf, inf)
                if inf > tol:
                    malignant.append({"event1": dataclasses.asdict(e1), "event2": dataclasses.asdict(e2), "record": rec, "branch_probability": qprob, "zero_sector_probability": cp, "logical_infidelity": inf})
                    pair_bad = True
                    break
            if pair_bad and len(malignant) >= 10:
                return {"effective_pairs": effective_pairs, "original_mult": original_mult, "nonzero_records": nonzero_records, "zero_sector_nonzero": zero_sector_nonzero, "malignant": malignant, "max_inf": max_inf}
    return {"effective_pairs": effective_pairs, "original_mult": original_mult, "nonzero_records": nonzero_records, "zero_sector_nonzero": zero_sector_nonzero, "malignant": malignant, "max_inf": max_inf}


def _balanced_ranges(n: int, parts: int):
    parts = max(1, min(parts, n))
    weights = [max(0, n - i - 1) for i in range(n)]
    total = sum(weights)
    if total == 0:
        return [(0, n)]
    target = total / parts
    ranges = []
    start = 0
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if len(ranges) < parts - 1 and acc >= target * (len(ranges) + 1):
            ranges.append((start, i + 1))
            start = i + 1
    ranges.append((start, n))
    return [r for r in ranges if r[0] < r[1]]


def _audit_second_half_pairs(workers: int, tol: float) -> dict[str, Any]:
    t0 = time.time()
    ranges = _balanced_ranges(len(SECOND_EVENTS), max(1, workers * 2))
    args = [(a, b, tol) for a, b in ranges]
    if workers <= 1:
        parts = []
        for k, x in enumerate(args, 1):
            print(f"S8-E pair chunk {k}/{len(args)}: i={x[0]}..{x[1]-1}", flush=True)
            parts.append(_pair_worker(x))
    else:
        parts = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_pair_worker, x) for x in args]
            for k, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                r = fut.result()
                parts.append(r)
                print(f"S8-E pair chunk completed {k}/{len(futs)}; malignant_so_far={sum(len(q['malignant']) for q in parts)}", flush=True)
    mal = [x for p in parts for x in p["malignant"]]
    # Two physical CAT readout faults in the same check are not represented as
    # two separate effective parity events because their parity flips cancel.
    # There are C(7,2)=21 such physical pairs/check. With no data action this is
    # exactly the zero-readout-fault trajectory and is therefore trivially safe.
    same_check_readout_pairs = 3 * math.comb(7, 2)  # second-half checks only
    return {
        "effective_second_half_events": len(SECOND_EVENTS),
        "effective_distinct_event_pairs_evaluated": sum(p["effective_pairs"] for p in parts),
        "original_fault_class_pair_multiplicity_represented": sum(p["original_mult"] for p in parts),
        "same_check_two_readout_fault_pairs_trivially_safe": same_check_readout_pairs,
        "nonzero_valid_record_branches": sum(p["nonzero_records"] for p in parts),
        "nonzero_zero_sector_branches": sum(p["zero_sector_nonzero"] for p in parts),
        "malignant_effective_pair_count": len(mal),
        "max_zero_sector_logical_infidelity": max([p["max_inf"] for p in parts] + [0.0]),
        "examples": mal[:30],
        "elapsed_seconds": time.time() - t0,
        "passed": not mal,
    }


def _event_inventory() -> dict[str, Any]:
    by_half = collections.Counter("first" if e.is_first_half else "second" for e in GATE_EVENTS)
    by_check_eff = collections.Counter(e.check_index for e in GATE_EVENTS)
    by_check_orig = collections.Counter()
    for e in GATE_EVENTS:
        by_check_orig[e.check_index] += e.multiplicity
    return {
        **GATE_EVENT_META,
        "effective_record_parity_events": len(RECORD_EVENTS),
        "record_event_multiplicity_each": 7,
        "effective_total_core_single_events": len(ALL_EVENTS),
        "effective_gate_events_by_half": dict(by_half),
        "effective_gate_events_by_check": {str(k): v for k, v in sorted(by_check_eff.items())},
        "original_gate_classes_by_check": {str(k): v for k, v in sorted(by_check_orig.items())},
        "effective_first_half_events_including_record": len(FIRST_EVENTS),
        "effective_second_half_events_including_record": len(SECOND_EVENTS),
    }


def _self_test() -> dict[str, Any]:
    inv = _event_inventory()
    if inv["physical_gate_locations"] != 322:
        raise AssertionError(inv)
    if inv["original_gate_pauli_classes"] != 4158:
        raise AssertionError(inv)
    if inv["effective_gate_events"] != 2856:
        raise AssertionError(inv)
    if len(FIRST_EVENTS) != 1431 or len(SECOND_EVENTS) != 1431:
        raise AssertionError((len(FIRST_EVENTS), len(SECOND_EVENTS)))
    r = _ideal_second_half_rank1_certificate()
    if not r["passed"]:
        raise AssertionError(r)
    return {"inventory": inv, "ideal_second_half_rank1": r, "passed": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=False)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--failure-tol", type=float, default=1e-10)
    ap.add_argument("--pair-workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--skip-pair-audit", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE / "s8e_results")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    st = _self_test()
    (a.out_dir / "s8e_self_test.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    if a.self_test:
        print(json.dumps(st, indent=2))
        print("S8-E SELF-TEST PASS")
        return
    if a.hirano_src is None:
        raise SystemExit("--hirano-src is required unless --self-test is used")

    t0 = time.time()
    b = s8d._bridge(a.hirano_src.resolve(), float(a.signature_p))
    sector_reduction = _faultfree_sector_acceptance_assertions(b)
    first_contract = _audit_first_half_single_midpoint(float(a.failure_tol))
    if not first_contract["passed"]:
        raise AssertionError(("first-half single midpoint contract failed", first_contract))
    single_summary, reachable = _audit_all_core_singles(b, float(a.failure_tol))
    if not single_summary["passed"]:
        raise AssertionError(("core single contract failed", single_summary))
    mixed, actual_table, actual_unsafe = _audit_core1_escape1(b, reachable)
    if not mixed["passed"]:
        raise AssertionError(("core1+escape1 certificate failed", mixed))

    pair_summary = None
    if not a.skip_pair_audit:
        pair_summary = _audit_second_half_pairs(int(a.pair_workers), float(a.failure_tol))

    escape_inherited = {
        "statement": "core0+escape<=2 inherited from exact S7-B4/S8-C catalogue used by _bridge",
        "unsafe_actual_syndrome_count": len(b["unsafe"]),
        "zero_actual_syndrome_is_safe": 0 not in b["unsafe"],
        "s8b_actual_signature_injective": bool(b["s8b"]["actual_signature_injective"]),
    }

    complete = pair_summary is not None
    passed = bool(
        escape_inherited["zero_actual_syndrome_is_safe"]
        and sector_reduction["zero_sector_accepted_and_correct"]
        and first_contract["passed"]
        and single_summary["passed"]
        and mixed["passed"]
        and (0 not in actual_unsafe)
        and (pair_summary is not None and pair_summary["passed"])
    )
    summary = {
        "stage": "S8-E",
        "scope": "preserved S8-D integrated core + dual k=2 live ZXZ model",
        "scope_warning": "midpoint ED remains ideal; preparation/CAT-preparation/physical-ED circuits are not fully expanded",
        "fault_budget": "total physical fault order <=2 in preserved integrated model; CAT record parity uses minimum one physical measurement fault per flipped check",
        "event_inventory": _event_inventory(),
        "escape_only_le2_inherited_certificate": escape_inherited,
        "faultfree_escape_sector_reduction": sector_reduction,
        "ideal_second_half_rank1_certificate": st["ideal_second_half_rank1"],
        "first_half_single_midpoint_contract": first_contract,
        "all_core_single_contract": single_summary,
        "core1_plus_escape1": mixed,
        "actual_budget_operational_policy_differs_from_S8C": mixed["additional_unsafe_prefixes_vs_S8C"] != 0,
        "second_half_core_pair_exhaustive": pair_summary,
        "pair_audit_was_skipped": bool(a.skip_pair_audit),
        "global_le2_certificate_complete": complete,
        "exact_accepted_harmful_events_order_le2": 0 if passed else None,
        "preserved_integrated_model_df_ge_3": passed,
        "combined_with_S8D3_malignant_triple_implies_df_eq_3": passed,
        "elapsed_seconds": time.time() - t0,
    }
    (a.out_dir / "s8e_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (a.out_dir / "s8e_reachable_single_core_sector_pairs.json").write_text(json.dumps([list(x) for x in sorted(reachable)], indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if passed:
        print("S8-E GLOBAL LOWER-BOUND PASS: no accepted harmful total-fault event of order <=2 exists in the preserved integrated model")
        print("S8-E + S8-D3: preserved integrated model has d_f = 3")
    elif a.skip_pair_audit:
        print("S8-E PRECHECK PASS/PARTIAL: singles and core1+escape1 closed; second-half core-pair exhaustive audit still required")
    else:
        print("S8-E FAIL/INCOMPLETE: inspect s8e_summary.json")


if __name__ == "__main__":
    main()
