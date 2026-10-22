#!/usr/bin/env python3
"""S8-D2: resolved logical-failure witness logging + deterministic branch replay.

Purpose
=======
S8-D observed a small number of *resolved* accepted logical failures at p=1e-3.
This script reproduces the S8-D sampler exactly (same RNG consumption) while
logging every such failure in enough detail to replay it without randomness.

The key question is fault order.  The preserved six-check core has 322
faultable check-gate locations, while its CAT7 record readout is represented by
an exact parity-flip probability q=[1-(1-2p)^7]/2 rather than seven explicitly
sampled measurement locations per check.  Therefore we report both

  gate_fault_order = core check-gate + escape-A + escape-B failed locations

and a rigorous lower bound on the physical order

  minimum_physical_fault_order = gate_fault_order
                                 + number of CAT-record parity-flipped checks.

Each parity-flipped check requires at least one physical measurement fault;
its true count may be 1,3,5,or7.  Thus if this minimum is >=3, that witness is
certainly an order>=3 physical event within the S8-D model.

This is still not a complete physical-factory proof: the midpoint ED remains an
ideal code-space projection and preparation/CAT/physical-ED circuits are not
expanded in this state-vector driver.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "src_current"
S8D_PATH = HERE / "direct_cs_integrated_core_dual_escape_mc.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s8d = _load("s8d2_base", S8D_PATH)
m = s8d.m
s8live = s8d.s8live
ACTUAL_BITS = s8d.ACTUAL_BITS
ACTUAL_MASK = s8d.ACTUAL_MASK
CORE_LOCATIONS = s8d.CORE_LOCATIONS
FRAME_LABELS = ("I", "X", "Z", "Y")


def _cvec(v: np.ndarray) -> list[list[float]]:
    return [[float(z.real), float(z.imag)] for z in np.asarray(v, dtype=complex).reshape(-1)]


def _vec_from_json(v: list[list[float]]) -> np.ndarray:
    return np.array([complex(float(a), float(b)) for a, b in v], dtype=complex)


def _phase_aligned_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, complex).reshape(-1)
    b = np.asarray(b, complex).reshape(-1)
    ov = np.vdot(a, b)
    if abs(ov) > 1e-15:
        b = b * np.exp(-1j * np.angle(ov))
    return float(np.linalg.norm(a - b))


def _core_fault_descriptor(lid: int, oi: int) -> dict[str, Any]:
    ci, ck, j, gi = m.locations[int(lid)]
    typ, q, _g = m.seqs[ck][gi]
    F, flip, label = m.opt_cache[(ck, gi)][int(oi)]
    return {
        "location_id": int(lid),
        "check_index": int(ci),
        "check": str(ck),
        "coordinate": int(j),
        "gate_index": int(gi),
        "gate": str(typ),
        "oneq_target_within_AB": (None if q is None else int(q)),
        "fault_option_index": int(oi),
        "fault_class": str(label),
        "record_flip_from_fault": int(flip),
    }


def _fault_book_trace(p: float, rng: np.random.Generator):
    # Exact copy of S8-D RNG consumption, with logging only.
    k = int(rng.binomial(CORE_LOCATIONS, p))
    fb = [collections.defaultdict(list) for _ in m.sequence]
    faults: list[dict[str, Any]] = []
    if k == 0:
        return fb, 0, faults
    for lid0 in np.asarray(rng.choice(CORE_LOCATIONS, size=k, replace=False), dtype=np.int64):
        lid = int(lid0)
        ci, ck, j, gi = m.locations[lid]
        oi = int(rng.integers(0, len(m.opt_cache[(ck, gi)])))
        fb[ci][j].append((gi, oi))
        faults.append(_core_fault_descriptor(lid, oi))
    return fb, k, faults


class _TraceRandomAdapter:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.calls: list[float] = []

    def random(self):
        u = float(self.rng.random())
        self.calls.append(u)
        return u


def _record_flip_from_faults(ck: str, fbc: dict[int, list[tuple[int, int]]]) -> int:
    rf = 0
    for _j, fs in fbc.items():
        for gi, oi in fs:
            _F, f, _lab = m.opt_cache[(ck, gi)][oi]
            rf ^= int(f)
    return int(rf)


def _forced_check(st: np.ndarray, ck: str, fbc: dict[int, list[tuple[int, int]]], raw_outcome: int):
    """Replay one core check with a forced quantum branch outcome."""
    m0, m1 = [], []
    rf = 0
    for j in range(7):
        fs = fbc.get(j, [])
        if fs:
            u0, r0 = m.coord(ck, 0, fs)
            u1, r1 = m.coord(ck, 1, fs)
            if r0 != r1:
                raise AssertionError((ck, j, r0, r1))
            rf ^= int(r0)
        else:
            u0, u1 = m.ideal_u[ck]
        m0.append(u0)
        m1.append(u1)
    b0 = m.prod(st, m0)
    b1 = m.prod(st, m1)
    kp = (b0 + b1) / 2
    km = (b0 - b1) / 2
    p0 = float(np.vdot(kp, kp).real)
    p1 = float(np.vdot(km, km).real)
    s = p0 + p1
    if s <= 0:
        raise AssertionError("zero branch norm")
    probs = (p0 / s, p1 / s)
    r = int(raw_outcome)
    out = kp if r == 0 else km
    pr = probs[r]
    if pr <= 1e-15:
        raise AssertionError(("forced zero-probability branch", ck, r, probs))
    out = out / np.linalg.norm(out)
    return out, int(rf), float(pr), probs


def _sample_sector_trace(C: np.ndarray, rng: np.random.Generator):
    probs = np.sum(np.abs(C) ** 2, axis=(1, 3))
    flat = probs.reshape(-1)
    total = float(flat.sum())
    if abs(total - 1) > 1e-9:
        raise AssertionError(total)
    idx = int(rng.choice(flat.size, p=flat / total))
    sa, sb = divmod(idx, 64)
    pr = float(probs[sa, sb])
    v = C[sa, :, sb, :].reshape(4) / math.sqrt(pr)
    sc = int(np.count_nonzero(probs > 1e-12))
    return int(sa), int(sb), v, sc, pr


def _sample_core_trace(p: float, rng: np.random.Generator):
    fb, nf, faults = _fault_book_trace(p, rng)
    q = s8d._q(p)

    trace: dict[str, Any] = {
        "core_check_fault_count": int(nf),
        "core_check_faults": faults,
        "cat_record_parity_flip_checks": [],
        "cat_record_parity_flip_count": 0,
        "checks": [],
        "fastpath": bool(nf == 0),
    }

    # Exact copy of S8-D fast-path RNG consumption.
    if nf == 0:
        u = float(rng.random())
        if u < 5 / 8:
            a, b = 0, 0
        elif u < 6 / 8:
            a, b = 0, 1
        elif u < 7 / 8:
            a, b = 1, 0
        else:
            a, b = 1, 1
        ideal_rec = s8d._valid_rec(a, b)
        rec: list[int] = []
        for ci, bit in enumerate(ideal_rec):
            ur = float(rng.random())
            pf = int(ur < q)
            if pf:
                trace["cat_record_parity_flip_checks"].append(int(ci))
            rec.append(int(bit ^ pf))
            trace["checks"].append({
                "check_index": int(ci),
                "check": str(m.sequence[ci]),
                "ideal_branch_bit": int(bit),
                "record_parity_flip": int(pf),
                "record_bit": int(rec[-1]),
                "rng_record_u": ur,
            })
        trace["cat_record_parity_flip_count"] = len(trace["cat_record_parity_flip_checks"])
        trace["ideal_branch_ab"] = [int(a), int(b)]
        trace["branch_rng_u"] = u
        trace["record"] = rec
        if not m.valid(rec):
            trace["rejection_stage"] = "record"
            return False, rec, None, None, nf, 1, True, trace
        trace["sector_pair"] = [0, 0]
        trace["sector_probability"] = 1.0
        trace["sector_count"] = 1
        lv = s8d._branch_vec(a, b)
        trace["logical_state_pre_escape"] = _cvec(lv)
        return True, rec, (0, 0), lv, nf, 1, True, trace

    rr = _TraceRandomAdapter(rng)
    st = m.psi0.copy()
    rec: list[int] = []
    for ci, ck in enumerate(m.sequence):
        before = len(rr.calls)
        st, meas = m.check(st, ck, fb[ci], rr, q)
        calls = rr.calls[before:]
        if len(calls) != 2:
            raise AssertionError((ci, ck, len(calls), calls))
        u_branch, u_record = calls
        pf = int(u_record < q)
        rf = _record_flip_from_faults(ck, fb[ci])
        raw = int(meas) ^ int(rf) ^ int(pf)
        if pf:
            trace["cat_record_parity_flip_checks"].append(int(ci))
        rec.append(int(meas))
        trace["checks"].append({
            "check_index": int(ci),
            "check": str(ck),
            "raw_quantum_outcome": int(raw),
            "record_flip_from_gate_faults": int(rf),
            "record_parity_flip": int(pf),
            "record_bit": int(meas),
            "rng_branch_u": float(u_branch),
            "rng_record_u": float(u_record),
        })
        if ci == 2:
            st, pr = m.proj(st)
            u_mid = float(rng.random())
            passed = bool(u_mid < pr)
            trace["midpoint_projection_probability"] = float(pr)
            trace["midpoint_projection_rng_u"] = u_mid
            trace["midpoint_projection_passed"] = passed
            if not passed:
                trace["cat_record_parity_flip_count"] = len(trace["cat_record_parity_flip_checks"])
                trace["record"] = rec
                trace["rejection_stage"] = "midpoint_projection"
                return False, rec, None, None, nf, 0, False, trace

    trace["cat_record_parity_flip_count"] = len(trace["cat_record_parity_flip_checks"])
    trace["record"] = rec
    if not m.valid(rec):
        trace["rejection_stage"] = "record"
        return False, rec, None, None, nf, 0, False, trace

    C, psum, recon, _ = s8live.exact_sector_tensor(st)
    if abs(psum - 1) > 1e-9 or recon > 1e-9:
        raise AssertionError((psum, recon))
    sa, sb, v, sc, spr = _sample_sector_trace(C, rng)
    trace["sector_pair"] = [sa, sb]
    trace["sector_probability"] = float(spr)
    trace["sector_count"] = int(sc)
    trace["logical_state_pre_escape"] = _cvec(v)
    trace["sector_reconstruction_error"] = float(recon)
    return True, rec, (sa, sb), v, nf, sc, False, trace


def _escape_rows_by_location(hirano_src: Path, signature_p: float, base_bridge: dict[str, Any]):
    paired, _analysis, _policy, _meta = s8d.s8c._escape_catalogue(hirano_src, signature_p)
    by: dict[int, list[Any]] = collections.defaultdict(list)
    for r in paired:
        by[int(r.location_id)].append(r)
    out: list[list[Any]] = []
    for lid in range(len(base_bridge["sigs"])):
        rows = sorted(by[lid], key=lambda r: int(r.fault_id))
        if len(rows) != len(base_bridge["sigs"][lid]):
            raise AssertionError((lid, len(rows), len(base_bridge["sigs"][lid])))
        # Verify class ordering exactly matches S8-D's signature arrays.
        for ci, r in enumerate(rows):
            if int(r.recovery_syndrome) != int(base_bridge["sigs"][lid][ci]):
                raise AssertionError(("signature order mismatch", lid, ci))
            if int(r.frame) != int(base_bridge["frames"][lid][ci]):
                raise AssertionError(("frame order mismatch", lid, ci))
        out.append(rows)
    return out


def _escape_fault_descriptor(r: Any, class_index: int) -> dict[str, Any]:
    return {
        "fault_id": int(r.fault_id),
        "location_id": int(r.location_id),
        "tick": int(r.tick),
        "gate": str(r.gate),
        "qubits": [int(x) for x in r.qubits],
        "fault_class": str(r.fault_class),
        "class_index_within_location": int(class_index),
        "full_recovery_syndrome_hex": hex(int(r.recovery_syndrome)),
        "terminal_frame": FRAME_LABELS[int(r.frame)],
        "terminal_frame_int": int(r.frame),
    }


def _sample_escape_trace(p: float, rng: np.random.Generator, b: dict[str, Any], rows_by_loc: list[list[Any]]):
    n = len(b["sigs"])
    k = int(rng.binomial(n, p))
    if k == 0:
        return 0, 0, 0, []
    full = 0
    frame = 0
    faults: list[dict[str, Any]] = []
    for lid0 in np.asarray(rng.choice(n, size=k, replace=False), dtype=np.int64):
        lid = int(lid0)
        choices = b["sigs"][lid]
        ci = 0 if len(choices) == 1 else int(rng.integers(0, len(choices)))
        full ^= int(choices[ci])
        frame ^= int(b["frames"][lid][ci])
        faults.append(_escape_fault_descriptor(rows_by_loc[lid][ci], ci))
    return int(full), int(frame), int(k), faults


def _faultbook_from_witness(faults: list[dict[str, Any]]):
    fb = [collections.defaultdict(list) for _ in m.sequence]
    for f in faults:
        lid = int(f["location_id"])
        oi = int(f["fault_option_index"])
        ci, ck, j, gi = m.locations[lid]
        if [int(ci), str(ck), int(j), int(gi)] != [
            int(f["check_index"]), str(f["check"]), int(f["coordinate"]), int(f["gate_index"])
        ]:
            raise AssertionError(("core descriptor mismatch", lid, f))
        fb[ci][j].append((gi, oi))
    return fb


def _replay_core(w: dict[str, Any]):
    ct = w["core"]
    rec_expected = [int(x) for x in ct["record"]]
    if ct["fastpath"]:
        a, b = map(int, ct["ideal_branch_ab"])
        rec = [int(x["record_bit"]) for x in ct["checks"]]
        if rec != rec_expected:
            raise AssertionError((rec, rec_expected))
        if not m.valid(rec):
            raise AssertionError("replayed witness record is invalid")
        return rec, (0, 0), s8d._branch_vec(a, b), {
            "logical_state_phase_aligned_error": _phase_aligned_error(
                s8d._branch_vec(a, b), _vec_from_json(ct["logical_state_pre_escape"])
            ),
            "sector_probability": 1.0,
        }

    fb = _faultbook_from_witness(ct["core_check_faults"])
    st = m.psi0.copy()
    rec: list[int] = []
    branch_probs: list[float] = []
    for ci, ck in enumerate(m.sequence):
        c = ct["checks"][ci]
        raw = int(c["raw_quantum_outcome"])
        st, rf, pr, _probs = _forced_check(st, ck, fb[ci], raw)
        branch_probs.append(float(pr))
        pf = int(c["record_parity_flip"])
        meas = raw ^ rf ^ pf
        if meas != int(c["record_bit"]):
            raise AssertionError(("record replay mismatch", ci, meas, c))
        rec.append(int(meas))
        if ci == 2:
            st, pproj = m.proj(st)
            if abs(float(pproj) - float(ct["midpoint_projection_probability"])) > 1e-9:
                raise AssertionError(("midpoint probability mismatch", pproj, ct["midpoint_projection_probability"]))
            if pproj <= 1e-15:
                raise AssertionError("accepted witness has zero midpoint projection probability")
    if rec != rec_expected or not m.valid(rec):
        raise AssertionError((rec, rec_expected))
    C, psum, recon, _ = s8live.exact_sector_tensor(st)
    if abs(psum - 1) > 1e-9 or recon > 1e-9:
        raise AssertionError((psum, recon))
    sa, sb = map(int, ct["sector_pair"])
    probs = np.sum(np.abs(C) ** 2, axis=(1, 3))
    spr = float(probs[sa, sb])
    if spr <= 1e-15:
        raise AssertionError(("zero replay sector probability", sa, sb))
    lv = C[sa, :, sb, :].reshape(4) / math.sqrt(spr)
    return rec, (sa, sb), lv, {
        "logical_state_phase_aligned_error": _phase_aligned_error(lv, _vec_from_json(ct["logical_state_pre_escape"])),
        "sector_probability": spr,
        "sector_probability_error": abs(spr - float(ct["sector_probability"])),
        "sector_reconstruction_error": float(recon),
        "forced_branch_probabilities": branch_probs,
    }


def _xor_escape_from_witness(faults: list[dict[str, Any]], b: dict[str, Any]):
    full = frame = 0
    for f in faults:
        lid = int(f["location_id"])
        ci = int(f["class_index_within_location"])
        full ^= int(b["sigs"][lid][ci])
        frame ^= int(b["frames"][lid][ci])
    return int(full), int(frame)


def replay_witness(w: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    rec, (sa, sb), lv, core_diag = _replay_core(w)
    af, afr = _xor_escape_from_witness(w["escape_A"]["faults"], b)
    bf, bfr = _xor_escape_from_witness(w["escape_B"]["faults"], b)
    sra = b["sector"][sa]
    srb = b["sector"][sb]
    fa = int(sra.full) ^ af
    fb = int(srb.full) ^ bf
    na = fa & ACTUAL_MASK
    nb = fb & ACTUAL_MASK
    if na in b["unsafe"] or nb in b["unsafe"]:
        raise AssertionError("logged accepted witness replays as rejected")
    pa = s8d._unique(b["table"].get(fa))
    pb = s8d._unique(b["table"].get(fb))
    if pa is None or pb is None:
        raise AssertionError("logged resolved witness replays as unresolved")
    aa = int(sra.frame) ^ afr
    ab = int(srb.frame) ^ bfr
    corr = s8d._apply(lv, pa ^ aa, pb ^ ab)
    fid = float(abs(np.vdot(s8d._target(rec), corr)) ** 2)
    fid = min(1.0, max(0.0, fid))
    inf = 1.0 - fid
    return {
        "replay_pass": bool(
            rec == [int(x) for x in w["core"]["record"]]
            and [sa, sb] == [int(x) for x in w["core"]["sector_pair"]]
            and abs(inf - float(w["result"]["logical_infidelity"])) < 1e-9
            and core_diag["logical_state_phase_aligned_error"] < 1e-9
        ),
        "core": core_diag,
        "full_syndrome_A_hex": hex(fa),
        "full_syndrome_B_hex": hex(fb),
        "actual_syndrome_A_hex": hex(na),
        "actual_syndrome_B_hex": hex(nb),
        "predicted_frame_A": FRAME_LABELS[int(pa)],
        "predicted_frame_B": FRAME_LABELS[int(pb)],
        "actual_frame_A": FRAME_LABELS[int(aa)],
        "actual_frame_B": FRAME_LABELS[int(ab)],
        "applied_residual_correction_A": FRAME_LABELS[int(pa ^ aa)],
        "applied_residual_correction_B": FRAME_LABELS[int(pb ^ ab)],
        "logical_state_after_recovery": _cvec(corr),
        "target_state": _cvec(s8d._target(rec)),
        "logical_fidelity": fid,
        "logical_infidelity": inf,
    }


def _one_run(
    *,
    p: float,
    shots: int,
    seed: int,
    b: dict[str, Any],
    rows_by_loc: list[list[Any]],
    failure_tol: float,
    max_witnesses: int,
    out_dir: Path,
):
    rng = np.random.default_rng(seed)
    core_acc = int_acc = resolved = unresolved = 0
    inf_sum = 0.0
    failure_count = 0
    all_failure_orders: list[int] = []
    gate_orders: list[int] = []
    witnesses: list[dict[str, Any]] = []

    for shot_index in range(shots):
        ok, rec, ss, lv, nfc, sc, fast, core_trace = _sample_core_trace(p, rng)
        if not ok:
            continue
        core_acc += 1
        sa, sb = ss

        af, afr, nfa, afaults = _sample_escape_trace(p, rng, b, rows_by_loc)
        bf, bfr, nfb, bfaults = _sample_escape_trace(p, rng, b, rows_by_loc)
        sra = b["sector"][sa]
        srb = b["sector"][sb]
        fa = int(sra.full) ^ af
        fb = int(srb.full) ^ bf
        na = fa & ACTUAL_MASK
        nb = fb & ACTUAL_MASK
        if na in b["unsafe"] or nb in b["unsafe"]:
            continue
        int_acc += 1
        pa = s8d._unique(b["table"].get(fa))
        pb = s8d._unique(b["table"].get(fb))
        if pa is None or pb is None:
            unresolved += 1
            continue
        resolved += 1
        aa = int(sra.frame) ^ afr
        ab = int(srb.frame) ^ bfr
        corr = s8d._apply(lv, pa ^ aa, pb ^ ab)
        fid = float(abs(np.vdot(s8d._target(rec), corr)) ** 2)
        fid = min(1.0, max(0.0, fid))
        inf = 1.0 - fid
        inf_sum += inf
        if inf <= failure_tol:
            continue

        failure_count += 1
        gate_order = int(nfc + nfa + nfb)
        parity_lb = int(core_trace["cat_record_parity_flip_count"])
        min_phys_order = gate_order + parity_lb
        gate_orders.append(gate_order)
        all_failure_orders.append(min_phys_order)

        witness = {
            "stage": "S8-D2",
            "shot_index_zero_based": int(shot_index),
            "p": float(p),
            "seed": int(seed),
            "scope_warning": "midpoint ED is still ideal; preparation/CAT/physical-ED circuits are not fully expanded",
            "fault_order": {
                "core_check_gate_fault_locations": int(nfc),
                "escape_A_fault_locations": int(nfa),
                "escape_B_fault_locations": int(nfb),
                "gate_fault_order": gate_order,
                "cat_record_parity_flipped_checks": parity_lb,
                "minimum_physical_fault_order": min_phys_order,
                "interpretation": "each parity-flipped CAT7 record bit requires >=1 physical measurement fault; true physical order can be higher",
            },
            "core": core_trace,
            "escape_A": {
                "fault_count": int(nfa),
                "faults": afaults,
                "fault_signature_full_hex": hex(int(af)),
                "fault_frame": FRAME_LABELS[int(afr)],
                "incoming_sector": int(sa),
                "incoming_sector_pauli": str(sra.pauli),
                "incoming_sector_full_signature_hex": hex(int(sra.full)),
                "combined_full_syndrome_hex": hex(int(fa)),
                "combined_actual_171_syndrome_hex": hex(int(na)),
                "predicted_recovery_frame": FRAME_LABELS[int(pa)],
                "actual_terminal_frame": FRAME_LABELS[int(aa)],
            },
            "escape_B": {
                "fault_count": int(nfb),
                "faults": bfaults,
                "fault_signature_full_hex": hex(int(bf)),
                "fault_frame": FRAME_LABELS[int(bfr)],
                "incoming_sector": int(sb),
                "incoming_sector_pauli": str(srb.pauli),
                "incoming_sector_full_signature_hex": hex(int(srb.full)),
                "combined_full_syndrome_hex": hex(int(fb)),
                "combined_actual_171_syndrome_hex": hex(int(nb)),
                "predicted_recovery_frame": FRAME_LABELS[int(pb)],
                "actual_terminal_frame": FRAME_LABELS[int(ab)],
            },
            "result": {
                "resolved": True,
                "logical_state_after_recovery": _cvec(corr),
                "target_state": _cvec(s8d._target(rec)),
                "logical_fidelity": fid,
                "logical_infidelity": inf,
            },
        }
        replay = replay_witness(witness, b)
        witness["deterministic_replay"] = replay
        if not replay["replay_pass"]:
            raise AssertionError(("witness deterministic replay failed", shot_index, replay))
        if len(witnesses) < max_witnesses:
            witnesses.append(witness)
            (out_dir / f"s8d2_witness_{len(witnesses):03d}.json").write_text(
                json.dumps(witness, indent=2), encoding="utf-8"
            )

    summary = {
        "stage": "S8-D2",
        "p": float(p),
        "shots": int(shots),
        "seed": int(seed),
        "core_live_accepted": int(core_acc),
        "integrated_accepted": int(int_acc),
        "resolved_accepted": int(resolved),
        "unresolved_accepted": int(unresolved),
        "resolved_logical_infidelity_sum": float(inf_sum),
        "resolved_failure_count": int(failure_count),
        "failure_tolerance": float(failure_tol),
        "logged_witness_count": int(len(witnesses)),
        "all_logged_witnesses_replay_pass": bool(all(w["deterministic_replay"]["replay_pass"] for w in witnesses)),
        "failure_gate_fault_orders": gate_orders,
        "failure_minimum_physical_fault_orders": all_failure_orders,
        "minimum_observed_failure_gate_fault_order": (min(gate_orders) if gate_orders else None),
        "minimum_observed_failure_physical_order_lower_bound": (min(all_failure_orders) if all_failure_orders else None),
        "all_observed_resolved_failures_guaranteed_order_ge3": bool(all(x >= 3 for x in all_failure_orders)),
        "scientific_scope": {
            "what_this_can_show": "the observed S8-D resolved failures are/are not compatible with leading physical order >=3 within the preserved core+dual-escape model",
            "what_this_does_not_prove": "absence of every possible <=2 full-factory mechanism; midpoint ED remains ideal and other physical modules are not expanded",
        },
    }
    (out_dir / "s8d2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, witnesses


def _selftest_core_replay():
    # Does not need Hirano.  Find a non-fast accepted core trajectory and verify
    # that forced branch replay reconstructs its logical boundary state.
    rng = np.random.default_rng(90210)
    for _ in range(2000):
        ok, rec, ss, lv, nf, sc, fast, tr = _sample_core_trace(1e-3, rng)
        if ok and not fast:
            dummy = {"core": tr}
            rr, rss, rlv, diag = _replay_core(dummy)
            if rr != rec or rss != ss or _phase_aligned_error(rlv, lv) > 1e-9:
                raise AssertionError((rr, rec, rss, ss, diag))
            return {
                "core_replay_selftest_pass": True,
                "sample_fault_count": int(nf),
                "sample_sector_count": int(sc),
                "sample_phase_aligned_error": _phase_aligned_error(rlv, lv),
            }
    raise AssertionError("could not find non-fast accepted core trajectory for self-test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--p", type=float, default=1e-3)
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--failure-tol", type=float, default=1e-10)
    ap.add_argument("--max-witnesses", type=int, default=100)
    ap.add_argument("--expect-integrated-accepted", type=int)
    ap.add_argument("--expect-resolved-failures", type=int)
    ap.add_argument("--expect-infidelity-sum", type=float)
    ap.add_argument("--out-dir", type=Path, default=HERE / "s8d2_results")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    st = _selftest_core_replay()
    (a.out_dir / "s8d2_core_replay_selftest.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    print(json.dumps(st, indent=2), flush=True)

    b = s8d._bridge(a.hirano_src.resolve(), a.signature_p)
    rows_by_loc = _escape_rows_by_location(a.hirano_src.resolve(), a.signature_p, b)
    summary, _w = _one_run(
        p=float(a.p),
        shots=int(a.shots),
        seed=int(a.seed),
        b=b,
        rows_by_loc=rows_by_loc,
        failure_tol=float(a.failure_tol),
        max_witnesses=int(a.max_witnesses),
        out_dir=a.out_dir,
    )

    if a.expect_integrated_accepted is not None and summary["integrated_accepted"] != a.expect_integrated_accepted:
        raise AssertionError(("integrated_accepted fingerprint mismatch", summary["integrated_accepted"], a.expect_integrated_accepted))
    if a.expect_resolved_failures is not None and summary["resolved_failure_count"] != a.expect_resolved_failures:
        raise AssertionError(("resolved_failure_count fingerprint mismatch", summary["resolved_failure_count"], a.expect_resolved_failures))
    if a.expect_infidelity_sum is not None and abs(summary["resolved_logical_infidelity_sum"] - a.expect_infidelity_sum) > 1e-9:
        raise AssertionError(("infidelity sum fingerprint mismatch", summary["resolved_logical_infidelity_sum"], a.expect_infidelity_sum))

    print(json.dumps(summary, indent=2), flush=True)
    if summary["resolved_failure_count"] and summary["all_observed_resolved_failures_guaranteed_order_ge3"]:
        print("S8-D2 OBSERVED-FAILURE PASS: every resolved logical-failure witness has minimum physical fault order >= 3")
    elif summary["resolved_failure_count"]:
        print("S8-D2 ALERT: at least one resolved logical-failure witness has minimum physical fault order <= 2")
    else:
        print("S8-D2 COMPLETE: no resolved logical failure observed in this run")
    print("S8-D2 PASS: witness logging + deterministic branch replay completed")


if __name__ == "__main__":
    main()
