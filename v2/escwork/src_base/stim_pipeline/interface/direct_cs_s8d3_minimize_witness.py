#!/usr/bin/env python3
"""S8-D3: deterministically minimize accepted logical-failure witnesses.

Given one or more S8-D2 witness JSON files, enumerate every subset of the
logged physical fault events.  For each subset, replay the same quantum
measurement branch outcomes used by the parent witness, enumerate every
nonzero Steane syndrome-sector branch at the live core boundary, and test the
actual k=2 operational ZXZ accept/reject policy and <=2 recovery lookup.

A subset is called malignant when there exists a nonzero-probability core
sector branch such that both live ZXZ blocks accept, the recovery lookup is
resolved, and the corrected logical state has infidelity > tolerance.

This gives an exact upper-bound witness within the preserved S8-D model.  It is
not a complete physical-factory theorem because the midpoint ED remains an
ideal code-space projection and prep/CAT/physical-ED are not fully expanded.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

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


d2 = _load("s8d3_d2", D2_PATH)
s8d = d2.s8d
m = d2.m
s8live = d2.s8live
ACTUAL_MASK = d2.ACTUAL_MASK
FRAME_LABELS = d2.FRAME_LABELS


def _faultbook(core_faults: list[dict[str, Any]]):
    fb = [collections.defaultdict(list) for _ in m.sequence]
    for f in core_faults:
        lid = int(f["location_id"])
        oi = int(f["fault_option_index"])
        ci, ck, j, gi = m.locations[lid]
        if (int(ci), str(ck), int(j), int(gi)) != (
            int(f["check_index"]), str(f["check"]), int(f["coordinate"]), int(f["gate_index"])
        ):
            raise AssertionError(("core descriptor mismatch", f))
        fb[ci][j].append((gi, oi))
    return fb


def _core_branches(parent: dict[str, Any], core_faults: list[dict[str, Any]], parity_checks: set[int]):
    """Replay parent raw quantum outcomes, then enumerate all nonzero final sectors."""
    ct = parent["core"]
    fb = _faultbook(core_faults)
    st = m.psi0.copy()
    rec: list[int] = []
    qbranch = 1.0
    midpoint_pr = 1.0

    for ci, ck in enumerate(m.sequence):
        c = ct["checks"][ci]
        if "raw_quantum_outcome" in c:
            raw = int(c["raw_quantum_outcome"])
        else:
            # Fast-path parent: ideal branch bit is the raw check eigenvalue bit.
            raw = int(c["ideal_branch_bit"])
        st, rf, pr, _probs = d2._forced_check(st, ck, fb[ci], raw)
        qbranch *= float(pr)
        pf = int(ci in parity_checks)
        meas = int(raw ^ rf ^ pf)
        rec.append(meas)
        if ci == 2:
            st, midpoint_pr = m.proj(st)
            midpoint_pr = float(midpoint_pr)
            if midpoint_pr <= 1e-15:
                return None
            qbranch *= midpoint_pr

    if not m.valid(rec):
        return None

    C, psum, recon, _ = s8live.exact_sector_tensor(st)
    if abs(float(psum) - 1.0) > 1e-9 or float(recon) > 1e-9:
        raise AssertionError((psum, recon))
    probs = np.sum(np.abs(C) ** 2, axis=(1, 3))
    branches = []
    for sa, sb in zip(*np.nonzero(probs > 1e-14)):
        spr = float(probs[sa, sb])
        lv = C[sa, :, sb, :].reshape(4) / math.sqrt(spr)
        branches.append((int(sa), int(sb), spr, lv))
    return {
        "record": rec,
        "quantum_branch_probability_excluding_readout_fault_probability": float(qbranch),
        "midpoint_projection_probability": midpoint_pr,
        "sector_reconstruction_error": float(recon),
        "branches": branches,
    }


def _fault_items(w: dict[str, Any]):
    items = []
    for i, f in enumerate(w["core"]["core_check_faults"]):
        items.append((f"C{i}", "core", i, f))
    for i, f in enumerate(w["escape_A"]["faults"]):
        items.append((f"A{i}", "A", i, f))
    for i, f in enumerate(w["escape_B"]["faults"]):
        items.append((f"B{i}", "B", i, f))
    for i, ci in enumerate(w["core"].get("cat_record_parity_flip_checks", [])):
        items.append((f"M{i}", "parity", i, int(ci)))
    return items


def _evaluate_subset(w: dict[str, Any], kept: list[tuple], b: dict[str, Any], tol: float):
    core_faults = [x[3] for x in kept if x[1] == "core"]
    afaults = [x[3] for x in kept if x[1] == "A"]
    bfaults = [x[3] for x in kept if x[1] == "B"]
    parity_checks = {int(x[3]) for x in kept if x[1] == "parity"}

    cb = _core_branches(w, core_faults, parity_checks)
    if cb is None:
        return {"malignant": False, "reason": "core branch rejected/zero-probability"}

    af, afr = d2._xor_escape_from_witness(afaults, b)
    bf, bfr = d2._xor_escape_from_witness(bfaults, b)
    target = s8d._target(cb["record"])
    malignant = []
    accepted = 0
    resolved = 0

    for sa, sb, spr, lv in cb["branches"]:
        sra = b["sector"][sa]
        srb = b["sector"][sb]
        fa = int(sra.full) ^ int(af)
        fb = int(srb.full) ^ int(bf)
        na = fa & ACTUAL_MASK
        nb = fb & ACTUAL_MASK
        if na in b["unsafe"] or nb in b["unsafe"]:
            continue
        accepted += 1
        pa = s8d._unique(b["table"].get(fa))
        pb = s8d._unique(b["table"].get(fb))
        if pa is None or pb is None:
            continue
        resolved += 1
        aa = int(sra.frame) ^ int(afr)
        ab = int(srb.frame) ^ int(bfr)
        corr = s8d._apply(lv, int(pa) ^ aa, int(pb) ^ ab)
        fid = float(abs(np.vdot(target, corr)) ** 2)
        fid = min(1.0, max(0.0, fid))
        inf = 1.0 - fid
        if inf > tol:
            malignant.append({
                "sector_pair": [sa, sb],
                "sector_probability": spr,
                "full_syndrome_A_hex": hex(fa),
                "full_syndrome_B_hex": hex(fb),
                "actual_syndrome_A_hex": hex(na),
                "actual_syndrome_B_hex": hex(nb),
                "predicted_frame_A": FRAME_LABELS[int(pa)],
                "predicted_frame_B": FRAME_LABELS[int(pb)],
                "actual_frame_A": FRAME_LABELS[int(aa)],
                "actual_frame_B": FRAME_LABELS[int(ab)],
                "applied_residual_correction_A": FRAME_LABELS[int(pa) ^ aa],
                "applied_residual_correction_B": FRAME_LABELS[int(pb) ^ ab],
                "logical_fidelity": fid,
                "logical_infidelity": inf,
                "logical_state_pre_escape": d2._cvec(lv),
                "logical_state_after_recovery": d2._cvec(corr),
            })

    return {
        "malignant": bool(malignant),
        "record": cb["record"],
        "quantum_branch_probability_excluding_readout_fault_probability": cb["quantum_branch_probability_excluding_readout_fault_probability"],
        "sector_branch_count": len(cb["branches"]),
        "accepted_sector_branch_count": accepted,
        "resolved_sector_branch_count": resolved,
        "malignant_branches": malignant,
    }


def _minimize_one(path: Path, b: dict[str, Any], tol: float):
    w = json.loads(path.read_text(encoding="utf-8"))
    items = _fault_items(w)
    all_results = []
    minimum = None
    minimum_results = []
    no_le2 = True

    for order in range(len(items) + 1):
        order_malignant = []
        for idxs in itertools.combinations(range(len(items)), order):
            kept = [items[i] for i in idxs]
            ev = _evaluate_subset(w, kept, b, tol)
            row = {
                "order": order,
                "kept_labels": [x[0] for x in kept],
                "kept_faults": [
                    {"label": x[0], "domain": x[1], "descriptor": x[3]} for x in kept
                ],
                **ev,
            }
            all_results.append(row)
            if ev["malignant"]:
                order_malignant.append(row)
        if order <= 2 and order_malignant:
            no_le2 = False
        if order_malignant and minimum is None:
            minimum = order
            minimum_results = order_malignant
            # Keep evaluating order <=2 status already complete because ascending.
            break

    return {
        "stage": "S8-D3",
        "source_witness": str(path),
        "parent_minimum_physical_fault_order": int(w["fault_order"]["minimum_physical_fault_order"]),
        "parent_logical_infidelity": float(w["result"]["logical_infidelity"]),
        "logged_fault_labels": [x[0] for x in items],
        "logged_fault_count": len(items),
        "minimum_malignant_subset_order": minimum,
        "minimum_malignant_subset_count": len(minimum_results),
        "no_malignant_subset_at_order_le2": bool(no_le2 and (minimum is None or minimum >= 3)),
        "minimum_malignant_subsets": minimum_results,
        "subsets_evaluated_until_first_malignant_order": len(all_results),
        "scope_warning": "subset minimization is exact for the preserved S8-D core+dual-escape model conditioned on the parent witness quantum branch; midpoint ED remains ideal",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", type=Path, required=True)
    ap.add_argument("--witness", type=Path, nargs="+", required=True)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--failure-tol", type=float, default=1e-10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "s8d3_results")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    b = s8d._bridge(a.hirano_src.resolve(), float(a.signature_p))
    results = []
    for i, p in enumerate(a.witness, 1):
        r = _minimize_one(p.resolve(), b, float(a.failure_tol))
        results.append(r)
        (a.out_dir / f"s8d3_minimized_{i:03d}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(json.dumps({
            "witness": str(p),
            "minimum_malignant_subset_order": r["minimum_malignant_subset_order"],
            "minimum_malignant_subset_count": r["minimum_malignant_subset_count"],
            "no_malignant_subset_at_order_le2": r["no_malignant_subset_at_order_le2"],
        }, indent=2), flush=True)

    summary = {
        "stage": "S8-D3",
        "witness_count": len(results),
        "minimum_orders": [r["minimum_malignant_subset_order"] for r in results],
        "global_minimum_observed_malignant_subset_order": min(
            r["minimum_malignant_subset_order"] for r in results if r["minimum_malignant_subset_order"] is not None
        ) if any(r["minimum_malignant_subset_order"] is not None for r in results) else None,
        "all_parent_witness_subset_searches_have_no_order_le2_malignant_subset": all(r["no_malignant_subset_at_order_le2"] for r in results),
        "interpretation": "If the global minimum is 3, these deterministic reduced witnesses establish d_f <= 3 for the preserved integrated model. A separate exhaustive <=2 audit is still required for d_f >= 3.",
    }
    (a.out_dir / "s8d3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["global_minimum_observed_malignant_subset_order"] == 3:
        print("S8-D3 UPPER-BOUND PASS: exact accepted malignant triple obtained by deterministic subset replay")
    if summary["all_parent_witness_subset_searches_have_no_order_le2_malignant_subset"]:
        print("S8-D3 LOCAL-LOWER-CHECK PASS: no <=2 malignant subset exists inside the logged witness fault sets")


if __name__ == "__main__":
    main()
