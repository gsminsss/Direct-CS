#!/usr/bin/env python3
"""S17-C/D2: logical-coset-constrained higher-order decoder.

Why D2 exists
=============
S17-C/D1 recovered a large amount of higher-order acceptance, but its confidence
metric was not reliable: one BP+OSD correction was produced per latent Steane
sector, and if the top-K sectors happened to produce only one logical Pauli,
the record was assigned an infinite confidence gap. Empirically, some of these
"infinite consensus" decisions were wrong.

D2 removes this failure mode by explicitly solving all four logical Pauli
cosets. Let H be the 470 x 6917 detector/effect matrix and let Lx,Lz be two
rows giving the logical Pauli label carried by each effect column. For each
latent Steane sector s and each desired FINAL Pauli q in {I,X,Z,Y}, D2 solves

    [ H  ] x = [ observed XOR sector_record[s] ]
    [ Lx ]     [ downstream_X(q XOR sector_pauli[s]) ]
    [ Lz ]     [ downstream_Z(q XOR sector_pauli[s]) ]

using BP+OSD on a 472 x 6917 augmented matrix. Returned effect sets are then
post-validated against the real 2407-location categorical fault model by
requiring a distinct physical-location assignment, exactly as in D1.

For each final Pauli q, candidates from the searched sectors are combined by a
log-sum-exp score. A record can enter the higher-order accepted tier only if all
four logical Pauli cosets were explicitly covered by at least one physically
valid candidate. Confidence is the finite score gap between the best and second
best Pauli cosets. Therefore D2 never interprets "no alternate Pauli found" as
infinite confidence.

The structural certificate is unchanged:
  Tier 0: frozen 14,197 certified ambiguous records -> reject.
  Tier 1: exact total-budget<=2 unique catalogue -> exact frame.
  Tier 2: only out-of-certificate records use this empirical higher-order D2.

This remains an empirical finite-p decoder. The exact d_f>=3 low-order result is
still the v13d/S17-C/B certificate, not this BP+OSD model.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_DETECTORS = 470
EXPECTED_LOCATIONS = 2407
EXPECTED_EFFECTS = 6918
PAULI_NAMES = {0: "I", 1: "X", 2: "Z", 3: "Y"}


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find(root: Path, name: str, prefer: tuple[str, ...] = ()) -> Path:
    xs = list(root.rglob(name))
    if not xs:
        raise FileNotFoundError(f"Could not find {name!r} under {root.resolve()}")
    xs.sort(key=lambda p: (
        tuple(0 if token in str(p) else 1 for token in prefer),
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)), str(p),
    ))
    return xs[0]


def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Can't import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _logsumexp(xs: list[float]) -> float:
    if not xs:
        return float("-inf")
    m = max(xs)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(x-m) for x in xs))


def _gf2_row_rank_sparse(h) -> int:
    """GF(2) row rank using Python-int bitsets; cheap for <=472 rows."""
    basis: dict[int, int] = {}
    h = h.tocsr()
    for i in range(h.shape[0]):
        x = 0
        for j in h.indices[h.indptr[i]:h.indptr[i+1]]:
            x ^= 1 << int(j)
        while x:
            p = x.bit_length() - 1
            if p in basis:
                x ^= basis[p]
            else:
                basis[p] = x
                break
    return len(basis)


@dataclass
class CosetCandidate:
    sector: int
    final_pauli: int
    downstream_pauli: int
    score: float
    correction_effect_count: int
    assigned_locations: tuple[int, ...]
    selected_columns: tuple[int, ...]
    prefix_distance: int
    source: str


@dataclass
class CosetDecodeResult:
    success: bool
    pauli: int | None
    gap: float
    coverage_paulis: tuple[int, ...]
    candidate_count: int
    invalid_candidate_count: int
    best: CosetCandidate | None
    best_different: CosetCandidate | None
    pauli_logsumexp: dict[int, float]
    pauli_best_score: dict[int, float]
    pauli_candidate_counts: dict[int, int]
    sectors_tested: tuple[int, ...]


def _build_augmented_matrix(h, cols):
    from scipy.sparse import csr_matrix, vstack
    x = np.asarray([[int(c.pauli) & 1 for c in cols]], dtype=np.uint8)
    z = np.asarray([[(int(c.pauli) >> 1) & 1 for c in cols]], dtype=np.uint8)
    h_aug = vstack([h, csr_matrix(x), csr_matrix(z)], format="csr")
    return h_aug


def _syndrome_and_pauli_from_selected(d1, cols, sel: np.ndarray) -> tuple[int, int]:
    s = d1._syndrome_from_selected(cols, sel)
    p = 0
    for j in np.flatnonzero(sel):
        p ^= int(cols[int(j)].pauli)
    return int(s), int(p)


def _collect_decoder_vectors(decoder, target: np.ndarray, ncols: int) -> list[tuple[str, np.ndarray]]:
    """Collect all decoder-exposed vectors that exactly satisfy the target.

    BpOsdDecoder exposes several useful post-decode vectors (osd0, osdw, BP).
    They are deduplicated later. This gives a small candidate list without
    pretending decoder multiplicity is probability mass.
    """
    out: list[tuple[str, np.ndarray]] = []
    raw = decoder.decode(np.asarray(target, dtype=np.uint8))
    attrs = [
        ("return", raw),
        ("osdw", getattr(decoder, "osdw_decoding", None)),
        ("osd0", getattr(decoder, "osd0_decoding", None)),
        ("decoding", getattr(decoder, "decoding", None)),
        ("bp", getattr(decoder, "bp_decoding", None)),
    ]
    seen: set[bytes] = set()
    for tag, v in attrs:
        if v is None:
            continue
        a = np.asarray(v, dtype=np.uint8).reshape(-1)
        if a.size != ncols:
            continue
        a = a & 1
        key = np.packbits(a, bitorder="little").tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((tag, a.copy()))
    return out


def _rank_sector_candidates(record: int, sectors: list[dict[str, Any]], prior: np.ndarray, k: int):
    # Reuse D1 ranking semantics: detector-driven prefix distance, prior tie-break.
    mask = (1 << 171) - 1
    scored = []
    for r in sectors:
        s = int(r["steane_syndrome"])
        rs = int(r["detector_470_hex"], 16)
        d = ((int(record) ^ rs) & mask).bit_count()
        scored.append((int(d), -math.log(max(float(prior[s]), 1e-300)), s))
    scored.sort()
    out = [s for _,_,s in scored[:max(1,int(k))]]
    if 0 not in out:
        out[-1] = 0
    return list(dict.fromkeys(out))


def _candidate_score(d1, sector_prior: float, p: float, cols, selected: tuple[int, ...]):
    assn = d1._assign_distinct_locations(cols, selected)
    if assn is None:
        return None
    locs, log_cond_weight = assn
    k = len(locs)
    score = math.log(max(float(sector_prior), 1e-300))
    if k:
        score += k * math.log(max(float(p),1e-300) / max(1.0-float(p),1e-300))
        score += float(log_cond_weight)
    return tuple(locs), float(score)


def _decode_unknown_coset(d1, record: int, sectors: list[dict[str, Any]], prior: np.ndarray, p: float,
                          decoder, h_aug, cols, sector_candidates: int,
                          require_all_paulis: bool = True) -> CosetDecodeResult:
    sector_list = _rank_sector_candidates(record, sectors, prior, sector_candidates)
    by_pauli: dict[int, list[CosetCandidate]] = {0:[],1:[],2:[],3:[]}
    invalid = 0
    ncols = len(cols)

    for s in sector_list:
        r = sectors[int(s)]
        rs = int(r["detector_470_hex"], 16)
        residual = int(record) ^ rs
        prefix_distance = ((int(record) ^ rs) & ((1<<171)-1)).bit_count()
        sector_pauli = int(r["raw_terminal_pauli_int"])
        det_bits = np.zeros(EXPECTED_DETECTORS, dtype=np.uint8)
        y = int(residual)
        while y:
            b = y & -y
            det_bits[b.bit_length()-1] = 1
            y ^= b

        for final_pauli in range(4):
            downstream_pauli = int(final_pauli) ^ int(sector_pauli)
            target = np.concatenate([
                det_bits,
                np.asarray([downstream_pauli & 1, (downstream_pauli >> 1) & 1], dtype=np.uint8)
            ])
            try:
                vectors = _collect_decoder_vectors(decoder, target, ncols)
            except Exception:
                vectors = []
            if not vectors:
                invalid += 1
                continue

            # Deduplicate physical effect sets for this (sector, final Pauli).
            seen_sel: set[tuple[int,...]] = set()
            found_valid = False
            for tag, vec in vectors:
                # Exact augmented-check validation. scipy result converted mod 2.
                chk = np.asarray(h_aug @ vec, dtype=np.uint8).reshape(-1) & 1
                if chk.size != EXPECTED_DETECTORS+2 or not np.array_equal(chk, target):
                    continue
                ds, dp = _syndrome_and_pauli_from_selected(d1, cols, vec)
                if ds != residual or dp != downstream_pauli:
                    continue
                selected = tuple(int(j) for j in np.flatnonzero(vec))
                if selected in seen_sel:
                    continue
                seen_sel.add(selected)
                scored = _candidate_score(d1, float(prior[int(s)]), p, cols, selected)
                if scored is None:
                    continue
                locs, score = scored
                found_valid = True
                by_pauli[int(final_pauli)].append(CosetCandidate(
                    sector=int(s), final_pauli=int(final_pauli), downstream_pauli=int(downstream_pauli),
                    score=float(score), correction_effect_count=len(locs), assigned_locations=tuple(locs),
                    selected_columns=selected, prefix_distance=int(prefix_distance), source=str(tag),
                ))
            if not found_valid:
                invalid += 1

    # Global dedup by actual sector+effect-set, not by decoder attribute.
    for q in range(4):
        uniq: dict[tuple[int,tuple[int,...]], CosetCandidate] = {}
        for c in by_pauli[q]:
            key = (int(c.sector), tuple(c.selected_columns))
            if key not in uniq or c.score > uniq[key].score:
                uniq[key] = c
        by_pauli[q] = list(uniq.values())

    coverage = tuple(q for q in range(4) if by_pauli[q])
    if require_all_paulis and len(coverage) < 4:
        return CosetDecodeResult(False, None, 0.0, coverage,
                                 sum(len(v) for v in by_pauli.values()), invalid,
                                 None, None, {}, {}, {q:len(by_pauli[q]) for q in range(4)},
                                 tuple(sector_list))
    if not coverage:
        return CosetDecodeResult(False, None, 0.0, coverage, 0, invalid,
                                 None, None, {}, {}, {q:0 for q in range(4)}, tuple(sector_list))

    lse = {q:_logsumexp([c.score for c in by_pauli[q]]) for q in coverage}
    bests = {q:max(by_pauli[q], key=lambda c:c.score) for q in coverage}
    ordered = sorted(coverage, key=lambda q:lse[q], reverse=True)
    q0 = int(ordered[0])
    q1 = int(ordered[1]) if len(ordered) > 1 else None
    gap = float(lse[q0] - lse[q1]) if q1 is not None else 0.0
    return CosetDecodeResult(
        True, q0, gap, coverage, sum(len(v) for v in by_pauli.values()), invalid,
        bests[q0], None if q1 is None else bests[q1],
        {int(q):float(lse[q]) for q in coverage},
        {int(q):float(bests[q].score) for q in coverage},
        {q:len(by_pauli[q]) for q in range(4)}, tuple(sector_list),
    )


def _to_d1_result(d1, d: CosetDecodeResult):
    """Adapter so D1's audited threshold/fidelity code can be reused."""
    def cv(c: CosetCandidate | None):
        if c is None:
            return None
        return d1.Candidate(
            sector=int(c.sector), pauli=int(c.final_pauli), score=float(c.score),
            correction_effect_count=int(c.correction_effect_count),
            assigned_locations=tuple(c.assigned_locations), selected_columns=tuple(c.selected_columns),
            prefix_distance=int(c.prefix_distance),
        )
    return d1.DecodeResult(
        bool(d.success), None if d.pauli is None else int(d.pauli), float(d.gap),
        int(d.candidate_count), cv(d.best), cv(d.best_different), int(d.invalid_candidate_count), False
    )


def _serialize(d: CosetDecodeResult):
    def cobj(c):
        if c is None:
            return None
        return {
            "sector": int(c.sector), "final_pauli": PAULI_NAMES[int(c.final_pauli)],
            "downstream_pauli": PAULI_NAMES[int(c.downstream_pauli)], "score": float(c.score),
            "correction_effect_count": int(c.correction_effect_count),
            "assigned_locations": list(map(int,c.assigned_locations)),
            "selected_columns": list(map(int,c.selected_columns)),
            "prefix_distance": int(c.prefix_distance), "source": c.source,
        }
    return {
        "success": bool(d.success),
        "pauli": None if d.pauli is None else PAULI_NAMES[int(d.pauli)],
        "finite_coset_gap": float(d.gap),
        "coverage_paulis": [PAULI_NAMES[int(q)] for q in d.coverage_paulis],
        "candidate_count": int(d.candidate_count),
        "invalid_candidate_count": int(d.invalid_candidate_count),
        "pauli_logsumexp": {PAULI_NAMES[int(q)]:float(v) for q,v in d.pauli_logsumexp.items()},
        "pauli_best_score": {PAULI_NAMES[int(q)]:float(v) for q,v in d.pauli_best_score.items()},
        "pauli_candidate_counts": {PAULI_NAMES[int(q)]:int(v) for q,v in d.pauli_candidate_counts.items()},
        "sectors_tested": list(map(int,d.sectors_tested)),
        "best": cobj(d.best), "best_different": cobj(d.best_different),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--d1-script", type=Path, default=None)
    ap.add_argument("--s17c-c-script", type=Path, default=None)
    ap.add_argument("--s10-script", type=Path, default=None)
    ap.add_argument("--v13d-script", type=Path, default=None)
    ap.add_argument("--sector-map", type=Path, default=None)
    ap.add_argument("--sector-aware-reject-policy", type=Path, default=None)
    ap.add_argument("--x-circuit", type=Path, default=None)
    ap.add_argument("--z-circuit", type=Path, default=None)
    ap.add_argument("--p-values", type=float, nargs="+", default=[5e-4,1e-3])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=None)
    ap.add_argument("--prior-calibration-shots", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd_cs")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--gap-thresholds", type=float, nargs="+", default=[2,4,6,8,10,12])
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--allow-incomplete-coset-coverage", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.search_root.resolve()

    d1_path = (args.d1_script or _find(root,"direct_cs_s17c_d_hybrid_bposd_abstention.py",("stim_pipeline/interface",))).resolve()
    c_path = (args.s17c_c_script or _find(root,"direct_cs_s17c_c_finite_p_e2e_certified_lookup.py",("stim_pipeline/interface",))).resolve()
    s10_path = (args.s10_script or _find(root,"direct_cs_s10_complete_physical_factory_mc.py",("stim_pipeline/interface",))).resolve()
    v13d_path = (args.v13d_script or _find(root,"direct_cs_s17b_v13d_joint_postselection_certificate.py",("stim_pipeline/interface",))).resolve()
    sector_path = (args.sector_map or _find(root,"s17c_a_sector_map.json",("s17c_a_results",))).resolve()
    reject_path = (args.sector_aware_reject_policy or _find(root,"s17c_b_sector_aware_reject_policy.json",("s17c_b_results",))).resolve()
    x_path = (args.x_circuit or _find(root,"s17b_v11_X_transition_circuit.stim",("s17b_v11_order2_results",))).resolve()
    z_path = (args.z_circuit or _find(root,"s17b_v11_Z_transition_circuit.stim",("s17b_v11_order2_results",))).resolve()

    d1 = _import_module("s17cd2_d1", d1_path)
    s17cc = _import_module("s17cd2_s17cc", c_path)
    s10mc = _import_module("s17cd2_s10mc", s10_path)
    v13d = _import_module("s17cd2_v13d", v13d_path)
    BpOsdDecoder = d1._require_ldpc()

    sectors = d1._sector_rows(sector_path)
    reject_obj = _load_json(reject_path)
    reject_set = {int(x,16) for x in reject_obj["total_sector_aware_rejected_syndromes_hex"]}
    if len(reject_set)!=14197 or 0 in reject_set:
        raise RuntimeError(f"Expected frozen 14,197 nonzero reject records, got {len(reject_set)}")

    downstream = s17cc._build_downstream_model(v13d, x_path, z_path)
    if len(downstream["effects"]) != EXPECTED_EFFECTS:
        raise RuntimeError((len(downstream["effects"]), EXPECTED_EFFECTS))
    cols = d1._effect_columns(downstream["effects"])
    h = d1._build_sparse_check_matrix(cols)
    h_aug = _build_augmented_matrix(h, cols)
    rank_h = _gf2_row_rank_sparse(h)
    rank_aug = _gf2_row_rank_sparse(h_aug)

    matrix_meta = {
        "detector_matrix_shape": list(map(int,h.shape)),
        "augmented_matrix_shape": list(map(int,h_aug.shape)),
        "detector_matrix_gf2_row_rank": int(rank_h),
        "augmented_matrix_gf2_row_rank": int(rank_aug),
        "logical_rows_independent_of_detector_rows": bool(rank_aug >= rank_h+2),
        "raw_joint_unique_effects": len(downstream["effects"]),
        "nonzero_syndrome_effect_columns": len(cols),
        "frozen_reject_records": len(reject_set),
        "physical_locations_per_patch": downstream["meta"]["physical_locations_per_patch"],
        "sector_count": len(sectors),
    }

    # Smoke-test all four logical cosets at zero detector syndrome. The augmented
    # system must return exact check-satisfying vectors; physical feasibility is
    # tested as well but non-identity zero-syndrome realizations may require
    # multiple effects and are used here only as solver tests, not as accepted
    # no-event faults.
    p_smoke = max(1e-6,min(args.p_values))
    pri_smoke = d1._effect_parity_priors(cols,p_smoke)
    dec_smoke = d1._make_bposd(BpOsdDecoder,h_aug,pri_smoke,args.bp_max_iter,args.bp_method,
                               args.ms_scaling_factor,args.osd_method,args.osd_order)
    smoke = {}
    for q in range(4):
        target = np.zeros(EXPECTED_DETECTORS+2,dtype=np.uint8)
        target[-2] = q & 1
        target[-1] = (q>>1)&1
        ok=False
        try:
            vs=_collect_decoder_vectors(dec_smoke,target,len(cols))
            for _,v in vs:
                chk=np.asarray(h_aug@v,dtype=np.uint8).reshape(-1)&1
                if np.array_equal(chk,target):
                    ok=True; break
        except Exception:
            ok=False
        smoke[PAULI_NAMES[q]]=bool(ok)

    st={
        "passed": bool(len(reject_set)==14197 and 0 not in reject_set and rank_aug>=rank_h+2 and smoke.get("I", False)),
        "matrix": matrix_meta,
        "zero_detector_four_logical_coset_solver_smoke_diagnostic": smoke,
        "decoder_class": getattr(BpOsdDecoder,"__name__",str(BpOsdDecoder)),
        "confidence_semantics": "finite competition between explicitly decoded I/X/Z/Y logical cosets; missing alternate Pauli is never treated as infinite confidence",
        "scope_warning": "Empirical higher-order decoder only. Shared-location correlations are not represented inside binary BP+OSD; returned effect sets are post-validated by distinct-location assignment."
    }
    _dump(args.out_dir/"s17c_d2_self_test.json",st)
    if not st["passed"]:
        raise RuntimeError(st)
    if args.self_test_only:
        print(json.dumps(st,indent=2),flush=True)
        print("S17-C/D2 SELF-TEST PASS",flush=True)
        return

    if args.shots_per_p is not None:
        if len(args.shots_per_p)!=len(args.p_values):
            raise SystemExit("--shots-per-p length must match --p-values")
        shot_list=[int(x) for x in args.shots_per_p]
    else:
        shot_list=[int(args.shots)]*len(args.p_values)

    all_rows=[]; prior_rows=[]; audits=[]
    t_all=time.time()
    sector_dict={int(r["steane_syndrome"]):r for r in sectors}

    for ip,(p,shots) in enumerate(zip(args.p_values,shot_list)):
        p=float(p)
        print(f"S17-C/D2: calibrating sector prior p={p:g} ...",flush=True)
        prior,prior_meta=d1._calibrate_sector_prior(p,args.prior_calibration_shots,args.seed+10000+ip,s10mc)
        prior_rows.append(prior_meta)

        print(f"S17-C/D2: sampling p={p:g}, shots={shots} ...",flush=True)
        sampled=s17cc._sample_one_p(p,int(shots),int(args.seed+ip),s10mc,sector_dict,downstream,args.max_examples)
        queries=set(sampled["queries"])
        print(f"S17-C/D2: exact <=2 resolving {len(queries)} observed patch records ...",flush=True)
        exact_decoder,exact_meta=s17cc._resolve_observed_records(queries,sectors,downstream["effects"],v13d,reject_set)
        unknown=sorted(q for q in queries if q not in reject_set and q not in exact_decoder)

        pri=d1._effect_parity_priors(cols,p)
        decoder=d1._make_bposd(BpOsdDecoder,h_aug,pri,args.bp_max_iter,args.bp_method,
                               args.ms_scaling_factor,args.osd_method,args.osd_order)
        high_raw: dict[int,CosetDecodeResult]={}
        high_adapter={}
        tdec=time.time()
        print(f"S17-C/D2: decoding {len(unknown)} unknown records; top-{args.sector_candidates} sectors x 4 logical cosets ...",flush=True)
        for qi,q in enumerate(unknown):
            d=_decode_unknown_coset(d1,q,sectors,prior,p,decoder,h_aug,cols,args.sector_candidates,
                                    require_all_paulis=not args.allow_incomplete_coset_coverage)
            high_raw[q]=d
            high_adapter[q]=_to_d1_result(d1,d)
            if (qi+1)%100==0:
                print(f"  decoded {qi+1}/{len(unknown)}",flush=True)
        dec_s=time.time()-tdec

        cover_hist=collections.Counter(len(d.coverage_paulis) for d in high_raw.values())
        success=sum(d.success for d in high_raw.values())
        gaps=[d.gap for d in high_raw.values() if d.success]
        cand_counts=[d.candidate_count for d in high_raw.values() if d.success]
        audits.append({
            "p":p,"observed_patch_records":len(queries),"exact_unique_records":len(exact_decoder),
            "frozen_reject_records_observed":sum(q in reject_set for q in queries),
            "out_of_certificate_unique_records":len(unknown),
            "coset_decode_success_records":int(success),"coset_decode_failure_records":int(len(unknown)-success),
            "logical_coset_coverage_histogram":{str(k):int(v) for k,v in sorted(cover_hist.items())},
            "finite_gap_summary": None if not gaps else {
                "count":len(gaps),"median":float(np.median(gaps)),
                "p10":float(np.quantile(gaps,.1)),"p90":float(np.quantile(gaps,.9)),
            },
            "candidate_count_summary": None if not cand_counts else {
                "median":float(np.median(cand_counts)),"p10":float(np.quantile(cand_counts,.1)),
                "p90":float(np.quantile(cand_counts,.9)),"max":int(max(cand_counts)),
            },
            "decode_seconds":dec_s,"exact_catalogue_meta":exact_meta,
            "effect_prior_min":float(np.min(pri)),"effect_prior_max":float(np.max(pri)),
        })

        sweeps=[]
        for gap in args.gap_thresholds:
            row=d1._finish_threshold(sampled,exact_decoder,reject_set,high_adapter,s10mc,float(gap),
                                     args.max_correction_effects,args.max_examples)
            sweeps.append(row)
            print(json.dumps({
                "p":p,"coset_gap":gap,"acceptance":row["end_to_end_acceptance"],
                "cond_infidelity":row["conditional_logical_infidelity"],
                "decoder_excess":row["decoder_excess_infidelity_conditional"],
                "harmful":row["harmful_accepted_samples"],
                "le2_gate":row["global_le2_regression_gate_pass"],
                "higher_order_accept":row["accepted_with_any_higher_order_decode"],
                "higher_order_abstain":row["higher_order_abstain"],
            },indent=2),flush=True)

        front={k:v for k,v in sampled["predecode"].items() if k!="examples"}
        all_rows.append({
            "p":p,"shots":int(shots),"seed":int(args.seed+ip),"frontend":front,
            "conservative_reference":{"core_accepted":int(front["core_accepted"]),
                                      "exact_unique_query_records":len(exact_decoder),
                                      "out_of_certificate_unique_query_records":len(unknown)},
            "threshold_sweep":sweeps,
        })
        sample_keys=unknown[:min(200,len(unknown))]
        _dump(args.out_dir/f"s17c_d2_coset_examples_p{p:g}.json",
              [{"record":hex(q),**_serialize(high_raw[q])} for q in sample_keys])

    global_le2=all(x["global_le2_regression_gate_pass"] for r in all_rows for x in r["threshold_sweep"])
    summary={
        "stage":"S17-C/D2 logical-coset-constrained BP+OSD list pilot",
        "decoder":{
            "tier0":"frozen 14,197 certified ambiguous records -> reject",
            "tier1":"exact total-budget<=2 unique catalogue -> exact frame",
            "tier2":"unknown records only: augmented [470 detector + 2 logical-label] x 6917 matrix; explicitly decode I/X/Z/Y cosets for each latent sector; require physical distinct-location realization; finite coset score-gap abstention",
            "sector_candidates":int(args.sector_candidates),"bp_max_iter":int(args.bp_max_iter),
            "bp_method":args.bp_method,"ms_scaling_factor":float(args.ms_scaling_factor),
            "osd_method":args.osd_method,"osd_order":int(args.osd_order),
            "gap_thresholds":list(map(float,args.gap_thresholds)),
            "require_all_four_logical_cosets":bool(not args.allow_incomplete_coset_coverage),
            "max_correction_effects":int(args.max_correction_effects),
            "sector_prior":f"independent core-only MC calibration, {args.prior_calibration_shots} attempts/p with Laplace smoothing",
        },
        "matrix":matrix_meta,"self_test":st,"sector_prior_calibration":prior_rows,
        "decoder_audit":audits,"finite_p_rows":all_rows,
        "main_gate":{
            "frozen_low_order_reject_policy_unchanged":True,
            "global_no_sampled_accepted_harmful_event_with_total_physical_fault_locations_le2":bool(global_le2),
            "no_infinite_gap_from_missing_logical_competitor_by_construction":True,
            "higher_order_decoder_is_empirical_not_structural":True,
        },
        "decision_rule":"Compare D2 decoder-excess infidelity and acceptance against D1 and conservative S17-C/C. If D2 sharply reduces false-confidence errors, validate on independent larger samples. If not, binary independent-effect marginals are the dominant limitation and the next decoder should model categorical physical locations directly.",
        "ideal_d13_close_warning":"470-bit boundary still uses an ideal downstream d13 close structurally; standalone noisy delivered-d13 performance remains unresolved.",
        "timing_seconds":time.time()-t_all,
    }
    _dump(args.out_dir/"s17c_d2_summary.json",summary)
    _dump(args.out_dir/"s17c_d2_finite_p_rows.json",all_rows)
    _dump(args.out_dir/"s17c_d2_decoder_audit.json",audits)
    _dump(args.out_dir/"s17c_d2_sector_priors.json",prior_rows)
    print(json.dumps({"global_le2_regression_gate_pass":bool(global_le2),"out_dir":str(args.out_dir)},indent=2),flush=True)


if __name__=="__main__":
    main()
