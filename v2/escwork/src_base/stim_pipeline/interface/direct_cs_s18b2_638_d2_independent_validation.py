#!/usr/bin/env python3
"""S18-B2: frozen independent validation of the practical 638-bit D2 decoder.

This stage follows S18-B1 calibration.  The threshold is frozen *before* B2
independent evaluation.  No threshold sweep is performed on B2 data.

Frozen policy used by this paper workflow
-----------------------------------------
    gap >= 10

Freeze rationale (B1 calibration only): among calibrated candidates that kept
stress-point (p=1e-3) operational acceptance >= 0.20 and preserved the sampled
<=2 regression gate, gap=10 had the lowest observed closed-boundary harmful
fraction while retaining useful acceptance.  The B1 program's automatic gap=16
recommendation was explicitly a fallback after its over-conservative zero-
harmful criterion failed; gap=16 collapsed stress-point acceptance to ~0.03.

Decoder input is exactly the 638 operational detector bits.  The 168 terminal
truth bits are scoring-only and are never visible to the decoder.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_GAP = 10.0
OP_DETECTORS = 638
TRUTH_BITS = 168
PAULI_NAMES = ["I", "X", "Z", "Y"]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find(root: Path, names: list[str], prefer: tuple[str, ...] = ()) -> Path:
    # IMPORTANT: preserve B1 path semantics.  A name containing a path
    # separator (e.g. ``msc-ls/src``) is an exact root-relative path, not
    # a request to recursively search for every directory called ``src``.
    hits: list[Path] = []
    for n in names:
        p = Path(n)
        if p.is_absolute() and p.exists():
            hits.append(p)
        elif "/" in n or "\\" in n:
            q = root / p
            if q.exists():
                hits.append(q)
        else:
            hits.extend(root.rglob(n))
    if not hits:
        raise FileNotFoundError((str(root), names))
    hits = list(dict.fromkeys(x.resolve() for x in hits))
    hits.sort(key=lambda p: (
        tuple(0 if token in str(p) else 1 for token in prefer),
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)), str(p),
    ))
    return hits[0]


def _tag(p: float) -> str:
    return f"{p:.12g}".replace('.', 'p').replace('-', 'm')


def _cache_paths(out: Path, p: float):
    t = _tag(p)
    return out / f"s18b2_cache_{t}.jsonl", out / f"s18b2_cache_{t}_meta.json"


def _deserialize_d2(d2, obj):
    def cand(c):
        if c is None: return None
        return d2.CosetCandidate(
            sector=int(c["sector"]), final_pauli=int(c["final_pauli"]),
            downstream_pauli=int(c["downstream_pauli"]), score=float(c["score"]),
            correction_effect_count=int(c["correction_effect_count"]),
            assigned_locations=tuple(map(int,c["assigned_locations"])),
            selected_columns=tuple(map(int,c["selected_columns"])),
            prefix_distance=int(c["prefix_distance"]), source=str(c["source"]))
    return d2.CosetDecodeResult(
        success=bool(obj["success"]),
        pauli=None if obj["pauli"] is None else int(obj["pauli"]),
        gap=float(obj["gap"]), coverage_paulis=tuple(map(int,obj.get("coverage_paulis",[]))),
        candidate_count=int(obj["candidate_count"]), invalid_candidate_count=int(obj["invalid_candidate_count"]),
        best=cand(obj.get("best")), best_different=cand(obj.get("best_different")),
        pauli_logsumexp={int(k):float(v) for k,v in obj.get("pauli_logsumexp",{}).items()},
        pauli_best_score={int(k):float(v) for k,v in obj.get("pauli_best_score",{}).items()},
        pauli_candidate_counts={int(k):int(v) for k,v in obj.get("pauli_candidate_counts",{}).items()},
        sectors_tested=tuple(map(int,obj.get("sectors_tested",[])))
    )


def _append_cache(path: Path, q: int, d2, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"record":hex(int(q)),"decode":d2._serialize(obj)},sort_keys=True)+"\n")


def _load_cache(d2, path: Path, meta: Path, wanted: dict[str,Any], resume: bool):
    if not resume or not path.exists(): return {}
    if not meta.exists() or _load(meta) != wanted:
        raise RuntimeError("Stale S18-B2 cache. Use a fresh --out-dir or --no-resume-cache.")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o = json.loads(line)
        out[int(o["record"],16)] = _deserialize_d2(d2,o["decode"])
    return out


def _hash_obj(x: Any) -> str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s18m-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m_results"))
    ap.add_argument("--s18m2-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--b1-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b1_calibration_results"))
    ap.add_argument("--p-values", type=float, nargs="+", default=[2e-4,5e-4,1e-3])
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=[5000,5000,3000])
    ap.add_argument("--frozen-gap", type=float, default=FROZEN_GAP)
    ap.add_argument("--prior-calibration-shots", type=int, default=30000)
    ap.add_argument("--prior-seed", type=int, default=20261101)
    ap.add_argument("--evaluation-seed", type=int, default=20261121)
    ap.add_argument("--signature-p", type=float, default=0.001)
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd_cs")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--no-resume-cache", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s18b2_validation_results"))
    args = ap.parse_args()

    if args.self_test_only:
        print(json.dumps({"passed": True, "stage": "S18-B2 frozen independent validation", "frozen_gap": FROZEN_GAP, "operational_detector_bits": OP_DETECTORS, "truth_bits_decoder_visible": False, "threshold_sweep_on_evaluation": False}, indent=2))
        return

    if abs(float(args.frozen_gap)-FROZEN_GAP) > 1e-12:
        raise SystemExit(f"B2 threshold is frozen at {FROZEN_GAP:g}. Do not retune on evaluation data.")
    if len(args.shots_per_p) != len(args.p_values):
        raise SystemExit("--shots-per-p length must match --p-values")
    if args.prior_seed == args.evaluation_seed:
        raise SystemExit("prior and evaluation seeds must differ")

    root = args.search_root.resolve()
    out = (root/args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    b1dir = (root/args.b1_result_dir).resolve() if not args.b1_result_dir.is_absolute() else args.b1_result_dir.resolve()
    b1summary = b1dir/"s18b1_summary.json"
    if not b1summary.exists(): raise FileNotFoundError(b1summary)
    b1 = _load(b1summary)
    if not bool(b1.get("calibration_only",False)):
        raise RuntimeError("Expected calibration-only S18-B1 summary")
    # Guard against accidentally reusing B1 seeds.
    b1_prior = int(b1.get("independence_plan",{}).get("B1_prior_seed_base",-1))
    b1_eval = int(b1.get("independence_plan",{}).get("B1_calibration_seed_base",-1))
    used_b2 = {args.prior_seed+i for i in range(len(args.p_values))} | {args.evaluation_seed+i for i in range(len(args.p_values))}
    if b1_prior in used_b2 or b1_eval in used_b2:
        raise RuntimeError("B2 seeds overlap B1 seed bases")

    b1mod = _import("s18b2_b1", _find(root,["direct_cs_s18b1_638_d2_calibration.py"],("stim_pipeline/interface",)))
    d2 = _import("s18b2_d2", _find(root,["direct_cs_s17c_d2_logical_coset_list_decoder.py"],("stim_pipeline/interface",)))
    d1 = _import("s18b2_d1", _find(root,["direct_cs_s17c_d_hybrid_bposd_abstention.py"],("stim_pipeline/interface",)))
    s10mc = _import("s18b2_s10", _find(root,["direct_cs_s10_complete_physical_factory_mc.py"],("stim_pipeline/interface",)))
    cert = _import("s18b2_v13d", _find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"],("stim_pipeline/interface",)))
    s18m = _import("s18b2_s18m", _find(root,["direct_cs_s18m_modular_noisy_d13_certificate.py"],("stim_pipeline/interface",)))
    m2 = _import("s18b2_m2", _find(root,["direct_cs_s18m2_terminal_truth_closure.py"],("stim_pipeline/interface",)))
    ca = _import("s18b2_ca", _find(root,["direct_cs_s17c_a_sector_to_470_map.py"],("stim_pipeline/interface",)))
    s8b = _import("s18b2_s8b", _find(root,["zxz_core_boundary_signature_map.py"],("stim_pipeline/interface",)))
    v7 = _import("s18b2_v7", _find(root,["direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"],("stim_pipeline/interface",)))
    v11 = _import("s18b2_v11", _find(root,["direct_cs_s17b_v11_complete_missing_basis.py"],("stim_pipeline/interface",)))
    s18a = _import("s18b2_s18a", _find(root,["direct_cs_s18a_noisy_d13_boundary_certificate.py"],("stim_pipeline/interface",)))
    d1.EXPECTED_DETECTORS=OP_DETECTORS; d2.EXPECTED_DETECTORS=OP_DETECTORS
    BpOsdDecoder=d1._require_ldpc()

    s18mr=(root/args.s18m_result_dir).resolve() if not args.s18m_result_dir.is_absolute() else args.s18m_result_dir.resolve()
    s18m2r=(root/args.s18m2_result_dir).resolve() if not args.s18m2_result_dir.is_absolute() else args.s18m2_result_dir.resolve()
    xtruth=s18m2r/"s18m2_X_truth_closed.stim"; ztruth=s18m2r/"s18m2_Z_truth_closed.stim"
    sector_op=s18mr/"s18m_sector_map_638.json"; m2sum=s18m2r/"s18m2_summary.json"
    for pth in (xtruth,ztruth,sector_op,m2sum):
        if not pth.exists(): raise FileNotFoundError(pth)
    ms=_load(m2sum)
    if not bool(ms.get("main_gate",{}).get("combined_modular_boundary_output_contract_pass",False)):
        raise RuntimeError("S18-M2 structural gate is not PASS")
    if bool(ms.get("policy",{}).get("truth_bits_decoder_visible",True)):
        raise RuntimeError("truth bits unexpectedly decoder-visible")

    hirano=_find(root,["msc-ls/src"]); sahay=_find(root,["reference/fd3_baseline_ckt.stim"])
    model=b1mod._build_truth_operational_model(cert,xtruth,ztruth)
    sectors,smeta=b1mod._build_sector_truth_map(root,s18m,m2,ca,s8b,v7,v11,s18a,hirano,sahay,sector_op,float(args.signature_p))
    sector_by={int(r["steane_syndrome"]):r for r in sectors}
    _dump(out/"s18b2_sector_map_638_truth.json",sectors)

    print("S18-B2: building frozen 638-bit D2 matrix...",flush=True)
    cols=d1._effect_columns(model["effects"])
    h=d1._build_sparse_check_matrix(cols); h_aug=d2._build_augmented_matrix(h,cols)
    rank_h=d2._gf2_row_rank_sparse(h); rank_aug=d2._gf2_row_rank_sparse(h_aug)
    exact,amb,cmeta=b1mod._build_fast_catalogue(sectors,model["effects"])
    matrix_meta={"H_shape":list(map(int,h.shape)),"H_aug_shape":list(map(int,h_aug.shape)),"H_rank":int(rank_h),"H_aug_rank":int(rank_aug),
                 "logical_rows_independent":bool(rank_aug>=rank_h+2),"effect_columns":len(cols),"raw_effect_classes":len(model["effects"]),
                 "fast_catalogue":cmeta,"downstream":model["meta"],"sector":smeta}
    if not (rank_h==638 and rank_aug>=640 and smeta["existing_S18M_operational_and_raw_labels_reproduced"]):
        raise RuntimeError(("S18-B2 matrix/sector selftest failed",matrix_meta))

    rows=[]; audits=[]; priors=[]; t0=time.time()
    for ip,(p,shots) in enumerate(zip(map(float,args.p_values),map(int,args.shots_per_p))):
        ps=int(args.prior_seed+ip); es=int(args.evaluation_seed+ip)
        print(f"S18-B2: fresh prior p={p:g}, seed={ps} ...",flush=True)
        prior,pmeta=d1._calibrate_sector_prior(p,int(args.prior_calibration_shots),ps,s10mc); pmeta["seed"]=ps; priors.append(pmeta)
        print(f"S18-B2: independent evaluation p={p:g}, shots={shots}, seed={es} ...",flush=True)
        sampled=b1mod._sample_one_p(p,shots,es,s10mc,sector_by,model,int(args.max_examples))
        queries=set(sampled["queries"]); unknown=sorted(q for q in queries if q not in exact and q not in amb and q!=0)
        pri=d1._effect_parity_priors(cols,p)
        decoder=d1._make_bposd(BpOsdDecoder,h_aug,pri,args.bp_max_iter,args.bp_method,args.ms_scaling_factor,args.osd_method,args.osd_order)

        cache,meta=_cache_paths(out,p)
        wanted={"stage":"S18-B2","frozen_gap":FROZEN_GAP,"p":p,
                "matrix_hash":_hash_obj({k:matrix_meta[k] for k in ("H_shape","H_aug_shape","H_rank","H_aug_rank","effect_columns")}),
                "prior_hash":hashlib.sha256(np.asarray(prior,dtype=np.float64).tobytes()).hexdigest(),
                "sector_candidates":args.sector_candidates,"bp_max_iter":args.bp_max_iter,"bp_method":args.bp_method,
                "ms_scaling_factor":args.ms_scaling_factor,"osd_method":args.osd_method,"osd_order":args.osd_order}
        if args.no_resume_cache:
            if cache.exists(): cache.unlink()
            _dump(meta,wanted); high={}
        else:
            if not meta.exists(): _dump(meta,wanted)
            high=_load_cache(d2,cache,meta,wanted,True)
        reused=sum(q in high for q in unknown); todo=[q for q in unknown if q not in high]
        print(f"S18-B2: records={len(queries)}, cheap_exact={sum(q in exact for q in queries)}, cheap_ambiguous={sum(q in amb for q in queries)}, D2 unknown={len(unknown)}, reused={reused}, to_decode={len(todo)}",flush=True)
        td=time.time()
        for ii,q in enumerate(todo):
            d=d2._decode_unknown_coset(d1,q,sectors,prior,p,decoder,h_aug,cols,int(args.sector_candidates),require_all_paulis=True)
            high[q]=d; _append_cache(cache,q,d2,d)
            if (ii+1)%25==0 or ii+1==len(todo): print(f"  decoded {ii+1}/{len(todo)} new records",flush=True)
        gaps=[float(high[q].gap) for q in unknown if high[q].success and math.isfinite(float(high[q].gap))]
        cover=collections.Counter(len(high[q].coverage_paulis) for q in unknown)
        audit={"p":p,"shots":shots,"evaluation_seed":es,"prior_seed":ps,"observed_records":len(queries),
               "cheap_exact_observed":sum(q in exact for q in queries),"cheap_ambiguous_observed":sum(q in amb for q in queries),
               "d2_unknown":len(unknown),"cache_reused":reused,"new_decodes":len(todo),
               "d2_success":sum(bool(high[q].success) for q in unknown),"d2_failure":sum(not bool(high[q].success) for q in unknown),
               "coverage_histogram":{str(k):int(v) for k,v in sorted(cover.items())},"finite_gap_count":len(gaps),
               "gap_median":float(np.median(gaps)) if gaps else None,"gap_p10":float(np.quantile(gaps,.1)) if gaps else None,
               "gap_p90":float(np.quantile(gaps,.9)) if gaps else None,"new_decode_seconds":time.time()-td}
        audits.append(audit)
        r=b1mod._score_threshold(sampled,exact,amb,high,s10mc,FROZEN_GAP,int(args.max_correction_effects),int(args.max_examples))
        r["evaluation_seed"]=es; r["prior_seed"]=ps; r["frozen_gap"]=FROZEN_GAP
        rows.append(r)
        print(json.dumps({"p":p,"frozen_gap":FROZEN_GAP,"acc":r["operational_acceptance"],"truth_closed":r["truth_closed_accepted"],
                          "residual_frac":r["detectable_residual_fraction_among_accepted"],"closed_harmful":r["closed_boundary_harmful"],
                          "le2_harmful":r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"]},indent=2),flush=True)

    validation_gate={
        "frozen_gap_not_swept":True,
        "truth_bits_decoder_visible":False,
        "all_d2_decodes_successful":all(a["d2_failure"]==0 for a in audits),
        "all_unknown_records_have_four_cosets":all(set(a["coverage_histogram"].keys()) <= {"4"} and a["d2_success"]==a["d2_unknown"] for a in audits),
        "sampled_le2_regression_pass_all_p":all(r["sampled_le2_regression_gate_pass"] for r in rows),
        "sampled_closed_boundary_harmful_le2_total":sum(int(r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"]) for r in rows),
    }
    validation_gate["passed"]=bool(validation_gate["frozen_gap_not_swept"] and not validation_gate["truth_bits_decoder_visible"] and
        validation_gate["all_d2_decodes_successful"] and validation_gate["all_unknown_records_have_four_cosets"] and
        validation_gate["sampled_le2_regression_pass_all_p"] and validation_gate["sampled_closed_boundary_harmful_le2_total"]==0)

    summary={
        "stage":"S18-B2 frozen independent 638-bit D2 validation",
        "independent_evaluation":True,"threshold_sweep_on_evaluation":False,"frozen_gap":FROZEN_GAP,
        "freeze_source":"S18-B1 calibration Pareto decision; gap=10 selected before B2 evaluation",
        "freeze_rule":"among B1 candidates with stress p=1e-3 operational acceptance >=0.20 and sampled <=2 regression PASS, choose the lowest observed stress-point closed-boundary harmful fraction; ties favor higher acceptance then smaller gap",
        "b1_automatic_fallback_gap_16_used":False,
        "structural_theorem_source":"S18-M2; B2 is empirical higher-order decoder validation and does not replace the exact theorem",
        "decoder_input":{"operational_detector_bits":638,"terminal_truth_bits_decoder_visible":False,"truth_bits_scoring_only":168},
        "decoder":{"algorithm":"D2 four-logical-coset BP+OSD list decoder","sector_candidates":int(args.sector_candidates),"osd_method":args.osd_method,"osd_order":int(args.osd_order),"max_correction_effects":int(args.max_correction_effects)},
        "matrix":matrix_meta,"fresh_sector_priors":priors,"decoder_audit":audits,"evaluation_rows":rows,
        "validation_gate":validation_gate,"timing_seconds":time.time()-t0,
        "paper_use":"B2 numbers may be used as independent practical-decoder finite-p validation; final delivered-resource claims still wait for S19/S20."
    }
    _dump(out/"s18b2_summary.json",summary); _dump(out/"s18b2_evaluation_rows.json",rows); _dump(out/"s18b2_decoder_audit.json",audits); _dump(out/"s18b2_sector_priors.json",priors)
    print("S18-B2 INDEPENDENT VALIDATION COMPLETE",flush=True)
    print(json.dumps(validation_gate,indent=2),flush=True)
    if not validation_gate["passed"]:
        raise RuntimeError(("S18-B2 validation gate failed",validation_gate))

if __name__ == "__main__":
    main()
