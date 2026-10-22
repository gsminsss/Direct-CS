#!/usr/bin/env python3
"""S19-A2B: paired decoder-prior stability audit for the exact-three S19 sample set.

Purpose
-------
This stage does NOT update/tune the S18 decoder and does NOT replace the S19-A2A
primary A3 estimate.  It replays the exact deterministic S19-A2A physical samples
for the union of:
  * S19-A1 active strata,
  * S19-A2A active strata, and
  * S19-A2A strata that exercised D2,
and rescored those SAME samples under the three already-generated frozen B3
sector priors p={2e-4,5e-4,1e-3}.

Only the D2 prior changes.  The exact <=2 firewall, cheap catalogue, gap=10,
BP+OSD algorithm/hyperparameters, physical samples, and shot allocation are
identical across paired comparisons.  This isolates decoder-prior sensitivity.

The p=2e-4 replay must reproduce the S19-A2A selected-strata result.  The script
reports paired A3 shifts and their paired standard errors.  It intentionally does
not use A2B outcomes to retune the decoder.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_GAP = 10.0
TOL = 1e-12
PAULI_NAMES = ("I", "X", "Z", "Y")
DEFAULT_PRIORS = (2e-4, 5e-4, 1e-3)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, (complex, np.complexfloating)):
        z = complex(x); return {"__complex__": [float(z.real), float(z.imag)]}
    raise TypeError(type(x).__name__)


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def _hash_obj(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=_json_default).encode()).hexdigest()


def _mean_se(xs):
    a = np.asarray(xs, dtype=float)
    if a.size == 0: return 0.0, 0.0
    m = float(a.mean())
    se = 0.0 if a.size <= 1 else float(a.std(ddof=1)/math.sqrt(a.size))
    return m, se


def _self_test():
    # Paired-difference arithmetic sanity test.
    a = np.array([0.0, 1.0, 0.25, 0.0])
    b = np.array([0.0, 0.5, 0.25, 0.0])
    d = b-a
    m, se = _mean_se(d)
    passed = abs(m + 0.125) < 1e-15 and se > 0 and FROZEN_GAP == 10.0
    return {
        "stage":"S19-A2B paired B3-prior stability",
        "passed":bool(passed),
        "frozen_gap":FROZEN_GAP,
        "paired_physical_samples":True,
        "threshold_sweep_on_S19":False,
        "truth_bits_decoder_visible":False,
        "decoder_policy":"zero -> B2R exact<=2 firewall -> cheap exact -> D2",
    }


def _decode_one(sample, pkey, high, a2, b3mod, b1mod, s10mc,
                fw_exact, fw_amb, cheap_exact, cheap_amb, max_effects):
    if not sample["accepted_predecode"]:
        return {"status":"pre_reject", "inf":0.0, "harmful":False,
                "oracle":0.0, "pred":None, "tiers":None}
    qa, qb = int(sample["record_a"]), int(sample["record_b"])
    oka, pa, ta = b3mod._decode_b3(qa, fw_exact, fw_amb, cheap_exact, cheap_amb,
                                  high, FROZEN_GAP, max_effects)
    okb, pb, tb = b3mod._decode_b3(qb, fw_exact, fw_amb, cheap_exact, cheap_amb,
                                  high, FROZEN_GAP, max_effects)
    if not oka or not okb:
        return {"status":"decoder_reject", "inf":0.0, "harmful":False,
                "oracle":0.0, "pred":None, "tiers":(ta,tb)}
    if int(sample["truth_a"]) != 0 or int(sample["truth_b"]) != 0:
        return {"status":"residual_accept", "inf":0.0, "harmful":False,
                "oracle":0.0, "pred":(int(pa),int(pb)), "tiers":(ta,tb)}
    tr = b1mod.Trajectory(
        qa, qb, int(sample["actual_pauli_a"]), int(sample["actual_pauli_b"]),
        int(sample["truth_a"]), int(sample["truth_b"]),
        np.asarray(sample["logical_vector"], complex),
        tuple(int(x) for x in sample["core_record"]), 3, 0, 0, 0,
        float(sample["intrinsic_oracle_infidelity"])
    )
    inf = float(b1mod._raw_corrected_infidelity(s10mc, tr, int(pa), int(pb)))
    return {"status":"closed_accept", "inf":inf, "harmful":bool(inf>TOL),
            "oracle":float(sample["intrinsic_oracle_infidelity"]),
            "pred":(int(pa),int(pb)), "tiers":(ta,tb)}


def _agg(rows, pstr):
    a3 = sum(float(r["per_prior"][pstr]["A3_inf"]) for r in rows)
    se = math.sqrt(sum(float(r["per_prior"][pstr]["A3_se"])**2 for r in rows))
    harm = sum(int(r["per_prior"][pstr]["harmful_samples"]) for r in rows)
    opacc = sum(int(r["per_prior"][pstr]["operational_accept"]) for r in rows)
    return {"A3_selected":a3, "A3_selected_naive_se":se,
            "harmful_samples_selected":harm, "operational_accept_samples_selected":opacc}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s18m-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m_results"))
    ap.add_argument("--s18m2-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--b2r-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b2r_firewall_results"))
    ap.add_argument("--b3-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b3_validation_results"))
    ap.add_argument("--s19a1-result-dir", type=Path, default=Path("stim_pipeline/interface/s19a1_results"))
    ap.add_argument("--s19a2a-result-dir", type=Path, default=Path("stim_pipeline/interface/s19a2a_results"))
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--decoder-ps", type=float, nargs="+", default=list(DEFAULT_PRIORS))
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd_cs")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s19a2b_results"))
    args=ap.parse_args()

    st=_self_test()
    if args.self_test_only:
        print(json.dumps(st,indent=2));
        if not st["passed"]: raise RuntimeError(st)
        return
    ps=tuple(float(x) for x in args.decoder_ps)
    if len(ps)<2 or not any(abs(x-2e-4)<1e-15 for x in ps):
        raise SystemExit("--decoder-ps must include baseline 0.0002 and at least one comparison prior")
    if any(x<=0 for x in ps): raise SystemExit("decoder priors must be >0")

    root=args.search_root.resolve()
    out=(root/args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    _dump(out/"s19a2b_self_test.json",st)

    # Import A2A helper implementation and the same frozen S18/S17 modules.
    a2=_import("s19a2b_a2", a2path := a2_find(root,["direct_cs_s19a2a_exact3_independent_highstat.py"]))
    a1=_import("s19a2b_a1", a2._find(root,["direct_cs_s11a1_exact3_stratified_pilot.py"],("stim_pipeline/interface",)))
    b1mod=_import("s19a2b_b1", a2._find(root,["direct_cs_s18b1_638_d2_calibration.py"],("stim_pipeline/interface",)))
    b3mod=_import("s19a2b_b3", a2._find(root,["direct_cs_s18b3_firewalled_fresh_validation.py"],("stim_pipeline/interface",)))
    b2rmod=_import("s19a2b_b2r", a2._find(root,["direct_cs_s18b2r_low_order_firewall.py"],("stim_pipeline/interface",)))
    d2=_import("s19a2b_d2", a2._find(root,["direct_cs_s17c_d2_logical_coset_list_decoder.py"],("stim_pipeline/interface",)))
    d1=_import("s19a2b_d1", a2._find(root,["direct_cs_s17c_d_hybrid_bposd_abstention.py"],("stim_pipeline/interface",)))
    s10mc=_import("s19a2b_s10", a2._find(root,["direct_cs_s10_complete_physical_factory_mc.py"],("stim_pipeline/interface",)))
    cert=_import("s19a2b_cert", a2._find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"],("stim_pipeline/interface",)))
    s18m=_import("s19a2b_s18m", a2._find(root,["direct_cs_s18m_modular_noisy_d13_certificate.py"],("stim_pipeline/interface",)))
    m2=_import("s19a2b_m2", a2._find(root,["direct_cs_s18m2_terminal_truth_closure.py"],("stim_pipeline/interface",)))
    ca=_import("s19a2b_ca", a2._find(root,["direct_cs_s17c_a_sector_to_470_map.py"],("stim_pipeline/interface",)))
    s8b=_import("s19a2b_s8b", a2._find(root,["zxz_core_boundary_signature_map.py"],("stim_pipeline/interface",)))
    v7=_import("s19a2b_v7", a2._find(root,["direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"],("stim_pipeline/interface",)))
    v11=_import("s19a2b_v11", a2._find(root,["direct_cs_s17b_v11_complete_missing_basis.py"],("stim_pipeline/interface",)))
    s18a=_import("s19a2b_s18a", a2._find(root,["direct_cs_s18a_noisy_d13_boundary_certificate.py"],("stim_pipeline/interface",)))
    d1.EXPECTED_DETECTORS=638; d2.EXPECTED_DETECTORS=638

    b3dir=(root/args.b3_result_dir).resolve(); b2rdir=(root/args.b2r_result_dir).resolve()
    b3sum=_load(b3dir/"s18b3_summary.json"); b3pri_path=b3dir/"s18b3_sector_priors.json"; b2rsum=_load(b2rdir/"s18b2r_summary.json")
    vg=b3sum.get("validation_gate",{})
    if not bool(vg.get("passed",False)) or int(vg.get("sampled_closed_boundary_harmful_le2_total",-1))!=0: raise RuntimeError("B3 not clean PASS")
    if abs(float(b3sum.get("frozen_gap",-1))-FROZEN_GAP)>1e-12: raise RuntimeError("gap drift")
    if not bool(b2rsum.get("validation_gate",{}).get("passed",False)): raise RuntimeError("B2R not PASS")

    a1dir=(root/args.s19a1_result_dir).resolve(); a2dir=(root/args.s19a2a_result_dir).resolve()
    a1active=_load(a1dir/"s19a1_active_strata.json")
    a2sum=_load(a2dir/"s19a2a_summary.json"); a2active=_load(a2dir/"s19a2a_active_strata.json"); a2rows=_load(a2dir/"s19a2a_strata.json"); a2plan=_load(a2dir/"s19a2a_plan.json")
    if str(a2sum.get("status"))!="INDEPENDENT_HIGHSTAT_PRIMARY" or int(a2sum.get("stratum_count",-1))!=220: raise RuntimeError("A2A incomplete")
    if abs(float(a2sum.get("decoder_p",-1))-2e-4)>1e-15 or abs(float(a2sum.get("frozen_gap",-1))-FROZEN_GAP)>1e-12: raise RuntimeError("A2A baseline drift")

    # Fixed selection from completed pre-A2B evidence only.
    a1keys={str(r["key"]) for r in a1active}
    a2keys={str(r["key"]) for r in a2active}
    d2keys={str(r["key"]) for r in a2rows if int(r.get("d2_unknown_records",0))>0}
    selected=sorted(a1keys|a2keys|d2keys)
    plan_by={str(r["key"]):r for r in a2plan["rows"]}
    selected_plan=[plan_by[k] for k in selected]
    selected_shots=sum(int(r["shots"]) for r in selected_plan)
    plan_obj={"stage":"S19-A2B paired prior-stability plan", "selection_frozen_before_A2B":True,
              "selection_source":"union(S19-A1 active, S19-A2A active, S19-A2A D2-relevant)",
              "selected_strata":len(selected),"selected_shots_replayed":selected_shots,
              "decoder_ps":list(ps),"baseline_decoder_p":2e-4,"a2a_seed":int(a2sum["seed"]),
              "selected_keys":selected,"rows":selected_plan}
    _dump(out/"s19a2b_plan.json",plan_obj)
    print(json.dumps({k:plan_obj[k] for k in ("stage","selected_strata","selected_shots_replayed","decoder_ps","a2a_seed")},indent=2),flush=True)

    # Rebuild exactly the A2A physical model.
    s18mr=(root/args.s18m_result_dir).resolve(); s18m2r=(root/args.s18m2_result_dir).resolve()
    xtruth=s18m2r/"s18m2_X_truth_closed.stim"; ztruth=s18m2r/"s18m2_Z_truth_closed.stim"; sector_op=s18mr/"s18m_sector_map_638.json"
    hirano=a2._find(root,["msc-ls/src"]); sahay=a2._find(root,["reference/fd3_baseline_ckt.stim"])
    print("S19-A2B: rebuilding paired physicalized model...",flush=True)
    model=b1mod._build_truth_operational_model(cert,xtruth,ztruth)
    sectors,smeta=b1mod._build_sector_truth_map(root,s18m,m2,ca,s8b,v7,v11,s18a,hirano,sahay,sector_op,float(args.signature_p))
    sector_by={int(r["steane_syndrome"]):r for r in sectors}
    groups=a2._build_groups(a1,model); strata=a2._all_strata(groups); strata_by={r["key"]:(i,r) for i,r in enumerate(strata)}
    if len(strata)!=220 or sum(int(r["location_triples"]) for r in strata)!=math.comb(7960,3): raise RuntimeError("partition drift")

    cols=d1._effect_columns(model["effects"]); h=d1._build_sparse_check_matrix(cols); h_aug=d2._build_augmented_matrix(h,cols)
    if int(d2._gf2_row_rank_sparse(h))!=638 or int(d2._gf2_row_rank_sparse(h_aug))<640: raise RuntimeError("matrix rank drift")
    cheap_exact,cheap_amb,cheap_meta=b1mod._build_fast_catalogue(sectors,model["effects"])
    print("S19-A2B: building frozen B2R firewall...",flush=True)
    fw_idx=b2rmod.build_firewall_index(sectors,model)
    fw_cache_path=out/"s19a2b_firewall_cache.jsonl"
    fw_cache=a2._load_fw_cache(fw_cache_path) if args.resume else {}
    if not args.resume and fw_cache_path.exists(): fw_cache_path.unlink(); fw_cache={}
    if 0 not in fw_cache:
        fw_cache[0]=b2rmod.resolve_query(0,fw_idx,max_examples=1); a2._append_fw_cache(fw_cache_path,0,fw_cache[0])
    if fw_cache[0]["status"]!="exact" or fw_cache[0]["pauli"]!="I": raise RuntimeError("firewall zero drift")

    # Frozen decoder instances for each already-generated B3 prior.
    BpOsdDecoder=d1._require_ldpc(); priors={}; decoders={}; prior_meta={}; d2caches={}; cache_paths={}
    for p in ps:
        pr,pm=a2._load_b3_prior(b3pri_path,p); priors[p]=pr; prior_meta[p]=pm
        pp=d1._effect_parity_priors(cols,p)
        decoders[p]=d1._make_bposd(BpOsdDecoder,h_aug,pp,args.bp_max_iter,args.bp_method,args.ms_scaling_factor,args.osd_method,args.osd_order)
        tag=f"{p:.4g}".replace(".","p").replace("-","m")
        cp=out/f"s19a2b_d2_cache_{tag}.jsonl"; cache_paths[p]=cp
        d2caches[p]=a2._load_d2_cache(b3mod,d2,cp) if args.resume else {}
        if not args.resume and cp.exists(): cp.unlink(); d2caches[p]={}

    rows_path=out/"s19a2b_rows.json"; rows=_load(rows_path) if args.resume and rows_path.exists() else []
    done={r["key"] for r in rows}
    if not args.resume and rows_path.exists(): rows_path.unlink(); rows=[]; done=set()

    base_p=min(ps,key=lambda x:abs(x-2e-4)); a2row_by={str(r["key"]):r for r in a2rows}
    for nk,key in enumerate(selected,1):
        if key in done: continue
        si,meta=strata_by[key]; nshot=int(plan_by[key]["shots"]); stratum_seed=int(a2sum["seed"])+si*1000003
        samples=[]
        for j in range(nshot):
            shot_seed=(stratum_seed+j*1000000007)&((1<<63)-1); rng=np.random.default_rng(shot_seed)
            locs=a2._sample_locations(meta,groups,rng)
            samples.append(a2._simulate_exact3(a1,b1mod,sector_by,model,locs,rng,want_trace=False))
        queries={int(r["record_a"]) for r in samples if r["accepted_predecode"]}|{int(r["record_b"]) for r in samples if r["accepted_predecode"]}
        for q in sorted(q for q in queries if q not in fw_cache):
            fw_cache[q]=b2rmod.resolve_query(q,fw_idx,max_examples=1); a2._append_fw_cache(fw_cache_path,q,fw_cache[q])
        fw_exact={q:int(fw_cache[q]["pauli_int"]) for q in queries if fw_cache[q]["status"]=="exact"}
        fw_amb={q for q in queries if fw_cache[q]["status"]=="ambiguous_reject"}
        fw_high={q for q in queries if fw_cache[q]["status"]=="higher_order_only"}
        unknown=sorted(q for q in queries if q!=0 and q not in fw_exact and q not in fw_amb and q not in cheap_exact and q not in cheap_amb)
        bad=[q for q in unknown if q not in fw_high]
        if bad: raise RuntimeError(("low-order query leaked to D2",key,[hex(q) for q in bad[:5]]))
        highs={}
        for p in ps:
            cache=d2caches[p]
            for q in [x for x in unknown if x not in cache]:
                dd=d2._decode_unknown_coset(d1,q,sectors,priors[p],p,decoders[p],h_aug,cols,int(args.sector_candidates),require_all_paulis=True)
                cache[q]=dd; a2._append_d2_cache(cache_paths[p],q,d2,dd)
            if any((not bool(cache[q].success)) or len(cache[q].coverage_paulis)!=4 for q in unknown): raise RuntimeError(("D2 completeness",p,key))
            highs[p]=cache

        per={}; vectors={}
        for p in ps:
            rr=[_decode_one(x,p,highs[p],a2,b3mod,b1mod,s10mc,fw_exact,fw_amb,cheap_exact,cheap_amb,int(args.max_correction_effects)) for x in samples]
            inf=np.asarray([x["inf"] for x in rr],float); m,se=_mean_se(inf); ways=int(meta["location_triples"])
            vectors[p]=inf
            per[str(p)]={"A3_inf":ways*m,"A3_se":ways*se,"harmful_samples":sum(x["harmful"] for x in rr),
                         "operational_accept":sum(x["status"] in ("closed_accept","residual_accept") for x in rr),
                         "closed_accept":sum(x["status"]=="closed_accept" for x in rr),
                         "decoder_reject":sum(x["status"]=="decoder_reject" for x in rr),
                         "d2_unknown_records":len(unknown)}
        paired={}
        base_rr=vectors[base_p]
        for p in ps:
            if p==base_p: continue
            diff=vectors[p]-base_rr; md,sed=_mean_se(diff); ways=int(meta["location_triples"])
            paired[str(p)]={"delta_A3_vs_0p0002":ways*md,"paired_se":ways*sed}
        row={"index":si,"key":key,"shots":nshot,"location_triples":int(meta["location_triples"]),"d2_unknown_records":len(unknown),"per_prior":per,"paired":paired}
        rows.append(row); rows.sort(key=lambda x:int(x["index"])); _dump(rows_path,rows)
        if nk%10==0 or any(per[str(p)]["harmful_samples"] for p in ps):
            print(f"S19-A2B {nk:2d}/{len(selected)} {key}: D2={len(unknown)} " + " ".join(f"p={p:g}:A3={per[str(p)]['A3_inf']:.4g}" for p in ps),flush=True)

    # Selected contribution and paired shifts.
    aggs={str(p):_agg(rows,str(p)) for p in ps}
    paired_agg={}
    for p in ps:
        if p==base_p: continue
        delta=sum(float(r["paired"][str(p)]["delta_A3_vs_0p0002"]) for r in rows)
        se=math.sqrt(sum(float(r["paired"][str(p)]["paired_se"])**2 for r in rows))
        paired_agg[str(p)]={"delta_A3_selected_vs_0p0002":delta,"paired_se":se,
                            "paired_z":(delta/se if se>0 else (0.0 if delta==0 else float("inf")))}

    # The unselected A2A samples did not exercise D2 by construction, so their
    # A2A contribution is prior-invariant on this paired sample set.  Reconstruct
    # a global rescored coefficient by replacing only the selected contribution.
    primary_global=float(a2sum["aggregate"]["A3_closed_boundary_infidelity_estimate"])
    base_selected=float(aggs[str(base_p)]["A3_selected"])
    global_rescored={str(p):primary_global+(float(aggs[str(p)]["A3_selected"])-base_selected) for p in ps}
    rel_shift={str(p):(global_rescored[str(p)]-primary_global)/primary_global if primary_global else 0.0 for p in ps}

    # Strict reproducibility check against the original A2A selected-strata rows.
    expected_selected=sum(float(a2row_by[k]["A3_closed_boundary_infidelity_estimate"]) for k in selected)
    replay_error=base_selected-expected_selected
    replay_tol=max(1e-6,1e-10*max(1.0,abs(expected_selected)))
    replay_pass=abs(replay_error)<=replay_tol

    # Pre-registered descriptive stability gate: no >50% global coefficient shift.
    # Paired z is reported separately; this gate is intentionally not a tuning rule.
    max_rel=max(abs(rel_shift[str(p)]) for p in ps if p!=base_p)
    stability_pass=bool(replay_pass and max_rel<=0.50)
    summary={"stage":"S19-A2B paired B3-prior stability audit","status":"PASS" if stability_pass else "REVIEW",
             "selection_frozen_before_A2B":True,"selected_strata":len(selected),"selected_shots_replayed":selected_shots,
             "decoder_ps":list(ps),"prior_seeds":{str(p):prior_meta[p].get("seed") for p in ps},
             "frozen_gap":FROZEN_GAP,"threshold_sweep_on_S19":False,"truth_bits_decoder_visible":False,
             "decoder_policy":"zero -> B2R exact<=2 firewall -> cheap exact -> D2",
             "A2A_primary_global_A3":primary_global,"selected_aggregate":aggs,"paired_deltas":paired_agg,
             "global_A3_rescored":global_rescored,"relative_global_shift_vs_0p0002":rel_shift,
             "baseline_replay":{"expected_selected_A2A_A3":expected_selected,"replayed_selected_A3":base_selected,
                                "difference":replay_error,"tolerance":replay_tol,"passed":replay_pass},
             "stability_gate":{"criterion":"max absolute paired global A3 shift <= 50% of A2A primary; descriptive robustness gate, no retuning","max_abs_relative_shift":max_rel,"passed":stability_pass},
             "interpretation":["A2B is a paired sensitivity analysis, not a new independent A3 estimate.",
                               "All physical samples and non-D2 decoder layers are identical across priors.",
                               "Only already-generated B3 sector/parity priors change; gap/firewall/OSD settings remain frozen."],
             "next_stage":"If stable, run S19-A3 targeted precision on variance-dominant rare strata (especially DOWN_HIRANO_D5^3) before quoting paper-final A3; then decoder-ablation MWPM/correlated-matching benchmark."}
    _dump(out/"s19a2b_summary.json",summary)
    print("S19-A2B COMPLETE",flush=True); print(json.dumps(summary,indent=2),flush=True)
    if not replay_pass: raise RuntimeError("A2B baseline paired replay failed to reproduce A2A")


def a2_find(root: Path, names: list[str]) -> Path:
    hits=[]
    for n in names:
        p=root/n
        if p.exists(): hits.append(p)
        hits.extend(root.rglob(n))
    # prefer interface source, not copied result folders
    hits=list(dict.fromkeys(hits))
    hits.sort(key=lambda p:(0 if "stim_pipeline" in p.parts and "interface" in p.parts else 1,len(p.parts),str(p)))
    if not hits: raise FileNotFoundError(names)
    return hits[0]


if __name__=="__main__": main()
