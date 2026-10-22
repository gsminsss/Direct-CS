#!/usr/bin/env python3
"""S21-OPT4B: optimized direct-d5->d13 exact-three pilot.

Purpose
-------
The optimized S21-OPT3 schedule changed the downstream circuit from
3535 locations/patch to 2132 locations/patch. Therefore the original S19
A3 coefficient must NOT be reused for a target-matched 3T comparison.

This stage performs a NEW, fixed-before-outcome exact-three pilot for the
optimized schedule.

Location partition
------------------
Pre-downstream factory: 890 locations, six groups
  FRONT 194
  CAT_PRE 105
  CORE_PRE 161
  MIDED 164
  CAT_POST 105
  CORE_POST 161

Optimized downstream:
  DOWN_DIRECT_D5_D13 = 2 * 2132 = 4264 locations

Total:
  N = 5154 locations
  7 groups
  C(7+3-1,3) = 84 exact-three strata
  C(5154,3) = 22,804,951,904 location triples

Decoder policy
--------------
The exact same physical samples are scored under BOTH already-frozen practical
decoder priors used by S21-OPT3C1:
  p_dec = 5e-4
  p_dec = 1e-3

For each prior:
  zero -> OPT3C0 exact <=2 firewall -> closure-conditioned cheap catalogue
       -> BP+OSD D2 (gap=10)

No threshold/gap/OSD tuning is performed.

Important interpretation
------------------------
This is a PILOT, not paper-final A3. Its purpose is to identify optimized
third-order harmful strata and determine whether a targeted high-stat follow-up
is needed. Zero observed events in a stratum are statistical non-observations,
not structural zeros.

The OPT3C0 168 future-truth bits remain scoring-only and are never decoder input.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_GAP=10.0
OP_DETECTORS=263
TRUTH_PER_PROBE=168
COMBINED_TRUTH_BITS=336
DOWNSTREAM_LOCATIONS_PER_PATCH=2132
PREDOWNSTREAM_LOCATIONS=890
TOTAL_POTENTIAL_LOCATIONS=5154
GROUPS=(
    "FRONT","CAT_PRE","CORE_PRE","MIDED","CAT_POST","CORE_POST",
    "DOWN_DIRECT_D5_D13",
)
PAULI_NAMES=("I","X","Z","Y")


def load(p:Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump(p:Path,obj:Any):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=json_default),encoding="utf-8")


def json_default(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return float(x)
    if isinstance(x,(complex,np.complexfloating)):
        z=complex(x); return {"__complex__":[float(z.real),float(z.imag)]}
    raise TypeError(type(x).__name__)


def imp(name:str,p:Path):
    spec=importlib.util.spec_from_file_location(name,str(p))
    if spec is None or spec.loader is None: raise ImportError(p)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m
    spec.loader.exec_module(m); return m


def find(root:Path,names:list[str],prefer:tuple[str,...]=()):
    hits=[]
    for n in names:
        p=Path(n)
        if p.is_absolute() and p.exists(): hits.append(p)
        elif "/" in n or "\\" in n:
            q=root/p
            if q.exists(): hits.append(q)
        else:
            hits.extend(root.rglob(n))
    if not hits: raise FileNotFoundError((str(root),names))
    hits=list(dict.fromkeys(x.resolve() for x in hits))
    hits.sort(key=lambda p:(
        tuple(0 if t in str(p) else 1 for t in prefer),
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)),str(p)))
    return hits[0]


def patch_globals(b1,d1,d2,b3,a2):
    for m in (b1,d1,d2,b3):
        setattr(m,"OP_DETECTORS",OP_DETECTORS)
        setattr(m,"TRUTH_PER_PROBE",TRUTH_PER_PROBE)
        setattr(m,"TRUTH_BITS",COMBINED_TRUTH_BITS)
        setattr(m,"TOTAL_LOCATIONS",DOWNSTREAM_LOCATIONS_PER_PATCH)
        setattr(m,"EXPECTED_DETECTORS",OP_DETECTORS)
    a2.GROUPS=GROUPS
    a2.OP_DETECTORS=OP_DETECTORS
    a2.TRUTH_BITS=COMBINED_TRUTH_BITS
    a2.DOWNSTREAM_LOCATIONS_PER_PATCH=DOWNSTREAM_LOCATIONS_PER_PATCH
    a2.PREDOWNSTREAM_LOCATIONS=PREDOWNSTREAM_LOCATIONS
    a2.TOTAL_POTENTIAL_LOCATIONS=TOTAL_POTENTIAL_LOCATIONS


def direct_stage(tick:int)->str:
    return "hirano_escape_plus_two_d5_rounds" if int(tick)<44 else "direct_d5_to_d13"


def sector_rows(path:Path):
    raw=load(path); out=[]
    for q0 in raw:
        q=dict(q0)
        q["detector_638_hex"]=q["detector_263_hex"]
        q["detector_638_weight"]=q["detector_263_weight"]
        q["detector_470_hex"]=q["detector_263_hex"]
        q["terminal_truth_168_hex"]=q["future_truth_168_hex"]
        q["terminal_truth_168_weight"]=q["future_truth_168_weight"]
        out.append(q)
    if len(out)!=64 or len({q["detector_638_hex"] for q in out})!=64:
        raise RuntimeError("bad 263-bit sector map")
    if any(int(q["future_truth_168_hex"],16)!=0 for q in out):
        raise RuntimeError("canonical sector future truth is nonzero")
    z=[q for q in out if int(q["detector_638_hex"],16)==0]
    if len(z)!=1 or z[0]["raw_terminal_pauli"]!="I":
        raise RuntimeError(("bad zero sector",z))
    return out


def closure_condition_decoder_effects(d1,effects):
    kept=[]; latent=[]
    for e in effects:
        try:
            d1._effect_columns([e]); kept.append(e)
        except RuntimeError as ex:
            if "zero-syndrome nonidentity downstream effect found" in str(ex):
                latent.append(e)
            else:
                raise
    if len(latent)!=3:
        raise RuntimeError(("optimized latent residual count drift",len(latent)))
    return kept,latent


def build_groups(a1,a2,model):
    out={g:[] for g in GROUPS}
    pre=a1._build_groups(0)
    for g in GROUPS[:6]:
        for loc in pre[g]:
            out[g].append(a2.S19Loc(group=g,kind="pre",pre_loc=loc))
    if len(model["loc_choices"])!=DOWNSTREAM_LOCATIONS_PER_PATCH:
        raise RuntimeError(("loc choices",len(model["loc_choices"])))
    for patch in (0,1):
        for li in range(DOWNSTREAM_LOCATIONS_PER_PATCH):
            out["DOWN_DIRECT_D5_D13"].append(
                a2.S19Loc(
                    group="DOWN_DIRECT_D5_D13",kind="downstream",
                    patch=patch,downstream_loc=li,stage="direct_d5_to_d13"))
    counts={g:len(out[g]) for g in GROUPS}
    expected={
        "FRONT":194,"CAT_PRE":105,"CORE_PRE":161,"MIDED":164,
        "CAT_POST":105,"CORE_POST":161,"DOWN_DIRECT_D5_D13":4264,
    }
    if counts!=expected: raise RuntimeError(("group inventory drift",counts,expected))
    return out


def all_strata(groups,a2):
    rows=a2._all_strata(groups)
    if len(rows)!=84: raise RuntimeError(("strata",len(rows)))
    total=sum(int(r["location_triples"]) for r in rows)
    expected=math.comb(TOTAL_POTENTIAL_LOCATIONS,3)
    if total!=expected or expected!=22804951904:
        raise RuntimeError(("C(N,3) partition",total,expected))
    return rows


def prior_from_c1(summary,p):
    for r in summary["prior_calibrations"]:
        if abs(float(r["p"])-float(p))<1e-15:
            sp=r["sector_probability"]
            arr=np.asarray([float(sp[str(i)]) for i in range(64)],dtype=float)
            if abs(float(arr.sum())-1)>1e-10: raise RuntimeError("prior norm")
            return arr,r
    raise RuntimeError(("prior missing",p))


def scan_firewall(path:Path,queries:set[int]):
    want=set(map(int,queries)); out={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for ii,line in enumerate(f,1):
            r=json.loads(line); q=int(r["record"],16)
            if q in want: out[q]=r
            if ii%1000000==0:
                print(f"  firewall {ii:,}: matched {len(out)}/{len(want)}",flush=True)
    return out


def load_base_d2(b3,d2,path:Path):
    out={}
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        out[int(r["record"],16)]=b3._deserialize_d2(d2,r["decode"])
    return out


def append_d2(path:Path,q:int,d2,obj):
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps({"record":hex(int(q)),"decode":d2._serialize(obj)},sort_keys=True)+"\n")


def score_aggregate(a2,rows):
    return a2._aggregate(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--search-root",type=Path,default=Path("."))
    ap.add_argument("--opt3b-result-dir",type=Path,default=Path("stim_pipeline/interface/s21opt3b_results"))
    ap.add_argument("--opt3c0-result-dir",type=Path,default=Path("stim_pipeline/interface/s21opt3c0_results"))
    ap.add_argument("--opt3c1-result-dir",type=Path,default=Path("stim_pipeline/interface/s21opt3c1_results"))
    ap.add_argument("--decoder-ps",type=float,nargs="+",default=[5e-4,1e-3])
    ap.add_argument("--shots-per-stratum",type=int,default=200)
    ap.add_argument("--seed",type=int,default=20270951)
    ap.add_argument("--sector-candidates",type=int,default=8)
    ap.add_argument("--bp-max-iter",type=int,default=30)
    ap.add_argument("--bp-method",default="minimum_sum")
    ap.add_argument("--ms-scaling-factor",type=float,default=0.625)
    ap.add_argument("--osd-method",default="osd_cs")
    ap.add_argument("--osd-order",type=int,default=2)
    ap.add_argument("--max-correction-effects",type=int,default=8)
    ap.add_argument("--max-examples-per-stratum",type=int,default=5)
    ap.add_argument("--plan-only",action="store_true")
    ap.add_argument("--out-dir",type=Path,default=Path("stim_pipeline/interface/s21opt4b_results"))
    args=ap.parse_args()

    if int(args.shots_per_stratum)<=0: raise SystemExit("--shots-per-stratum must be >0")
    if tuple(map(float,args.decoder_ps))!=(5e-4,1e-3):
        raise SystemExit("OPT4B pilot freezes decoder priors at exactly 0.0005 and 0.001")
    root=args.search_root.resolve()
    def R(p):
        return p.resolve() if p.is_absolute() else (root/p).resolve()
    bdir,c0dir,c1dir,out=map(R,(args.opt3b_result_dir,args.opt3c0_result_dir,args.opt3c1_result_dir,args.out_dir))
    out.mkdir(parents=True,exist_ok=True)

    # Frozen source stack.
    a2=imp("s21opt4b_a2",find(root,["direct_cs_s19a2a_exact3_independent_highstat.py"],("stim_pipeline/interface","s19work")))
    a1=imp("s21opt4b_a1",find(root,["direct_cs_s11a1_exact3_stratified_pilot.py"],("stim_pipeline/interface",)))
    b1=imp("s21opt4b_b1",find(root,["direct_cs_s18b1_638_d2_calibration.py"],("stim_pipeline/interface",)))
    b3=imp("s21opt4b_b3",find(root,["direct_cs_s18b3_firewalled_fresh_validation.py"],("stim_pipeline/interface",)))
    d1=imp("s21opt4b_d1",find(root,["direct_cs_s17c_d_hybrid_bposd_abstention.py"],("stim_pipeline/interface",)))
    d2=imp("s21opt4b_d2",find(root,["direct_cs_s17c_d2_logical_coset_list_decoder.py"],("stim_pipeline/interface",)))
    s10=imp("s21opt4b_s10",find(root,["direct_cs_s10_complete_physical_factory_mc.py"],("stim_pipeline/interface",)))
    cert=imp("s21opt4b_cert",find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"],("stim_pipeline/interface",)))
    patch_globals(b1,d1,d2,b3,a2)
    b1._stage_from_tick=direct_stage
    cert.EXPECTED_LOCATIONS=DOWNSTREAM_LOCATIONS_PER_PATCH
    cert.EXPECTED_DETECTORS=431

    # Required optimized results.
    c1sum=load(c1dir/"s21opt3c1_summary.json")
    if c1sum.get("status")!="PASS" or not c1sum.get("main_gate",{}).get("passed",False):
        raise RuntimeError("OPT3C1 not PASS")
    fwpath=c0dir/"s21opt3c0_firewall.jsonl.gz"
    sectors=sector_rows(c0dir/"s21opt3c0_sector_map_263_truth.json")
    sector_by={int(r["steane_syndrome"]):r for r in sectors}
    xtruth=bdir/"s21opt3b_X_direct_d5_d13_noisy.stim"
    ztruth=bdir/"s21opt3b_Z_direct_d5_d13_noisy.stim"
    for p in (fwpath,xtruth,ztruth):
        if not p.exists(): raise FileNotFoundError(p)

    print("S21-OPT4B: rebuilding optimized physical model...",flush=True)
    model=b1._build_truth_operational_model(cert,xtruth,ztruth)
    decoder_effects,latent=closure_condition_decoder_effects(d1,model["effects"])
    groups=build_groups(a1,a2,model)
    strata=all_strata(groups,a2)
    inventory={g:len(groups[g]) for g in GROUPS}
    inventory["total"]=sum(inventory.values())

    plan={
        "stage":"S21-OPT4B optimized exact-three fixed pilot plan",
        "status":"PLAN_FIXED_BEFORE_OUTCOMES",
        "seed":int(args.seed),
        "shots_per_stratum":int(args.shots_per_stratum),
        "total_conditional_shots":len(strata)*int(args.shots_per_stratum),
        "groups":inventory,
        "stratum_count":len(strata),
        "total_location_triples":math.comb(TOTAL_POTENTIAL_LOCATIONS,3),
        "decoder_ps":list(map(float,args.decoder_ps)),
        "threshold_sweep":False,
        "rows":[{
            "index":i,"key":r["key"],
            "counts":{GROUPS[j]:int(k) for j,k in enumerate(r["counts"]) if int(k)},
            "location_triples":int(r["location_triples"]),
            "shots":int(args.shots_per_stratum),
        } for i,r in enumerate(strata)],
    }
    dump(out/"s21opt4b_plan.json",plan)
    print(json.dumps({k:plan[k] for k in ("stratum_count","total_conditional_shots","groups","total_location_triples","decoder_ps")},indent=2),flush=True)
    if args.plan_only:
        print("S21-OPT4B plan-only PASS"); return

    # Closure-conditioned practical decoder matrix.
    cols=d1._effect_columns(decoder_effects)
    h=d1._build_sparse_check_matrix(cols)
    h_aug=d2._build_augmented_matrix(h,cols)
    rh=int(d2._gf2_row_rank_sparse(h)); ra=int(d2._gf2_row_rank_sparse(h_aug))
    if rh!=263 or ra<265: raise RuntimeError(("matrix drift",rh,ra))
    cheap_exact,cheap_amb,cheap_meta=b1._build_fast_catalogue(sectors,decoder_effects)

    # Sample every exact-three stratum once, fixed uniform pilot allocation.
    sampled={}
    all_queries=set()
    t0=time.time()
    for si,meta in enumerate(strata):
        sseed=int(args.seed)+si*1000003
        arr=[]
        for j in range(int(args.shots_per_stratum)):
            shot_seed=(sseed+j*1000000007)&((1<<63)-1)
            rng=np.random.default_rng(shot_seed)
            locs=a2._sample_locations(meta,groups,rng)
            rr=a2._simulate_exact3(a1,b1,sector_by,model,locs,rng,want_trace=False)
            arr.append(rr)
            if rr.get("accepted_predecode",False):
                all_queries.add(int(rr["record_a"])); all_queries.add(int(rr["record_b"]))
        sampled[meta["key"]]=arr
        if (si+1)%10==0:
            print(f"S21-OPT4B sampled {si+1}/{len(strata)} strata; queries={len(all_queries)}",flush=True)

    print(f"S21-OPT4B: scanning OPT3C0 firewall for {len(all_queries)} observed records...",flush=True)
    fw=scan_firewall(fwpath,all_queries)
    fw_exact={q:int(r["pauli_int"]) for q,r in fw.items() if r["status"]=="exact"}
    fw_amb={q for q,r in fw.items() if r["status"]=="ambiguous_reject"}
    if 0 not in fw_exact or fw_exact[0]!=0: raise RuntimeError("zero firewall drift")
    unknown=sorted(q for q in all_queries if q!=0 and q not in fw_exact and q not in fw_amb and q not in cheap_exact and q not in cheap_amb)

    # Build paired p-specific practical decoders, seeding caches from OPT3C1.
    BpOsdDecoder=d1._require_ldpc()
    priors={}; decoders={}; caches={}; cache_paths={}
    for p in map(float,args.decoder_ps):
        prior,pmeta=prior_from_c1(c1sum,p); priors[p]=prior
        pp=d1._effect_parity_priors(cols,p)
        decoders[p]=d1._make_bposd(
            BpOsdDecoder,h_aug,pp,
            int(args.bp_max_iter),args.bp_method,float(args.ms_scaling_factor),
            args.osd_method,int(args.osd_order))
        tag=f"{p:.12g}".replace(".","p").replace("-","m")
        base=c1dir/f"s21opt3c1_d2_cache_{tag}.jsonl"
        caches[p]=load_base_d2(b3,d2,base)
        cp=out/f"s21opt4b_d2_cache_{tag}.jsonl"; cache_paths[p]=cp
        if cp.exists(): cp.unlink()
        print(f"S21-OPT4B pdec={p:g}: inherited D2 cache={len(caches[p])}",flush=True)
        todo=[q for q in unknown if q not in caches[p]]
        for ii,q in enumerate(todo):
            dd=d2._decode_unknown_coset(
                d1,q,sectors,prior,p,decoders[p],h_aug,cols,
                int(args.sector_candidates),require_all_paulis=True)
            caches[p][q]=dd; append_d2(cp,q,d2,dd)
            if (ii+1)%25==0 or ii+1==len(todo):
                print(f"  pdec={p:g} new D2 {ii+1}/{len(todo)}",flush=True)
        bad=[q for q in unknown if (not bool(caches[p][q].success)) or len(caches[p][q].coverage_paulis)!=4]
        if bad: raise RuntimeError(("D2 completeness failure",p,[hex(x) for x in bad[:10]]))

    # Score the SAME physical samples under both p-specific decoder priors.
    summaries={}
    rows_by_p={}
    witnesses=[]
    for p in map(float,args.decoder_ps):
        rows=[]
        for si,meta in enumerate(strata):
            row=a2._score_samples(
                sampled[meta["key"]],b3,b1,s10,
                fw_exact,fw_amb,cheap_exact,cheap_amb,caches[p],
                int(args.max_correction_effects),int(args.max_examples_per_stratum),meta)
            row["index"]=si; row["decoder_p"]=p
            # Deterministic replay for harmful examples.
            for ex in row.get("harmful_examples",[]):
                j=int(ex["sample_index"]); sseed=int(args.seed)+si*1000003
                shot_seed=(sseed+j*1000000007)&((1<<63)-1)
                rng=np.random.default_rng(shot_seed)
                locs=a2._sample_locations(meta,groups,rng)
                rep=a2._simulate_exact3(a1,b1,sector_by,model,locs,rng,want_trace=True)
                if int(rep["record_a"])!=int(ex["record_A"],16) or int(rep["record_b"])!=int(ex["record_B"],16):
                    raise RuntimeError(("witness replay drift",p,meta["key"],j))
                ex["trace"]=rep.get("trace"); ex["shot_seed"]=int(shot_seed); ex["decoder_p"]=p
                witnesses.append(ex)
            rows.append(row)
            if row["closed_boundary_harmful_samples"]:
                print(f"  pdec={p:g} {meta['key']}: harmful={row['closed_boundary_harmful_samples']}/{row['shots']} A3={row['A3_closed_boundary_infidelity_estimate']:.6g}",flush=True)
        agg=score_aggregate(a2,rows)
        active=[r for r in rows if int(r["closed_boundary_harmful_samples"])>0 or float(r["A3_closed_boundary_infidelity_estimate"])>0]
        active.sort(key=lambda r:float(r["A3_closed_boundary_infidelity_estimate"]),reverse=True)
        summaries[str(p)]={
            "decoder_p":p,
            "aggregate":agg,
            "active_strata_count":len(active),
            "harmful_strata_count":sum(int(r["closed_boundary_harmful_samples"])>0 for r in rows),
            "top_active_strata":[{
                k:r[k] for k in (
                    "key","counts","location_triples","shots",
                    "closed_boundary_harmful_samples",
                    "A3_closed_boundary_infidelity_estimate",
                    "A3_closed_boundary_infidelity_se",
                    "A3_intrinsic_oracle_estimate",
                    "decoder_reject",
                )
            } for r in active[:25]],
        }
        rows_by_p[str(p)]=rows

    # Paired prior comparison from the exact same physical samples.
    p0,p1=map(float,args.decoder_ps)
    a0=float(summaries[str(p0)]["aggregate"]["A3_closed_boundary_infidelity_estimate"])
    a1v=float(summaries[str(p1)]["aggregate"]["A3_closed_boundary_infidelity_estimate"])
    s0=float(summaries[str(p0)]["aggregate"]["A3_closed_boundary_infidelity_naive_se"])
    s1=float(summaries[str(p1)]["aggregate"]["A3_closed_boundary_infidelity_naive_se"])

    result={
        "stage":"S21-OPT4B optimized direct-d5->d13 exact-three pilot",
        "status":"PILOT_NOT_PAPER_FINAL",
        "structural_source":"S21-OPT3B + OPT3C0; pilot does not replace <=2 certificate",
        "decoder_policy":"zero -> OPT3C0 exact<=2 firewall -> closure-conditioned cheap exact -> D2 gap10",
        "truth_bits_decoder_visible":False,
        "threshold_sweep":False,
        "location_inventory":inventory,
        "total_potential_locations":TOTAL_POTENTIAL_LOCATIONS,
        "C_N_3":math.comb(TOTAL_POTENTIAL_LOCATIONS,3),
        "stratum_count":len(strata),
        "shots_per_stratum":int(args.shots_per_stratum),
        "total_conditional_shots":len(strata)*int(args.shots_per_stratum),
        "seed":int(args.seed),
        "matrix":{"H_rank":rh,"H_aug_rank":ra,"effect_columns":len(cols),
                  "raw_effect_classes":len(model["effects"]),
                  "closure_conditioned_effect_classes":len(decoder_effects),
                  "latent_zero_op_residuals":len(latent)},
        "firewall_observed_query_matches":len(fw),
        "observed_queries":len(all_queries),
        "D2_unknown_records":len(unknown),
        "decoder_results":summaries,
        "paired_decoder_prior_difference":{
            "pdec_low":p0,"pdec_high":p1,
            "A3_low":a0,"A3_high":a1v,"delta_high_minus_low":a1v-a0,
            "naive_independent_se_for_delta":math.sqrt(s0*s0+s1*s1),
            "note":"Same physical samples are used; this SE is conservative because covariance is not subtracted."
        },
        "harmful_witness_count":len(witnesses),
        "elapsed_seconds":time.time()-t0,
        "next_stage_rule":{
            "if_harmful_strata_found":"Run S21-OPT4C targeted high-stat only on the dominant harmful/uncertain strata, with the same paired priors.",
            "if_no_harmful_events":"Do not call A3 zero. Increase targeted/statistically efficient exact-three sampling using intrinsic/decoder-active strata.",
        },
        "paper_use":"Pilot only. Do not use the global A3 estimate as paper-final until targeted precision/prior-stability is closed.",
    }
    dump(out/"s21opt4b_summary.json",result)
    dump(out/"s21opt4b_rows_by_decoder_p.json",rows_by_p)
    dump(out/"s21opt4b_harmful_witnesses.json",witnesses)
    print("S21-OPT4B PILOT COMPLETE")
    print(json.dumps({
        "status":result["status"],
        "C_N_3":result["C_N_3"],
        "queries":len(all_queries),
        "D2_unknown":len(unknown),
        "decoder_results":{
            k:{
                "A3":v["aggregate"]["A3_closed_boundary_infidelity_estimate"],
                "A3_se":v["aggregate"]["A3_closed_boundary_infidelity_naive_se"],
                "harmful_samples":v["aggregate"]["closed_boundary_harmful_samples"],
                "active_strata":v["active_strata_count"],
            } for k,v in summaries.items()
        },
        "paired":result["paired_decoder_prior_difference"],
    },indent=2))


if __name__=="__main__":
    main()
