#!/usr/bin/env python3
"""S21-OPT3C1: freeze 263-bit practical BP+OSD decoder and run fresh finite-p validation.

This stage consumes the PASS S21-OPT3B direct d5->d13 circuits and the
S21-OPT3C0 263-operational/168-future-truth firewall.

Scientific policy
-----------------
* Decoder-visible bits: first 263 detector bits only.
* Future 168 truth detectors: scoring only, never decoder-visible.
* Exact <=2 firewall has priority over the cheap catalogue and BP+OSD.
* Practical higher-order decoder policy is NOT retuned:
    gap = 10
    sector candidates = 8
    BP max iter = 30
    BP method = minimum_sum
    min-sum scale = 0.625
    OSD = osd_cs, order 2
    max correction effects = 8
* Fresh p points only: 5e-4 and 1e-3 by default.
* Fresh independent prior/evaluation seeds.
* Matrix rank is audited before any finite-p decoding.
* No threshold sweep.

The direct-splice downstream contains 2132 active locations per patch instead
of the old 3535.  `_score_b3` is intentionally reused after monkeypatching the
B1 downstream location constant to 2132, so its stage-level active-location
ledger is automatically updated.

This script is a resource/performance validation, not a new A3 rare-event
coefficient calculation.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

OP_DETECTORS=263
TRUTH_BITS=168
TOTAL_DETECTORS=431
DOWNSTREAM_LOCATIONS_PER_PATCH=2132
PREDOWNSTREAM_LOCATIONS=890
FULL_FACTORY_POTENTIAL_LOCATIONS=PREDOWNSTREAM_LOCATIONS+2*DOWNSTREAM_LOCATIONS_PER_PATCH
FROZEN_GAP=10.0
PAULI_NAMES=["I","X","Z","Y"]


def load(p:Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump(p:Path,obj:Any):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str),encoding="utf-8")


def imp(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m


def find(root:Path,names:list[str],prefer:tuple[str,...]=()):
    hits=[]
    for n in names:
        p=Path(n)
        if p.is_absolute() and p.exists():
            hits.append(p)
        elif "/" in n or "\\" in n:
            q=root/p
            if q.exists(): hits.append(q)
        else:
            hits.extend(root.rglob(n))
    if not hits:
        raise FileNotFoundError((str(root),names))
    hits=list(dict.fromkeys(p.resolve() for p in hits))
    hits.sort(key=lambda p:(
        tuple(0 if t in str(p) else 1 for t in prefer),
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)),str(p)))
    return hits[0]


def patch_module_constants(m):
    # Historical S18-B1 used
    #   638 operational + 84 truth/probe = 722 detectors/probe,
    # and combined the X/Z probe truth masks into 168 scoring bits.
    #
    # The direct-expansion circuit instead has
    #   263 operational + 168 future-truth detectors PER probe = 431.
    # B1 therefore must see TRUTH_PER_PROBE=168 explicitly. The combined
    # scoring truth carried by its Trajectory object can occupy 336 bits.
    vals={
        "OP_DETECTORS":OP_DETECTORS,
        "TRUTH_PER_PROBE":TRUTH_BITS,
        "TRUTH_BITS":2*TRUTH_BITS,
        "TOTAL_DETECTORS":TOTAL_DETECTORS,
        "TOTAL_LOCATIONS":DOWNSTREAM_LOCATIONS_PER_PATCH,
        "EXPECTED_DETECTORS":OP_DETECTORS,
    }
    for k,v in vals.items():
        # Setting an extra module global is harmless and makes wrappers robust
        # to different historical versions that refer to different names.
        setattr(m,k,v)


def sector_live_rows(rows):
    out=[]
    for r0 in rows:
        r=dict(r0)
        # Historical D1/D2/B1 functions use detector_638_hex as the operational
        # record field name.  In this 263-bit wrapper it is explicitly an alias.
        r["detector_638_hex"]=r["detector_263_hex"]
        r["detector_638_weight"]=r["detector_263_weight"]
        r["detector_470_hex"]=r["detector_263_hex"]
        # Historical B1 sampler expects this field name. For the canonical
        # incoming sectors OPT3C0 has already shown all future-truth bits are 0.
        r["terminal_truth_168_hex"]=r["future_truth_168_hex"]
        r["terminal_truth_168_weight"]=r["future_truth_168_weight"]
        out.append(r)
    if len(out)!=64:
        raise RuntimeError(("sector count",len(out)))
    if len({r["detector_638_hex"] for r in out})!=64:
        raise RuntimeError("263-bit canonical sector signatures are not unique")
    if any(int(r["future_truth_168_hex"],16)!=0 for r in out):
        raise RuntimeError("canonical sector unexpectedly has future-truth support")
    z=[r for r in out if int(r["detector_638_hex"],16)==0]
    if len(z)!=1 or z[0]["raw_terminal_pauli"]!="I":
        raise RuntimeError(("zero sector contract",z))
    return out



def closure_condition_decoder_effects(d1, effects):
    """Remove only latent zero-operational-syndrome nonidentity residual columns.

    Scientific justification:
      OPT3C0's exact <=2 firewall proves that operational record 0 AND
      future-truth closed implies logical I only. Therefore any *single-effect*
      object rejected here by the historical D1 guard
          "zero-syndrome nonidentity downstream effect found"
      cannot be a truth-closed single-fault logical error under the frozen
      direct-expansion contract. It is a truth-open/detectable residual whose
      physical trajectory remains present in sampling and scoring.

    These latent residuals are excluded only from the BP+OSD parity-check
    column model, because an unobserved zero-syndrome column cannot be inferred
    from the operational syndrome anyway. They are NOT removed from the
    physical model or from future-truth scoring.
    """
    kept=[]
    latent=[]
    other_errors=[]
    for i,e in enumerate(effects):
        try:
            _ = d1._effect_columns([e])
            kept.append(e)
        except RuntimeError as ex:
            msg=str(ex)
            if "zero-syndrome nonidentity downstream effect found" in msg:
                latent.append(e)
            else:
                other_errors.append((i,msg,repr(e)))
    if other_errors:
        raise RuntimeError(("unexpected effect-column validation errors",other_errors[:20]))
    if not latent:
        return kept, {
            "raw_effect_classes":len(effects),
            "decoder_effect_classes":len(kept),
            "latent_zero_syndrome_nonidentity_excluded":0,
            "policy":"none needed",
            "examples":[],
        }
    # Keep only short reprs for diagnostics; exact physical safety is certified
    # by the OPT3C0 firewall, not by these repr strings.
    return kept, {
        "raw_effect_classes":len(effects),
        "decoder_effect_classes":len(kept),
        "latent_zero_syndrome_nonidentity_excluded":len(latent),
        "policy":"excluded from decoder columns only; retained in physical sampling/truth scoring",
        "OPT3C0_zero_record_truth_closed_identity_only_required":True,
        "examples":[repr(e)[:500] for e in latent[:20]],
    }


def firewall_zero_gate(path:Path):
    import gzip,json
    zero=None
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if '"record": "0x0"' not in line and '"record":"0x0"' not in line:
                continue
            r=json.loads(line)
            if r.get("record")=="0x0":
                zero=r
                break
    if zero is None:
        # Sorting normally places 0x0 first, but do a robust full fallback.
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line)
                if r.get("record")=="0x0":
                    zero=r; break
    ok=zero is not None and zero.get("status")=="exact" and zero.get("possible_paulis")==["I"]
    if not ok:
        raise RuntimeError(("OPT3C0 zero firewall gate fail",zero))
    return zero


def resolve_needed_firewall(path:Path,needed:set[int]):
    """Stream the 4.8M-row firewall once; absent records are higher-order-only."""
    wanted=set(map(int,needed))
    out={}
    t0=time.time()
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for ii,line in enumerate(f,1):
            r=json.loads(line)
            q=int(r["record"],16)
            if q in wanted:
                out[q]=r
            if ii%1000000==0:
                print(f"  firewall scanned {ii:,} rows; matched {len(out)}/{len(wanted)} observed queries",flush=True)
    return out,time.time()-t0


def tag(p:float):
    s=f"{p:.12g}".replace(".","p").replace("-","m")
    return s


def cache_paths(out:Path,p:float):
    t=tag(p)
    return out/f"s21opt3c1_d2_cache_{t}.jsonl",out/f"s21opt3c1_d2_cache_{t}_meta.json"


def hash_obj(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def load_d2_cache(b3mod,d2,path:Path,meta:Path,wanted:dict):
    if not path.exists():
        return {}
    if not meta.exists() or load(meta)!=wanted:
        raise RuntimeError("stale OPT3C1 D2 cache; use a fresh --out-dir or delete old cache")
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o=json.loads(line)
        out[int(o["record"],16)]=b3mod._deserialize_d2(d2,o["decode"])
    return out


def append_d2_cache(d2,path:Path,q:int,d):
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps({"record":hex(int(q)),"decode":d2._serialize(d)},sort_keys=True)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--search-root",type=Path,default=Path("."))
    ap.add_argument("--opt3b-result-dir",type=Path,
                    default=Path("stim_pipeline/interface/s21opt3b_results"))
    ap.add_argument("--opt3c0-result-dir",type=Path,
                    default=Path("stim_pipeline/interface/s21opt3c0_results"))
    ap.add_argument("--p-values",type=float,nargs="+",default=[5e-4,1e-3])
    ap.add_argument("--shots-per-p",type=int,nargs="+",default=[5000,5000])
    ap.add_argument("--prior-calibration-shots",type=int,default=30000)
    ap.add_argument("--prior-seed",type=int,default=20270901)
    ap.add_argument("--evaluation-seed",type=int,default=20270921)
    ap.add_argument("--signature-p",type=float,default=0.001)
    ap.add_argument("--sector-candidates",type=int,default=8)
    ap.add_argument("--bp-max-iter",type=int,default=30)
    ap.add_argument("--bp-method",default="minimum_sum")
    ap.add_argument("--ms-scaling-factor",type=float,default=0.625)
    ap.add_argument("--osd-method",default="osd_cs")
    ap.add_argument("--osd-order",type=int,default=2)
    ap.add_argument("--max-correction-effects",type=int,default=8)
    ap.add_argument("--frozen-gap",type=float,default=FROZEN_GAP)
    ap.add_argument("--max-examples",type=int,default=10)
    ap.add_argument("--out-dir",type=Path,
                    default=Path("stim_pipeline/interface/s21opt3c1_results"))
    ap.add_argument("--matrix-only",action="store_true")
    args=ap.parse_args()

    if abs(float(args.frozen_gap)-FROZEN_GAP)>1e-12:
        raise SystemExit("OPT3C1 gap is frozen at 10; no evaluation-time retuning")
    if len(args.p_values)!=len(args.shots_per_p):
        raise SystemExit("--p-values and --shots-per-p length mismatch")
    if args.prior_seed==args.evaluation_seed:
        raise SystemExit("prior/evaluation seeds must differ")

    root=args.search_root.resolve()
    bdir=args.opt3b_result_dir if args.opt3b_result_dir.is_absolute() else (root/args.opt3b_result_dir).resolve()
    c0dir=args.opt3c0_result_dir if args.opt3c0_result_dir.is_absolute() else (root/args.opt3c0_result_dir).resolve()
    out=args.out_dir if args.out_dir.is_absolute() else (root/args.out_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)

    xtruth=bdir/"s21opt3b_X_direct_d5_d13_noisy.stim"
    ztruth=bdir/"s21opt3b_Z_direct_d5_d13_noisy.stim"
    opt3bs=bdir/"s21opt3b_summary.json"
    pre=c0dir/"s21opt3c0_preflight.json"
    sectors_p=c0dir/"s21opt3c0_sector_map_263_truth.json"
    firewall_p=c0dir/"s21opt3c0_firewall.jsonl.gz"
    for p in (xtruth,ztruth,opt3bs,pre,sectors_p,firewall_p):
        if not p.exists(): raise FileNotFoundError(p)

    bs=load(opt3bs); ps=load(pre)
    if bs.get("status")!="PASS" or not bs.get("main_gate",{}).get("passed",False):
        raise RuntimeError("OPT3B structural gate not PASS")
    if not ps.get("passed",False) or int(ps.get("decoder_input_bits",-1))!=OP_DETECTORS:
        raise RuntimeError(("OPT3C0 preflight not PASS",ps))
    if ps.get("future_truth_bits_decoder_visible",True):
        raise RuntimeError("future truth bits decoder-visible")
    zfw=firewall_zero_gate(firewall_p)

    raw_sectors=load(sectors_p)
    sectors=sector_live_rows(raw_sectors)
    sector_by={int(r["steane_syndrome"]):r for r in sectors}

    # Locate and import frozen S18 decoder stack.
    b1p=find(root,["direct_cs_s18b1_638_d2_calibration.py"],("stim_pipeline/interface",))
    d1p=find(root,["direct_cs_s17c_d_hybrid_bposd_abstention.py"],("stim_pipeline/interface",))
    d2p=find(root,["direct_cs_s17c_d2_logical_coset_list_decoder.py"],("stim_pipeline/interface",))
    s10p=find(root,["direct_cs_s10_complete_physical_factory_mc.py"],("stim_pipeline/interface",))
    certp=find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"],("stim_pipeline/interface",))
    b3p=find(root,["direct_cs_s18b3_firewalled_fresh_validation.py"],("stim_pipeline/interface",))

    b1=imp("s21opt3c1_b1",b1p)
    d1=imp("s21opt3c1_d1",d1p)
    d2=imp("s21opt3c1_d2",d2p)
    s10=imp("s21opt3c1_s10",s10p)
    cert=imp("s21opt3c1_cert",certp)
    b3=imp("s21opt3c1_b3",b3p)

    for m in (b1,d1,d2,b3):
        patch_module_constants(m)

    # Historical B1 labels tick>=44 as the old Sahay d5->d7/d7->d13/close
    # stages. In the optimized circuit all post-Hirano faultable ticks belong
    # to the single direct d5->d13 expansion stage. This affects diagnostics
    # only, but keep the resource ledger semantically correct.
    def _direct_stage_from_tick(tick:int)->str:
        return (
            "hirano_escape_plus_two_d5_rounds"
            if int(tick) < 44 else
            "direct_d5_to_d13"
        )
    b1._stage_from_tick = _direct_stage_from_tick

    cert.EXPECTED_LOCATIONS=DOWNSTREAM_LOCATIONS_PER_PATCH
    cert.EXPECTED_DETECTORS=TOTAL_DETECTORS
    BpOsdDecoder=d1._require_ldpc()

    print("S21-OPT3C1: rebuilding 263+168 truth-operational model...",flush=True)
    model=b1._build_truth_operational_model(cert,xtruth,ztruth)
    if len(model.get("loc_choices",[]))!=DOWNSTREAM_LOCATIONS_PER_PATCH:
        raise RuntimeError(("downstream location model size",len(model.get("loc_choices",[]))))
    if not model.get("effects"):
        raise RuntimeError("empty direct-expansion decoder effect model")

    print("S21-OPT3C1: closure-conditioning decoder effect columns...",flush=True)
    decoder_effects,latent_meta=closure_condition_decoder_effects(d1,model["effects"])
    if not decoder_effects:
        raise RuntimeError("closure-conditioned decoder effect model is empty")
    print(
        f"S21-OPT3C1: raw effects={len(model['effects'])}, "
        f"decoder effects={len(decoder_effects)}, "
        f"latent zero-op residuals excluded={latent_meta['latent_zero_syndrome_nonidentity_excluded']}",
        flush=True
    )

    print("S21-OPT3C1: matrix/rank audit...",flush=True)
    cols=d1._effect_columns(decoder_effects)
    h=d1._build_sparse_check_matrix(cols)
    h_aug=d2._build_augmented_matrix(h,cols)
    rank_h=d2._gf2_row_rank_sparse(h)
    rank_aug=d2._gf2_row_rank_sparse(h_aug)
    exact,amb,cmeta=b1._build_fast_catalogue(sectors,decoder_effects)
    matrix={
        "H_shape":list(map(int,h.shape)),
        "H_aug_shape":list(map(int,h_aug.shape)),
        "H_rank":int(rank_h),
        "H_aug_rank":int(rank_aug),
        "logical_rows_independent":bool(rank_aug>=rank_h+2),
        "effect_columns":len(cols),
        "raw_effect_classes":len(model["effects"]),
        "closure_conditioned_decoder_effect_classes":len(decoder_effects),
        "latent_residual_filter":latent_meta,
        "fast_catalogue":cmeta,
        "model_meta":model.get("meta",{}),
    }
    matrix_pass=bool(
        rank_h==OP_DETECTORS
        and rank_aug>=OP_DETECTORS+2
        and len(decoder_effects)>0
    )
    dump(out/"s21opt3c1_matrix_audit.json",matrix)
    if not matrix_pass:
        raise RuntimeError(("263-bit matrix gate failed",matrix))

    preflight={
        "stage":"S21-OPT3C1 practical decoder freeze",
        "matrix_pass":True,
        "detector_contract":{"operational":263,"future_truth":168,"truth_visible":False},
        "downstream_locations_per_patch":DOWNSTREAM_LOCATIONS_PER_PATCH,
        "full_factory_potential_locations":FULL_FACTORY_POTENTIAL_LOCATIONS,
        "firewall_zero":zfw,
        "closure_conditioned_decoder_model":latent_meta,
        "frozen_decoder":{
            "gap":FROZEN_GAP,
            "sector_candidates":int(args.sector_candidates),
            "bp_max_iter":int(args.bp_max_iter),
            "bp_method":args.bp_method,
            "ms_scaling_factor":float(args.ms_scaling_factor),
            "osd_method":args.osd_method,
            "osd_order":int(args.osd_order),
            "max_correction_effects":int(args.max_correction_effects),
            "threshold_sweep":False,
        },
        "dependencies":{"b1":str(b1p),"d1":str(d1p),"d2":str(d2p),"s10":str(s10p),"cert":str(certp),"b3":str(b3p)},
    }
    dump(out/"s21opt3c1_preflight.json",preflight)
    if args.matrix_only:
        print(json.dumps({"matrix":matrix,"preflight":preflight},indent=2))
        print("S21-OPT3C1 matrix-only PASS")
        return

    # First generate all fresh physical samples, so the precomputed 4.8M-row
    # firewall can be streamed exactly once for the union of observed records.
    sampled_rows=[]
    prior_rows=[]
    all_queries=set()
    for ip,(p,shots) in enumerate(zip(map(float,args.p_values),map(int,args.shots_per_p))):
        psd=int(args.prior_seed+ip); esd=int(args.evaluation_seed+ip)
        print(f"S21-OPT3C1: fresh sector prior p={p:g}, seed={psd} ...",flush=True)
        prior,pmeta=d1._calibrate_sector_prior(p,int(args.prior_calibration_shots),psd,s10)
        pmeta["seed"]=psd
        prior_rows.append({"p":p,"prior":prior,"meta":pmeta})
        print(f"S21-OPT3C1: fresh physical evaluation p={p:g}, shots={shots}, seed={esd} ...",flush=True)
        sampled=b1._sample_one_p(p,shots,esd,s10,sector_by,model,int(args.max_examples))
        sampled_rows.append(sampled)
        all_queries.update(map(int,sampled["queries"]))

    print(f"S21-OPT3C1: scanning exact OPT3C0 firewall for {len(all_queries)} observed records...",flush=True)
    fw_rows,fw_seconds=resolve_needed_firewall(firewall_p,all_queries)
    zero_seen=fw_rows.get(0,zfw)
    if zero_seen.get("status")!="exact" or zero_seen.get("pauli")!="I":
        raise RuntimeError(("zero firewall drift",zero_seen))

    rows=[]; audits=[]
    for ip,(sampled,pr) in enumerate(zip(sampled_rows,prior_rows)):
        p=float(sampled["p"]); prior=pr["prior"]
        queries=set(map(int,sampled["queries"]))
        fw_exact={q:int(fw_rows[q]["pauli_int"]) for q in queries
                  if q in fw_rows and fw_rows[q]["status"]=="exact"}
        fw_amb={q for q in queries if q in fw_rows and fw_rows[q]["status"]=="ambiguous_reject"}
        fw_high={q for q in queries if q not in fw_rows}

        # Firewall has priority. Cheap catalogue is only a higher-order fallback.
        unknown=sorted(q for q in queries
                       if q!=0 and q not in fw_exact and q not in fw_amb
                       and q not in exact and q not in amb)
        if any(q not in fw_high for q in unknown):
            raise RuntimeError("low-order firewall-resolvable record leaked to D2")

        pri=d1._effect_parity_priors(cols,p)
        decoder=d1._make_bposd(
            BpOsdDecoder,h_aug,pri,
            int(args.bp_max_iter),args.bp_method,float(args.ms_scaling_factor),
            args.osd_method,int(args.osd_order)
        )

        cp,cm=cache_paths(out,p)
        wanted={
            "stage":"S21-OPT3C1",
            "p":p,
            "matrix_hash":hash_obj({k:matrix[k] for k in ("H_shape","H_aug_shape","H_rank","H_aug_rank","effect_columns")}),
            "prior_seed":int(pr["meta"]["seed"]),
            "prior_hash":hashlib.sha256(__import__("numpy").asarray(prior,dtype="float64").tobytes()).hexdigest(),
            "gap":FROZEN_GAP,
            "sector_candidates":int(args.sector_candidates),
            "bp_max_iter":int(args.bp_max_iter),
            "bp_method":args.bp_method,
            "ms_scaling_factor":float(args.ms_scaling_factor),
            "osd_method":args.osd_method,
            "osd_order":int(args.osd_order),
            "max_correction_effects":int(args.max_correction_effects),
        }
        if not cm.exists(): dump(cm,wanted)
        high=load_d2_cache(b3,d2,cp,cm,wanted)
        todo=[q for q in unknown if q not in high]
        print(
            f"S21-OPT3C1 p={p:g}: records={len(queries)}, "
            f"firewall exact={len(fw_exact)}, ambiguous={len(fw_amb)}, higher={len(fw_high)}, "
            f"D2 unknown={len(unknown)}, cached={len(unknown)-len(todo)}, decode={len(todo)}",
            flush=True
        )
        td=time.time()
        for ii,q in enumerate(todo):
            d=d2._decode_unknown_coset(
                d1,q,sectors,prior,p,decoder,h_aug,cols,
                int(args.sector_candidates),require_all_paulis=True
            )
            high[q]=d
            append_d2_cache(d2,cp,q,d)
            if (ii+1)%25==0 or ii+1==len(todo):
                print(f"  decoded {ii+1}/{len(todo)}",flush=True)

        cover=collections.Counter(len(high[q].coverage_paulis) for q in unknown)
        gaps=[float(high[q].gap) for q in unknown
              if bool(high[q].success) and math.isfinite(float(high[q].gap))]
        audit={
            "p":p,
            "shots":int(sampled["shots"]),
            "prior_seed":int(pr["meta"]["seed"]),
            "evaluation_seed":int(sampled["seed"]),
            "observed_records":len(queries),
            "firewall_exact":len(fw_exact),
            "firewall_ambiguous":len(fw_amb),
            "firewall_higher_order_only":len(fw_high),
            "cheap_exact_after_firewall":sum(q in exact and q not in fw_exact and q not in fw_amb for q in queries),
            "cheap_ambiguous_after_firewall":sum(q in amb and q not in fw_exact and q not in fw_amb for q in queries),
            "d2_unknown":len(unknown),
            "d2_success":sum(bool(high[q].success) for q in unknown),
            "d2_failure":sum(not bool(high[q].success) for q in unknown),
            "coverage_histogram":{str(k):int(v) for k,v in sorted(cover.items())},
            "finite_gap_count":len(gaps),
            "gap_median":float(__import__("numpy").median(gaps)) if gaps else None,
            "new_decode_seconds":time.time()-td,
        }
        audits.append(audit)

        r=b3._score_b3(
            sampled,fw_exact,fw_amb,exact,amb,high,b1,s10,
            FROZEN_GAP,int(args.max_correction_effects),int(args.max_examples)
        )
        r["prior_seed"]=int(pr["meta"]["seed"])
        r["evaluation_seed"]=int(sampled["seed"])
        r["downstream_locations_per_patch"]=DOWNSTREAM_LOCATIONS_PER_PATCH
        r["full_factory_potential_locations"]=FULL_FACTORY_POTENTIAL_LOCATIONS
        rows.append(r)

    validation_pass=all(
        bool(r["sampled_le2_regression_gate_pass"]) for r in rows
    )
    # Decoder construction additionally requires all four logical cosets to be
    # represented for every successful higher-order D2 decode.
    coverage_pass=all(
        all((not d) or True for d in [True]) and
        all(int(k)==4 or int(v)==0 for k,v in
            ((int(k),int(v)) for k,v in a["coverage_histogram"].items()))
        for a in audits
    )
    # More direct robust form: no successful D2 records with coverage !=4.
    coverage_pass=all(
        sum(v for k,v in ((int(k),int(v)) for k,v in a["coverage_histogram"].items()) if k!=4)==0
        for a in audits
    )

    main_gate={
        "OPT3B_structural_pass":True,
        "OPT3C0_263_plus_168_split_pass":True,
        "truth_bits_decoder_visible":False,
        "matrix_rank_pass":matrix_pass,
        "no_threshold_sweep":True,
        "fresh_prior_and_evaluation_seeds":True,
        "sampled_closed_boundary_harmful_le2_total":
            sum(int(r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"]) for r in rows),
        "sampled_le2_regression_pass":validation_pass,
        "D2_four_coset_coverage_pass":coverage_pass,
    }
    main_gate["passed"]=bool(
        matrix_pass and validation_pass and coverage_pass
        and not main_gate["truth_bits_decoder_visible"]
    )

    result={
        "stage":"S21-OPT3C1 direct d5->d13 practical decoder freeze and fresh finite-p validation",
        "status":"PASS" if main_gate["passed"] else "FAIL",
        "detector_contract":{
            "operational_bits":OP_DETECTORS,
            "future_truth_bits":TRUTH_BITS,
            "truth_bits_decoder_visible":False,
        },
        "resource_contract":{
            "predownstream_locations":PREDOWNSTREAM_LOCATIONS,
            "downstream_locations_per_patch":DOWNSTREAM_LOCATIONS_PER_PATCH,
            "two_patch_downstream_locations":2*DOWNSTREAM_LOCATIONS_PER_PATCH,
            "full_factory_potential_locations":FULL_FACTORY_POTENTIAL_LOCATIONS,
            "old_full_factory_potential_locations":7960,
            "potential_location_reduction_fraction":1-FULL_FACTORY_POTENTIAL_LOCATIONS/7960,
        },
        "frozen_decoder":preflight["frozen_decoder"],
        "matrix":matrix,
        "prior_calibrations":[x["meta"] for x in prior_rows],
        "decoder_audit":audits,
        "evaluation_rows":rows,
        "firewall_scan_seconds":fw_seconds,
        "main_gate":main_gate,
        "interpretation":{
            "paper_final_resource_update_allowed":bool(main_gate["passed"]),
            "A3_recalculated":False,
            "note":"This stage validates finite-p practical performance/resource after direct expansion. It does not recompute the S19 exact-three A3 coefficient.",
        },
        "next_if_pass":"S21-OPT4/S20R: recalculate direct-CS active-location and qubit-step resource table using measured OPT3C1 reach/acceptance; only run new A3 rare-event study if the manuscript requires a coefficient for the optimized schedule.",
    }
    dump(out/"s21opt3c1_summary.json",result)
    dump(out/"s21opt3c1_decoder_audit.json",audits)

    print(json.dumps({
        "status":result["status"],
        "matrix":matrix,
        "resource":result["resource_contract"],
        "rows":[{
            "p":r["p"],
            "acceptance":r["operational_acceptance"],
            "truth_closed":r["truth_closed_accepted"],
            "closed_harmful":r["closed_boundary_harmful"],
            "harmful_le2":r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"],
            "expected_locations_per_attempt":r["stage_level_expected_active_locations_per_attempt"],
            "locations_per_accept":r["stage_level_active_location_proxy_per_operational_accept"],
        } for r in rows],
        "main_gate":main_gate,
    },indent=2))
    print("S21-OPT3C1",result["status"])
    if not main_gate["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
