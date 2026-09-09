#!/usr/bin/env python3
"""S21-OPT3C0: split operational-vs-future-truth detectors and build exact <=2 firewall.

Why this stage exists
---------------------
S21-OPT3B proved a strong structural closure for the live direct d5->d13 splice,
but its exact zero-record join used all 431 detector bits. Direct inspection of
the generated clean circuits shows that these split as

    263 operational detectors
    168 detectors created by the ideal stationary follow-up round

The latter are future/truth closure bits and must NOT be decoder-visible.
Therefore OPT3C0 reinterprets the already-generated OPT3B circuits exactly as
S18-M2/B2R did:

    operational record = first 263 detector bits
    scoring-only truth = final 168 detector bits

It then rebuilds the exact total-fault-order <=2 firewall over the 263-bit
operational record, with the 168 truth bits used only to decide whether the
delivered boundary is closed.

No finite-p Monte Carlo is performed here.

Exact low-order universe
------------------------
Total order <=2 consists of:
  * canonical incoming boundary sector weight 0/1/2;
  * zero or one downstream physical location for weight-1 boundary sectors;
  * zero, one, or two DISTINCT downstream physical locations for weight-0;
  * no downstream physical fault for weight-2 sectors.

For every truth-closed realization, the firewall records the set of possible
logical Paulis for its 263-bit operational record:
  unique Pauli -> exact
  >1 Pauli     -> ambiguous_reject
  no <=2 realization -> higher_order_only at query time

The theorem gate is:
  zero operational record + truth closed => identity Pauli for every <=2 event.

Outputs
-------
  s21opt3c0_summary.json
  s21opt3c0_firewall.jsonl.gz
  s21opt3c0_sector_map_263_truth.json
  s21opt3c0_preflight.json
  s21opt3c0_self_test.json

The compressed firewall is intended for S21-OPT3C finite-p validation.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAULI_NAMES = ("I","X","Z","Y")
EXPECTED_TOTAL_DETECTORS = 431
EXPECTED_TRUTH_BITS = 168
EXPECTED_OP_DETECTORS = 263
EXPECTED_LOCATIONS = 2132


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def imp(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m


def find(root: Path, names: list[str]) -> Path:
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
    hits=list(dict.fromkeys(x.resolve() for x in hits))
    hits.sort(key=lambda p:(0 if "stim_pipeline" in str(p) else 1,len(str(p)),str(p)))
    return hits[0]


def detector_count_in_instruction_range(path: Path, a: int, b: int) -> int:
    lines=path.read_text(encoding="utf-8").splitlines()
    return sum(1 for s in lines[a:b] if s.lstrip().startswith("DETECTOR"))


def derive_partition(clean_x: Path, clean_z: Path, summary: dict) -> dict[str,Any]:
    vals={}
    for ch,path in (("X",clean_x),("Z",clean_z)):
        sp=summary["schedule"][ch]["splice"]
        end=int(sp["expansion_faultable_instruction_end"])
        follow=int(sp["ideal_followup_instruction_end"])
        lines=path.read_text(encoding="utf-8").splitlines()
        total=sum(1 for s in lines if s.lstrip().startswith("DETECTOR"))
        op=detector_count_in_instruction_range(path,0,end)
        truth=detector_count_in_instruction_range(path,end,follow)
        tail=detector_count_in_instruction_range(path,follow,len(lines))
        vals[ch]={"total":total,"operational":op,"truth":truth,"tail":tail,
                  "faultable_instruction_end":end,"ideal_followup_instruction_end":follow}
    if vals["X"] != vals["Z"]:
        raise RuntimeError(("X/Z partition mismatch",vals))
    v=vals["X"]
    if (v["total"],v["operational"],v["truth"],v["tail"]) != (
        EXPECTED_TOTAL_DETECTORS,EXPECTED_OP_DETECTORS,EXPECTED_TRUTH_BITS,0
    ):
        raise RuntimeError(("unexpected detector partition",vals))
    return vals


@dataclass
class Effect:
    op: int
    truth: int
    pauli: int
    locations: set[int]
    example: Any


def distinct_exists(a: Effect, b: Effect) -> bool:
    if a is b:
        return len(a.locations) >= 2
    if len(a.locations) == 1 and len(b.locations) == 1:
        return next(iter(a.locations)) != next(iter(b.locations))
    return any(x != y for x in a.locations for y in b.locations)


def distinct_witness(a: Effect, b: Effect):
    for x in sorted(a.locations):
        for y in sorted(b.locations):
            if x != y:
                return int(x),int(y)
    return None


def build_model(cert, xnoisy: Path, znoisy: Path, op_bits: int, truth_bits: int):
    import stim
    xc=stim.Circuit(xnoisy.read_text(encoding="utf-8"))
    zc=stim.Circuit(znoisy.read_text(encoding="utf-8"))
    if int(xc.num_detectors)!=op_bits+truth_bits or int(zc.num_detectors)!=op_bits+truth_bits:
        raise RuntimeError((xc.num_detectors,zc.num_detectors,op_bits,truth_bits))

    cert.EXPECTED_LOCATIONS=EXPECTED_LOCATIONS
    cert.EXPECTED_DETECTORS=op_bits+truth_bits

    # Stage labels only affect diagnostics.
    cert._stage_from_tick=lambda tick: (
        "hirano_escape_plus_two_d5_rounds" if int(tick)<44 else "direct_d5_to_d13"
    )

    xl,xnative,xmeta=cert._noise_locations_from_text(xnoisy)
    zl,znative,zmeta=cert._noise_locations_from_text(znoisy)
    if len(xl)!=EXPECTED_LOCATIONS or len(zl)!=EXPECTED_LOCATIONS:
        raise RuntimeError(("location count",len(xl),len(zl)))
    xs=[(xmeta[i]["tick"],xmeta[i]["gate"],tuple(xmeta[i]["qubits"])) for i in xl]
    zs=[(zmeta[i]["tick"],zmeta[i]["gate"],tuple(zmeta[i]["qubits"])) for i in zl]
    if xs!=zs:
        raise RuntimeError("X/Z stochastic schedules differ")
    cert._LOCATION_META.clear()
    cert._LOCATION_META.update(xmeta)

    print("S21-OPT3C0: recovering X mechanisms...",flush=True)
    xp,xr=cert._recover_mechanism_map(xc,cert._raw_dem(xc),"X",xnative)
    print("S21-OPT3C0: recovering Z mechanisms...",flush=True)
    zp,zr=cert._recover_mechanism_map(zc,cert._raw_dem(zc),"Z",znative)
    universe=cert._all_mechanism_keys(xl)

    mask_op=(1<<op_bits)-1
    mask_truth=(1<<truth_bits)-1
    groups={}
    xop={}
    zop={}
    loc_weight=collections.defaultdict(float)
    mismatch=[]
    for m in sorted(universe,key=str):
        mx,lx=xp.get(m,(0,0))
        mz,lz=zp.get(m,(0,0))
        opx=int(mx)&mask_op
        opz=int(mz)&mask_op
        tx=(int(mx)>>op_bits)&mask_truth
        tz=(int(mz)>>op_bits)&mask_truth
        if opx!=opz or tx!=tz:
            if len(mismatch)<20:
                mismatch.append({
                    "mechanism":cert._mechanism_json(m),
                    "opX":hex(opx),"opZ":hex(opz),
                    "truthX":hex(tx),"truthZ":hex(tz),
                })
            continue
        pauli=int(lz)|(int(lx)<<1)
        loc=int(m[0])
        key=(opx,tx,pauli)
        if key not in groups:
            groups[key]=Effect(opx,tx,pauli,set(),m)
        groups[key].locations.add(loc)
        xop[m]=(opx,int(lx))
        zop[m]=(opz,int(lz))
        loc_weight[loc]+=float(cert._mechanism_weight(cert._location_gate(loc)))

    if mismatch:
        raise RuntimeError(("probe-dependent operational/truth syndrome",mismatch))
    bad=[(i,w) for i,w in sorted(loc_weight.items()) if abs(w-1.0)>1e-10]
    if bad or len(loc_weight)!=EXPECTED_LOCATIONS:
        raise RuntimeError(("mechanism weight/location coverage",len(loc_weight),bad[:20]))

    # Decoder effect classes: operational syndrome + logical Pauli only.
    effects,emeta=cert._build_joint_effects(xop,zop)
    return list(groups.values()),effects,{
        "X_recovery":xr,"Z_recovery":zr,
        "concrete_mechanisms":len(universe),
        "truth_aware_effect_classes":len(groups),
        "decoder_joint_effect_classes":len(effects),
        "decoder_joint_effect_meta":emeta,
        "physical_locations":EXPECTED_LOCATIONS,
    }


def sector_split(summary: dict, op_bits: int, truth_bits: int):
    mask_op=(1<<op_bits)-1
    mask_truth=(1<<truth_bits)-1
    rows=[]
    for r in summary["sector_map"]["rows"]:
        full=int(r["detector_638_hex"],16)
        op=full&mask_op
        truth=(full>>op_bits)&mask_truth
        q={
            **r,
            "detector_263_hex":hex(op),
            "detector_263_weight":int(op.bit_count()),
            "future_truth_168_hex":hex(truth),
            "future_truth_168_weight":int(truth.bit_count()),
            # D2 legacy alias used by unchanged ranking implementation.
            "detector_470_hex":hex(op),
        }
        rows.append(q)
    if len(rows)!=64:
        raise RuntimeError(len(rows))
    return rows,{
        "sector_count":64,
        "canonical_weight_histogram":dict(collections.Counter(str(int(r["canonical_weight"])) for r in rows)),
        "unique_operational_signatures":len({r["detector_263_hex"] for r in rows}),
        "unique_full_431_signatures":len({r["detector_638_hex"] for r in rows}),
        "nonzero_future_truth_sectors":sum(int(r["future_truth_168_hex"],16)!=0 for r in rows),
    }


def add_mask(index: dict[int,int], op: int, pauli: int):
    index[int(op)] = int(index.get(int(op),0)) | (1<<int(pauli))


def build_firewall(cert, sectors: list[dict], effects: list[Effect], max_examples: int):
    """Enumerate all truth-closed total-order <=2 realizations by effect-class joins."""
    idx={}
    counts=collections.Counter()
    zero_bad=[]
    examples=[]

    by_truth=collections.defaultdict(list)
    for e in effects:
        by_truth[e.truth].append(e)

    # Boundary alone: weights 0,1,2.
    for r in sectors:
        bw=int(r["canonical_weight"])
        if bw>2: continue
        op=int(r["detector_263_hex"],16)
        truth=int(r["future_truth_168_hex"],16)
        pa=int(r["raw_terminal_pauli_int"])
        if truth==0:
            add_mask(idx,op,pa)
            counts[f"boundary_w{bw}_alone_truthclosed"]+=1
            if op==0 and pa!=0 and len(zero_bad)<max_examples:
                zero_bad.append({"kind":"boundary_alone","weight":bw,"sector":int(r["steane_syndrome"]),
                                 "pauli":PAULI_NAMES[pa]})

    # Boundary weight 0 + physical 1, and boundary weight 1 + physical 1.
    for r in sectors:
        bw=int(r["canonical_weight"])
        if bw not in (0,1): continue
        so=int(r["detector_263_hex"],16)
        st=int(r["future_truth_168_hex"],16)
        sp=int(r["raw_terminal_pauli_int"])
        for e in by_truth.get(st,()):
            op=so^e.op
            pa=sp^e.pauli
            add_mask(idx,op,pa)
            counts[f"boundary_w{bw}_plus_phys1_truthclosed_effectclasses"]+=1
            if op==0 and pa!=0 and len(zero_bad)<max_examples:
                zero_bad.append({
                    "kind":f"boundary_w{bw}_plus_phys1",
                    "sector":int(r["steane_syndrome"]),
                    "pauli":PAULI_NAMES[pa],
                    "physical_mechanism":cert._mechanism_json(e.example),
                })

    # Boundary weight 0 is the identity sector; physical1+physical2 must have
    # equal truth masks to be truth closed.
    zero_sector=[r for r in sectors if int(r["canonical_weight"])==0]
    if len(zero_sector)!=1:
        raise RuntimeError("no unique weight-0 sector")
    z=zero_sector[0]
    z_op=int(z["detector_263_hex"],16)
    z_truth=int(z["future_truth_168_hex"],16)
    z_pa=int(z["raw_terminal_pauli_int"])
    if z_truth!=0 or z_op!=0 or z_pa!=0:
        raise RuntimeError(("weight0 sector not identity",z))

    truth_groups=sorted(by_truth.items(),key=lambda kv:(len(kv[1]),kv[0]))
    pair_class_tests=0
    pair_distinct_classes=0
    for gi,(truth,es) in enumerate(truth_groups):
        m=len(es)
        # Same-class pairs are possible only if class spans >=2 locations.
        for i,a in enumerate(es):
            if len(a.locations)>=2:
                pair_class_tests+=1
                add_mask(idx,0,0)  # same class XOR itself
                counts["phys2_same_effectclass_distinct_location"]+=1
            for j in range(i+1,m):
                b=es[j]
                pair_class_tests+=1
                if not distinct_exists(a,b):
                    continue
                pair_distinct_classes+=1
                op=a.op^b.op
                pa=a.pauli^b.pauli
                add_mask(idx,op,pa)
                counts["phys2_distinct_effectclasses_truthclosed"]+=1
                if op==0 and pa!=0 and len(zero_bad)<max_examples:
                    w=distinct_witness(a,b)
                    zero_bad.append({
                        "kind":"physical2",
                        "truth_hex":hex(int(truth)),
                        "pauli":PAULI_NAMES[pa],
                        "locations":list(w) if w else None,
                        "mechanism_A":cert._mechanism_json(a.example),
                        "mechanism_B":cert._mechanism_json(b.example),
                    })
        if (gi+1)%100==0 or gi+1==len(truth_groups):
            print(f"  truth groups joined {gi+1}/{len(truth_groups)}; firewall records={len(idx)}",flush=True)

    hist=collections.Counter()
    exact=amb=0
    for mask in idx.values():
        n=int(mask).bit_count()
        hist[n]+=1
        if n==1: exact+=1
        else: amb+=1

    zero_mask=int(idx.get(0,0))
    zero_paulis=[PAULI_NAMES[p] for p in range(4) if zero_mask&(1<<p)]
    passed=(zero_mask==(1<<0) and not zero_bad)
    return idx,{
        "truth_group_count":len(truth_groups),
        "physical_pair_effect_class_tests":pair_class_tests,
        "physical_pair_distinct_effect_class_joins":pair_distinct_classes,
        "firewall_record_count":len(idx),
        "exact_record_count":exact,
        "ambiguous_record_count":amb,
        "pauli_multiplicity_histogram":{str(k):int(v) for k,v in sorted(hist.items())},
        "zero_record_paulis":zero_paulis,
        "zero_record_identity_only":zero_mask==(1<<0),
        "zero_record_harmful_examples":zero_bad,
        "zero_record_low_order_gate_pass":passed,
        "realization_counts":dict(sorted(counts.items())),
    }


def write_firewall(path: Path, idx: dict[int,int]):
    with gzip.open(path,"wt",encoding="utf-8") as f:
        for op,mask in sorted(idx.items()):
            paulis=[p for p in range(4) if mask&(1<<p)]
            status="exact" if len(paulis)==1 else "ambiguous_reject"
            obj={
                "record":hex(int(op)),
                "status":status,
                "pauli_int":paulis[0] if len(paulis)==1 else None,
                "pauli":PAULI_NAMES[paulis[0]] if len(paulis)==1 else None,
                "possible_paulis":[PAULI_NAMES[p] for p in paulis],
            }
            f.write(json.dumps(obj,sort_keys=True)+"\n")


def matrix_audit(root: Path, effects: list[Any], op_bits: int):
    d1p=find(root,["direct_cs_s17c_d_hybrid_bposd_abstention.py"])
    d2p=find(root,["direct_cs_s17c_d2_logical_coset_list_decoder.py"])
    d1=imp("s21opt3c0_d1",d1p)
    d2=imp("s21opt3c0_d2",d2p)
    d1.EXPECTED_DETECTORS=op_bits
    d2.EXPECTED_DETECTORS=op_bits
    cols=d1._effect_columns(effects)
    h=d1._build_sparse_check_matrix(cols)
    h_aug=d2._build_augmented_matrix(h,cols)
    rh=d2._gf2_row_rank_sparse(h)
    ra=d2._gf2_row_rank_sparse(h_aug)
    return {
        "H_shape":list(map(int,h.shape)),
        "H_aug_shape":list(map(int,h_aug.shape)),
        "H_rank":int(rh),
        "H_aug_rank":int(ra),
        "logical_rows_independent":bool(ra>=rh+2),
        "effect_columns":len(cols),
    }


def self_test():
    idx={}
    add_mask(idx,3,0); add_mask(idx,3,2)
    a=Effect(1,5,1,{1},("a",))
    b=Effect(1,5,2,{2},("b",))
    c=Effect(1,5,1,{1},("c",))
    return {
        "passed": idx[3].bit_count()==2 and distinct_exists(a,b) and not distinct_exists(a,c),
        "multi_pauli_record_becomes_ambiguous":True,
        "distinct_location_requirement_enforced":True,
        "truth_bits_decoder_visible":False,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--search-root",type=Path,default=Path("."))
    ap.add_argument("--opt3b-result-dir",type=Path,
                    default=Path("stim_pipeline/interface/s21opt3b_results"))
    ap.add_argument("--out-dir",type=Path,
                    default=Path("stim_pipeline/interface/s21opt3c0_results"))
    ap.add_argument("--max-examples",type=int,default=20)
    ap.add_argument("--preflight-only",action="store_true")
    ap.add_argument("--self-test-only",action="store_true")
    args=ap.parse_args()

    if args.self_test_only:
        r=self_test(); print(json.dumps(r,indent=2))
        if not r["passed"]: raise RuntimeError("self test fail")
        return

    root=args.search_root.resolve()
    rdir=args.opt3b_result_dir
    if not rdir.is_absolute(): rdir=(root/rdir).resolve()
    out=args.out_dir
    if not out.is_absolute(): out=(root/out).resolve()
    out.mkdir(parents=True,exist_ok=True)

    summary_p=rdir/"s21opt3b_summary.json"
    xc=rdir/"s21opt3b_X_direct_d5_d13_clean.stim"
    zc=rdir/"s21opt3b_Z_direct_d5_d13_clean.stim"
    xn=rdir/"s21opt3b_X_direct_d5_d13_noisy.stim"
    zn=rdir/"s21opt3b_Z_direct_d5_d13_noisy.stim"
    for p in (summary_p,xc,zc,xn,zn):
        if not p.exists(): raise FileNotFoundError(p)

    s=load(summary_p)
    if s.get("status")!="PASS" or not bool(s.get("main_gate",{}).get("passed",False)):
        raise RuntimeError("OPT3B is not PASS")
    if int(s["schedule"]["physical_locations"])!=EXPECTED_LOCATIONS:
        raise RuntimeError(("OPT3B location drift",s["schedule"]["physical_locations"]))

    partition=derive_partition(xc,zc,s)
    certp=find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"])
    preflight={
        "stage":"S21-OPT3C0 preflight",
        "passed":True,
        "opt3b_status":"PASS",
        "detector_partition":partition,
        "decoder_input_bits":EXPECTED_OP_DETECTORS,
        "future_truth_bits_decoder_visible":False,
        "physical_locations_per_patch":EXPECTED_LOCATIONS,
        "certificate":str(certp),
    }
    dump(out/"s21opt3c0_preflight.json",preflight)
    if args.preflight_only:
        print(json.dumps(preflight,indent=2))
        print("S21-OPT3C0 preflight PASS")
        return

    cert=imp("s21opt3c0_cert",certp)
    groups,decoder_effects,mmeta=build_model(
        cert,xn,zn,EXPECTED_OP_DETECTORS,EXPECTED_TRUTH_BITS
    )
    sectors,smeta=sector_split(s,EXPECTED_OP_DETECTORS,EXPECTED_TRUTH_BITS)
    dump(out/"s21opt3c0_sector_map_263_truth.json",sectors)

    print("S21-OPT3C0: building exact <=2 263-bit firewall...",flush=True)
    t0=time.time()
    fw,fmeta=build_firewall(cert,sectors,groups,int(args.max_examples))
    fmeta["build_seconds"]=time.time()-t0
    fwpath=out/"s21opt3c0_firewall.jsonl.gz"
    write_firewall(fwpath,fw)

    print("S21-OPT3C0: auditing 263-bit D2 matrix rank...",flush=True)
    matrix=matrix_audit(root,decoder_effects,EXPECTED_OP_DETECTORS)

    gate={
        "opt3b_prior_structural_closure_pass":True,
        "detector_partition_exact_263_plus_168":True,
        "truth_bits_decoder_visible":False,
        "X_Z_operational_and_truth_masks_mechanismwise_identical":True,
        "zero_operational_record_truth_closed_identity_only_through_total_order2":
            bool(fmeta["zero_record_low_order_gate_pass"]),
        "logical_rows_independent":bool(matrix["logical_rows_independent"]),
    }
    gate["passed"]=all([
        gate["opt3b_prior_structural_closure_pass"],
        gate["detector_partition_exact_263_plus_168"],
        not gate["truth_bits_decoder_visible"],
        gate["X_Z_operational_and_truth_masks_mechanismwise_identical"],
        gate["zero_operational_record_truth_closed_identity_only_through_total_order2"],
        gate["logical_rows_independent"],
    ])

    result={
        "stage":"S21-OPT3C0 direct d5->d13 operational/truth split and exact <=2 firewall",
        "status":"PASS" if gate["passed"] else "FAIL",
        "detector_contract":{
            "total_detectors_per_probe":EXPECTED_TOTAL_DETECTORS,
            "operational_decoder_bits":EXPECTED_OP_DETECTORS,
            "future_truth_scoring_bits":EXPECTED_TRUTH_BITS,
            "truth_bits_decoder_visible":False,
            "partition_source":"instruction boundary before ideal stationary follow-up in OPT3B clean circuits",
        },
        "physical_model":mmeta,
        "sector_map_meta":smeta,
        "firewall":fmeta,
        "matrix":matrix,
        "main_gate":gate,
        "resource_context":{
            "old_downstream_locations_per_patch":3535,
            "new_downstream_locations_per_patch":EXPECTED_LOCATIONS,
            "per_patch_location_saving":3535-EXPECTED_LOCATIONS,
            "two_patch_location_saving":2*(3535-EXPECTED_LOCATIONS),
            "old_full_factory_potential_locations":7960,
            "new_full_factory_potential_locations":890+2*EXPECTED_LOCATIONS,
        },
        "next_if_pass":"S21-OPT3C: freeze a 263-bit practical BP+OSD decoder using the exact <=2 firewall, then run fresh finite-p validation at p=5e-4 and 1e-3 before updating S20.",
        "stop_if_fail":"Do not run finite-p validation. Inspect zero-record truth-closed witnesses or decoder-matrix rank failure.",
    }
    dump(out/"s21opt3c0_summary.json",result)
    dump(out/"s21opt3c0_self_test.json",self_test())

    print(json.dumps({
        "status":result["status"],
        "detectors":result["detector_contract"],
        "locations":result["resource_context"],
        "firewall":{
            "records":fmeta["firewall_record_count"],
            "exact":fmeta["exact_record_count"],
            "ambiguous":fmeta["ambiguous_record_count"],
            "zero_record_paulis":fmeta["zero_record_paulis"],
        },
        "matrix":matrix,
        "main_gate":gate,
    },indent=2))
    print("S21-OPT3C0",result["status"])
    if not gate["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
