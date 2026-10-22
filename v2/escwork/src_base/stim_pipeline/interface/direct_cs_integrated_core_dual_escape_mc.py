#!/usr/bin/env python3
"""S8-D: integrated direct-|CS> check-core -> dual live-ZXZ escape Monte Carlo.

This is the first PERFORMANCE bridge with the old final ideal Steane projection
removed. It combines the exact 14-data-qubit six-check core trajectory, the
exact S8-A coherent Steane syndrome-sector measurement, and two independent
Hirano ZXZ + 2 faultable surface-QEC live-handoff blocks using the S8-C/S7-B4
171-bit operational selective policy.

This is NOT yet the full physical factory: the preserved core still models the
midpoint ED boundary by an ideal code-space projection, and separate physical
preparation/CAT/ED modules are not expanded inside this state-vector driver.
"""
from __future__ import annotations
import argparse, collections, importlib.util, json, math, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
SRC=ROOT/'src_current'

def _load(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot import {name} from {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

s8live=_load('s8d_s8live',SRC/'direct_cs_step8_live_boundary.py')
s8b=_load('s8d_s8b',HERE/'zxz_core_boundary_signature_map.py')
s8c=_load('s8d_s8c',HERE/'zxz_core_sector_escape_composability.py')
s7b5=_load('s8d_s7b5',HERE/'zxz_operational_handoff_mc.py')
m=s8live.m
ACTUAL_BITS=171; ACTUAL_MASK=(1<<ACTUAL_BITS)-1
CORE_LOCATIONS=len(m.locations)
if CORE_LOCATIONS!=322: raise AssertionError(CORE_LOCATIONS)

@dataclass
class Row:
    p:float;shots:int;seed:int
    core_live_accepted:int;core_live_acceptance:float
    integrated_accepted:int;integrated_acceptance:float
    conditional_escape_acceptance_given_core:float|None
    rejected_block_a:int;rejected_block_b:int;rejected_either_block:int
    resolved_accepted:int;unresolved_accepted:int;resolved_fraction_among_accepted:float|None
    resolved_logical_infidelity_sum:float
    conditional_logical_infidelity_lower_bound:float|None
    conditional_logical_infidelity_upper_bound:float|None
    resolved_shots_with_infidelity_gt_1e_10:int
    mean_core_fault_locations:float;mean_escape_fault_locations_a:float;mean_escape_fault_locations_b:float
    max_total_fault_locations:int
    core_zero_physical_fault_shots:int;core_zero_physical_fault_fastpath_accepted:int
    sampled_multisector_core_accepted:int;sampled_nonzero_sector_pair_after_core:int

class _RandomAdapter:
    def __init__(self,rng):self.rng=rng
    def random(self):return float(self.rng.random())

def _branch_vec(a,b):return m.cs*np.array([1,(-1)**b,(-1)**a,(-1)**(a^b)],complex)
def _target(rec):return _branch_vec(int(rec[0]),int(rec[1]))
def _valid_rec(a,b):c=a^b;return [a,b,c,a,b,c]
def _q(p):return (1-(1-2*p)**7)/2

def _fault_book(p,rng):
    k=int(rng.binomial(CORE_LOCATIONS,p));fb=[collections.defaultdict(list) for _ in m.sequence]
    if k==0:return fb,0
    for lid in np.asarray(rng.choice(CORE_LOCATIONS,size=k,replace=False),dtype=np.int64):
        ci,ck,j,gi=m.locations[int(lid)];oi=int(rng.integers(0,len(m.opt_cache[(ck,gi)])));fb[ci][j].append((gi,oi))
    return fb,k

def _sample_sector(C,rng):
    probs=np.sum(np.abs(C)**2,axis=(1,3));flat=probs.reshape(-1);total=float(flat.sum())
    if abs(total-1)>1e-9:raise AssertionError(total)
    idx=int(rng.choice(flat.size,p=flat/total));sa,sb=divmod(idx,64);pr=float(probs[sa,sb])
    v=C[sa,:,sb,:].reshape(4)/math.sqrt(pr)
    return sa,sb,v,int(np.count_nonzero(probs>1e-12))

def _sample_core(p,rng):
    fb,nf=_fault_book(p,rng);q=_q(p)
    if nf==0:
        u=float(rng.random())
        if u<5/8:a,b=0,0
        elif u<6/8:a,b=0,1
        elif u<7/8:a,b=1,0
        else:a,b=1,1
        rec=[int(x^(rng.random()<q)) for x in _valid_rec(a,b)]
        if not m.valid(rec):return False,rec,None,None,nf,1,True
        return True,rec,(0,0),_branch_vec(a,b),nf,1,True
    rr=_RandomAdapter(rng);st=m.psi0.copy();rec=[]
    for ci,ck in enumerate(m.sequence):
        st,meas=m.check(st,ck,fb[ci],rr,q);rec.append(int(meas))
        if ci==2:
            st,pr=m.proj(st)
            if rng.random()>=pr:return False,rec,None,None,nf,0,False
    if not m.valid(rec):return False,rec,None,None,nf,0,False
    C,psum,recon,_=s8live.exact_sector_tensor(st)
    if abs(psum-1)>1e-9 or recon>1e-9:raise AssertionError((psum,recon))
    sa,sb,v,sc=_sample_sector(C,rng);return True,rec,(sa,sb),v,nf,sc,False

def _words_int(arr):
    out=[]
    for row in np.asarray(arr,dtype=np.uint64):out.append(int(row[0])|(int(row[1])<<64)|(int(row[2])<<128)|(int(row[3])<<192))
    return out

def _sample_escape(p,rng,sigs,frames):
    n=len(sigs);k=int(rng.binomial(n,p))
    if k==0:return 0,0,0
    full=frame=0
    for lid in np.asarray(rng.choice(n,size=k,replace=False),dtype=np.int64):
        choices=sigs[int(lid)];ci=0 if len(choices)==1 else int(rng.integers(0,len(choices)))
        full^=int(choices[ci]);frame^=int(frames[int(lid)][ci])
    return full,frame,k

def _unique(flags):
    if flags is None:return None
    f=int(flags)
    return f.bit_length()-1 if f and f.bit_count()==1 else None

def _apply(v,fa,fb):
    mats=(m.I2,m.X,m.Z,m.Y);return np.kron(mats[int(fa)],mats[int(fb)])@np.asarray(v,complex)

def _bridge(hirano,signature_p):
    words,frames,clean_unsafe,_clean_state,meta=s7b5._build_exact_catalogue(hirano,signature_p)
    sectors,s8bs=s8c._sector_rows(hirano,signature_p)
    paired,clean_analysis,_clean_policy,_=s8c._escape_catalogue(hirano,signature_p)
    table,_,_=s8c._event_source_table(sectors,paired,clean_analysis);pol=s8c._policy_from_table(table)
    unsafe=set(pol['unsafe_actual'])
    if unsafe!=set(clean_unsafe):raise AssertionError(('S8-C unsafe table changed',len(unsafe),len(clean_unsafe)))
    if not pol['boundary_operational_certified_policy_exists'] or 0 in unsafe:raise AssertionError('S8-C gate failed')
    sm={int(s.syndrome):s for s in sectors}
    return {'sigs':[_words_int(x) for x in words],'frames':frames,'unsafe':unsafe,'table':{int(k):int(v) for k,v in table.items()},'sector':sm,'meta':meta,'s8b':s8bs}

def _run(p,shots,seed,b):
    rng=np.random.default_rng(seed);core_acc=int_acc=rej_a=rej_b=rej_either=0;resolved=unresolved=0;inf_sum=0.;inf_nz=0
    core_nf_sum=ea_sum=eb_sum=0;max_total=0;core_zero=core_zero_acc=multisector=nonzero_sector=0
    for _ in range(shots):
        ok,rec,ss,lv,nfc,sc,fast=_sample_core(p,rng);core_nf_sum+=nfc;core_zero+=int(nfc==0)
        if not ok:continue
        core_acc+=1;core_zero_acc+=int(nfc==0);multisector+=int(sc>1);sa,sb=ss;nonzero_sector+=int(sa!=0 or sb!=0)
        af,afr,nfa=_sample_escape(p,rng,b['sigs'],b['frames']);bf,bfr,nfb=_sample_escape(p,rng,b['sigs'],b['frames'])
        ea_sum+=nfa;eb_sum+=nfb;max_total=max(max_total,nfc+nfa+nfb)
        sra=b['sector'][sa];srb=b['sector'][sb];fa=int(sra.full)^af;fb=int(srb.full)^bf;na=fa&ACTUAL_MASK;nb=fb&ACTUAL_MASK
        ra=na in b['unsafe'];rb=nb in b['unsafe'];rej_a+=int(ra);rej_b+=int(rb);rej_either+=int(ra or rb)
        if ra or rb:continue
        int_acc+=1;pa=_unique(b['table'].get(fa));pb=_unique(b['table'].get(fb))
        if pa is None or pb is None:unresolved+=1;continue
        resolved+=1;aa=int(sra.frame)^afr;ab=int(srb.frame)^bfr;corr=_apply(lv,pa^aa,pb^ab);fid=float(abs(np.vdot(_target(rec),corr))**2);fid=min(1.,max(0.,fid));inf=1-fid;inf_sum+=inf;inf_nz+=int(inf>1e-10)
    return Row(p,shots,seed,core_acc,core_acc/shots,int_acc,int_acc/shots,(int_acc/core_acc if core_acc else None),rej_a,rej_b,rej_either,resolved,unresolved,(resolved/int_acc if int_acc else None),inf_sum,(inf_sum/int_acc if int_acc else None),((inf_sum+unresolved)/int_acc if int_acc else None),inf_nz,core_nf_sum/shots,ea_sum/max(core_acc,1),eb_sum/max(core_acc,1),max_total,core_zero,core_zero_acc,multisector,nonzero_sector)

def _selftest():
    v=_branch_vec(0,0)
    for f in range(4):
        o=_apply(_apply(v,f,0),f,0)
        if 1-abs(np.vdot(v,o))**2>1e-10:raise AssertionError(f)
    rng=np.random.default_rng(1234);cnt=collections.Counter()
    for _ in range(2000):
        ok,rec,ss,lv,nf,sc,fast=_sample_core(0.,rng)
        if not(ok and ss==(0,0) and nf==0 and sc==1 and fast):raise AssertionError((ok,ss,nf,sc,fast))
        if 1-abs(np.vdot(_target(rec),lv))**2>1e-10:raise AssertionError(rec)
        cnt[(rec[0],rec[1])]+=1
    return {'fastpath_p0_pass':True,'sampled_branch_counts_2000':{str(k):v for k,v in cnt.items()}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hirano-src',type=Path,required=True);ap.add_argument('--p-values',type=float,nargs='+',default=[2e-4,5e-4,1e-3,2e-3,5e-3]);ap.add_argument('--shots',type=int,default=5000);ap.add_argument('--seed',type=int,default=20260820);ap.add_argument('--signature-p',type=float,default=1e-3);ap.add_argument('--self-test',action='store_true');ap.add_argument('--out-dir',type=Path,default=HERE/'s8d_results');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    st=_selftest();(a.out_dir/'s8d_self_test.json').write_text(json.dumps(st,indent=2),encoding='utf-8')
    if a.self_test:print(json.dumps(st,indent=2));print('S8-D SELF-TEST PASS');return
    b=_bridge(a.hirano_src.resolve(),a.signature_p);meta={'stage':'S8-D','scope':'integrated six-check core + dual k=2 live ZXZ escape; not yet full physical factory','core_check_locations':CORE_LOCATIONS,'escape_locations_per_block':int(b['meta']['physical_locations']),'operational_detector_bits_per_block':ACTUAL_BITS,'unsafe_actual_syndrome_count':len(b['unsafe']),'s8b_actual_signature_injective':bool(b['s8b']['actual_signature_injective']),'final_ideal_core_projection_removed':True,'midpoint_ideal_projection_retained':True,'physical_preparation_cat_midED_modules_not_expanded_here':True};(a.out_dir/'s8d_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    rows=[]
    for i,p in enumerate(a.p_values):r=_run(float(p),a.shots,a.seed+i,b);rows.append(asdict(r));print(json.dumps(asdict(r),indent=2),flush=True)
    (a.out_dir/'s8d_sweep.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print('S8-D PASS: integrated live core -> dual operational ZXZ Monte Carlo completed')
if __name__=='__main__':main()
