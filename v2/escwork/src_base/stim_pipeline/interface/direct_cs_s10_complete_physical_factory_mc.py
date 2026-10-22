#!/usr/bin/env python3
"""S10-C: complete physical-front-end direct-|CS> factory Monte Carlo.

Physicalized modules in this driver:
  * two encoded |+> Steane preparations: 7 PREP + 8-CNOT encoder + 82-loc initial ED,
  * six verified CAT7 ancillas: 28-loc prep/verification each,
  * explicit seven physical CAT X-readout locations per logical check,
  * 322-location non-Clifford six-check core,
  * two explicit 82-location midpoint Steane EDs,
  * two k=2 live ZXZ escape/handoff blocks (1028 locations each).

The CAT preparation/verification circuit is Clifford, so under the active-
location stochastic Pauli model its exact postselected action is represented
by the propagated CAT Pauli frame (x-mask, phase parity). This is not an ideal
CAT abstraction: all 28+7 physical CAT locations are sampled explicitly.
"""
from __future__ import annotations
import argparse, collections, importlib.util, json, math, sys
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;SRC=ROOT/'src_current'

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError((name,path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

s10=_load('s10mc_front',SRC/'direct_cs_step10_physical_frontend.py')
s8live=s10.s8live;s9ed=s10.s9ed;m=s10.m
ACTUAL_BITS=171;ACTUAL_MASK=(1<<ACTUAL_BITS)-1
CORE_LOCATIONS=len(m.locations);MIDED_LOCATIONS=len(s9ed.ED_OPS)
FRONT_LOCATIONS_PER_BLOCK=s10.ENCODER_LOCATION_COUNT+MIDED_LOCATIONS
CAT_PREP_LOCATIONS=s10.CAT7_LOCATION_COUNT;CAT_READOUT_LOCATIONS=s10.CAT_READOUT_LOCATIONS
if (CORE_LOCATIONS,MIDED_LOCATIONS,FRONT_LOCATIONS_PER_BLOCK,CAT_PREP_LOCATIONS,CAT_READOUT_LOCATIONS)!=(322,82,97,28,7):raise AssertionError('inventory')

@dataclass
class Row:
    p:float;shots:int;seed:int
    frontend_accepted:int;frontend_acceptance:float
    core_midED_accepted:int;core_midED_acceptance_given_frontend:float|None
    integrated_accepted:int;integrated_acceptance:float
    conditional_escape_acceptance_given_core_midED:float|None
    resolved_accepted:int;unresolved_accepted:int;resolved_fraction_among_accepted:float|None
    resolved_logical_infidelity_sum:float;conditional_logical_infidelity_lower_bound:float|None;conditional_logical_infidelity_upper_bound:float|None
    resolved_shots_with_infidelity_gt_1e_10:int
    mean_frontend_fault_locations_a:float;mean_frontend_fault_locations_b:float
    mean_CAT_prepverify_fault_locations:float;mean_CAT_readout_fault_locations:float
    mean_core_check_fault_locations:float;mean_midED_fault_locations_a:float;mean_midED_fault_locations_b:float
    mean_escape_fault_locations_a:float;mean_escape_fault_locations_b:float
    max_total_modeled_fault_locations:int
    fastpath_core_shots:int;sampled_multisector_final_core_accepted:int;sampled_nonzero_sector_pair_after_core:int

class _RandomAdapter:
    def __init__(self,rng):self.rng=rng
    def random(self):return float(self.rng.random())

def _branch_vec(a,b):return m.cs*np.array([1,(-1)**b,(-1)**a,(-1)**(a^b)],complex)
def _target(rec):return _branch_vec(int(rec[0]),int(rec[1]))
def _valid_rec(a,b):c=a^b;return [a,b,c,a,b,c]

def _fault_book(p,rng):
    k=int(rng.binomial(CORE_LOCATIONS,p));fb=[collections.defaultdict(list) for _ in m.sequence]
    if k==0:return fb,0
    for lid in np.asarray(rng.choice(CORE_LOCATIONS,size=k,replace=False),dtype=np.int64):
        ci,ck,j,gi=m.locations[int(lid)];oi=int(rng.integers(0,len(m.opt_cache[(ck,gi)])));fb[ci][j].append((gi,oi))
    return fb,k

def _sample_midED(p,rng):
    k=int(rng.binomial(MIDED_LOCATIONS,p));f={}
    if k:
        for lid in np.asarray(rng.choice(MIDED_LOCATIONS,size=k,replace=False),dtype=np.int64):
            i=int(lid);f[i]=int(rng.integers(0,len(s9ed.ED_OPTIONS[i])))
    return f,k

def _sample_sector(C,rng):
    probs=np.sum(np.abs(C)**2,axis=(1,3));flat=probs.reshape(-1);total=float(flat.sum())
    if abs(total-1)>1e-9:raise AssertionError(total)
    idx=int(rng.choice(flat.size,p=flat/total));sa,sb=divmod(idx,64);pr=float(probs[sa,sb]);v=C[sa,:,sb,:].reshape(4)/math.sqrt(pr)
    return sa,sb,v,int(np.count_nonzero(probs>1e-12))

def _front_block(p,rng):
    r=s10.sample_block_frontend(p,rng)
    # Accepted branch probability is 0/1 for stabilizer input + Pauli faults, but retain exact sampling semantics.
    if r['accepted'] and r['accept_probability']<1-1e-12:
        if float(rng.random())>=r['accept_probability']:r['accepted']=False
    return r

def _sample_cats(p,rng):
    cats=[];pv=ro=0
    for _ in range(6):
        ce,rows=s10.sample_cat_preverification(p,rng);rp,k=s10.sample_cat_x_readout(p,rng);pv+=len(rows);ro+=k;cats.append((ce,rp))
    return cats,pv,ro

def _is_plus_block(r):
    if not r['accepted'] or int(r['syndrome'])!=0:return False
    v=np.asarray(r['logical_vector'],complex);q=np.array([1,1],complex)/math.sqrt(2)
    return 1-float(abs(np.vdot(q,v))**2)<1e-12

def _sample_core_fullfront(p,rng,fa,fb):
    cats,ncat,nro=_sample_cats(p,rng);fbk,nfc=_fault_book(p,rng);eda,neda=_sample_midED(p,rng);edb,nedb=_sample_midED(p,rng)
    for ci,(ce,_rp) in enumerate(cats):
        if not ce.accepted:return False,f'CAT{ci}',[],None,None,nfc,neda,nedb,ncat,nro,0,False

    # Exact analytic path whenever the quantum input is ideal, CAT x masks are zero,
    # core gates are fault-free, and midpoint ED is fault-free. Physical CAT phase/readout
    # faults remain explicit record flips in this path.
    fast=(_is_plus_block(fa) and _is_plus_block(fb) and nfc==0 and neda==0 and nedb==0 and all(ce.x_mask==0 for ce,_ in cats))
    if fast:
        u=float(rng.random())
        if u<5/8:a,b=0,0
        elif u<6/8:a,b=0,1
        elif u<7/8:a,b=1,0
        else:a,b=1,1
        rec=[]
        for ci,x in enumerate(_valid_rec(a,b)):
            ce,rp=cats[ci];rec.append(int(x)^int(ce.phase_flip)^int(rp))
        if not m.valid(rec):return False,'record',rec,None,None,nfc,neda,nedb,ncat,nro,1,True
        return True,'acc',rec,(0,0),_branch_vec(a,b),nfc,neda,nedb,ncat,nro,1,True

    va=s10.block_state_from_sector(int(fa['syndrome']),np.asarray(fa['logical_vector'],complex));vb=s10.block_state_from_sector(int(fb['syndrome']),np.asarray(fb['logical_vector'],complex));st=s10.pair_block_states(va,vb)
    rr=_RandomAdapter(rng);rec=[]
    for ci,ck in enumerate(m.sequence):
        ce,rp=cats[ci];st,meas=s10.check_with_physical_cat(st,ck,fbk[ci],ce,rp,rr);rec.append(int(meas))
        if ci==2:
            st,pr,_=s9ed.apply_two_block_midpoint_compact(st,eda,edb)
            if pr<=1e-20 or float(rng.random())>=pr:return False,'midED',rec,None,None,nfc,neda,nedb,ncat,nro,0,False
    if not m.valid(rec):return False,'record',rec,None,None,nfc,neda,nedb,ncat,nro,0,False
    C,psum,recon,_=s8live.exact_sector_tensor(st)
    if abs(psum-1)>1e-9 or recon>1e-9:raise AssertionError((psum,recon))
    sa,sb,v,sc=_sample_sector(C,rng)
    return True,'acc',rec,(sa,sb),v,nfc,neda,nedb,ncat,nro,sc,False

def _words_int(arr):
    return [int(row[0])|(int(row[1])<<64)|(int(row[2])<<128)|(int(row[3])<<192) for row in np.asarray(arr,dtype=np.uint64)]

def _sample_escape(p,rng,sigs,frames):
    n=len(sigs);k=int(rng.binomial(n,p));full=frame=0
    if k:
        for lid in np.asarray(rng.choice(n,size=k,replace=False),dtype=np.int64):
            choices=sigs[int(lid)];ci=0 if len(choices)==1 else int(rng.integers(0,len(choices)));full^=int(choices[ci]);frame^=int(frames[int(lid)][ci])
    return full,frame,k

def _unique(flags):
    if flags is None:return None
    f=int(flags);return f.bit_length()-1 if f and f.bit_count()==1 else None

def _apply(v,fa,fb):
    mats=(m.I2,m.X,m.Z,m.Y);return np.kron(mats[int(fa)],mats[int(fb)])@np.asarray(v,complex)

def _bridge(hirano,signature_p):
    s8c=_load('s10mc_s8c',HERE/'zxz_core_sector_escape_composability.py');s7b5=_load('s10mc_s7b5',HERE/'zxz_operational_handoff_mc.py')
    words,frames,clean_unsafe,_clean_state,meta=s7b5._build_exact_catalogue(hirano,signature_p);sectors,_=s8c._sector_rows(hirano,signature_p);paired,clean_analysis,_clean_policy,_=s8c._escape_catalogue(hirano,signature_p);table,_,_=s8c._event_source_table(sectors,paired,clean_analysis);pol=s8c._policy_from_table(table)
    unsafe=set(pol['unsafe_actual'])
    if unsafe!=set(clean_unsafe) or not pol['boundary_operational_certified_policy_exists'] or 0 in unsafe:raise AssertionError('S8-C policy gate')
    return {'sigs':[_words_int(x) for x in words],'frames':frames,'unsafe':unsafe,'table':{int(k):int(v) for k,v in table.items()},'sector':{int(s.syndrome):s for s in sectors},'meta':meta}

def _run(p,shots,seed,b):
    rng=np.random.default_rng(seed);front_acc=core_acc=int_acc=resolved=unresolved=0;inf_sum=0.;inf_nz=0;reasons=collections.Counter();sum_fa=sum_fb=sum_cat=sum_ro=sum_core=sum_ma=sum_mb=sum_ea=sum_eb=0;max_tot=0;fastn=multi=nonzero=0
    for _ in range(shots):
        fa=_front_block(p,rng);fb=_front_block(p,rng);nfa=fa['encoder_fault_count']+fa['initial_ed_fault_count'];nfb=fb['encoder_fault_count']+fb['initial_ed_fault_count'];sum_fa+=nfa;sum_fb+=nfb
        if not fa['accepted'] or not fb['accepted']:
            reasons['initialED']+=1;continue
        front_acc+=1
        ok,why,rec,ss,lv,nfc,nma,nmb,ncat,nro,sc,fast=_sample_core_fullfront(p,rng,fa,fb);sum_core+=nfc;sum_ma+=nma;sum_mb+=nmb;sum_cat+=ncat;sum_ro+=nro;fastn+=int(fast);reasons[why]+=1
        if not ok:continue
        core_acc+=1;multi+=int(sc>1);sa,sb=ss;nonzero+=int(sa!=0 or sb!=0)
        af,afr,nea=_sample_escape(p,rng,b['sigs'],b['frames']);bf,bfr,neb=_sample_escape(p,rng,b['sigs'],b['frames']);sum_ea+=nea;sum_eb+=neb;max_tot=max(max_tot,nfa+nfb+ncat+nro+nfc+nma+nmb+nea+neb)
        sra=b['sector'][sa];srb=b['sector'][sb];fulla=int(sra.full)^af;fullb=int(srb.full)^bf;na=fulla&ACTUAL_MASK;nb=fullb&ACTUAL_MASK
        if na in b['unsafe'] or nb in b['unsafe']:continue
        int_acc+=1;pa=_unique(b['table'].get(fulla));pb=_unique(b['table'].get(fullb))
        if pa is None or pb is None:unresolved+=1;continue
        resolved+=1;aa=int(sra.frame)^afr;ab=int(srb.frame)^bfr;corr=_apply(lv,pa^aa,pb^ab);fid=float(abs(np.vdot(_target(rec),corr))**2);fid=min(1.,max(0.,fid));inf=1-fid;inf_sum+=inf;inf_nz+=int(inf>1e-10)
    return asdict(Row(p,shots,seed,front_acc,front_acc/shots,core_acc,(core_acc/front_acc if front_acc else None),int_acc,int_acc/shots,(int_acc/core_acc if core_acc else None),resolved,unresolved,(resolved/int_acc if int_acc else None),inf_sum,(inf_sum/int_acc if int_acc else None),((inf_sum+unresolved)/int_acc if int_acc else None),inf_nz,sum_fa/shots,sum_fb/shots,sum_cat/max(front_acc,1),sum_ro/max(front_acc,1),sum_core/max(front_acc,1),sum_ma/max(front_acc,1),sum_mb/max(front_acc,1),sum_ea/max(core_acc,1),sum_eb/max(core_acc,1),max_tot,fastn,multi,nonzero))|{'rejection_reasons':dict(reasons)}

def _selftest():
    f=s10.self_test(include_cat_pairs=False);rng=np.random.default_rng(20260820)
    for _ in range(200):
        a=_front_block(0.,rng);b=_front_block(0.,rng)
        if not(_is_plus_block(a) and _is_plus_block(b)):raise AssertionError('p0 front')
        ok,why,rec,ss,lv,nfc,nma,nmb,ncat,nro,sc,fast=_sample_core_fullfront(0.,rng,a,b)
        if not(ok and why=='acc' and ss==(0,0) and nfc==nma==nmb==ncat==nro==0 and sc==1 and fast):raise AssertionError((ok,why,ss,nfc,nma,nmb,ncat,nro,sc,fast))
        if 1-float(abs(np.vdot(_target(rec),lv))**2)>1e-10:raise AssertionError(rec)
    return {'passed':True,'front_end_self_test':f,'p0_integrated_core_fastpath_pass':True,'modeled_locations_before_escape':2*FRONT_LOCATIONS_PER_BLOCK+6*(CAT_PREP_LOCATIONS+CAT_READOUT_LOCATIONS)+CORE_LOCATIONS+2*MIDED_LOCATIONS,'modeled_locations_with_dual_escape':2*FRONT_LOCATIONS_PER_BLOCK+6*(CAT_PREP_LOCATIONS+CAT_READOUT_LOCATIONS)+CORE_LOCATIONS+2*MIDED_LOCATIONS+2*1028,'logical_check_CAT_readout_parity_abstraction_retained':False,'ideal_initial_encoded_state_abstraction_retained':False,'ideal_midpoint_projection_retained':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hirano-src',type=Path);ap.add_argument('--p-values',type=float,nargs='+',default=[2e-4,5e-4,1e-3]);ap.add_argument('--shots',type=int,default=5000);ap.add_argument('--seed',type=int,default=20260820);ap.add_argument('--signature-p',type=float,default=1e-3);ap.add_argument('--self-test',action='store_true');ap.add_argument('--out-dir',type=Path,default=HERE/'s10_results');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    st=_selftest();(a.out_dir/'s10_self_test.json').write_text(json.dumps(st,indent=2),encoding='utf-8')
    if a.self_test:print(json.dumps(st,indent=2));print('S10 SELF-TEST PASS: complete physical front end integrated');return
    if a.hirano_src is None:raise SystemExit('--hirano-src required unless --self-test')
    b=_bridge(a.hirano_src.resolve(),a.signature_p);meta={'stage':'S10-C','scope':'physical encoded preparation + initial ED + six physical verified CAT7 gadgets/readouts + physical midpoint ED + dual k=2 live ZXZ','location_inventory':{'front_two_blocks':194,'six_CAT_prep_verification':168,'six_CAT_X_readout':42,'six_check_core':322,'midpoint_ED_two_blocks':164,'dual_escape':2*int(b['meta']['physical_locations']),'total':194+168+42+322+164+2*int(b['meta']['physical_locations'])},'CAT_effect_representation':'exact propagated Pauli frame of explicit Clifford CAT prep/verification circuit under stochastic Pauli faults','remaining_unmodeled':['idle errors','leakage','coherent/calibration errors']}
    (a.out_dir/'s10_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');rows=[]
    for i,p in enumerate(a.p_values):
        r=_run(float(p),a.shots,a.seed+i,b);rows.append(r);print(json.dumps(r,indent=2),flush=True)
    (a.out_dir/'s10_sweep.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print('S10-C PASS: complete physical-front-end factory Monte Carlo completed')
if __name__=='__main__':main()
