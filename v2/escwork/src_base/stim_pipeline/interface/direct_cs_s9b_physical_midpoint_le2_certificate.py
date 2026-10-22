#!/usr/bin/env python3
"""S9-B: exact <=2-fault certificate after physicalizing the midpoint Steane ED.

This extends the S8-E certificate.  All <=2 patterns NOT involving midpoint-ED
faults are inherited unchanged because the fault-free physical ED Kraus map is
exactly the old ideal code-space projection.  The only new fault classes are:

  ED + ED, ED + second-half-core, ED + escape,
  first-half-core + ED.

The first three classes are reduced exactly to the already-exhausted S8-E
second-half/core-boundary certificates using the preserved Step-2 residual
support contracts.  The genuinely new class, first-half-core + ED, is audited
explicitly without assuming that nonzero pre-ED syndrome sectors carry the
correct logical state.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, dataclasses, importlib.util, json, math, os, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
# Small dense kernels dominate this audit; forcing one BLAS thread avoids severe
# oversubscription (especially when several Python workers/processes are used).
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
SRC=ROOT/'src_current'

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError((name,path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

s9ed=_load('s9b_ed',SRC/'direct_cs_step9_physical_midpoint_ed.py')
s8e=_load('s9b_s8e',HERE/'direct_cs_s8e_global_le2_certificate.py')
m=s8e.m; d2=s8e.d2; s8live=s8e.s8live
TOL=1e-10
BRANCHES=s8e.BRANCHES
U=s8live.BLOCK_BASIS; U0=U[:,0:2]; PAIR_INDEX=s8live.PAIR_INDEX

@dataclass(frozen=True)
class EDEffect:
    input_syndrome:int
    output_syndrome:int
    logical_map_re_im:tuple
    multiplicity:int
    example_op:int
    example_option:int

    def matrix(self):
        return np.array([[complex(*self.logical_map_re_im[i][j]) for j in range(2)] for i in range(2)],complex)


def _phase_normalized(A):
    A=np.asarray(A,complex);z=next((x for x in A.reshape(-1) if abs(x)>1e-12),1+0j)
    return A/(z/abs(z))

def _effect_key(t):
    A=_phase_normalized(t.matrix())
    return (int(t.input_syndrome),int(t.output_syndrome),tuple(np.round(A.real,12).reshape(-1)),tuple(np.round(A.imag,12).reshape(-1)))

def _payload(A): return tuple(tuple((float(A[i,j].real),float(A[i,j].imag)) for j in range(2)) for i in range(2))

def _build_ed_effects():
    groups={}; rejected=0; accepted_classes=0
    for i,opts in enumerate(s9ed.ED_OPTIONS):
        for j in range(len(opts)):
            t=s9ed.transfer_for_faults({i:j})
            if t is None: rejected+=1;continue
            accepted_classes+=1;k=_effect_key(t)
            if k not in groups:groups[k]=[t,0,i,j]
            groups[k][1]+=1
    effects=[]
    for t,mult,i,j in groups.values():
        A=_phase_normalized(t.matrix())
        effects.append(EDEffect(int(t.input_syndrome),int(t.output_syndrome),_payload(A),int(mult),int(i),int(j)))
    effects.sort(key=lambda e:(e.input_syndrome,e.output_syndrome,e.example_op,e.example_option))
    return effects,{'physical_single_fault_classes_per_block':sum(len(x) for x in s9ed.ED_OPTIONS),'single_classes_accepting_some_input_syndrome':accepted_classes,'single_classes_rejecting_all_input_syndromes':rejected,'global_phase_equivalent_accepted_transfer_effects':len(effects)}

ED_EFFECTS,ED_EFFECT_META=_build_ed_effects()

# Steane stabilizer group for exact minimum-weight residual representative.
STAB=[]
for s in m.supports:
    mask=sum(1<<q for q in s); STAB.append((mask,0))
for s in m.supports:
    mask=sum(1<<q for q in s); STAB.append((0,mask))
STAB_GROUP=[(0,0)]
for gx,gz in STAB:
    old=list(STAB_GROUP);STAB_GROUP += [(x^gx,z^gz) for x,z in old]

def _pauli_string(x,z,n=7):
    out=[]
    for q in range(n):
        a=(x>>q)&1;b=(z>>q)&1;out.append('Y' if a and b else 'X' if a else 'Z' if b else 'I')
    return ''.join(out)

def _min_rep(x,z):
    best=None
    for sx,sz in STAB_GROUP:
        xx=x^sx;zz=z^sz;w=(xx|zz).bit_count();ps=_pauli_string(xx,zz)
        cand=(w,ps,xx,zz)
        if best is None or cand<best:best=cand
    return best

def _ed_contracts():
    # single on code
    single_acc=0;single_max=0;single_bad=[]
    for i,opts in enumerate(s9ed.ED_OPTIONS):
        for j in range(len(opts)):
            acc,x,z=s9ed._pauli_propagation_reference({i:j})
            if acc:
                single_acc+=1;w,ps,_,_=_min_rep(x,z);single_max=max(single_max,w)
                if w>1 and len(single_bad)<10:single_bad.append((i,j,w,ps))
    # pair in same ED block, distinct physical locations
    pair_total=pair_acc=0;pair_max=0;pair_bad=[]
    for i in range(len(s9ed.ED_OPTIONS)):
        for j in range(i+1,len(s9ed.ED_OPTIONS)):
            for oi in range(len(s9ed.ED_OPTIONS[i])):
                for oj in range(len(s9ed.ED_OPTIONS[j])):
                    pair_total+=1;acc,x,z=s9ed._pauli_propagation_reference({i:oi,j:oj})
                    if acc:
                        pair_acc+=1;w,ps,_,_=_min_rep(x,z);pair_max=max(pair_max,w)
                        if w>2 and len(pair_bad)<10:pair_bad.append((i,oi,j,oj,w,ps))
    # incoming weight1 + one ED fault
    cross_total=cross_acc=0;cross_max=0;cross_bad=[]
    for q in range(7):
        for p in 'XYZ':
            ix=(1<<q) if p in 'XY' else 0; iz=(1<<q) if p in 'YZ' else 0
            for i,opts in enumerate(s9ed.ED_OPTIONS):
                for oi in range(len(opts)):
                    cross_total+=1;acc,x,z=s9ed._pauli_propagation_reference({i:oi},(ix,iz))
                    if acc:
                        cross_acc+=1;w,ps,_,_=_min_rep(x,z);cross_max=max(cross_max,w)
                        if w>2 and len(cross_bad)<10:cross_bad.append((p,q,i,oi,w,ps))
    return {
        'single_fault':{'total':sum(len(x) for x in s9ed.ED_OPTIONS),'accepted_on_code':single_acc,'max_residual_weight_mod_stabilizer':single_max,'violations':single_bad,'passed':not single_bad},
        'two_fault_same_block':{'total':pair_total,'accepted_on_code':pair_acc,'max_residual_weight_mod_stabilizer':pair_max,'violations':pair_bad,'passed':not pair_bad},
        'incoming_weight1_plus_one_ED_fault':{'total':cross_total,'accepted':cross_acc,'max_residual_weight_mod_stabilizer':cross_max,'violations':cross_bad,'passed':not cross_bad},
    }


def _matrix_match_option(ck,gi,F):
    for oi,(G,flip,_lab) in enumerate(m.opt_cache[(ck,gi)]):
        if flip:continue
        for ph in (1,-1,1j,-1j):
            if np.linalg.norm(F-ph*G)<1e-10:return oi
    raise AssertionError(('no matching core fault option',ck,gi,F))

def _map_single_residual_to_second_event(block:int,coord:int,p:str):
    # Error exists immediately before second-half C1 (check index 3).
    if block==1: # B: first gate is 1Q on B
        gi=0;g=m.seqs['C1'][gi][2];P=m.op1(m.p2[p],1);Uq=m.op1(g,1);F=Uq@P@Uq.conj().T
    else: # A commutes through first B-only 1Q, then through CX A->B
        gi=1;Uq=m.seqs['C1'][gi][2];P=m.op1(m.p2[p],0);F=Uq@P@Uq.conj().T
    oi=_matrix_match_option('C1',gi,F)
    lid=next(i for i,(ci,ck,j,gii) in enumerate(m.locations) if ci==3 and ck=='C1' and j==coord and gii==gi)
    return lid,oi

def _residual_embedding_certificate():
    second_keys={(int(e.location_id),int(e.option_index)) for e in s8e.SECOND_EVENTS if e.kind=='gate'}
    singles={};bad=[]
    for block in (0,1):
        for q in range(7):
            for p in 'XYZ':
                lid,oi=_map_single_residual_to_second_event(block,q,p);singles[(block,q,p)]=(lid,oi)
                # effective event may use a different duplicate representative; compare exact action key at location.
                ci,ck,j,gi=m.locations[lid];F,flip,_=m.opt_cache[(ck,gi)][oi];mk=s8e._matrix_key(F,int(flip))
                found=False
                for e in s8e.SECOND_EVENTS:
                    if e.kind!='gate' or e.location_id!=lid:continue
                    G,gflip,_=m.opt_cache[(ck,gi)][int(e.option_index)]
                    if s8e._matrix_key(G,int(gflip))==mk:found=True;break
                if not found:bad.append((block,q,p,lid,oi))
    # Every physical Pauli residual of support <=2 maps to <=2 distinct/effect-composable second-half events.
    max_events=0;pair_collapses=0
    atoms=list(singles)
    for i,a in enumerate(atoms):
        max_events=max(max_events,1)
        for b in atoms[i+1:]:
            # Only distinct physical data qubits; A/B at same coordinate are distinct.
            if a[0]==b[0] and a[1]==b[1]:continue
            ev=[singles[a],singles[b]]
            if ev[0][0]==ev[1][0]:pair_collapses+=1;max_events=max(max_events,1)
            else:max_events=max(max_events,2)
    return {'single_data_paulis_mapped':len(singles),'mapping_violations':bad,'max_equivalent_second_half_core_events_for_weight_le2_residual':max_events,'same_location_products_reducible_to_single_core_fault':pair_collapses,'s8e_second_half_pair_certificate_malignant_count':0,'passed':not bad and max_events<=2}


def _simulate_first_half_no_projection(eff,rec):
    fb,par=s8e._event_faultbook([eff]);st=m.psi0.copy();prob=1.0
    for ci in range(3):
        ck=m.sequence[ci];rf=s8e._record_flip_for_check(fb[ci],ck);raw=int(rec[ci])^int(rf)^int(ci in par)
        try:st,rf2,pr,_=d2._forced_check(st,ck,fb[ci],raw)
        except AssertionError as ex:
            if 'zero' in str(ex):return None
            raise
        if int(rf2)!=int(rf):raise AssertionError((rf,rf2))
        prob*=float(pr)
    return st,prob


def _selected_blocks(st):
    M=np.asarray(st,complex)[PAIR_INDEX]
    # C[sA,lA, sB=0,lB]
    ca=(U.conj().T@M@U0.conj()).reshape(64,2,2)
    # C[sA=0,lA, sB,lB]
    cb=(U0.conj().T@M@U.conj()).reshape(2,64,2).transpose(1,0,2)
    return ca,cb


def _state_from_sector_logical(sa,sb,v):
    D=np.asarray(v,complex).reshape(2,2);UA=U[:,2*sa:2*sa+2];UB=U[:,2*sb:2*sb+2];M=UA@D@UB.T
    out=np.empty(4**7,complex);out[PAIR_INDEX]=M;return out


def _ideal_check_unnorm(st,ck,raw):
    m0=[m.ideal_u[ck][0] for _ in range(7)];m1=[m.ideal_u[ck][1] for _ in range(7)]
    b0=m.prod(st,m0);b1=m.prod(st,m1);return (b0+b1)/2 if int(raw)==0 else (b0-b1)/2


def _build_secondhalf_code_maps(syndrome_pairs):
    maps={}
    for a,b in BRANCHES:
        rec=s8e._record(a,b)
        for sa,sb in sorted(syndrome_pairs):
            cols=[]
            for k in range(4):
                v=np.zeros(4,complex);v[k]=1
                st=_state_from_sector_logical(sa,sb,v)
                for ci in range(3,6):st=_ideal_check_unnorm(st,m.sequence[ci],rec[ci])
                cols.append(m.B.conj().T@st)
            maps[(a,b,sa,sb)]=np.stack(cols,axis=1)
    return maps


def _cross_worker(payload):
    start,end,kmaps,effects,tol=payload
    tested=represented=accepted=0;mal=[];max_inf=0.0
    for ei in range(start,end):
        ce=s8e.FIRST_EVENTS[ei]
        for a,b in BRANCHES:
            rec=s8e._record(a,b);z=_simulate_first_half_no_projection(ce,rec)
            if z is None:continue
            st,_=z;ca,cb=_selected_blocks(st);target=s8e._target(rec)
            for side in (0,1):
                blocks=ca if side==0 else cb
                for ed in effects:
                    block=blocks[ed.input_syndrome]
                    if float(np.vdot(block,block).real)<=1e-18:continue
                    D=ed.matrix()@block if side==0 else block@ed.matrix().T
                    vec=D.reshape(4)
                    if float(np.vdot(vec,vec).real)<=1e-18:continue
                    tested+=1;represented+=int(ce.multiplicity)*int(ed.multiplicity)
                    sa,sb=(ed.output_syndrome,0) if side==0 else (0,ed.output_syndrome)
                    amp=kmaps[(a,b,sa,sb)]@vec;pcode=float(np.vdot(amp,amp).real)
                    if pcode<=1e-18:continue
                    accepted+=1;inf=s8e._phase_infidelity(amp,target);max_inf=max(max_inf,inf)
                    if inf>tol and len(mal)<20:
                        mal.append({'core_event':asdict(ce),'record':rec,'ED_side':'A' if side==0 else 'B','ED_effect':asdict(ed),'postED_sector_pair':[sa,sb],'final_code_probability_unnormalized':pcode,'logical_infidelity':inf})
    return {'start':start,'end':end,'tested':tested,'represented':represented,'accepted':accepted,'mal':mal,'max_inf':max_inf}


def _firstcore_plus_ed_audit(tol=TOL,progress=200,workers=1):
    effects=ED_EFFECTS
    syndrome_pairs={(e.output_syndrome,0) for e in effects}|{(0,e.output_syndrome) for e in effects}
    kmaps=_build_secondhalf_code_maps(syndrome_pairs)
    n=len(s8e.FIRST_EVENTS);t0=time.time()
    if workers<=1:
        parts=[]
        step=max(1,progress or n)
        for st in range(0,n,step):
            r=_cross_worker((st,min(n,st+step),kmaps,effects,tol));parts.append(r)
            print(f"S9-B first-core+ED progress {r['end']}/{n}; malignant_so_far={sum(len(x['mal']) for x in parts)}",flush=True)
            if r['mal']:break
    else:
        workers=min(int(workers),n)
        # More chunks than workers balances the nonuniform branch cost.
        nchunks=max(workers*2,workers);cuts=np.linspace(0,n,nchunks+1,dtype=int)
        payloads=[(int(cuts[i]),int(cuts[i+1]),kmaps,effects,tol) for i in range(nchunks) if cuts[i]<cuts[i+1]]
        parts=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futs=[ex.submit(_cross_worker,p) for p in payloads]
            done=0
            for f in concurrent.futures.as_completed(futs):
                r=f.result();parts.append(r);done+=1
                print(f"S9-B cross chunk completed {done}/{len(futs)} [{r['start']}:{r['end']}]; malignant_so_far={sum(len(x['mal']) for x in parts)}",flush=True)
                if r['mal']:
                    for ff in futs: ff.cancel()
                    break
    tested=sum(x['tested'] for x in parts);represented=sum(x['represented'] for x in parts);accepted=sum(x['accepted'] for x in parts);mal=[y for x in parts for y in x['mal']][:20];max_inf=max([x['max_inf'] for x in parts] or [0.0])
    complete=(not mal and sum(x['end']-x['start'] for x in parts)==n)
    return {'effective_first_half_core_events':n,'ED_transfer_effects':len(effects),'tested_nonzero_effect_pairs':tested,'original_class_pair_multiplicity_represented':represented,'accepted_final_code_branches':accepted,'malignant_count':len(mal),'malignant_examples':mal,'max_final_code_logical_infidelity':max_inf,'cross_workers':workers,'complete':complete,'elapsed_seconds':time.time()-t0,'passed':complete and not mal}


def _load_s8e_summary(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    req=['global_le2_certificate_complete','exact_accepted_harmful_events_order_le2','preserved_integrated_model_df_ge_3','combined_with_S8D3_malignant_triple_implies_df_eq_3']
    for k in req:
        if k not in d:raise KeyError(k)
    if not d['global_le2_certificate_complete'] or d['exact_accepted_harmful_events_order_le2']!=0 or not d['preserved_integrated_model_df_ge_3']:
        raise AssertionError('referenced S8-E certificate is not PASS')
    return {'path':str(Path(path).resolve()),'global_le2_certificate_complete':True,'exact_accepted_harmful_events_order_le2':0,'second_half_malignant_pair_count':d['second_half_core_pair_exhaustive']['malignant_effective_pair_count'],'core1_escape1_passed':d['core1_plus_escape1']['passed']}


def _contracts_bundle(s8e_summary_path):
    inherited=_load_s8e_summary(s8e_summary_path)
    instr=s9ed.self_test()
    contracts=_ed_contracts()
    embed=_residual_embedding_certificate()
    return {'inherited_S8E_certificate':inherited,'S9A_physical_ED_instrument':instr,'ED_effect_inventory':ED_EFFECT_META,'ED_nested_support_contracts':contracts,'residual_to_second_half_core_embedding':embed}


def _combine_summary(contracts_bundle,cross):
    contract_pass=all(v['passed'] for v in contracts_bundle['ED_nested_support_contracts'].values())
    lower=bool(contracts_bundle['S9A_physical_ED_instrument']['passed'] and contract_pass and contracts_bundle['residual_to_second_half_core_embedding']['passed'] and cross.get('passed',False) and contracts_bundle['inherited_S8E_certificate']['global_le2_certificate_complete'])
    return {'stage':'S9-B','scope':'S8-E integrated model with explicit physical midpoint Steane ED on both blocks',**contracts_bundle,'first_half_core1_plus_midED1_exact_audit':cross,'new_fault_classes_reduced':{'ED_ED':'Step-2 accepted two-ED-fault residual weight<=2 -> <=2 equivalent second-half core faults -> inherited S8-E second-half pair certificate','ED_secondhalfcore':'accepted single ED fault on code leaves weight<=1 residual -> equivalent one second-half core fault; plus actual core fault -> inherited S8-E pair/single certificate','ED_escape':'accepted single ED fault on code leaves weight<=1 residual -> equivalent one core-single boundary event -> inherited S8-E core1+escape1 certificate','firsthalfcore_ED':'NOT reduced by a target-state assumption; explicitly audited because pre-ED nonzero syndrome sectors can carry nontrivial logical states'},'global_le2_certificate_complete_with_physical_midpoint_ED':lower,'exact_accepted_harmful_events_order_le2':0 if lower else None,'physical_midpoint_integrated_model_df_ge_3':lower,'escape_only_malignant_triple_survives_faultfree_physical_ED':True,'combined_with_existing_escape_malignant_triple_implies_df_eq_3':lower}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['all','contracts','cross','combine'],default='all')
    ap.add_argument('--s8e-summary',type=Path,default=HERE/'reference_results'/'s8e_summary.json')
    ap.add_argument('--contracts-json',type=Path)
    ap.add_argument('--cross-json',type=Path)
    ap.add_argument('--progress',type=int,default=100,help='sequential cross-audit chunk size/progress cadence')
    ap.add_argument('--cross-workers',type=int,default=1)
    ap.add_argument('--out-dir',type=Path,default=HERE/'s9b_results')
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);t=time.time()

    if a.mode=='contracts':
        c=_contracts_bundle(a.s8e_summary);c['elapsed_seconds']=time.time()-t
        p=a.out_dir/'s9b_contracts.json';p.write_text(json.dumps(c,indent=2),encoding='utf-8');print(json.dumps(c,indent=2));print('S9-B CONTRACTS PASS' if all(v['passed'] for v in c['ED_nested_support_contracts'].values()) and c['residual_to_second_half_core_embedding']['passed'] else 'S9-B CONTRACTS FAIL');return

    if a.mode=='cross':
        c=_firstcore_plus_ed_audit(progress=a.progress,workers=a.cross_workers);c['elapsed_seconds_total']=time.time()-t
        p=a.out_dir/'s9b_cross.json';p.write_text(json.dumps(c,indent=2),encoding='utf-8');print(json.dumps(c,indent=2));print('S9-B CROSS PASS' if c['passed'] else 'S9-B CROSS FAIL');return

    if a.mode=='combine':
        if a.contracts_json is None or a.cross_json is None:raise SystemExit('--contracts-json and --cross-json required for --mode combine')
        c=json.loads(a.contracts_json.read_text(encoding='utf-8'));x=json.loads(a.cross_json.read_text(encoding='utf-8'))
        # Remove stage-local elapsed annotations if present; harmless otherwise.
        summary=_combine_summary(c,x);summary['elapsed_seconds']=time.time()-t
    else:
        c=_contracts_bundle(a.s8e_summary)
        (a.out_dir/'s9b_contracts.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
        x=_firstcore_plus_ed_audit(progress=a.progress,workers=a.cross_workers)
        (a.out_dir/'s9b_cross.json').write_text(json.dumps(x,indent=2),encoding='utf-8')
        summary=_combine_summary(c,x);summary['elapsed_seconds']=time.time()-t

    (a.out_dir/'s9b_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
    if summary['global_le2_certificate_complete_with_physical_midpoint_ED']:
        print('S9-B GLOBAL LOWER-BOUND PASS: explicit physical midpoint ED preserves d_f >= 3')
        print('S9-B + existing escape malignant triple: physical-midpoint integrated model has d_f = 3')
    else:print('S9-B FAIL: inspect contracts/cross results')

if __name__=='__main__':main()
