#!/usr/bin/env python3
"""S8-A: exact live-boundary exporter for the 14-qubit direct-|CS> core.

Key change from Step 7
======================
The midpoint ideal Steane projection is retained as the existing idealized
midpoint-ED module.  The *final* ideal projection is removed.

The live 14-qubit state is expanded exactly in the complete basis

    E_sA |l_A>_L  tensor  E_sB |l_B>_L,

where sA,sB are the 64 Steane stabilizer syndromes, l_A,l_B in {0,1}, and E_s
is a deterministic minimum-weight physical Pauli representative.  Therefore
this exporter remains exact even when a core trajectory is a coherent
superposition of multiple Steane syndrome sectors.

The exported sparse sector amplitudes are a complete pure-state boundary
contract.  They must NOT be reduced to one Pauli label unless
`sector_is_sharp == true`.
"""
from __future__ import annotations

import argparse, collections, importlib.util, json, math, random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CORE_PATH=HERE/'direct_cs_step7_core_mc.py'
_spec=importlib.util.spec_from_file_location('direct_cs_step7_core_mc',CORE_PATH)
if _spec is None or _spec.loader is None: raise RuntimeError(CORE_PATH)
m=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(m)

STAB_SPECS=tuple([('X',tuple(sorted(s))) for s in m.supports]+[('Z',tuple(sorted(s))) for s in m.supports])
PAULIS=('I','X','Y','Z');NON_ID=('X','Y','Z')

@dataclass(frozen=True)
class RecoveryRepresentative:
    syndrome:int;weight:int;pauli_string:str

@dataclass
class LiveBoundary:
    record:list[int]
    sector_is_sharp:bool
    dominant_syndrome_a:int;dominant_syndrome_b:int
    dominant_sector_probability:float
    dominant_recovery_a:str;dominant_recovery_b:str
    syndrome_sector_count:int
    syndrome_sector_components:list[dict]
    logical_density_after_ideal_syndrome_recovery_re_im:list[list[list[float]]]
    target_fidelity_after_ideal_syndrome_recovery:float
    basis_probability_sum:float
    exact_basis_reconstruction_error:float
    live_norm:float

def _single_pauli_syndrome(ps):
    if len(ps)!=7 or any(p not in PAULIS for p in ps):raise ValueError(ps)
    syn=0
    for bi,(sp,support) in enumerate(STAB_SPECS):
        anti=0
        for j in support:
            p=ps[j]
            if p=='I':continue
            anti^=int(p in ('Z','Y') if sp=='X' else p in ('X','Y'))
        syn|=anti<<bi
    return syn

def _build_recovery_table():
    best={}
    def consider(chars):
        s=''.join(chars);syn=_single_pauli_syndrome(s);w=sum(c!='I' for c in s);r=RecoveryRepresentative(syn,w,s);cur=best.get(syn)
        if cur is None or (w,s)<(cur.weight,cur.pauli_string):best[syn]=r
    consider(['I']*7)
    for j in range(7):
        for p in NON_ID:
            c=['I']*7;c[j]=p;consider(c)
    for j in range(7):
        for k in range(j+1,7):
            for p in NON_ID:
                for q in NON_ID:
                    c=['I']*7;c[j]=p;c[k]=q;consider(c)
    if set(best)!=set(range(64)):raise AssertionError((len(best),sorted(set(range(64))-set(best))))
    return best
RECOVERY_TABLE=_build_recovery_table()

def _apply_pauli_7q(v,ps):
    out=np.asarray(v,dtype=complex).copy()
    for q,p in enumerate(ps):
        if p=='I':continue
        nxt=np.zeros_like(out);bit=1<<q
        for x,a in enumerate(out):
            if abs(a)<1e-18:continue
            b=(x>>q)&1
            if p=='X':nxt[x^bit]+=a
            elif p=='Z':nxt[x]+=(-1 if b else 1)*a
            elif p=='Y':nxt[x^bit]+=(1j if b==0 else -1j)*a
            else:raise ValueError(p)
        out=nxt
    return out

def _build_block_boundary_basis():
    U=np.empty((128,128),complex)
    for s in range(64):
        e=RECOVERY_TABLE[s].pauli_string
        U[:,2*s]=_apply_pauli_7q(m.zero,e)
        U[:,2*s+1]=_apply_pauli_7q(m.one,e)
    gram=U.conj().T@U
    dev=float(np.max(np.abs(gram-np.eye(128))))
    if dev>1e-10:raise AssertionError(('boundary basis nonunitary',dev))
    return U,dev
BLOCK_BASIS,BLOCK_BASIS_UNITARITY_DEV=_build_block_boundary_basis()
PAIR_INDEX=np.array([[m.pidx(a,b) for b in range(128)] for a in range(128)],dtype=np.int64)

def _cplx(z):return [float(z.real),float(z.imag)]
def _vec(v):return [_cplx(z) for z in v]
def _rho(r):return [[_cplx(z) for z in row] for row in r]

def exact_sector_tensor(st):
    """Return C[sA,lA,sB,lB] in the complete live-boundary basis."""
    n=float(np.vdot(st,st).real)
    if n<=1e-20:raise ValueError('zero state')
    st=np.asarray(st,complex)/math.sqrt(n)
    M=st[PAIR_INDEX]
    # U columns are |s,l>.  For coefficient matrix of a bipartite ket:
    # C = U^dagger M U^*.
    C2=BLOCK_BASIS.conj().T@M@BLOCK_BASIS.conj()
    C=C2.reshape(64,2,64,2)
    prob=float(np.vdot(C.reshape(-1),C.reshape(-1)).real)
    # exact back-transform check in the original interleaved pair ordering
    M2=BLOCK_BASIS@C2@BLOCK_BASIS.T
    recon=np.empty_like(st);recon[PAIR_INDEX]=M2
    err=float(np.linalg.norm(recon-st))
    return C,prob,err,n

def analyze_live_boundary(st,record,tol=1e-12):
    if len(record)!=6:raise ValueError(record)
    C,psum,recon,n=exact_sector_tensor(st)
    comps=[];rho=np.zeros((4,4),complex)
    dominant=(-1.0,0,0)
    for sa in range(64):
        for sb in range(64):
            v=C[sa,:,sb,:].reshape(4)
            p=float(np.vdot(v,v).real)
            if p<=tol:continue
            rho+=np.outer(v,v.conj())
            if p>dominant[0]:dominant=(p,sa,sb)
            comps.append({'syndrome_a':sa,'syndrome_b':sb,'probability':p,'recovery_a':RECOVERY_TABLE[sa].pauli_string,'recovery_b':RECOVERY_TABLE[sb].pauli_string,'recovery_weight_a':RECOVERY_TABLE[sa].weight,'recovery_weight_b':RECOVERY_TABLE[sb].weight,'logical_amplitudes_unnormalized_re_im':_vec(v),'logical_amplitudes_conditional_re_im':_vec(v/math.sqrt(p))})
    comps.sort(key=lambda x:-x['probability'])
    pdom,sa,sb=dominant
    sharp=(len(comps)==1 and abs(pdom-1)<1e-9)
    tgt=m.cs*np.array([1,(-1)**int(record[1]),(-1)**int(record[0]),(-1)**int(record[0]^record[1])],complex)
    fid=float(np.vdot(tgt,rho@tgt).real)
    return LiveBoundary(list(map(int,record)),sharp,sa,sb,pdom,RECOVERY_TABLE[sa].pauli_string,RECOVERY_TABLE[sb].pauli_string,len(comps),comps,_rho(rho),fid,psum,recon,n)

def _sample_fault_book(p,rng):
    fb=[collections.defaultdict(list) for _ in m.sequence];nf=0;faults=[]
    for ci,ck,j,gi in m.locations:
        if rng.random()<p:
            oi=rng.randrange(len(m.opt_cache[(ck,gi)]));nf+=1;fb[ci][j].append((gi,oi));faults.append({'check_index':ci,'check':ck,'coordinate':j,'gate_index':gi,'fault_option_index':oi,'fault_label':m.opt_cache[(ck,gi)][oi][2]})
    return fb,nf,faults

def shot_live(p,rng):
    fb,nf,faults=_sample_fault_book(p,rng);q=(1-(1-2*p)**7)/2;st=m.psi0.copy();rec=[];midpr=None
    for ci,ck in enumerate(m.sequence):
        st,meas=m.check(st,ck,fb[ci],rng,q);rec.append(int(meas))
        if ci==2:
            st,pr=m.proj(st);midpr=float(pr)
            if rng.random()>=pr:return False,'mid',nf,rec,midpr,None,faults
    if not m.valid(rec):return False,'record',nf,rec,midpr,None,faults
    return True,'acc',nf,rec,midpr,analyze_live_boundary(st,rec),faults

def _find_option(ck,gi,label):
    h=[i for i,(_,_,s) in enumerate(m.opt_cache[(ck,gi)]) if s==label]
    if len(h)!=1:raise AssertionError((ck,gi,label,h))
    return h[0]

def _fixed_check(st,ck,fbc,recorded):
    m0=[];m1=[];rf=0
    for j in range(7):
        fs=fbc.get(j,[])
        if fs:
            u0,r0=m.coord(ck,0,fs);u1,r1=m.coord(ck,1,fs)
            if r0!=r1:raise AssertionError((ck,j,r0,r1))
            rf^=r0
        else:u0,u1=m.ideal_u[ck]
        m0.append(u0);m1.append(u1)
    b0=m.prod(st,m0);b1=m.prod(st,m1);out=(b0+b1)/2 if (int(recorded)^rf)==0 else (b0-b1)/2;z=np.linalg.norm(out)
    if z<=1e-20:raise AssertionError('zero fixed branch')
    return out/z

def explicit_malignant_triple_boundary():
    st=m.target(0,0).copy();f=[collections.defaultdict(list) for _ in range(3)]
    f[0][0].append((4,_find_option('C1',4,'IYI')));f[1][1].append((1,_find_option('C2',1,'YZ')));f[2][2].append((3,_find_option('C12',3,'Z')))
    for wi,ck in enumerate(['C1','C2','C12']):st=_fixed_check(st,ck,f[wi],0)
    return analyze_live_boundary(st,[0,0,0,0,0,0])

def explicit_nonpauli_triple_boundary():
    # S7-A witness before the now-removed final projection.
    st=m.target(0,0).copy();f=[collections.defaultdict(list) for _ in range(3)]
    f[1][4].append((7,_find_option('C2',7,'XX')))
    f[1][3].append((6,_find_option('C2',6,'X')))
    f[2][0].append((2,_find_option('C12',2,'ZXZ')))
    for wi,ck in enumerate(['C1','C2','C12']):st=_fixed_check(st,ck,f[wi],0)
    return analyze_live_boundary(st,[0,0,0,0,0,0])

def _from_payload(v):return np.array([complex(r,i) for r,i in v],complex)

def self_test():
    if len(RECOVERY_TABLE)!=64:raise AssertionError(len(RECOVERY_TABLE))
    ff=[]
    for a,b in [(0,0),(0,1),(1,0),(1,1)]:
        bd=analyze_live_boundary(m.target(a,b),[a,b,a^b,a,b,a^b])
        if not bd.sector_is_sharp or bd.dominant_syndrome_a or bd.dominant_syndrome_b or abs(1-bd.target_fidelity_after_ideal_syndrome_recovery)>1e-10:raise AssertionError(asdict(bd))
        ff.append({'branch':[a,b],'fidelity':bd.target_fidelity_after_ideal_syndrome_recovery})
    # Exhaustively verify all 64 canonical syndrome basis states on each block.
    base=m.target(0,0);max_recon=0.;max_inf=0.
    # use old physical-pair helper for exact injected Pauli on the 14q state
    def apply_block(st,block,ps):
        out=st
        for j,p in enumerate(ps):
            if p!='I':out=m.local4(out,m.op1(m.p2[p],block),j)
        return out
    for block in [0,1]:
        for syn,r in RECOVERY_TABLE.items():
            bd=analyze_live_boundary(apply_block(base,block,r.pauli_string),[0,0,0,0,0,0])
            got=bd.dominant_syndrome_a if block==0 else bd.dominant_syndrome_b
            if not bd.sector_is_sharp or got!=syn:raise AssertionError((block,syn,got,asdict(bd)))
            max_recon=max(max_recon,bd.exact_basis_reconstruction_error);max_inf=max(max_inf,abs(1-bd.target_fidelity_after_ideal_syndrome_recovery))
    tri=explicit_malignant_triple_boundary();comp=tri.syndrome_sector_components[0];lv=_from_payload(comp['logical_amplitudes_conditional_re_im']);fza=float(abs(np.vdot(m.op1(m.Z,0)@m.cs,lv))**2)
    if not tri.sector_is_sharp or tri.dominant_syndrome_a or tri.dominant_syndrome_b or abs(1-fza)>1e-10:raise AssertionError((asdict(tri),fza))
    npw=explicit_nonpauli_triple_boundary()
    probs=sorted([c['probability'] for c in npw.syndrome_sector_components])
    if npw.syndrome_sector_count!=4 or max(abs(q-0.25) for q in probs)>1e-10:raise AssertionError(asdict(npw))
    zero=[c for c in npw.syndrome_sector_components if c['syndrome_a']==0 and c['syndrome_b']==0]
    if len(zero)!=1:raise AssertionError(asdict(npw))
    bell=np.array([1,0,0,1j],complex)/np.sqrt(2);zv=_from_payload(zero[0]['logical_amplitudes_conditional_re_im']);fbell=float(abs(np.vdot(bell,zv))**2)
    if abs(1-fbell)>1e-10:raise AssertionError((asdict(npw),fbell))
    return {'boundary_basis_unitarity_max_deviation':BLOCK_BASIS_UNITARITY_DEV,'recovery_table_entries':64,'recovery_weight_histogram':dict(collections.Counter(r.weight for r in RECOVERY_TABLE.values())),'fault_free_branches':ff,'canonical_basis_max_reconstruction_error':max_recon,'canonical_basis_max_logical_infidelity':max_inf,'malignant_triple':{'sector_count':tri.syndrome_sector_count,'syndrome_a':tri.dominant_syndrome_a,'syndrome_b':tri.dominant_syndrome_b,'logical_ZA_fidelity':fza,'target_CS_fidelity':tri.target_fidelity_after_ideal_syndrome_recovery},'nonpauli_triple_live_boundary':{'sector_count':npw.syndrome_sector_count,'sector_probabilities':probs,'zero_zero_sector_bell_fidelity':fbell,'target_fidelity_after_ideal_syndrome_recovery':npw.target_fidelity_after_ideal_syndrome_recovery}}

def run_mc(p,shots,seed,export_limit=100):
    rng=random.Random(seed);reasons=collections.Counter();nfh=collections.Counter();acc_nfh=collections.Counter();multisector_nfh=collections.Counter();acc=sharp=0;inf=0.;sector_counts=collections.Counter();dom=collections.Counter();samples=[];max_recon=0.;max_probdev=0.
    for _ in range(shots):
        ok,why,nf,rec,midpr,bd,faults=shot_live(p,rng);reasons[why]+=1;nfh[nf]+=1
        if not ok:continue
        acc+=1;acc_nfh[nf]+=1;sharp+=int(bd.sector_is_sharp);multisector_nfh[nf]+=int(not bd.sector_is_sharp);inf+=max(0.,1-bd.target_fidelity_after_ideal_syndrome_recovery);sector_counts[bd.syndrome_sector_count]+=1;dom[(bd.dominant_syndrome_a,bd.dominant_syndrome_b)]+=1;max_recon=max(max_recon,bd.exact_basis_reconstruction_error);max_probdev=max(max_probdev,abs(1-bd.basis_probability_sum))
        if len(samples)<export_limit:
            row=asdict(bd);row.update({'fault_count':nf,'faults':faults,'midpoint_projection_probability':midpr});samples.append(row)
    return {'p':p,'shots':shots,'seed':seed,'live_acceptance':acc/shots,'accepted':acc,'reasons':dict(reasons),'fault_count_histogram':{str(k):v for k,v in sorted(nfh.items())},'accepted_fault_count_histogram':{str(k):v for k,v in sorted(acc_nfh.items())},'multisector_accepted_by_fault_count':{str(k):v for k,v in sorted(multisector_nfh.items()) if v},'accepted_sharp_single_sector_fraction':sharp/acc if acc else None,'accepted_multisector_fraction':1-sharp/acc if acc else None,'accepted_sector_count_histogram':{str(k):v for k,v in sorted(sector_counts.items())},'conditional_target_infidelity_after_ideal_syndrome_recovery':inf/acc if acc else None,'max_exact_basis_reconstruction_error':max_recon,'max_basis_probability_sum_deviation':max_probdev,'distinct_dominant_syndrome_pairs':len(dom),'top_dominant_syndrome_pairs':[{'syndrome_a':a,'syndrome_b':b,'count':c} for (a,b),c in dom.most_common(20)],'final_ideal_projection_removed':True,'midpoint_ideal_projection_retained':True},samples

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--p',type=float,default=1e-3);ap.add_argument('--shots',type=int,default=200);ap.add_argument('--seed',type=int,default=20260820);ap.add_argument('--export-limit',type=int,default=100);ap.add_argument('--out-dir',type=Path,default=HERE.parent/'results_s8');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        x=self_test();(a.out_dir/'s8a_self_test.json').write_text(json.dumps(x,indent=2),encoding='utf-8');print(json.dumps(x,indent=2))
    summary,samples=run_mc(a.p,a.shots,a.seed,a.export_limit);(a.out_dir/'s8a_mc_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (a.out_dir/'s8a_boundary_samples.jsonl').open('w',encoding='utf-8') as f:
        for x in samples:f.write(json.dumps(x)+'\n')
    (a.out_dir/'s8a_recovery_table.json').write_text(json.dumps({str(s):asdict(r) for s,r in sorted(RECOVERY_TABLE.items())},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print('S8-A PASS: exact coherent live-boundary syndrome/logical-state expansion exported')
if __name__=='__main__':main()
