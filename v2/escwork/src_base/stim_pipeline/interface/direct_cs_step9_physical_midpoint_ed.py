#!/usr/bin/env python3
"""S9-A: exact physical midpoint Steane-ED instrument.

Replaces the ideal midpoint code-space projection used by Step 7/8 with the
explicit 82-active-location one-block Steane error-detection circuit preserved
in direct_cs_step2_certificate.py:

  Z snapshot (0145, 0235, 0246)
  transversal H^7
  Z snapshot again (therefore X-syndrome in the original frame)
  transversal H^7 back.

The instrument acts on an arbitrary 7-qubit input state. Ancillas are appended
only while their individual superdense-syndrome gadget is live and are removed
when measured, so the exact state-vector never exceeds 10 active qubits.

Fault model: one active location fails with probability p; PREP/H faults are
uniform X/Y/Z after the operation, CX faults are uniform over the 15 non-II
Paulis after the gate, and measurement faults flip the recorded bit. The
postselected ED branch accepts iff all 14 recorded measurement outcomes are 0.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent


def _find_core() -> Path:
    for p in [HERE/'direct_cs_step7_core_mc.py', Path.cwd()/'src_current'/'direct_cs_step7_core_mc.py']:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError('direct_cs_step7_core_mc.py not found')

_spec = importlib.util.spec_from_file_location('s9_core', _find_core())
if _spec is None or _spec.loader is None:
    raise RuntimeError('cannot import core')
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)

def _find_s8live() -> Path:
    for p in [HERE/'direct_cs_step8_live_boundary.py', Path.cwd()/'src_current'/'direct_cs_step8_live_boundary.py']:
        if p.is_file(): return p.resolve()
    raise FileNotFoundError('direct_cs_step8_live_boundary.py not found')
_s8spec=importlib.util.spec_from_file_location('s9_s8live',_find_s8live())
if _s8spec is None or _s8spec.loader is None: raise RuntimeError('cannot import S8 live')
s8live=importlib.util.module_from_spec(_s8spec); sys.modules['s9_s8live']=s8live; _s8spec.loader.exec_module(s8live)
PAIR_INDEX=np.array([[m.pidx(a,b) for b in range(128)] for a in range(128)],dtype=np.int64)

I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.diag([1,-1]).astype(complex)
H = (X+Z)/math.sqrt(2)
P1 = {'I':I,'X':X,'Y':Y,'Z':Z}

@dataclass(frozen=True)
class Op:
    typ: str
    qs: tuple[int, ...]
    basis: str|None = None
    label: str|None = None


def z_snapshot_ops(anc0: int=7, prefix: str='Z'):
    ops=[]; aidx=anc0
    a,b=aidx,aidx+1; aidx+=2
    ops += [
        Op('PREP',(a,),'X',prefix+'0145_pa'), Op('PREP',(b,),'Z',prefix+'0145_pb'),
        Op('CX',(a,b),label=prefix+'0145_ent'), Op('CX',(1,b),label=prefix+'0145_d1'),
        Op('CX',(4,a),label=prefix+'0145_d4'), Op('CX',(5,b),label=prefix+'0145_d5'),
        Op('CX',(0,b),label=prefix+'0145_d0'), Op('CX',(a,b),label=prefix+'0145_dis'),
        Op('MEAS',(a,),'X',prefix+'0145_flag'), Op('MEAS',(b,),'Z',prefix+'0145_syn'),
    ]
    a,b=aidx,aidx+1; aidx+=2
    ops += [
        Op('PREP',(a,),'X',prefix+'0235_pa'), Op('PREP',(b,),'Z',prefix+'0235_pb'),
        Op('CX',(a,b),label=prefix+'0235_ent'), Op('CX',(5,b),label=prefix+'0235_d5'),
        Op('CX',(3,b),label=prefix+'0235_d3'), Op('CX',(0,b),label=prefix+'0235_d0'),
        Op('CX',(2,a),label=prefix+'0235_d2'), Op('CX',(a,b),label=prefix+'0235_dis'),
        Op('MEAS',(a,),'X',prefix+'0235_flag'), Op('MEAS',(b,),'Z',prefix+'0235_syn'),
    ]
    a,b,c=aidx,aidx+1,aidx+2; aidx+=3
    ops += [
        Op('PREP',(a,),'Z',prefix+'0246_pa'), Op('PREP',(b,),'X',prefix+'0246_pb'), Op('PREP',(c,),'X',prefix+'0246_pc'),
        Op('CX',(b,a),label=prefix+'0246_entb'), Op('CX',(c,a),label=prefix+'0246_entc'),
        Op('CX',(0,a),label=prefix+'0246_d0'), Op('CX',(4,b),label=prefix+'0246_d4'),
        Op('CX',(2,c),label=prefix+'0246_d2'), Op('CX',(6,b),label=prefix+'0246_d6'),
        Op('CX',(b,a),label=prefix+'0246_disb'), Op('CX',(c,a),label=prefix+'0246_disc'),
        Op('MEAS',(a,),'Z',prefix+'0246_syn'), Op('MEAS',(b,),'X',prefix+'0246_flagb'), Op('MEAS',(c,),'X',prefix+'0246_flagc'),
    ]
    return ops,aidx

ED_OPS,next_anc = z_snapshot_ops(7,'Z')
for q in range(7): ED_OPS.append(Op('H',(q,),label=f'H1_{q}'))
_second,next_anc = z_snapshot_ops(next_anc,'XviaH'); ED_OPS += _second
for q in range(7): ED_OPS.append(Op('H',(q,),label=f'H2_{q}'))


def options(op: Op):
    if op.typ in ('PREP','H'):
        return tuple((p,) for p in 'XYZ')
    if op.typ == 'CX':
        return tuple((a+b,) for a in 'IXYZ' for b in 'IXYZ' if a!='I' or b!='I')
    if op.typ == 'MEAS':
        return (('M',),)
    raise ValueError(op.typ)

ED_OPTIONS = tuple(options(o) for o in ED_OPS)


def inventory():
    from collections import Counter
    c=Counter(o.typ for o in ED_OPS)
    return {
        'physical_locations_per_block':len(ED_OPS),
        'location_types':dict(c),
        'single_fault_classes_per_block':sum(len(x) for x in ED_OPTIONS),
        'two_blocks_physical_locations':2*len(ED_OPS),
    }


def _apply_1q(A: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
    """Apply U on bit q (LSB convention) to row-state matrix A[2^n,batch]."""
    dim=A.shape[0]; idx=np.arange(dim,dtype=np.int64); z=idx[((idx>>q)&1)==0]; o=z|(1<<q)
    out=A.copy()
    a0=A[z]; a1=A[o]
    out[z]=U[0,0]*a0 + U[0,1]*a1
    out[o]=U[1,0]*a0 + U[1,1]*a1
    return out


def _apply_cx(A: np.ndarray, c: int, t: int) -> np.ndarray:
    idx=np.arange(A.shape[0],dtype=np.int64)
    perm=idx ^ ((((idx>>c)&1).astype(np.int64))<<t)
    return A[perm]


def _append_qubit(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.vstack((v[0]*A, v[1]*A))


def _project_remove(A: np.ndarray, q: int, basis: str, outcome: int) -> np.ndarray:
    if basis == 'X':
        A=_apply_1q(A,H,q)
    elif basis != 'Z':
        raise ValueError(basis)
    dim=A.shape[0]; idx=np.arange(dim,dtype=np.int64); keep=idx[((idx>>q)&1)==int(outcome)]
    # Removing bit q maps old indices to contiguous new indices.
    low=keep & ((1<<q)-1); high=keep>>(q+1); new_idx=low|(high<<q)
    out=np.empty((dim//2,A.shape[1]),dtype=complex)
    out[new_idx]=A[keep]
    return out


def accepted_apply(inputs: np.ndarray, faults: dict[int,int]|None=None) -> np.ndarray:
    """Apply the unnormalized all-recorded-zero ED branch to one or more inputs.

    inputs may be shape (128,) or (128,batch). The output has the same shape.
    """
    faults={} if faults is None else {int(k):int(v) for k,v in faults.items()}
    A=np.asarray(inputs,dtype=complex)
    squeeze=(A.ndim==1)
    if squeeze: A=A[:,None]
    if A.ndim!=2 or A.shape[0]!=128: raise ValueError(A.shape)
    labels=list(range(7))  # bit position -> global circuit qubit label
    for oi,op in enumerate(ED_OPS):
        fidx=faults.get(oi)
        if fidx is not None and not (0 <= fidx < len(ED_OPTIONS[oi])):
            raise IndexError((oi,fidx,len(ED_OPTIONS[oi])))
        if op.typ == 'PREP':
            v=np.array([1,1],complex)/math.sqrt(2) if op.basis=='X' else np.array([1,0],complex)
            A=_append_qubit(A,v); labels.append(op.qs[0]); q=len(labels)-1
            if fidx is not None:
                p=ED_OPTIONS[oi][fidx][0]; A=_apply_1q(A,P1[p],q)
        elif op.typ == 'H':
            q=labels.index(op.qs[0]); A=_apply_1q(A,H,q)
            if fidx is not None:
                p=ED_OPTIONS[oi][fidx][0]; A=_apply_1q(A,P1[p],q)
        elif op.typ == 'CX':
            c=labels.index(op.qs[0]); t=labels.index(op.qs[1]); A=_apply_cx(A,c,t)
            if fidx is not None:
                lab=ED_OPTIONS[oi][fidx][0]
                A=_apply_1q(A,P1[lab[0]],c); A=_apply_1q(A,P1[lab[1]],t)
        elif op.typ == 'MEAS':
            q=labels.index(op.qs[0])
            # A measurement fault flips the recorded bit. Conditioned on recorded
            # zero, the underlying quantum outcome must therefore be one.
            actual_outcome=1 if fidx is not None else 0
            A=_project_remove(A,q,op.basis,actual_outcome)
            labels.pop(q)
        else:
            raise ValueError(op.typ)
    if labels != list(range(7)) or A.shape[0] != 128:
        raise AssertionError((labels,A.shape))
    return A[:,0] if squeeze else A


def accepted_kraus(faults: dict[int,int]|None=None) -> np.ndarray:
    """Return the unnormalized all-recorded-zero Kraus operator on 7 data qubits."""
    return accepted_apply(np.eye(128,dtype=complex), faults)


@functools.lru_cache(maxsize=4096)
def accepted_kraus_cached(key: tuple[tuple[int,int], ...]) -> np.ndarray:
    return accepted_kraus(dict(key))


def kraus_for_faults(faults: dict[int,int]|None=None) -> np.ndarray:
    key=tuple(sorted((int(k),int(v)) for k,v in (faults or {}).items()))
    return accepted_kraus_cached(key)


def apply_two_block_midpoint(st: np.ndarray, faults_a: dict[int,int]|None=None, faults_b: dict[int,int]|None=None):
    """Apply physical ED accept Kraus on both Steane blocks to core interleaved state."""
    try:
        # Reuse exact pair-index convention from S8 if available without importing.
        pair_index=PAIR_INDEX
        M=np.asarray(st,complex)[pair_index]
        KA=kraus_for_faults(faults_a); KB=kraus_for_faults(faults_b)
        M2=KA@M@KB.T
        out=np.empty_like(np.asarray(st,complex)); out[pair_index]=M2
        pr=float(np.vdot(out,out).real)
        if pr>1e-20: out=out/math.sqrt(pr)
        return out,pr
    except Exception:
        raise



@dataclass(frozen=True)
class EDTransfer:
    input_syndrome: int
    output_syndrome: int
    logical_map_re_im: tuple[tuple[tuple[float,float],tuple[float,float]],tuple[tuple[float,float],tuple[float,float]]]
    selected_input_probability_scale: float

    def matrix(self) -> np.ndarray:
        return np.array([[complex(*self.logical_map_re_im[i][j]) for j in range(2)] for i in range(2)],complex)


def _ps_to_xz(ps: str):
    x=z=0
    for q,p in enumerate(ps):
        if p in 'XY': x|=1<<q
        if p in 'YZ': z|=1<<q
    return x,z


def _mat_payload(A: np.ndarray):
    return tuple(tuple((float(A[i,j].real),float(A[i,j].imag)) for j in range(2)) for i in range(2))


@functools.lru_cache(maxsize=8192)
def transfer_cached(key: tuple[tuple[int,int], ...]):
    faults=dict(key)
    accepted_inputs=[]
    for syn in range(64):
        ps=s8live.RECOVERY_TABLE[syn].pauli_string
        if _pauli_propagation_reference(faults,_ps_to_xz(ps))[0]: accepted_inputs.append(syn)
    if not accepted_inputs:
        return None
    if len(accepted_inputs)!=1:
        raise AssertionError(('ED all-zero branch must select at most one input syndrome',key,accepted_inputs))
    si=int(accepted_inputs[0])
    inp=s8live.BLOCK_BASIS[:,2*si:2*si+2]
    out=accepted_apply(inp,faults)
    coeff=s8live.BLOCK_BASIS.conj().T@out
    weights=np.sum(np.abs(coeff.reshape(64,2,2))**2,axis=(1,2))
    nz=np.nonzero(weights>1e-18)[0]
    if len(nz)!=1:
        raise AssertionError(('ED output not syndrome-sharp',key,si,nz.tolist(),weights[nz].tolist()))
    so=int(nz[0]); A=coeff[2*so:2*so+2,:]
    # Logical map should be proportional to a unitary because this is a
    # postselected Pauli-fault branch of a Clifford syndrome instrument.
    gram=A.conj().T@A; scale=float(np.trace(gram).real/2)
    if np.linalg.norm(gram-scale*np.eye(2))>1e-9:
        raise AssertionError(('logical map not proportional to unitary',key,si,so,gram))
    return EDTransfer(si,so,_mat_payload(A),scale)


def transfer_for_faults(faults: dict[int,int]|None=None):
    key=tuple(sorted((int(k),int(v)) for k,v in (faults or {}).items()))
    return transfer_cached(key)


def apply_two_block_midpoint_compact(st: np.ndarray, faults_a: dict[int,int]|None=None, faults_b: dict[int,int]|None=None):
    """Exact physical ED using syndrome-sector transfer maps instead of 128x128 Kraus matrices."""
    ta=transfer_for_faults(faults_a); tb=transfer_for_faults(faults_b)
    if ta is None or tb is None: return np.asarray(st,complex)*0,0.0,None
    C,psum,recon,_=s8live.exact_sector_tensor(st)
    if abs(float(psum)-1)>1e-9 or float(recon)>1e-9: raise AssertionError((psum,recon))
    block=C[ta.input_syndrome,:,tb.input_syndrome,:]
    D=ta.matrix()@block@tb.matrix().T
    pr=float(np.vdot(D,D).real)
    if pr<=1e-20: return np.asarray(st,complex)*0,0.0,(ta,tb)
    UA=s8live.BLOCK_BASIS[:,2*ta.output_syndrome:2*ta.output_syndrome+2]
    UB=s8live.BLOCK_BASIS[:,2*tb.output_syndrome:2*tb.output_syndrome+2]
    M=UA@D@UB.T
    out=np.empty_like(np.asarray(st,complex)); out[PAIR_INDEX]=M; out/=math.sqrt(pr)
    return out,pr,(ta,tb)


def sample_ed_faults(p: float, rng: random.Random):
    faults={}; rows=[]
    for i,op in enumerate(ED_OPS):
        if rng.random() < p:
            j=rng.randrange(len(ED_OPTIONS[i])); faults[i]=j
            rows.append({'op_index':i,'op_type':op.typ,'qubits':list(op.qs),'basis':op.basis,'label':op.label,'fault_class':ED_OPTIONS[i][j][0],'fault_option_index':j})
    return faults,rows


def _apply_pauli_vec(v: np.ndarray, x: int, z: int) -> np.ndarray:
    out=np.asarray(v,complex).copy(); idx=np.arange(128,dtype=np.int64)
    # X permutation first, then Z/Y phase convention equivalent up to global phase.
    perm=idx ^ int(x)
    out=out[perm]
    # phase for Z support evaluated on original output basis; global i phases from Y irrelevant for comparison.
    phase=np.ones(128,complex)
    zz=int(z)
    for q in range(7):
        if (zz>>q)&1: phase*=np.where(((idx>>q)&1)==0,1,-1)
    return phase*out


def _pauli_propagation_reference(faults: dict[int,int], input_pauli=(0,0)):
    """Step-2 Pauli-frame reference, duplicated without importing its expensive top-level audit."""
    x,z=map(int,input_pauli); clicks=[]
    for idx,op in enumerate(ED_OPS):
        fi=faults.get(idx)
        if op.typ=='PREP':
            q=op.qs[0]; x &= ~(1<<q); z &= ~(1<<q)
            if fi is not None:
                p=ED_OPTIONS[idx][fi][0]
                if p in 'XY': x ^= 1<<q
                if p in 'YZ': z ^= 1<<q
        elif op.typ=='H':
            q=op.qs[0]; xb=(x>>q)&1; zb=(z>>q)&1
            if xb!=zb: x^=1<<q; z^=1<<q
            if fi is not None:
                p=ED_OPTIONS[idx][fi][0]
                if p in 'XY': x ^= 1<<q
                if p in 'YZ': z ^= 1<<q
        elif op.typ=='CX':
            c,t=op.qs
            if (x>>c)&1: x ^= 1<<t
            if (z>>t)&1: z ^= 1<<c
            if fi is not None:
                lab=ED_OPTIONS[idx][fi][0]
                for p,q in zip(lab,(c,t)):
                    if p in 'XY': x ^= 1<<q
                    if p in 'YZ': z ^= 1<<q
        elif op.typ=='MEAS':
            q=op.qs[0]; flip=((x>>q)&1) if op.basis=='Z' else ((z>>q)&1)
            if fi is not None: flip^=1
            clicks.append(int(flip)); x &= ~(1<<q); z &= ~(1<<q)
    return all(v==0 for v in clicks), x&0x7f, z&0x7f


def self_test(single_limit: int|None=None):
    P=np.outer(m.zero,m.zero.conj())+np.outer(m.one,m.one.conj())
    K0=kraus_for_faults({})
    # Projectors may differ by an overall phase only.
    ov=np.vdot(P.reshape(-1),K0.reshape(-1)); phase=ov/abs(ov) if abs(ov)>1e-15 else 1
    projector_error=float(np.linalg.norm(K0-phase*P))
    if projector_error>1e-10: raise AssertionError(('fault-free ED != code projector',projector_error))

    incoming=[]
    for p in 'XYZ':
        for q in range(7):
            x=(1<<q) if p in 'XY' else 0; z=(1<<q) if p in 'YZ' else 0
            v=_apply_pauli_vec(m.zero,x,z); n=float(np.linalg.norm(K0@v))
            if n>1e-10: incoming.append((p,q,n))
    if incoming: raise AssertionError(('undetected incoming weight1',incoming[:5]))

    cases=[]; accepted=0; mismatches=[]; max_action_err=0.0; ncase=0
    for i,opts in enumerate(ED_OPTIONS):
        for j in range(len(opts)):
            ncase+=1
            if single_limit is not None and ncase>single_limit: break
            ref_acc,x,z=_pauli_propagation_reference({i:j})
            outs=accepted_apply(np.stack([m.zero,m.one],axis=1),{i:j})
            norms=[float(np.linalg.norm(outs[:,k])) for k in range(2)]
            exact_acc=max(norms)>1e-10
            if exact_acc != ref_acc:
                mismatches.append({'op':i,'opt':j,'ref_accept':ref_acc,'norms':norms})
                continue
            if ref_acc:
                accepted+=1
                # Compare the accepted logical action against the propagated residual Pauli on both code basis states.
                cols=[]; refs=[]
                for k,v in enumerate((m.zero,m.one)):
                    a=outs[:,k]; r=_apply_pauli_vec(v,x,z)
                    if np.linalg.norm(a)>1e-15: a=a/np.linalg.norm(a)
                    r=r/np.linalg.norm(r); cols.append(a); refs.append(r)
                # fit one common global phase using both columns
                vv=np.concatenate(cols); rr=np.concatenate(refs); o=np.vdot(rr,vv); ph=o/abs(o) if abs(o)>1e-15 else 1
                err=float(np.linalg.norm(vv-ph*rr)); max_action_err=max(max_action_err,err)
                if err>1e-9:
                    mismatches.append({'op':i,'opt':j,'action_error':err,'residual_x':x,'residual_z':z,'label':opts[j][0]})
            if len(cases)<10: cases.append({'op':i,'opt':j,'label':opts[j][0],'accepted':ref_acc,'residual_x':x,'residual_z':z,'norms':norms})
        if single_limit is not None and ncase>single_limit: break
    if mismatches: raise AssertionError(('single-fault exact/reference mismatch',mismatches[:3]))

    # Two-block fault-free physical ED must reproduce the old ideal midpoint projection.
    rng=np.random.default_rng(20260820); max_two_block=0.0; max_pr=0.0
    for _ in range(4):
        st=rng.normal(size=4**7)+1j*rng.normal(size=4**7); st=st/np.linalg.norm(st)
        a,pa=apply_two_block_midpoint(st); b,pb=m.proj(st)
        if pa>1e-15 and pb>1e-15:
            o=np.vdot(b,a); ph=o/abs(o) if abs(o)>1e-15 else 1; max_two_block=max(max_two_block,float(np.linalg.norm(a-ph*b)))
        max_pr=max(max_pr,abs(pa-pb))
    if max_two_block>1e-9 or max_pr>1e-9: raise AssertionError((max_two_block,max_pr))

    # Compact syndrome-transfer implementation must equal full Kraus application.
    rng2=np.random.default_rng(914); compact_state_err=0.0; compact_pr_err=0.0; compact_cases=0
    candidates=[]
    for i,opts in enumerate(ED_OPTIONS):
        for j in range(len(opts)):
            if _pauli_propagation_reference({i:j})[0]: candidates.append((i,j))
    for i,j in candidates[::max(1,len(candidates)//20)][:20]:
        # Build an input with support on many syndrome sectors.
        st=rng2.normal(size=4**7)+1j*rng2.normal(size=4**7); st/=np.linalg.norm(st)
        fa={i:j}; fb={}
        aa,pa=apply_two_block_midpoint(st,fa,fb); bb,pb,_=apply_two_block_midpoint_compact(st,fa,fb)
        compact_pr_err=max(compact_pr_err,abs(pa-pb))
        if pa>1e-15 and pb>1e-15:
            o=np.vdot(aa,bb); ph=o/abs(o) if abs(o)>1e-15 else 1
            compact_state_err=max(compact_state_err,float(np.linalg.norm(aa-ph*bb)))
        compact_cases+=1
    if compact_state_err>1e-9 or compact_pr_err>1e-9: raise AssertionError(('compact transfer mismatch',compact_state_err,compact_pr_err))

    return {
        'stage':'S9-A', 'inventory':inventory(), 'faultfree_kraus_projector_error':projector_error,
        'incoming_weight1_undetected':incoming, 'single_fault_cases_checked':min(ncase,sum(len(x) for x in ED_OPTIONS)),
        'single_fault_accepted':accepted, 'single_fault_reference_mismatches':len(mismatches),
        'max_accepted_single_action_error_vs_step2_pauli_reference':max_action_err,
        'two_block_faultfree_vs_old_ideal_projection_state_error':max_two_block,
        'two_block_faultfree_vs_old_ideal_projection_probability_error':max_pr,
        'compact_transfer_cases_checked':compact_cases,
        'compact_vs_full_kraus_state_error':compact_state_err,
        'compact_vs_full_kraus_probability_error':compact_pr_err,
        'passed':True,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--single-limit',type=int,default=None)
    ap.add_argument('--out-dir',type=Path,default=HERE.parent/'s9a_results')
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        r=self_test(a.single_limit); (a.out_dir/'s9a_self_test.json').write_text(json.dumps(r,indent=2),encoding='utf-8'); print(json.dumps(r,indent=2)); print('S9-A PASS: explicit physical Steane-ED accept instrument matches the preserved Step-2 contract')
    else:
        print(json.dumps(inventory(),indent=2))

if __name__=='__main__': main()
