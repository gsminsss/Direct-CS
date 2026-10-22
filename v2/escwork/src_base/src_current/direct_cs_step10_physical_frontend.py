#!/usr/bin/env python3
"""S10-A physical front-end primitives for the direct-|CS> factory.

Adds the two remaining front-end abstractions left after S9:

1) encoded |+>_L preparation per Steane block
   7 physical preparations -> 8-CNOT encoder -> explicit 82-location initial ED,
2) verified CAT7 ancilla preparation/verification and explicit seven-bit X readout
   for each of the six logical-check gadgets.

All faults are stochastic Pauli-after-location faults, matching the preserved
Step-2/Step-5 conventions.  This module supplies exact effective states/effects
for the non-Clifford core driver; it does not itself simulate the ZXZ escape.
"""
from __future__ import annotations
import collections, functools, importlib.util, json, math, sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError((name,path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

m=_load('s10_front_core',HERE/'direct_cs_step7_core_mc.py')
s8live=_load('s10_front_s8live',HERE/'direct_cs_step8_live_boundary.py')
s9ed=_load('s10_front_s9ed',HERE/'direct_cs_step9_physical_midpoint_ed.py')

# -----------------------------------------------------------------------------
# A. Physical encoded-|+> preparation: 7 preps + 8-CNOT encoder + initial ED.
# -----------------------------------------------------------------------------
ENC=((0,1),(5,3),(6,2),(4,1),(0,2),(6,3),(5,1),(4,6))
PREP_BASES=('X','Z','Z','Z','X','X','X')
PAULIS=('I','X','Y','Z')
NONID=('X','Y','Z')

def _cnot_prop(x,z,c,t):
    if (x>>c)&1:x^=1<<t
    if (z>>t)&1:z^=1<<c
    return x,z

def _prop_encoder(x,z,start):
    for c,t in ENC[start:]:x,z=_cnot_prop(x,z,c,t)
    return x,z

@dataclass(frozen=True)
class EncoderLocation:
    kind:str
    qubits:tuple[int,...]
    options:tuple[tuple[int,int,str],...]

ENCODER_LOCATIONS=[]
for q in range(7):
    opts=[]
    for p in NONID:
        x=(1<<q) if p in 'XY' else 0;z=(1<<q) if p in 'YZ' else 0
        x,z=_prop_encoder(x,z,0);opts.append((x,z,p))
    ENCODER_LOCATIONS.append(EncoderLocation('PREP',(q,),tuple(opts)))
for gi,(c,t) in enumerate(ENC):
    opts=[]
    for pc in PAULIS:
        for pt in PAULIS:
            if pc=='I' and pt=='I':continue
            x=z=0
            if pc in 'XY':x|=1<<c
            if pc in 'YZ':z|=1<<c
            if pt in 'XY':x|=1<<t
            if pt in 'YZ':z|=1<<t
            x,z=_prop_encoder(x,z,gi+1);opts.append((x,z,pc+pt))
    ENCODER_LOCATIONS.append(EncoderLocation('CX',(c,t),tuple(opts)))
ENCODER_LOCATION_COUNT=len(ENCODER_LOCATIONS)
if ENCODER_LOCATION_COUNT!=15:raise AssertionError(ENCODER_LOCATION_COUNT)


def encoder_error(faults:dict[int,int]|None=None):
    x=z=0
    for li,oi in (faults or {}).items():
        a,b,_=ENCODER_LOCATIONS[int(li)].options[int(oi)];x^=a;z^=b
    return int(x),int(z)

@functools.lru_cache(maxsize=16384)
def pauli_plus_sector_logical(x:int,z:int):
    """Return sharp syndrome and normalized logical amplitudes of E|+_L>."""
    ps=[]
    for q in range(7):
        a=(x>>q)&1;b=(z>>q)&1;ps.append('Y' if a and b else 'X' if a else 'Z' if b else 'I')
    v=s8live._apply_pauli_7q(m.plus,''.join(ps))
    c=s8live.BLOCK_BASIS.conj().T@v;c=c.reshape(64,2)
    w=np.sum(np.abs(c)**2,axis=1);nz=np.nonzero(w>1e-12)[0]
    if len(nz)!=1:raise AssertionError(('Pauli image not syndrome-sharp',x,z,nz.tolist()))
    s=int(nz[0]);lv=np.asarray(c[s],complex);n=float(np.linalg.norm(lv));lv/=n
    return s,lv


def apply_initial_ed_compact(encoder_x:int,encoder_z:int,ed_faults:dict[int,int]|None=None):
    """Exact accepted all-zero initial-ED branch on encoded E|+_L>.

    Returns (accepted, output_syndrome, normalized logical_vector, probability).
    """
    si,lv=pauli_plus_sector_logical(int(encoder_x),int(encoder_z))
    t=s9ed.transfer_for_faults(ed_faults or {})
    if t is None or int(t.input_syndrome)!=si:return False,None,None,0.0
    out=t.matrix()@lv;pr=float(np.vdot(out,out).real)
    if pr<=1e-18:return False,None,None,0.0
    return True,int(t.output_syndrome),out/math.sqrt(pr),pr


def block_state_from_sector(s:int,lv:np.ndarray):
    return s8live.BLOCK_BASIS[:,2*int(s):2*int(s)+2]@np.asarray(lv,complex)


def pair_block_states(a:np.ndarray,b:np.ndarray):
    return m.pair(np.asarray(a,complex),np.asarray(b,complex))


def sample_encoder_faults(p:float,rng):
    k=int(rng.binomial(ENCODER_LOCATION_COUNT,p));f={};rows=[]
    if k:
        for lid in np.asarray(rng.choice(ENCODER_LOCATION_COUNT,size=k,replace=False),dtype=np.int64):
            li=int(lid);oi=int(rng.integers(0,len(ENCODER_LOCATIONS[li].options)));f[li]=oi
            loc=ENCODER_LOCATIONS[li];rows.append({'location_id':li,'kind':loc.kind,'qubits':list(loc.qubits),'option_index':oi,'fault_class':loc.options[oi][2]})
    return f,rows


def sample_initial_ed_faults(p:float,rng):
    k=int(rng.binomial(len(s9ed.ED_OPS),p));f={};rows=[]
    if k:
        for lid in np.asarray(rng.choice(len(s9ed.ED_OPS),size=k,replace=False),dtype=np.int64):
            li=int(lid);oi=int(rng.integers(0,len(s9ed.ED_OPTIONS[li])));f[li]=oi;op=s9ed.ED_OPS[li]
            rows.append({'op_index':li,'op_type':op.typ,'qubits':list(op.qs),'basis':op.basis,'label':op.label,'option_index':oi,'fault_class':s9ed.ED_OPTIONS[li][oi][0]})
    return f,rows


def sample_block_frontend(p:float,rng):
    ef,erows=sample_encoder_faults(p,rng);df,drows=sample_initial_ed_faults(p,rng);x,z=encoder_error(ef)
    ok,s,lv,pr=apply_initial_ed_compact(x,z,df)
    return {'accepted':bool(ok),'syndrome':s,'logical_vector':lv,'accept_probability':float(pr),'encoder_faults':erows,'initial_ed_faults':drows,'encoder_fault_count':len(erows),'initial_ed_fault_count':len(drows),'encoder_error_x':x,'encoder_error_z':z}

# -----------------------------------------------------------------------------
# B. Verified CAT7 and explicit physical seven-bit readout.
# -----------------------------------------------------------------------------
CAT7_GATES=((0,4),(0,2),(4,6),(0,1),(2,3),(4,5),(7,9),(7,8),(9,10),(0,7),(2,9),(4,8),(5,10))
CAT7_NQ=11

def _cat_prop(x,z,start):
    for c,t in CAT7_GATES[start:]:x,z=_cnot_prop(x,z,c,t)
    return x,z

@dataclass(frozen=True)
class CATLocation:
    kind:str
    qubits:tuple[int,...]
    options:tuple[tuple[int,int,int,str],...]

CAT7_LOCATIONS=[]
for q in range(CAT7_NQ):
    opts=[]
    for p in NONID:
        x=(1<<q) if p in 'XY' else 0;z=(1<<q) if p in 'YZ' else 0;x,z=_cat_prop(x,z,0);opts.append((x,z,0,p))
    CAT7_LOCATIONS.append(CATLocation('PREP',(q,),tuple(opts)))
for gi,(c,t) in enumerate(CAT7_GATES):
    opts=[]
    for pc in PAULIS:
        for pt in PAULIS:
            if pc=='I' and pt=='I':continue
            x=z=0
            if pc in 'XY':x|=1<<c
            if pc in 'YZ':z|=1<<c
            if pt in 'XY':x|=1<<t
            if pt in 'YZ':z|=1<<t
            x,z=_cat_prop(x,z,gi+1);opts.append((x,z,0,pc+pt))
    CAT7_LOCATIONS.append(CATLocation('CX',(c,t),tuple(opts)))
for i,q in enumerate(range(7,11)):
    CAT7_LOCATIONS.append(CATLocation('MZ',(q,),((0,0,1<<i,'M'),)))
CAT7_LOCATION_COUNT=len(CAT7_LOCATIONS)
if CAT7_LOCATION_COUNT!=28:raise AssertionError(CAT7_LOCATION_COUNT)
CAT_READOUT_LOCATIONS=7

@dataclass(frozen=True)
class CATEffect:
    accepted:bool
    x_mask:int
    phase_flip:int
    verifier_pattern:int


def _canonical_cat_xmask(x:int):
    x=int(x)&0x7f
    # X^7 stabilizes the ideal GHZ/CAT state, so a mask and its complement are equivalent.
    y=x^0x7f
    return y if (y.bit_count(),y)<(x.bit_count(),x) else x


def cat_effect(faults:dict[int,int]|None=None):
    x=z=mf=0
    for li,oi in (faults or {}).items():
        a,b,c,_=CAT7_LOCATIONS[int(li)].options[int(oi)];x^=a;z^=b;mf^=c
    data_x=_canonical_cat_xmask(x&0x7f);verifier_x=(x>>7)&0xf;measured=verifier_x^mf
    acc=measured in (0,0xf);phase=(z&0x7f).bit_count()&1
    return CATEffect(bool(acc),int(data_x),int(phase),int(measured))


def sample_cat_preverification(p:float,rng):
    k=int(rng.binomial(CAT7_LOCATION_COUNT,p));f={};rows=[]
    if k:
        for lid in np.asarray(rng.choice(CAT7_LOCATION_COUNT,size=k,replace=False),dtype=np.int64):
            li=int(lid);oi=int(rng.integers(0,len(CAT7_LOCATIONS[li].options)));f[li]=oi;loc=CAT7_LOCATIONS[li]
            rows.append({'location_id':li,'kind':loc.kind,'qubits':list(loc.qubits),'option_index':oi,'fault_class':loc.options[oi][3]})
    return cat_effect(f),rows


def sample_cat_x_readout(p:float,rng):
    k=int(rng.binomial(CAT_READOUT_LOCATIONS,p))
    return int(k&1),k


def check_with_physical_cat(st,ck,fbc,cat:CATEffect,readout_parity:int,rng):
    """One exact logical-check instrument with verified-CAT residual effect.

    `cat.x_mask` toggles the controlled branch on affected coordinates.  A CAT
    Z-parity error and physical X-readout parity errors flip the recorded check
    bit.  Per-coordinate compiled-gate faults are the same `fbc` used by S7-S9.
    """
    if not cat.accepted:raise ValueError('cannot run rejected CAT')
    m0=[];m1=[];rf=0
    for j in range(7):
        fs=fbc.get(j,[]);xj=(int(cat.x_mask)>>j)&1
        if fs:
            u0,r0=m.coord(ck,0^xj,fs);u1,r1=m.coord(ck,1^xj,fs)
            if r0!=r1:raise AssertionError(('branch-dependent record flip',ck,j,r0,r1))
            rf^=int(r0)
        else:
            u0=m.ideal_u[ck][0^xj];u1=m.ideal_u[ck][1^xj]
        m0.append(u0);m1.append(u1)
    b0=m.prod(st,m0);b1=m.prod(st,m1);kp=(b0+b1)/2;km=(b0-b1)/2
    p0=float(np.vdot(kp,kp).real);p1=float(np.vdot(km,km).real);tot=p0+p1
    if tot<=1e-20:raise AssertionError('zero check probability')
    p0/=tot;raw=0 if float(rng.random())<p0 else 1;out=kp if raw==0 else km;out/=np.linalg.norm(out)
    recorded=int(raw)^int(rf)^int(cat.phase_flip)^int(readout_parity)
    return out,recorded


def forced_check_with_physical_cat(st,ck,fbc,cat:CATEffect,recorded:int,readout_parity:int=0):
    """Unnormalized deterministic branch for exact certification."""
    if not cat.accepted:return None,0.0
    m0=[];m1=[];rf=0
    for j in range(7):
        fs=fbc.get(j,[]);xj=(int(cat.x_mask)>>j)&1
        if fs:
            u0,r0=m.coord(ck,0^xj,fs);u1,r1=m.coord(ck,1^xj,fs)
            if r0!=r1: raise AssertionError((ck,j,r0,r1))
            rf^=int(r0)
        else:u0=m.ideal_u[ck][0^xj];u1=m.ideal_u[ck][1^xj]
        m0.append(u0);m1.append(u1)
    b0=m.prod(st,m0);b1=m.prod(st,m1)
    raw=int(recorded)^int(rf)^int(cat.phase_flip)^int(readout_parity)
    out=(b0+b1)/2 if raw==0 else (b0-b1)/2;pr=float(np.vdot(out,out).real)
    return out,pr


def cat_single_effect_catalogue():
    ctr=collections.Counter();rej=0
    for li,loc in enumerate(CAT7_LOCATIONS):
        for oi in range(len(loc.options)):
            e=cat_effect({li:oi})
            if not e.accepted:rej+=1;continue
            ctr[(e.x_mask,e.phase_flip)]+=1
    return ctr,rej


def cat_pair_effect_catalogue():
    ctr=collections.Counter();rej=0;total=0
    for i in range(CAT7_LOCATION_COUNT):
        for j in range(i+1,CAT7_LOCATION_COUNT):
            for oi in range(len(CAT7_LOCATIONS[i].options)):
                for oj in range(len(CAT7_LOCATIONS[j].options)):
                    total+=1;e=cat_effect({i:oi,j:oj})
                    if not e.accepted:rej+=1;continue
                    ctr[(e.x_mask,e.phase_flip)]+=1
    return ctr,rej,total


def inventory():
    return {
        'encoded_prep_locations_per_block':ENCODER_LOCATION_COUNT,
        'initial_ED_locations_per_block':len(s9ed.ED_OPS),
        'physical_frontend_locations_per_block':ENCODER_LOCATION_COUNT+len(s9ed.ED_OPS),
        'physical_frontend_locations_two_blocks':2*(ENCODER_LOCATION_COUNT+len(s9ed.ED_OPS)),
        'CAT_prep_verification_locations_per_check':CAT7_LOCATION_COUNT,
        'CAT_X_readout_locations_per_check':CAT_READOUT_LOCATIONS,
        'CAT_total_locations_per_check':CAT7_LOCATION_COUNT+CAT_READOUT_LOCATIONS,
        'six_CAT_total_locations':6*(CAT7_LOCATION_COUNT+CAT_READOUT_LOCATIONS),
    }


def self_test(include_cat_pairs=True):
    # Fault-free encoded prep + initial ED is exactly |+_L>.
    ok,s,lv,pr=apply_initial_ed_compact(0,0,{})
    plus_lv=np.array([1,1],complex)/math.sqrt(2)
    if not ok or s!=0 or 1-abs(np.vdot(plus_lv,lv))**2>1e-12 or abs(pr-1)>1e-12:raise AssertionError((ok,s,lv,pr))
    v=block_state_from_sector(s,lv);prep_err=float(np.linalg.norm(v-m.plus))
    if prep_err>1e-10:raise AssertionError(prep_err)

    # Full physical single encoder classes through fault-free initial ED.
    ec=collections.Counter();nenc=0
    for li,loc in enumerate(ENCODER_LOCATIONS):
        for oi in range(len(loc.options)):
            nenc+=1;x,z=encoder_error({li:oi});a,so,lvo,_=apply_initial_ed_compact(x,z,{})
            if not a:ec['reject']+=1
            else:
                # classify whether the accepted state is exactly ideal |+_L>.
                out=block_state_from_sector(so,lvo);fid=float(abs(np.vdot(m.plus,out))**2)
                ec['accepted_ideal' if 1-fid<1e-10 else 'accepted_nonideal']+=1
    if ec['accepted_nonideal']!=0:raise AssertionError(ec)

    # CAT contracts, exactly matching preserved Step-2 totals.
    c1,rej1=cat_single_effect_catalogue();acc1=sum(c1.values());bad1=[(k,v) for k,v in c1.items() if k[0].bit_count()>1]
    if (sum(len(x.options) for x in CAT7_LOCATIONS),acc1,rej1)!=(232,96,136) or bad1:raise AssertionError((acc1,rej1,bad1))
    pair=None
    if include_cat_pairs:
        c2,rej2,tot2=cat_pair_effect_catalogue();acc2=sum(c2.values());bad2=[(k,v) for k,v in c2.items() if k[0].bit_count()>2]
        if (tot2,acc2)!=(25398,5524) or bad2:raise AssertionError((tot2,acc2,bad2[:3]))
        pair={'total':tot2,'accepted':acc2,'rejected':rej2,'unique_accepted_effects':len(c2),'max_x_support':max(k[0].bit_count() for k in c2)}

    # Fault-free physical CAT check is exactly the previous ideal check.
    class RR:
        def __init__(self,x):self.x=list(x)
        def random(self):return self.x.pop(0)
    st=m.psi0.copy();r1=RR([0.123,0.456]);r2=RR([0.123,0.456])
    a,ma=m.check(st,'C1',collections.defaultdict(list),r1,0.0)
    b,mb=check_with_physical_cat(st,'C1',collections.defaultdict(list),CATEffect(True,0,0,0),0,r2)
    ov=np.vdot(a,b);ph=ov/abs(ov);chkerr=float(np.linalg.norm(a-ph*b))
    if ma!=mb or chkerr>1e-10:raise AssertionError((ma,mb,chkerr))

    return {'passed':True,'inventory':inventory(),'faultfree_prep_state_error':prep_err,'physical_encoder_single_classes':nenc,'encoder_single_through_faultfree_initial_ED':dict(ec),'CAT_single':{'total':232,'accepted':acc1,'rejected':rej1,'unique_accepted_effects':len(c1),'max_x_support':max(k[0].bit_count() for k in c1)},'CAT_pair':pair,'faultfree_CAT_check_state_error':chkerr}

if __name__=='__main__':
    r=self_test();print(json.dumps(r,indent=2));print('S10-A FRONT-END SELF-TEST PASS')
