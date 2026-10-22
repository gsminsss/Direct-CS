from __future__ import annotations
import re, json, math, hashlib
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict

SRC = Path('/mnt/data/src2/2. Source code/reference/fd3_baseline_ckt.stim')
LINES = SRC.read_text().splitlines()
G57_START, G57_END = 603, 832
G713_START, G713_END = 833, 855


def opname(s:str)->str:
    return s.split()[0].split('(')[0] if s.strip() else ''

def ints_after_op(s:str):
    return [int(x) for x in s.split()[1:] if re.fullmatch(r'-?\d+', x)]

def parse_targets(s:str):
    # active op target ints only (no rec targets in growth stages)
    return [int(x) for x in s.split()[1:]]

def bitpar(x:int)->int:
    return x.bit_count() & 1

def mask(qs):
    m=0
    for q in qs:m |= 1<<q
    return m

# Qubit coordinates and final d13 data/stabilizer supports.
coords={}
for s in LINES:
    m=re.match(r'QUBIT_COORDS\(([^)]*)\)\s+(\d+)',s)
    if m:
        coords[int(m.group(2))]=tuple(map(float,m.group(1).split(',')))

# Extract final ideal decoder stabilizer supports from CNOT layers 858,860,862,864.
stab_support=defaultdict(set)
for ln in [858,860,862,864]:
    s=LINES[ln-1]
    assert opname(s)=='CX'
    t=parse_targets(s)
    assert len(t)%2==0
    for a,b in zip(t[::2],t[1::2]):
        if a>=300 and b<300:
            stab_support[a].add(b)
        elif b>=300 and a<300:
            stab_support[b].add(a)
        else:
            raise ValueError((ln,a,b))
X_STABS={a:mask(v) for a,v in stab_support.items() if 300<=a<=383}
Z_STABS={a:mask(v) for a,v in stab_support.items() if 384<=a<=467}
assert len(X_STABS)==84 and len(Z_STABS)==84
DATA=set()
for v in stab_support.values(): DATA |= v
assert len(DATA)==169
DATA_MASK=mask(DATA)
XLOG_MASK=mask([q for q in DATA if abs(coords[q][0]-0.0)<1e-9])
ZLOG_MASK=mask([q for q in DATA if abs(coords[q][1]-0.0)<1e-9])
assert XLOG_MASK.bit_count()==13 and ZLOG_MASK.bit_count()==13

D7_DATA=set(range(0,97,2))
G57_NEW_DATA=set(parse_targets(LINES[G57_START-1]))
OLD_D5_DATA=sorted(D7_DATA-G57_NEW_DATA)
assert len(OLD_D5_DATA)==25

@dataclass(frozen=True)
class Fault:
    line:int
    kind:str
    sub:int
    pauli:str
    denom:int


def active_faults(start:int,end:int):
    fs=[]
    counts=Counter()
    for ln in range(start,end+1):
        s=LINES[ln-1].strip()
        if not s: continue
        op=opname(s)
        if op=='R':
            qs=parse_targets(s)
            counts['PREP_RESET'] += len(qs)
            for j,q in enumerate(qs): fs.append(Fault(ln,'R',j,'X',1))
        elif op in ('H','S'):
            qs=parse_targets(s)
            counts['1Q'] += len(qs)
            for j,q in enumerate(qs):
                for p in 'XYZ': fs.append(Fault(ln,op,j,p,3))
        elif op in ('CX','CY','CZ'):
            qs=parse_targets(s)
            assert len(qs)%2==0
            counts[op] += len(qs)//2
            paulis=['I','X','Y','Z']
            choices=[a+b for a in paulis for b in paulis if a+b!='II']
            for j in range(len(qs)//2):
                for p in choices: fs.append(Fault(ln,op,j,p,15))
        elif op in ('M','MX','MY'):
            qs=parse_targets(s)
            counts['MEAS'] += len(qs)
            for j,q in enumerate(qs): fs.append(Fault(ln,op,j,'flip',1))
        elif op=='MR':
            qs=parse_targets(s)
            # not present in growth; would be 2 active locations under project convention.
            counts['MEAS'] += len(qs); counts['PREP_RESET'] += len(qs)
            for j,q in enumerate(qs):
                fs.append(Fault(ln,'MR_M',j,'flip',1))
                fs.append(Fault(ln,'MR_R',j,'X',1))
    return fs, counts

G57_FAULTS,G57_COUNTS=active_faults(G57_START,G57_END)
G713_FAULTS,G713_COUNTS=active_faults(G713_START,G713_END)
assert sum(G57_COUNTS.values())==971, G57_COUNTS
assert sum(G713_COUNTS.values())==408, G713_COUNTS


def inject_single(x:int,z:int,q:int,p:str):
    if p=='I': return x,z
    if p in ('X','Y'): x ^= 1<<q
    if p in ('Z','Y'): z ^= 1<<q
    return x,z

def inject_pair(x,z,q1,q2,p):
    x,z=inject_single(x,z,q1,p[0])
    x,z=inject_single(x,z,q2,p[1])
    return x,z

REC_RE=re.compile(r'rec\[-(\d+)\]')

def simulate_fault(f:Fault):
    x=z=0
    if f.kind=='INPUT':
        x,z=inject_single(x,z,OLD_D5_DATA[f.sub],f.pauli)
    rec=[]
    detectors=[]
    for ln in range(G57_START,G713_END+1):
        s=LINES[ln-1].strip()
        if not s: continue
        op=opname(s)
        if op in ('QUBIT_COORDS','TICK','SHIFT_COORDS','DEPOLARIZE1','DEPOLARIZE2','X_ERROR','Z_ERROR','Y_ERROR') or op=='I':
            continue
        if op=='R':
            qs=parse_targets(s); m=mask(qs); x &= ~m; z &= ~m
            if ln==f.line and f.kind=='R': x,z=inject_single(x,z,qs[f.sub],f.pauli)
        elif op=='H':
            qs=parse_targets(s); m=mask(qs); a=(x^z)&m; x^=a; z^=a
            if ln==f.line and f.kind=='H': x,z=inject_single(x,z,qs[f.sub],f.pauli)
        elif op=='S':
            qs=parse_targets(s); m=mask(qs); z ^= x&m
            if ln==f.line and f.kind=='S': x,z=inject_single(x,z,qs[f.sub],f.pauli)
        elif op in ('CX','CY','CZ'):
            qs=parse_targets(s); pairs=list(zip(qs[::2],qs[1::2]))
            for a,b in pairs:
                if op=='CX':
                    # all pairs in a layer are disjoint in these circuits
                    xa=(x>>a)&1; zb=(z>>b)&1
                    if xa: x ^= 1<<b
                    if zb: z ^= 1<<a
                elif op=='CZ':
                    xa=(x>>a)&1; xb=(x>>b)&1
                    if xb: z ^= 1<<a
                    if xa: z ^= 1<<b
                else: # CY = S_t CX S_t^dag, symplectic S/Sdag identical
                    # pre S on target
                    if (x>>b)&1: z ^= 1<<b
                    xa=(x>>a)&1; zb=(z>>b)&1
                    if xa: x ^= 1<<b
                    if zb: z ^= 1<<a
                    if (x>>b)&1: z ^= 1<<b
            if ln==f.line and f.kind==op:
                a,b=pairs[f.sub]; x,z=inject_pair(x,z,a,b,f.pauli)
        elif op in ('M','MX','MY'):
            qs=parse_targets(s)
            for j,q in enumerate(qs):
                xb=(x>>q)&1; zb=(z>>q)&1
                flip = xb if op=='M' else (zb if op=='MX' else xb^zb)
                if ln==f.line and f.kind==op and f.sub==j: flip ^= 1
                rec.append(int(flip))
                x &= ~(1<<q); z &= ~(1<<q)
        elif op=='MR':
            qs=parse_targets(s)
            for j,q in enumerate(qs):
                flip=(x>>q)&1
                if ln==f.line and f.kind=='MR_M' and f.sub==j: flip^=1
                rec.append(int(flip))
                x &= ~(1<<q); z &= ~(1<<q)
                if ln==f.line and f.kind=='MR_R' and f.sub==j: x ^= 1<<q
        elif op=='DETECTOR':
            ks=[int(v) for v in REC_RE.findall(s)]
            val=0
            for k in ks:
                if k>len(rec):
                    # References before G57 are noise-difference zero.
                    continue
                val ^= rec[-k]
            detectors.append(val)
        elif op=='OBSERVABLE_INCLUDE':
            pass
        else:
            raise ValueError((ln,op,s[:100]))
    x &= DATA_MASK; z &= DATA_MASK
    sx=tuple(a for a,m in X_STABS.items() if bitpar(z&m))  # syndrome from Z-errors
    sz=tuple(a for a,m in Z_STABS.items() if bitpar(x&m))  # syndrome from X-errors
    lx=bitpar(x & ZLOG_MASK)
    lz=bitpar(z & XLOG_MASK)
    return {
        'detector_weight':sum(detectors),
        'detector_count':len(detectors),
        'detector_bits':sum((b&1)<<i for i,b in enumerate(detectors)),
        'sx_weight':len(sx), 'sz_weight':len(sz),
        'sx_bits':sum(1<<i for i,a in enumerate(sorted(X_STABS)) if a in sx),
        'sz_bits':sum(1<<i for i,a in enumerate(sorted(Z_STABS)) if a in sz),
        'zero_syndrome':not sx and not sz,
        'logical_x_if_zero_syndrome':lx if not sx and not sz else None,
        'logical_z_if_zero_syndrome':lz if not sx and not sz else None,
        'logical_x_commutation':lx,
        'logical_z_commutation':lz,
        'data_weight':((x|z)&DATA_MASK).bit_count(),
        'x_weight':(x&DATA_MASK).bit_count(),
        'z_weight':(z&DATA_MASK).bit_count(),
    }


def audit(fs,name):
    agg=Counter(); weighted=Counter(); maxw=0; examples=[]
    bykind=defaultdict(lambda:Counter())
    record_map=defaultdict(set); final_syn_map=defaultdict(set); all_record_map=defaultdict(list)
    for i,f in enumerate(fs):
        r=simulate_fault(f)
        maxw=max(maxw,r['data_weight'])
        rej=r['detector_weight']>0
        zero=r['zero_syndrome']
        log=zero and ((r['logical_x_if_zero_syndrome'] or 0) or (r['logical_z_if_zero_syndrome'] or 0))
        agg['classes']+=1
        agg['detected_classes']+=int(rej)
        agg['zero_detector_classes']+=int(not rej)
        agg['zero_syndrome_classes']+=int(zero)
        agg['zero_syndrome_logical_classes']+=int(log)
        agg['nonzero_final_syndrome_classes']+=int(not zero)
        # first-order coefficient: each location fault occurs p, uniformly among classes at location.
        w=1/f.denom
        weighted['all_location_equivalents'] += w
        weighted['detector_reject_coeff'] += w*rej
        weighted['zero_detector_coeff'] += w*(not rej)
        weighted['zero_syndrome_logical_coeff'] += w*log
        bykind[f.kind]['classes'] += 1
        bykind[f.kind]['detected'] += int(rej)
        bykind[f.kind]['zero_syndrome_logical'] += int(log)
        lsig=(int(r['logical_x_if_zero_syndrome']) if zero else bitpar(0), int(r['logical_z_if_zero_syndrome']) if zero else bitpar(0))
        # Logical commutation signatures are needed for nonzero syndrome too; recompute by a helper run return fields below.
        lsig=(r['logical_x_commutation'], r['logical_z_commutation'])
        key_record=(r['detector_bits'],r['sx_bits'],r['sz_bits'])
        key_final=(r['sx_bits'],r['sz_bits'])
        record_map[key_record].add(lsig); final_syn_map[key_final].add(lsig)
        all_record_map[key_record].append((f,lsig))
        if (log or (not rej and not zero)) and len(examples)<12:
            examples.append({'fault':f.__dict__,'result':r})
    amb_rec=[k for k,v in record_map.items() if len(v)>1]
    amb_final=[k for k,v in final_syn_map.items() if len(v)>1]
    # Include no-fault all-zero record/syndrome with logical signature (0,0).
    zero_key_record=(0,0,0); zero_key_final=(0,0)
    record_map[zero_key_record].add((0,0)); final_syn_map[zero_key_final].add((0,0))
    amb_rec=[k for k,v in record_map.items() if len(v)>1]
    amb_final=[k for k,v in final_syn_map.items() if len(v)>1]
    return {
        'name':name,
        'aggregate':dict(agg),
        'weighted_first_order':{k:float(v) for k,v in weighted.items()},
        'max_final_data_weight_from_single_fault':maxw,
        'decoder_ambiguity':{
            'unique_record_keys':len(record_map),
            'ambiguous_full_record_keys':len(amb_rec),
            'unique_final_syndrome_keys':len(final_syn_map),
            'ambiguous_final_syndrome_keys':len(amb_final),
            'full_record_single_fault_correctable':len(amb_rec)==0,
            'final_syndrome_only_single_fault_correctable':len(amb_final)==0,
        },
        'by_kind':{k:dict(v) for k,v in bykind.items()},
        'examples':examples,
    }

if __name__=='__main__':
    a57=audit(G57_FAULTS,'d5_to_d7')
    a713=audit(G713_FAULTS,'d7_to_d13')
    out={
      'source':str(SRC),
      'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
      'ranges':{'g57':[G57_START,G57_END],'g713':[G713_START,G713_END]},
      'active_counts':{'g57':dict(G57_COUNTS),'g57_total':sum(G57_COUNTS.values()),'g713':dict(G713_COUNTS),'g713_total':sum(G713_COUNTS.values())},
      'final_code':{'data_qubits':len(DATA),'X_stabilizers':len(X_STABS),'Z_stabilizers':len(Z_STABS),'logical_X_support_weight':XLOG_MASK.bit_count(),'logical_Z_support_weight':ZLOG_MASK.bit_count()},
      'audit':{'g57':a57,'g713':a713},
    }
    print(json.dumps(out,indent=2))
