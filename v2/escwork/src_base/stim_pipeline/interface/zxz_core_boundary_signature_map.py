#!/usr/bin/env python3
"""S8-B: map all 64 Steane boundary syndrome representatives into the k=2 ZXZ handoff.

The S8-A core boundary can be a coherent superposition of Steane syndrome
sectors.  This script asks whether the *observed ZXZ detector record* resolves
those sectors.

For each of the 21 incoming single-qubit Steane Pauli errors, immediately after
Hirano's ideal encoded-Steane initialization and before the first ZXZ
instruction, obtain the exact detector signature and complementary X/Z logical
probe flips.  Because the downstream circuit is Clifford, a canonical
weight-0/1/2 representative E_s has the GF(2) XOR signature of its constituents.

The operational schedule is the frozen S7-B4/B5 candidate:
    ZXZ + 2 faultable surface-QEC rounds -> live handoff (171 detector bits)
    + 1 noiseless audit tail round (24 extra bits, audit only)
    + terminal complementary probe (scoring only)

If all 64 canonical input syndromes have distinct 171-bit signatures, then for
a fixed stochastic escape-fault realization the measured detector record
separates the coherent S8-A syndrome sectors.  Since the escape noise model is
a classical stochastic Pauli mixture, the core boundary may then be treated as
an exact classical mixture over its S8-A syndrome-sector components *after the
ZXZ detector measurement*, while retaining each sector's arbitrary logical
state.
"""
from __future__ import annotations

import argparse, importlib, importlib.util, json, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
ZXZ_DIR=HERE.parent/'zxz'

def _find_src_current() -> Path:
    # Canonical layout: <project>/stim_pipeline/interface/this.py and
    # <project>/src_current/direct_cs_step8_live_boundary.py.
    candidates = [
        ROOT/'src_current',
        HERE.parent.parent/'src_current',
        HERE.parent.parent.parent/'src_current',
        Path.cwd()/'src_current',
    ]
    seen=set()
    for c in candidates:
        c=c.resolve()
        if c in seen:continue
        seen.add(c)
        if (c/'direct_cs_step8_live_boundary.py').is_file():
            return c
    raise FileNotFoundError(
        'Cannot locate src_current/direct_cs_step8_live_boundary.py. Tried: ' +
        ', '.join(str(x) for x in candidates)
    )

SRC_CURRENT=_find_src_current()
for p in (ZXZ_DIR,HERE,SRC_CURRENT):
    if str(p) not in sys.path:sys.path.append(str(p))

import zxz_faultable_handoff_recovery_audit as s7b3  # type: ignore
import zxz_online_handoff_audit as s7b  # type: ignore
import hirano_zxz_generator as s1  # type: ignore

spec=importlib.util.spec_from_file_location('s8live',SRC_CURRENT/'direct_cs_step8_live_boundary.py')
if spec is None or spec.loader is None:raise RuntimeError('cannot load S8-A')
s8live=importlib.util.module_from_spec(spec);sys.modules['s8live']=s8live;spec.loader.exec_module(s8live)

FAULTABLE_ROUNDS=2
IDEAL_TAIL_ROUNDS=1
ACTUAL_BITS=171
FULL_RECOVERY_BITS=195
FRAME_LABELS=('I','X','Z','Y')
PAULI1=('X','Y','Z')

@dataclass(frozen=True)
class IncomingSingle:
    coordinate:int;qubit_id:int;pauli:str
    actual_syndrome:int;full_recovery_syndrome:int;frame:int


def _flag(x:Any,name:str)->bool:
    v=getattr(x,name,False);return bool(v() if callable(v) else v)

def _val(x:Any)->int:
    if hasattr(x,'val'):return int(x.val)
    v=getattr(x,'value');return int(v() if callable(v) else v)

def _dem_signature(circuit:Any)->tuple[int,int]:
    dem=circuit.detector_error_model(decompose_errors=False).flattened()
    errs=[i for i in dem if getattr(i,'type',None)=='error']
    if not errs:return 0,0
    if len(errs)!=1:raise AssertionError(('expected <=1 boundary error DEM mechanism',len(errs),dem))
    d=o=0
    for t in errs[0].targets_copy():
        if _flag(t,'is_relative_detector_id'):d^=1<<_val(t)
        elif _flag(t,'is_logical_observable_id'):
            if _val(t)!=0:raise AssertionError(t)
            o^=1
        elif _flag(t,'is_separator'):raise AssertionError(('unexpected separator',errs[0]))
        else:raise AssertionError(t)
    return d,o

def _insert(circuit,idx,q,p,prob):
    import stim
    out=stim.Circuit();out+=circuit[:idx];out.append(f'{p}_ERROR',[q],prob);out+=circuit[idx:]
    if out.without_noise()!=circuit:raise AssertionError('boundary insertion changes ideal circuit')
    return out

def _steane_qubits(hirano_src:Path):
    _,util=s1._load_hirano(hirano_src);sc=importlib.import_module('steane_code');mapping=util.QubitMapping(30,30)
    pos=[tuple(getattr(sc,f'STEANE_{i}')) for i in range(7)];ids=[int(mapping.get_id(*x)) for x in pos]
    if len(set(ids))!=7:raise AssertionError(ids)
    return pos,ids

def _single_rows(hirano_src:Path,channel:str,prob:float):
    circuit,init_range,_faultable_end,recovery_ids,terminal_ids,_rec=s7b3._build_circuit(hirano_src,channel,FAULTABLE_ROUNDS,IDEAL_TAIL_ROUNDS)
    if len(recovery_ids)!=FULL_RECOVERY_BITS:raise AssertionError(len(recovery_ids))
    if len(terminal_ids)!=12:raise AssertionError(terminal_ids)
    boundary=int(init_range[1]);pos,ids=_steane_qubits(hirano_src);rows={}
    for coord,q in enumerate(ids):
        for p in PAULI1:
            d,o=_dem_signature(_insert(circuit,boundary,q,p,prob));rec=s7b._project_mask(d,recovery_ids);actual=rec&((1<<ACTUAL_BITS)-1)
            rows[(coord,p)]={'coordinate':coord,'position':list(pos[coord]),'qubit_id':q,'pauli':p,'actual_syndrome':actual,'full_recovery_syndrome':rec,'observable_flip':o}
    return rows,{'boundary_instruction_index':boundary,'steane_positions':[list(x) for x in pos],'steane_qubit_ids':ids,'num_detectors_total':int(circuit.num_detectors),'recovery_detector_bits':len(recovery_ids)}

def build_map(hirano_src:Path,prob:float):
    xr,xm=_single_rows(hirano_src,'X',prob);zr,zm=_single_rows(hirano_src,'Z',prob)
    singles={};mismatch=0
    for key in sorted(xr):
        x,z=xr[key],zr[key]
        if x['actual_syndrome']!=z['actual_syndrome'] or x['full_recovery_syndrome']!=z['full_recovery_syndrome']:mismatch+=1
        frame=int(z['observable_flip'])|(int(x['observable_flip'])<<1)
        singles[key]=IncomingSingle(key[0],x['qubit_id'],key[1],int(x['actual_syndrome']),int(x['full_recovery_syndrome']),frame)
    if mismatch:raise AssertionError(('X/Z incoming syndrome mismatch',mismatch))

    rows=[]
    for syn in range(64):
        rep=s8live.RECOVERY_TABLE[syn]
        actual=full=frame=0
        constituents=[]
        for coord,p in enumerate(rep.pauli_string):
            if p=='I':continue
            r=singles[(coord,p)];actual^=r.actual_syndrome;full^=r.full_recovery_syndrome;frame^=r.frame;constituents.append({'coordinate':coord,'pauli':p,'qubit_id':r.qubit_id})
        rows.append({'steane_syndrome':syn,'steane_syndrome_bits':format(syn,'06b'),'canonical_pauli':rep.pauli_string,'canonical_weight':rep.weight,'constituents':constituents,'actual_171_syndrome_hex':hex(actual),'actual_171_weight':actual.bit_count(),'full_195_syndrome_hex':hex(full),'full_195_weight':full.bit_count(),'logical_frame':FRAME_LABELS[frame],'logical_frame_int':frame})

    def collisions(field):
        g={}
        for r in rows:g.setdefault(r[field],[]).append(r['steane_syndrome'])
        return {k:v for k,v in g.items() if len(v)>1}
    ca=collisions('actual_171_syndrome_hex');cf=collisions('full_195_syndrome_hex')
    dmin=min((int(a['actual_171_syndrome_hex'],16)^int(b['actual_171_syndrome_hex'],16)).bit_count() for i,a in enumerate(rows) for b in rows[i+1:])
    summary={'faultable_handoff_rounds':FAULTABLE_ROUNDS,'ideal_tail_rounds':IDEAL_TAIL_ROUNDS,'actual_detector_bits':ACTUAL_BITS,'full_recovery_detector_bits':FULL_RECOVERY_BITS,'canonical_input_syndromes':64,'incoming_single_pauli_classes':21,'xz_incoming_syndrome_mismatches':mismatch,'actual_signature_unique_count':len({r['actual_171_syndrome_hex'] for r in rows}),'actual_signature_injective':not ca,'full_signature_unique_count':len({r['full_195_syndrome_hex'] for r in rows}),'full_signature_injective':not cf,'nontrivial_input_syndromes_with_zero_actual_signature':sum(r['steane_syndrome']!=0 and int(r['actual_171_syndrome_hex'],16)==0 for r in rows),'minimum_actual_signature_hamming_distance':dmin,'zero_input_signature_is_zero':int(rows[0]['actual_171_syndrome_hex'],16)==0 and rows[0]['logical_frame']=='I','coherent_sector_dephasing_by_actual_detector_record_certified':not ca,'actual_collision_groups':ca,'full_collision_groups':cf,'x_meta':xm,'z_meta':zm}
    return summary,rows,singles

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hirano-src',type=Path,required=True);ap.add_argument('--signature-p',type=float,default=1e-3);ap.add_argument('--out-dir',type=Path,default=HERE/'s8b_results');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    summary,rows,singles=build_map(a.hirano_src.resolve(),a.signature_p)
    (a.out_dir/'s8b_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8');(a.out_dir/'s8b_canonical_input_map.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(a.out_dir/'s8b_incoming_single_map.json').write_text(json.dumps([asdict(v)|{'actual_syndrome_hex':hex(v.actual_syndrome),'full_recovery_syndrome_hex':hex(v.full_recovery_syndrome),'frame_label':FRAME_LABELS[v.frame]} for _,v in sorted(singles.items())],indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2,sort_keys=True))
    if summary['actual_signature_injective']:
        print('S8-B PASS: all 64 Steane syndrome sectors are resolved by the 171-bit operational ZXZ detector record')
    else:
        print('S8-B RESULT: operational detector record has core-syndrome collisions; coherent-sector bridge remains unresolved')
if __name__=='__main__':main()
