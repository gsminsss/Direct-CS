from __future__ import annotations
import importlib.util, sys, json
from collections import defaultdict
from pathlib import Path
P=Path('/mnt/data/s15_work/direct_cs_s15_single_fault_audit.py')
spec=importlib.util.spec_from_file_location('s15mix',P); m=importlib.util.module_from_spec(spec); sys.modules['s15mix']=m; spec.loader.exec_module(m)
input_faults=[m.Fault(602,'INPUT',i,p,3) for i,q in enumerate(m.OLD_D5_DATA) for p in 'XYZ']
all_single=[('input',f) for f in input_faults]+[('g57',f) for f in m.G57_FAULTS]+[('g713',f) for f in m.G713_FAULTS]
# Build syndrome -> logical signature for all generalized order-1 boundary/growth events.
sig=defaultdict(lambda:{'ls':set(),'origins':set(),'classes':0})
input_stats={'classes':len(input_faults),'zero_syndrome_logical':0,'max_final_data_weight':0}
for origin,f in all_single:
    r=m.simulate_fault(f)
    s=(r['sx_bits'],r['sz_bits']); l=(r['logical_x_commutation'],r['logical_z_commutation'])
    sig[s]['ls'].add(l); sig[s]['origins'].add(origin); sig[s]['classes']+=1
    if origin=='input':
        input_stats['max_final_data_weight']=max(input_stats['max_final_data_weight'],r['data_weight'])
        if r['zero_syndrome'] and l!=(0,0): input_stats['zero_syndrome_logical']+=1
# no fault
sig[(0,0)]['ls'].add((0,0)); sig[(0,0)]['origins'].add('none'); sig[(0,0)]['classes']+=1
single_amb={s:v['ls'] for s,v in sig.items() if len(v['ls'])>1}
# For a conservative order-2 generalized audit, enumerate XOR of every unique single signature.
# We do not enforce detailed front-fault realizability; arbitrary weight-1 d5 input Paulis are included as one boundary event.
items=[]
for s,v in sig.items():
    if len(v['ls'])==1:
        items.append((s,next(iter(v['ls'])),v['origins']))
pairmap=defaultdict(set)
N=len(items)
for i in range(N):
    s1,l1,o1=items[i]
    for j in range(i,N):
        s2,l2,o2=items[j]
        so=(s1[0]^s2[0],s1[1]^s2[1]); lo=(l1[0]^l2[0],l1[1]^l2[1])
        pairmap[so].add(lo)
pair_amb={s:v for s,v in pairmap.items() if len(v)>1}
res={
 'boundary_model':'arbitrary one-qubit X/Y/Z error on any of the 25 d=5 data qubits at the d5->d7 growth input',
 'd5_input_data_qubits':m.OLD_D5_DATA,
 'input_single_fault_stats':input_stats,
 'generalized_order1_classes':len(all_single),
 'generalized_order1_unique_final_syndromes':len(sig),
 'generalized_order1_ambiguous_syndromes':len(single_amb),
 'generalized_order2_output_syndromes':len(pairmap),
 'generalized_order2_ambiguous_syndromes':len(pair_amb),
 'zero_syndrome_logical_signatures_order2':sorted(list(pairmap.get((0,0),set()))),
 'mixed_boundary_growth_order2_final_syndrome_decoder_exists':len(single_amb)==0 and len(pair_amb)==0,
 'scope_warning':'This is a conservative boundary-error-class audit, not a replacement for injecting the exact front-end S12/S11 emitted boundary classes with their probabilities. It certifies syndrome/coset distinguishability for arbitrary weight-1 d5 Pauli boundary errors plus growth faults under the linear Clifford model.'
}
print(json.dumps(res,indent=2))
Path('/mnt/data/s15_work/s15_mixed_boundary_order2.json').write_text(json.dumps(res,indent=2))
