from __future__ import annotations
import importlib.util, sys, json
from collections import defaultdict
from pathlib import Path
P=Path('/mnt/data/s15_work/direct_cs_s15_single_fault_audit.py')
spec=importlib.util.spec_from_file_location('s15sf',P); m=importlib.util.module_from_spec(spec); sys.modules['s15sf']=m; spec.loader.exec_module(m)

# Map unique final syndrome signature -> logical commutation signature, plus locations realizing it.
sig=defaultdict(lambda: {'ls':set(),'locs':set(),'classes':0})
for stage,fs in [('g57',m.G57_FAULTS),('g713',m.G713_FAULTS)]:
    for f in fs:
        r=m.simulate_fault(f)
        s=(r['sx_bits'],r['sz_bits'])
        l=(r['logical_x_commutation'],r['logical_z_commutation'])
        d=sig[s]; d['ls'].add(l); d['locs'].add((stage,f.line,f.kind,f.sub)); d['classes']+=1
# no fault
sig[(0,0)]['ls'].add((0,0)); sig[(0,0)]['locs'].add(('none',0,'none',0)); sig[(0,0)]['classes']+=1
assert all(len(v['ls'])==1 for v in sig.values())
items=[]
for s,v in sig.items():
    l=next(iter(v['ls'])); items.append((s,l,v['locs'],v['classes']))

# Enumerate unordered pairs of unique signatures, excluding no-fault+no-fault.
# This is a conservative signature-level audit: same-location feasibility is not required initially.
pair_map=defaultdict(set)
pair_sources=defaultdict(list)
N=len(items)
for i in range(N):
    (s1,l1,loc1,c1)=items[i]
    for j in range(i,N):
        (s2,l2,loc2,c2)=items[j]
        # require at least one way to choose two distinct physical locations, unless one is no-fault.
        if s1==(0,0) and l1==(0,0) and ('none',0,'none',0) in loc1:
            # This is really order <=1, retain for consistency.
            feasible=True
        elif s2==(0,0) and l2==(0,0) and ('none',0,'none',0) in loc2:
            feasible=True
        else:
            feasible = (len(loc1|loc2)>=2) and not (len(loc1)==1 and loc1==loc2)
        if not feasible: continue
        so=(s1[0]^s2[0],s1[1]^s2[1]); lo=(l1[0]^l2[0],l1[1]^l2[1])
        pair_map[so].add(lo)
        if len(pair_map[so])>1 and len(pair_sources[so])<8:
            pair_sources[so].append({'i':i,'j':j,'s1':s1,'l1':l1,'s2':s2,'l2':l2})

amb={str(k):sorted(list(v)) for k,v in pair_map.items() if len(v)>1}
zero_syn=pair_map.get((0,0),set())
res={
 'single_unique_syndromes_including_no_fault':N,
 'order_le_2_output_syndromes':len(pair_map),
 'ambiguous_syndromes_order_le_2':len(amb),
 'zero_syndrome_logical_signatures_order_le_2':sorted(list(zero_syn)),
 'zero_syndrome_contains_nontrivial_logical':any(x!=(0,0) for x in zero_syn),
 'order2_final_syndrome_decoder_exists':len(amb)==0,
 'interpretation':'If false, final syndrome alone is insufficient to correct all <=2 growth faults. This does not yet test whether d5->d7 detector history resolves the ambiguity.',
}
print(json.dumps(res,indent=2))
Path('/mnt/data/s15_work/s15_order2_signature_audit.json').write_text(json.dumps(res,indent=2))
# Save only a few ambiguity source examples if any.
if amb:
    ex=[]
    for so,v in pair_map.items():
        if len(v)>1:
            ex.append({'syndrome':so,'logical_signatures':sorted(list(v)),'examples':pair_sources.get(so,[])})
            if len(ex)>=10: break
    Path('/mnt/data/s15_work/s15_order2_ambiguity_examples.json').write_text(json.dumps(ex,indent=2))
