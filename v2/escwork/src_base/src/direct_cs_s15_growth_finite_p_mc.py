from __future__ import annotations
import importlib.util,sys,math,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
P=Path('/mnt/data/s15_work/direct_cs_s15_single_fault_audit.py')
spec=importlib.util.spec_from_file_location('s15mcbase',P); m=importlib.util.module_from_spec(spec); sys.modules['s15mcbase']=m; spec.loader.exec_module(m)

def pack(r):
    return r['sx_bits'] | (r['sz_bits']<<84), r['logical_x_commutation'] | (r['logical_z_commutation']<<1)
# group classes by physical location
locdict=defaultdict(list)
for stage,fs in [('g57',m.G57_FAULTS),('g713',m.G713_FAULTS)]:
    for f in fs:
        locdict[(stage,f.line,f.kind,f.sub)].append(pack(m.simulate_fault(f)))
locs=list(locdict.values()); L=len(locs); assert L==1379
# exact <=2 minimum-order syndrome -> logical coset table from unique single signatures
single={0:0}
for cs in locs:
    for s,l in cs:
        if s in single and single[s]!=l: raise AssertionError('single ambiguity')
        single[s]=l
items=list(single.items())
dec={}
for i,(s1,l1) in enumerate(items):
    for s2,l2 in items[i:]:
        s=s1^s2; l=l1^l2
        old=dec.get(s)
        if old is not None and old!=l: raise AssertionError('order2 ambiguity')
        dec[s]=l

def run(p,shots,seed):
    rng=np.random.default_rng(seed)
    karr=rng.binomial(L,p,size=shots)
    resolved=errors=unresolved=0
    byk=defaultdict(lambda:[0,0,0]) # shots, errors, unresolved
    # Random choice of locations conditional on K. Python per-shot, but K small in operating range.
    py_rng=np.random.default_rng(seed+99991)
    for k in karr:
        k=int(k); byk[k][0]+=1
        if k==0:
            ss=ll=0
        else:
            inds=py_rng.choice(L,size=k,replace=False)
            ss=ll=0
            for idx in inds:
                cs=locs[int(idx)]
                s,l=cs[int(py_rng.integers(len(cs)))]
                ss^=s; ll^=l
        dl=dec.get(ss)
        if dl is None:
            unresolved+=1; byk[k][2]+=1
        else:
            resolved+=1
            if dl!=ll:
                errors+=1; byk[k][1]+=1
    lower=errors/shots
    upper=(errors+unresolved)/shots
    # conditional on deterministic growth acceptance =1, same as per attempt
    return {
      'p':p,'shots':shots,'active_locations_per_patch':L,
      'resolved':resolved,'logical_errors_resolved':errors,'unresolved':unresolved,
      'logical_lower':lower,'logical_upper':upper,
      'unresolved_fraction':unresolved/shots,
      'two_patch_union_lower':1-(1-lower)**2,
      'two_patch_union_upper':1-(1-upper)**2,
      'fault_count_histogram':{str(k):{'shots':v[0],'resolved_errors':v[1],'unresolved':v[2]} for k,v in sorted(byk.items())}
    }

if __name__=='__main__':
    specs=[(1e-4,200000),(2e-4,200000),(5e-4,300000),(1e-3,500000),(2e-3,300000)]
    rows=[]
    t=time.time()
    for i,(p,n) in enumerate(specs):
        r=run(p,n,20260821+i*1000); rows.append(r); print('done',p,r['logical_lower'],r['logical_upper'],r['unresolved_fraction'],flush=True)
    out={'stage':'S15 growth-only finite-p MC','model':'1379 active locations per patch = public Sahay d5->d7 (971) + d7->d13 (408), local stochastic Pauli; no idle; no additional growth postselection',
         'decoder':'exact final-syndrome logical-coset lookup for all order<=2 growth faults; higher-order syndromes absent from table are unresolved',
         'rows':rows,'elapsed_seconds':time.time()-t,
         'interpretation':'lower counts only resolved logical failures; upper treats every unresolved higher-order syndrome as logical failure. This is a rigorous decoder-table bracket for the growth-only module, not yet the complete S12-front + growth E2E result.'}
    Path('/mnt/data/s15_work/s15_growth_finite_p_mc.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
