from __future__ import annotations
import importlib.util,sys,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
P=Path('/mnt/data/s15_work/direct_cs_s15_single_fault_audit.py')
spec=importlib.util.spec_from_file_location('s15o3base',P); m=importlib.util.module_from_spec(spec); sys.modules['s15o3base']=m; spec.loader.exec_module(m)

def pack(r): return r['sx_bits'] | (r['sz_bits']<<84), r['logical_x_commutation'] | (r['logical_z_commutation']<<1)
locdict=defaultdict(list)
for stage,fs in [('g57',m.G57_FAULTS),('g713',m.G713_FAULTS)]:
 for f in fs: locdict[(stage,f.line,f.kind,f.sub)].append(pack(m.simulate_fault(f)))
locs=list(locdict.values()); L=len(locs); assert L==1379
single={0:0}
for cs in locs:
 for s,l in cs:
  if s in single and single[s]!=l: raise AssertionError
  single[s]=l
items=list(single.items())
dec2={}
for i,(s1,l1) in enumerate(items):
 for s2,l2 in items[i:]:
  s=s1^s2; l=l1^l2
  if s in dec2 and dec2[s]!=l: raise AssertionError('o2 amb')
  dec2[s]=l
cache={}
def decode3(s):
 if s in dec2: return ('o2',dec2[s])
 if s in cache:return cache[s]
 vals=set()
 for s1,l1 in items:
  q=s^s1
  if q in dec2:
   vals.add(l1^dec2[q])
   if len(vals)>1:
    cache[s]=('amb',None); return cache[s]
 if len(vals)==1:
  cache[s]=('o3',next(iter(vals)))
 else:
  cache[s]=('none',None)
 return cache[s]

def run(p,shots,seed):
 rng=np.random.default_rng(seed); ks=rng.binomial(L,p,size=shots); rr=np.random.default_rng(seed+77)
 counts=defaultdict(int); byk=defaultdict(lambda:defaultdict(int))
 for k0 in ks:
  k=int(k0); ss=ll=0
  if k:
   inds=rr.choice(L,size=k,replace=False)
   for idx in inds:
    cs=locs[int(idx)]; s,l=cs[int(rr.integers(len(cs)))]; ss^=s; ll^=l
  level,dl=decode3(ss)
  byk[k]['shots']+=1; counts[level]+=1
  if dl is None:
   counts['unresolved']+=1; byk[k]['unresolved']+=1
  elif dl!=ll:
   counts['errors']+=1; byk[k]['errors']+=1
 lower=counts['errors']/shots; upper=(counts['errors']+counts['unresolved'])/shots
 return {'p':p,'shots':shots,'counts':dict(counts),'logical_lower':lower,'logical_upper':upper,'unresolved_fraction':counts['unresolved']/shots,
         'two_patch_union_lower':1-(1-lower)**2,'two_patch_union_upper':1-(1-upper)**2,
         'by_fault_count':{str(k):dict(v) for k,v in sorted(byk.items())},'decode3_cache_size':len(cache)}
if __name__=='__main__':
 rows=[]; t=time.time()
 for i,(p,n) in enumerate([(1e-3,200000)]):
  r=run(p,n,20260831+i*1000); rows.append(r); print('done',p,r['logical_lower'],r['logical_upper'],r['counts'],flush=True)
 out={'stage':'S15 growth finite-p with conservative exact <=3 coset decoder','decoder':'order<=2 exact table; unseen syndrome searched as single XOR order<=2. Unique logical coset -> resolved; multiple/none -> unresolved. This is conservative because physical-location feasibility is not used to eliminate spurious decompositions.',
      'rows':rows,'elapsed_seconds':time.time()-t}
 Path('/mnt/data/s15_work/s15_growth_order3_decoder_mc.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
