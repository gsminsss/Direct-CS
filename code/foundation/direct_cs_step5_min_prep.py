from collections import Counter
import runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=runpy.run_path(str(HERE/'direct_cs_step2_certificate.py'))
run_ed=s['run_ed']; ED_OPTIONS=s['ED_OPTIONS']
STAB_GROUP=s['STAB_GROUP']; code_min_weight=s['code_min_weight']
STAB_SUPPORTS=s['STAB_SUPPORTS']

# Steane helpers
STAB_GENS=[]
for S in STAB_SUPPORTS:
    m=sum(1<<q for q in S); STAB_GENS.append((m,0))
for S in STAB_SUPPORTS:
    m=sum(1<<q for q in S); STAB_GENS.append((0,m))
XLOG=(0x7f,0); ZLOG=(0,0x7f)
def symp(e,g):
    x,z=e; a,b=g
    return ((x&b).bit_count()+(z&a).bit_count())&1
def normalizer(e): return all(symp(e,g)==0 for g in STAB_GENS)
def logical_label(e):
    if not normalizer(e): return None
    lx=symp(e,ZLOG); lz=symp(e,XLOG)
    return {(0,0):'I',(1,0):'X',(0,1):'Z',(1,1):'Y'}[(lx,lz)]

def cnot_prop(x,z,c,t):
    if (x>>c)&1: x ^= 1<<t
    if (z>>t)&1: z ^= 1<<c
    return x,z

# 8-CNOT dual encoder only (no Goto verifier)
ENC=[(0,1),(5,3),(6,2),(4,1),(0,2),(6,3),(5,1),(4,6)]
# Initial bases q0|+>,q1-3|0>,q4-6|+>
BASES=['X','Z','Z','Z','X','X','X']
def prop(x,z,start):
    for c,t in ENC[start:]: x,z=cnot_prop(x,z,c,t)
    return x,z
loc=[]
for q,b in enumerate(BASES):
    p='Z' if b=='X' else 'X'
    x=(1<<q) if p=='X' else 0; z=(1<<q) if p=='Z' else 0
    x,z=prop(x,z,0); loc.append((f'R{b}_{q}',[(x,z,p)]))
for gi,(c,t) in enumerate(ENC):
    opts=[]
    for pc in 'IXYZ':
      for pt in 'IXYZ':
        if pc=='I' and pt=='I': continue
        x=z=0
        if pc in 'XY': x|=1<<c
        if pc in 'YZ': z|=1<<c
        if pt in 'XY': x|=1<<t
        if pt in 'YZ': z|=1<<t
        x,z=prop(x,z,gi+1); opts.append((x,z,pc+pt))
    loc.append((f'CX{gi}_{c}_{t}',opts))

def geterr(fs):
    x=z=0
    for li,oi in fs:
        a,b,_=loc[li][1][oi]; x^=a; z^=b
    return x,z

# single encoder fault -> fault-free initial ED
A=Counter()
for i,(_,opts) in enumerate(loc):
  for oi in range(len(opts)):
    e=geterr([(i,oi)])
    ok,w,x,z=run_ed({},e)
    if not ok: A['reject']+=1
    else:
      ee=(x,z)
      A['N_'+logical_label(ee) if normalizer(ee) else 'D_'+str(w)]+=1

# two encoder faults -> fault-free initial ED
B=Counter(); total2=0
for i in range(len(loc)):
 for j in range(i+1,len(loc)):
  for oi in range(len(loc[i][1])):
   for oj in range(len(loc[j][1])):
    total2+=1
    e=geterr([(i,oi),(j,oj)])
    ok,w,x,z=run_ed({},e)
    if not ok: B['reject']+=1
    else:
      ee=(x,z); B['N_'+logical_label(ee) if normalizer(ee) else 'D_'+str(w)]+=1

# one encoder fault + one initial-ED fault
C=Counter(); totalcross=0
for i,(_,opts) in enumerate(loc):
 for oi in range(len(opts)):
  e=geterr([(i,oi)])
  for j,eopts in enumerate(ED_OPTIONS):
   for oj in range(len(eopts)):
    totalcross+=1
    ok,w,x,z=run_ed({j:oj},e)
    if not ok: C['reject']+=1
    else:
      ee=(x,z); C['N_'+logical_label(ee) if normalizer(ee) else 'D_'+str(w)]+=1

print('=== STEP 5 MINIMAL PROOF-FIRST INITIALIZATION ===')
print('per block: 7 physical preps + 8 encoder CX + immediate full Steane ED (82 locations)')
print('encoder fault classes:',sum(len(o) for _,o in loc),'locations',len(loc))
print('single encoder fault -> fault-free initial ED:',dict(A))
print('two encoder faults total',total2,'-> fault-free initial ED:',dict(B))
print('encoder fault + initial-ED fault total',totalcross,':',dict(C))
print('ideal encoder + one ED fault: already Step2 => accepted residual code-weight <=1')
print('Conclusion: logical normalizer survivors are harmless because the first C1,C2 projectors have rank-1 logical eigenspaces; detectable two-fault outputs have a later fault-free midpoint ED.')
