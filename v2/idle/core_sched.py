"""Idle census of the Direct front end + core under ASAP and ALAP layer schedules.
Primitive convention as in S13-A: PREP/M/H/1Q/CX/CZZ/MX each one layer; idles not charged."""
import collections
ENC=((0,1),(5,3),(6,2),(4,1),(0,2),(6,3),(5,1),(4,6))
CAT7_GATES=((0,4),(0,2),(4,6),(0,1),(2,3),(4,5),(7,9),(7,8),(9,10),(0,7),(2,9),(4,8),(5,10))
ops=[]   # (kind, qubits tuple, is_location)
fresh=[0]
def new(tag): fresh[0]+=1; return f"{tag}{fresh[0]}"
def D(b,j): return f"{b}{j}"
def zsnap(b, ops):
    # mirrors s9 z_snapshot_ops with fresh ancillas
    for grp in (("0145",[(1,'b'),(4,'a'),(5,'b'),(0,'b')]),("0235",[(5,'b'),(3,'b'),(0,'b'),(2,'a')])):
        a,bb=new('ed'),new('ed'); L={'a':a,'b':bb}
        ops+= [("PREP",(a,)),("PREP",(bb,)),("CX",(a,bb))]+[("CX",(D(b,d),L[t])) for d,t in grp[1]]+[("CX",(a,bb)),("M",(a,)),("M",(bb,))]
    a,bb,c=new('ed'),new('ed'),new('ed')
    ops+= [("PREP",(a,)),("PREP",(bb,)),("PREP",(c,)),("CX",(bb,a)),("CX",(c,a)),("CX",(D(b,0),a)),("CX",(D(b,4),bb)),("CX",(D(b,2),c)),("CX",(D(b,6),bb)),("CX",(bb,a)),("CX",(c,a)),("M",(a,)),("M",(bb,)),("M",(c,))]
def ED(b):
    o=[]; zsnap(b,o); o+=[("H",(D(b,q),)) for q in range(7)]; zsnap(b,o); o+=[("H",(D(b,q),)) for q in range(7)]; return o
for b in "AB":
    ops+= [("PREP",(D(b,q),)) for q in range(7)] + [("CX",(D(b,c),D(b,t))) for c,t in ENC]
    ops+= ED(b)
SEQ={'C1':['1Q1','CX','1Q0','1Q1','CZZ','1Q0','1Q1','CX','1Q1'],
     'C2':['1Q0','CX','1Q0','1Q1','CZZ','1Q0','1Q1','CX','1Q0'],
     'C12':['1Q0','1Q1','CZZ','1Q0','1Q1']}
def check(ck):
    cat=[new('cat') for _ in range(11)]
    o=[("PREP",(q,)) for q in cat]+[("CX",(cat[c],cat[t])) for c,t in CAT7_GATES]+[("M",(cat[q],)) for q in range(7,11)]
    for j in range(7):
        a,bq=D('A',j),D('B',j)
        for g in SEQ[ck]:
            if g=='1Q0': o.append(("1Q",(a,)))
            elif g=='1Q1': o.append(("1Q",(bq,)))
            elif g=='CX': o.append(("CX",(a,bq)))
            else: o.append(("CZZ",(a,bq,cat[j])))
    o+=[("MX",(cat[j],)) for j in range(7)]
    return o
for i,ck in enumerate(['C1','C2','C12','C1','C2','C12']):
    ops+=check(ck)
    if i==2: ops+=ED('A')+ED('B')
assert len(ops)==890, len(ops)
def asap(ops):
    ready=collections.defaultdict(int); t=[]
    for k,qs in ops:
        s=max(ready[q] for q in qs); t.append(s)
        for q in qs: ready[q]=s+1
    return t
def alap(ops):
    T=max(asap(ops))
    ready=collections.defaultdict(lambda:T); t=[0]*len(ops)
    for i in range(len(ops)-1,-1,-1):
        k,qs=ops[i]; s=min(ready[q] for q in qs); t[i]=s
        for q in qs: ready[q]=s-1
    return t
def census(t):
    life=collections.defaultdict(list)
    for (k,qs),s in zip(ops,t):
        for q in qs: life[q].append(s)
    idle=collections.Counter(); act=collections.Counter()
    for q,ts in life.items():
        cls='data' if q[0] in 'AB' else q.rstrip('0123456789')
        idle[cls]+=max(ts)-min(ts)+1-len(ts); act[cls]+=len(ts)
    return dict(depth=max(t)+1, idle=dict(idle), idle_total=sum(idle.values()), active_slots=sum(act.values()))
# hybrid: data ASAP but ancilla ops as late as possible (ALAP) -> take per-class min of the two
a,b=census(asap(ops)),census(alap(ops))
print("ASAP",a); print("ALAP",b)
hyb={c:min(a['idle'][c],b['idle'][c]) for c in a['idle']}
print("per-class min",hyb,sum(hyb.values()))
def tight(ops):
    t=asap(ops)
    isdata=lambda q:q[0] in 'AB'
    nxt=collections.defaultdict(lambda:10**9)
    t2=list(t)
    for i in range(len(ops)-1,-1,-1):
        k,qs=ops[i]
        if all(not isdata(q) for q in qs) and k not in ("M","MX"):
            lim=min(nxt[q] for q in qs)-1
            if lim<10**8: t2[i]=max(t[i],lim)
        for q in qs: nxt[q]=min(nxt[q],t2[i])
    # validity: per-qubit strictly increasing in program order
    last={}
    for (k,qs),s in zip(ops,t2):
        for q in qs:
            assert s>last.get(q,-1),(k,qs); last[q]=s
    return t2
c=census(tight(ops)); print("TIGHT (ancilla prep just-in-time, measure ASAP)",c)
def tight2(ops):
    t=asap(ops); isdata=lambda q:q[0] in 'AB'
    # time of first data-touching use for each cat qubit (CZZ)
    use={}
    for (k,qs),s in zip(ops,t):
        if k=="CZZ":
            for q in qs:
                if q.startswith('cat'): use[q]=s
    # flag qubits share block with the cat: map flag -> min CZZ time of its block (cat ids consecutive per block)
    def blk(q): return (int(q[3:])-fresh_base)//11
    t2=list(t); nxt=collections.defaultdict(lambda:10**9)
    blkuse=collections.defaultdict(lambda:10**9)
    for q,s in use.items(): blkuse[blk(q)]=min(blkuse[blk(q)],s)
    for i in range(len(ops)-1,-1,-1):
        k,qs=ops[i]
        if all(not isdata(q) for q in qs):
            if k in ("M","MX") and all(nxt[q]>=10**8 for q in qs):
                if qs[0].startswith('cat') and k=="M":
                    t2[i]=max(t[i],blkuse[blk(qs[0])]-1)
            else:
                lim=min(nxt[q] for q in qs)-1
                if lim<10**8: t2[i]=max(t[i],lim)
        for q in qs: nxt[q]=min(nxt[q],t2[i])
    last={}
    for (k,qs),s in zip(ops,t2):
        for q in qs:
            assert s>last.get(q,-1),(k,qs,s,last.get(q)); last[q]=s
    return t2
cats=sorted(int(q[3:]) for k,qs in ops for q in qs if q.startswith('cat'))
fresh_base=cats[0]
c=census(tight2(ops)); print("TIGHT2 (cat+flag verification just before CZZ)",c)
