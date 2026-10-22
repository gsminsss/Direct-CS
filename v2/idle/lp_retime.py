"""Certificate-preserving retiming: per-qubit operation order fixed, only layer times change.
min sum_q (t_last(q) - t_first(q))  s.t.  t_next >= t_prev + 1 on every qubit.
Difference constraints -> totally unimodular -> LP optimum is integral."""
import collections, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
MEAS={"M","MX","MY","MZ","MEAS"}; RESET={"R","RX","RY","RZ","PREP"}
def free_gap(ka,kb):
    """A measured qubit that is next reset carries no state: that gap is not an idle location."""
    return ka in MEAS and kb in RESET
def retime(ops):
    n=len(ops); seq=collections.defaultdict(list)
    for i,(k,qs) in enumerate(ops):
        for q in qs: seq[q].append(i)
    rows=[];cols=[];vals=[];b=[]
    r=0
    for q,s in seq.items():
        for a,c in zip(s,s[1:]):
            rows+= [r,r]; cols+=[a,c]; vals+=[1,-1]; b.append(-1); r+=1   # t_a - t_c <= -1
    A=sp.csr_matrix((vals,(rows,cols)),shape=(r,n))
    c=np.zeros(n)
    for q,s in seq.items():
        c[s[-1]]+=1; c[s[0]]-=1
        for a_,b_ in zip(s,s[1:]):
            if free_gap(ops[a_][0],ops[b_][0]): c[a_]+=1; c[b_]-=1   # segment ends at a, restarts at b
    res=linprog(c,A_ub=A,b_ub=b,bounds=[(0,None)]*n,method="highs")
    t=np.round(res.x).astype(int); t-=t.min()
    assert abs(res.fun-(c@t))<1e-6
    return t, seq
def census(ops,t,cls=lambda q:'data' if q[0] in 'AB' else q.rstrip('0123456789')):
    life=collections.defaultdict(list)
    for (k,qs),s in zip(ops,t):
        for q in qs: life[q].append((int(s),k))
    idle=collections.Counter(); act=collections.Counter(); qt=0
    for q,ts in life.items():
        L=len(ts); act[cls(q)]+=len(ts)
        for (ta,ka),(tb,kb) in zip(ts,ts[1:]):
            if not free_gap(ka,kb): L+=tb-ta-1; idle[cls(q)]+=tb-ta-1
        qt+=L
    return dict(depth=int(max(t))+1, qubit_ticks=qt, active_slots=sum(act.values()), idle=dict(idle), idle_total=sum(idle.values()))
if __name__=="__main__":
    import core_sched as cs
    t,_=retime(cs.ops); print("LP retiming:",census(cs.ops,t))
