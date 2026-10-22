import stim, numpy as np, sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'idle')
import count_conv as cc, retime3t
SRC="snap/cc/direct_cs_review/source/reference/fd3_baseline_ckt.stim"
ev, pr, nt = cc.parse(SRC); _, nat = cc.count(ev, pr, lambda t: t)
cum=np.cumsum([nat.get(t,0) for t in range(nt+1)]); FULL=1400; last=89
base=stim.Circuit.from_file(SRC); tick=0; dt=[]; co=base.get_detector_coordinates()
for inst in base.flattened():
    if inst.name=="TICK": tick+=1
    elif inst.name=="DETECTOR": dt.append(tick)
dt=np.array(dt); ps=np.array([(dt[i]<=52) and tuple(co[i][2:])!=(0.0,1.0) for i in range(len(dt))])
order=np.argsort(dt); ps_idx=[i for i in order if ps[i]]
res={}
for p in (5e-4,1e-3):
    for s in ("orig","lp"):
        L,n=retime3t.lines(p,s); c=stim.Circuit("\n".join(L))
        d=c.compile_detector_sampler(seed=7).sample(2000000)[:,ps_idx]; fired=d.any(1)
        first=np.where(fired, dt[ps_idx][np.argmax(d,1)], last)
        cost=cum[np.minimum(first,last)].astype(float); cost[~fired]=FULL
        res[f"{p:.0e}_{s}"]=dict(idle_slots=n,hard_discard=float(fired.mean()),rho=float(cost.mean()/FULL),mean_abort_tick=float(first[fired].mean()))
        print(p,s,res[f"{p:.0e}_{s}"],flush=True)
json.dump(res,open("idle/abort_lp.json","w"),indent=1)
