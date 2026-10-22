import stim, collections, sys
from lp_retime import retime, census
NOISE={"DEPOLARIZE1","DEPOLARIZE2","X_ERROR","Y_ERROR","Z_ERROR","PAULI_CHANNEL_1","PAULI_CHANNEL_2"}
ANN={"DETECTOR","OBSERVABLE_INCLUDE","QUBIT_COORDS","SHIFT_COORDS","TICK","MPAD"}
def ops_of(path):
    c=stim.Circuit(open(path).read()).flattened(); tick=0; ops=[]; last_noise=0; tk=[]
    for inst in c:
        n=inst.name
        if n=="TICK": tick+=1; continue
        if n in NOISE: last_noise=tick; continue
        if n in ANN: continue
        tg=[t.value for t in inst.targets_copy() if t.is_qubit_target]
        if n in ("CX","CZ","CY","SWAP"):
            for a,b in zip(tg[::2],tg[1::2]): ops.append((n,(f"q{a}",f"q{b}"))); tk.append(tick)
        elif n=="MPP":
            ops.append((n,tuple(f"q{a}" for a in tg))); tk.append(tick)
        else:
            for a in tg: ops.append((n,(f"q{a}",))); tk.append(tick)
    keep=[i for i,t in enumerate(tk) if t<=last_noise]
    return [ops[i] for i in keep],[tk[i] for i in keep]
if __name__=="__main__":
    for p in sys.argv[1:]:
        ops,tk=ops_of(p)
        cls=lambda q:'q'
        print(p.split('/')[-3], "original:",census(ops,tk,cls))
        mpp=[o for o in ops if o[0]=="MPP"]; print("  MPP ops:",len(mpp), [len(o[1]) for o in mpp])
        t,_=retime(ops); print("  LP retimed:",census(ops,t,cls))
