import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import stim, sys, collections, re
sys.path.insert(0,str(_V2) + "/gapdec")
NOISE={"DEPOLARIZE1","DEPOLARIZE2","X_ERROR","Y_ERROR","Z_ERROR","PAULI_CHANNEL_1","PAULI_CHANNEL_2"}
ANN={"DETECTOR","OBSERVABLE_INCLUDE","QUBIT_COORDS","SHIFT_COORDS","TICK","MPAD"}
def census(c, cutoff=None):
    c=c.flattened(); tick=0; act=collections.defaultdict(set); lastnoise=0; noisy_meas=0
    for inst in c:
        n=inst.name
        if n=="TICK": tick+=1; continue
        if n in NOISE: lastnoise=tick; continue
        if n in ANN: continue
        if n in ("M","MX","MY","MR","MRX") and inst.gate_args_copy() and inst.gate_args_copy()[0]>0: lastnoise=tick
        for t in inst.targets_copy():
            if t.is_qubit_target: act[t.value].add(tick)
    T = cutoff if cutoff is not None else lastnoise
    life=0; slots=0
    for q,ts in act.items():
        ts=[t for t in ts if t<=T]
        if not ts: continue
        life+=max(ts)-min(ts)+1; slots+=len(ts)
    return dict(last_noisy_tick=T, qticks=life, active_slots=slots, idle_slots=life-slots)
if __name__=="__main__":
    import common
    S=str(_V2) + "/"
    full=stim.Circuit("\n".join(common.variant_text(1e-3,True)))
    noid=stim.Circuit("\n".join(common.variant_text(1e-3,False)))
    # idle noise count = DEPOLARIZE1 targets difference
    def d1(c): return sum(len(i.targets_copy()) for i in c.flattened() if i.name=="DEPOLARIZE1")
    print("3T idle DEPOLARIZE1 targets:", d1(full)-d1(noid), " census noisy region:", census(noid))
    def nloc(c): 
        k=collections.Counter()
        for i in c.flattened():
            if i.name in NOISE: k[i.name]+=len(i.targets_copy())//(2 if i.name.endswith("2") else 1)
        return dict(k)
    print("3T noise channels (no idle):", nloc(noid)); print("3T with idle:", nloc(full))
    for v in ("g3w0","opt3b_w0"):
        c=stim.Circuit(open(S+f"escwork/gen/{v}/s21opt3b_Z_direct_d5_d13_noisy.stim").read())
        print(v, census(c), nloc(c))
