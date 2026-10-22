#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import argparse, json

NOISE={"DEPOLARIZE1","DEPOLARIZE2","X_ERROR","Y_ERROR","Z_ERROR","PAULI_CHANNEL_1","PAULI_CHANNEL_2"}
ANN={"QUBIT_COORDS","DETECTOR","OBSERVABLE_INCLUDE","SHIFT_COORDS","TICK"}
ONE={"R","RX","RY","RZ","H","S","S_DAG","X","Y","Z","I","M","MX","MY","MZ","MR","MRX","MRY","MRZ"}
TWO={"CX","CY","CZ","XCZ","YCZ","SWAP","ISWAP","ISWAP_DAG"}

def audit(path):
    c=Counter(); noise=Counter()
    for line in Path(path).read_text().splitlines():
        s=line.strip()
        if not s or s.startswith("#"): continue
        name=s.split()[0].split("(")[0]
        args=s.split()[1:]
        if name in ANN: continue
        if name in NOISE:
            noise[name] += len(args)//2 if name.endswith("2") else len(args)
            continue
        if name in ONE:
            c[name] += len(args)
            continue
        if name in TWO:
            assert len(args)%2==0
            for i in range(0,len(args),2):
                a,b=args[i:i+2]
                ca=a.startswith("rec[") or a.startswith("sweep[")
                cb=b.startswith("rec[") or b.startswith("sweep[")
                if ca ^ cb:
                    c[name+"_FEEDBACK_1Q"] += 1
                elif not ca and not cb:
                    c[name] += 1
                else:
                    raise ValueError((name,a,b))
            continue
        raise ValueError(("unknown",name))
    return {
        "gate_counts":dict(c),
        "active_including_I":sum(c.values()),
        "I_idle":c["I"],
        "active_no_idle":sum(c.values())-c["I"],
        "noise_counts":dict(noise),
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stim_file")
    ap.add_argument("--out",default="gate_audit.json")
    a=ap.parse_args()
    x=audit(a.stim_file)
    Path(a.out).write_text(json.dumps(x,indent=2))
    print(json.dumps(x,indent=2))
