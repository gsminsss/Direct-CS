#!/usr/bin/env python3
"""Audit a Sahay Stim circuit in the direct-CS active-location convention."""
from pathlib import Path
from collections import Counter
import argparse, json, re

NOISE={"DEPOLARIZE1","DEPOLARIZE2","X_ERROR","Y_ERROR","Z_ERROR",
       "PAULI_CHANNEL_1","PAULI_CHANNEL_2","CORRELATED_ERROR","ELSE_CORRELATED_ERROR"}
META={"QUBIT_COORDS","DETECTOR","OBSERVABLE_INCLUDE","SHIFT_COORDS","TICK"}

def count(lines,a,b):
    c=Counter(); ticks=0
    for s in lines[a-1:min(b,len(lines))]:
        s=s.strip()
        if not s or s.startswith("#"): continue
        parts=s.split(); op=parts[0].split("(")[0]; args=parts[1:]
        if op=="TICK": ticks+=1
        q=[int(x) for x in args if re.fullmatch(r"-?\d+",x)]
        if op in NOISE or op in META: continue
        classical=any(x.startswith("rec[") or x.startswith("sweep[") for x in args)
        if op in {"CX","CY","CZ"}:
            if classical: c["FRAME_FEEDBACK"]+=len(q)
            else: c[op]+=len(q)//2
        elif op=="MR":
            c["MEAS"]+=len(q); c["PREP_RESET"]+=len(q)
        elif op in {"M","MX","MY","MZ"}:
            c["MEAS"]+=len(q)
        elif op in {"R","RX","RY","RZ"}:
            c["PREP_RESET"]+=len(q)
        elif op=="I":
            c["IDLE_EXPLICIT"]+=len(q)
        else:
            c["1Q"]+=len(q)
    active=sum(v for k,v in c.items() if k not in {"IDLE_EXPLICIT","FRAME_FEEDBACK"})
    return {"ticks":ticks,"breakdown":dict(c),"active_locations":active}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stim", required=True)
    ap.add_argument("--out", default="s13e_gate_count_recomputed.json")
    a=ap.parse_args()
    lines=Path(a.stim).read_text().splitlines()
    out={
      "d5_factory":count(lines,1,602),
      "d5_to_d7":count(lines,603,832),
      "d7_to_d13":count(lines,833,855),
      "ideal_decoder_tail_excluded":count(lines,856,1037),
      "terminal_readout_excluded":count(lines,1038,1042),
    }
    out["physical_total"]=sum(out[k]["active_locations"] for k in ["d5_factory","d5_to_d7","d7_to_d13"])
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
