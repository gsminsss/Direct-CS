#!/usr/bin/env python3
"""Reconstruct Sahay f=3 p=1e-3 gap tradeoff and target-match 3T->CS."""
from pathlib import Path
import argparse, csv, json

def load_row(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            meta = json.loads(r[6])
            if (meta.get("p") == 0.001 and meta.get("f") == 3
                    and meta.get("prep") == "unitstab"
                    and meta.get("ourbase") == 1
                    and meta.get("dfinal") == 13):
                return {
                    "shots": int(r[0].strip()),
                    "errors": int(r[1].strip()),
                    "discards": int(r[2].strip()),
                    "meta": meta,
                    "counts": json.loads(r[7]),
                }
    raise RuntimeError("main f=3 p=1e-3 row not found")

def curve(row):
    gc = {}
    for k,v in row["counts"].items():
        gc.setdefault(int(k[1:]), {"I":0,"X":0})[k[0]] += int(v)
    gs=sorted(gc)
    out=[]
    for t in range(min(gs), max(gs)+1):
        I=sum(gc[g]["I"] for g in gs if g>=t)
        X=sum(gc[g]["X"] for g in gs if g>=t)
        kept=I+X
        if kept:
            out.append({
                "gap_threshold_dB":t,
                "survival":kept/row["shots"],
                "expected_attempts":row["shots"]/kept,
                "LER":X/kept,
            })
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sahay-csv", required=True)
    ap.add_argument("--cs-target-error", type=float, required=True)
    ap.add_argument("--baseline-sto", type=float, default=3970.735827931926)
    ap.add_argument("--out", default="s13c_recomputed.json")
    a=ap.parse_args()
    row=load_row(a.sahay_csv)
    c=curve(row)
    base=c[0]
    t_target=a.cs_target_error/3
    m=min([x for x in c if x["LER"]<=t_target], key=lambda x:x["expected_attempts"])
    t_sto=a.baseline_sto*m["expected_attempts"]/base["expected_attempts"]
    out={
        "T_error_budget":t_target,
        "matched_point":m,
        "one_T_expected_STO":t_sto,
        "three_T_production_expected_STO":3*t_sto,
        "threeT_leading_CS_error":3*m["LER"],
    }
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
