#!/usr/bin/env python3
"""S21-OPT3A: direct d=5 -> d=13 expansion feasibility smoke test.

This is a *module feasibility* test, not an integrated direct-CS result.

It uses the already-vendored public `msc-ls/src/surface_code_expansion.py`
generator, which supports arbitrary distance1 <= distance2, to instantiate a
direct 5->13 expansion. The goal is to answer one narrow question before
touching the certified S18 pipeline:

    Does the public code asset construct a deterministic, decodable direct
    d=5 -> d=13 expansion circuit at all?

Two geometric orientations (UPWARD/DOWNWARD) are built. No parameter sweep is
performed. The default is one post-expansion syndrome round. A small optional
standalone Monte Carlo is provided, but its logical error must NOT be quoted as
the direct-CS delivered-state error because the interface to our Hirano d=5
history is not yet spliced in.

If this smoke test passes, S21-OPT3B should splice the direct expansion into the
existing 171-detector Hirano boundary and rerun the S18 low-order certificate.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, math, time
from pathlib import Path

def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8")

def find_src(root:Path)->Path:
    exact=root/"msc-ls"/"src"/"surface_code_expansion.py"
    if exact.exists(): return exact.resolve()
    hits=list(root.rglob("surface_code_expansion.py"))
    if not hits: raise FileNotFoundError("surface_code_expansion.py")
    hits.sort(key=lambda p:(0 if "msc-ls" in str(p) else 1,len(str(p))))
    return hits[0].resolve()

def load_expansion(path:Path):
    src=path.parent
    if str(src) not in sys.path: sys.path.insert(0,str(src))
    spec=importlib.util.spec_from_file_location("s21opt3_surface_code_expansion",path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def op_counts(c):
    # Flatten repeat blocks if present, count Stim instruction target multiplicities.
    try: flat=c.flattened()
    except Exception: flat=c
    counts={}
    ticks=0
    for inst in flat:
        if not hasattr(inst,"name"): continue
        n=inst.name
        if n=="TICK": ticks+=1
        counts[n]=counts.get(n,0)+1
    return {"instruction_rows":counts,"ticks":ticks}

def build_one(sce,p,pattern_name,rounds):
    import stim
    mapping=sce.QubitMapping(40,40)
    circ=sce.Circuit(mapping,float(p))
    patt=sce.ExpansionPattern[pattern_name]
    patch=sce.SurfaceCodePatch(circ,5,13,int(rounds),patt)
    patch.build()
    sc=circ.circuit
    # Required structural smoke gates.
    dem=sc.detector_error_model(decompose_errors=False,allow_gauge_detectors=False)
    graphlike=sc.detector_error_model(decompose_errors=True,allow_gauge_detectors=False)
    return {
      "pattern":pattern_name,
      "p":float(p),
      "distance1":5,
      "distance2":13,
      "rounds_for_gap":int(rounds),
      "num_qubits":int(sc.num_qubits),
      "num_detectors":int(sc.num_detectors),
      "num_measurements":int(sc.num_measurements),
      "num_observables":int(sc.num_observables),
      "dem_error_terms":sum(1 for x in dem if getattr(x,"type",None)=="error"),
      "graphlike_dem_build_pass":True,
      "circuit_counts":op_counts(sc),
      "postselection_detector_count":len(circ.detectors_for_post_selection),
      "complementary_gap_detector_id":(
          int(patch.detector_for_complementary_gap.id)
          if patch.detector_for_complementary_gap is not None else None
      ),
      "_circuit":sc,
      "_patch":patch,
      "_wrapper":circ,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--search-root",type=Path,default=Path("."))
    ap.add_argument("--p",type=float,default=0.001)
    ap.add_argument("--rounds-for-gap",type=int,default=1)
    ap.add_argument("--shots",type=int,default=0,
                    help="optional standalone expansion MC; 0 = structural smoke only")
    ap.add_argument("--threshold-gap",type=float,default=0.0)
    ap.add_argument("--out-dir",type=Path,default=Path("stim_pipeline/interface/s21opt3a_results"))
    args=ap.parse_args()

    root=args.search_root.resolve()
    src=find_src(root)
    sce=load_expansion(src)
    out=(root/args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)

    rows=[]
    built={}
    t0=time.time()
    for patt in ("DOWNWARD","UPWARD"):
        r=build_one(sce,args.p,patt,args.rounds_for_gap)
        built[patt]=r
        pub={k:v for k,v in r.items() if not k.startswith("_")}
        rows.append(pub)
        # Persist circuits for later splicing inspection.
        (out/f"s21opt3a_d5_d13_{patt.lower()}.stim").write_text(str(r["_circuit"]),encoding="utf-8")

    mc={}
    if args.shots>0:
        # This is standalone expansion MC only.
        for patt,r in built.items():
            patch=r["_patch"]; wrapper=r["_wrapper"]
            det=patch.detector_for_complementary_gap
            if det is None: raise RuntimeError("missing complementary-gap detector")
            [rs]=sce.perform_simulation(
                wrapper.circuit,int(args.shots),det,
                [(float(args.threshold_gap),float(args.threshold_gap))],
                wrapper.detectors_for_post_selection)
            kept=rs.num_valid_samples+rs.num_wrong_samples
            mc[patt]={
              "shots":int(args.shots),
              "valid":int(rs.num_valid_samples),
              "wrong":int(rs.num_wrong_samples),
              "discarded":int(rs.num_discarded_samples),
              "kept":int(kept),
              "standalone_wrong_fraction_among_kept":(
                  rs.num_wrong_samples/kept if kept else None
              ),
              "warning":"Standalone direct-expansion diagnostic only; not integrated direct-CS logical error."
            }

    summary={
      "stage":"S21-OPT3A direct d5->d13 expansion feasibility",
      "status":"PASS" if all(r["graphlike_dem_build_pass"] for r in rows) else "FAIL",
      "source":str(src),
      "contract":{
        "distance1":5,"distance2":13,
        "rounds_for_gap":int(args.rounds_for_gap),
        "parameter_sweep":False,
        "patterns_tested":["DOWNWARD","UPWARD"],
        "integrated_Hirano_interface":False,
        "paper_final":False,
      },
      "rows":rows,
      "standalone_mc":mc,
      "resource_context":{
        "current_two_patch_d5_to_d7_qsteps":4656,
        "current_two_patch_d7_to_d13_unitary_qsteps":1352,
        "current_two_patch_first_d13_SE_qsteps":5392,
        "current_post_d5_growth_plus_close_qsteps":11400,
        "note":"A direct one-round expansion could only claim savings after the integrated detector/fault-distance audit. Do not subtract 5392 merely by relabeling the required first d13 syndrome round."
      },
      "next_if_pass":"S21-OPT3B: splice one direct d5->d13 expansion into each Hirano d5 output, preserve/derive interface detectors, then rerun exact <=2 structural certification before finite-p MC.",
      "timing_seconds":time.time()-t0,
    }
    dump(out/"s21opt3a_summary.json",summary)
    print(json.dumps(summary,indent=2))
    if summary["status"]!="PASS": raise RuntimeError("S21-OPT3A failed")
    print("S21-OPT3A PASS")

if __name__=="__main__":
    main()
