"""usage: run_opt4b.py SRC OPT3B_DIR OPT3C0_DIR OPT3C1_DIR OUT [extra]"""
import sys, json, importlib.util
from pathlib import Path
import stim
src,bdir,c0,c1,out=[Path(a).resolve() for a in sys.argv[1:6]]; extra=sys.argv[6:]
p=src/"stim_pipeline/interface/zz_opt4b_general.py"
spec=importlib.util.spec_from_file_location("opt4bg",p); m=importlib.util.module_from_spec(spec); sys.modules["opt4bg"]=m; spec.loader.exec_module(m)
s=json.load(open(bdir/"s21opt3b_summary.json")); loc=int(s["schedule"]["physical_locations"])
tot=stim.Circuit.from_file(str(bdir/"s21opt3b_X_direct_d5_d13_noisy.stim")).num_detectors
import os
m.ALLOC_BUDGET=int(os.environ.get("ALLOC_BUDGET","0")); m.ALLOC_FLOOR=int(os.environ.get("ALLOC_FLOOR","100"))
am=os.environ.get("ALLOC_MAP")
if am: m.ALLOC_MAP=json.load(open(am))
m.OP_DETECTORS=tot-168; m.DOWNSTREAM_LOCATIONS_PER_PATCH=loc; m.TOTAL_POTENTIAL_LOCATIONS=890+2*loc
print("OPT4B constants",m.OP_DETECTORS,loc,m.TOTAL_POTENTIAL_LOCATIONS,flush=True)
sys.argv=[str(p),"--search-root",str(src),"--opt3b-result-dir",str(bdir),"--opt3c0-result-dir",str(c0),"--opt3c1-result-dir",str(c1),"--out-dir",str(out)]+extra
m.main()
