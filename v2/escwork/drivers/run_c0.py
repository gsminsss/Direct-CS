"""Driver: run OPT3C0 on an OPT3B result dir with overridden detector/location constants.
The decoder-matrix audit excludes nonidentity effect classes with zero operational
syndrome (the paper's boundary restriction; 3 such classes in the baseline) and
records how many were excluded.  Code of the original module is unchanged.
usage: run_c0.py SRC_ROOT OPT3B_DIR OUT_DIR"""
import sys, json, importlib.util
from pathlib import Path
src, rdir, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
s = json.load(open(rdir / "s21opt3b_summary.json"))
p = src / "stim_pipeline/interface/direct_cs_s21opt3c0_operational_truth_firewall.py"
spec = importlib.util.spec_from_file_location("opt3c0", p); m = importlib.util.module_from_spec(spec)
sys.modules["opt3c0"] = m; spec.loader.exec_module(m)
tot = int(s["detectors"]) if "detectors" in s else None
loc = int(s["schedule"]["physical_locations"])
import stim
tot = stim.Circuit.from_file(str(rdir / "s21opt3b_X_direct_d5_d13_clean.stim")).num_detectors
m.EXPECTED_TOTAL_DETECTORS = tot; m.EXPECTED_TRUTH_BITS = 168; m.EXPECTED_OP_DETECTORS = tot - 168; m.EXPECTED_LOCATIONS = loc
orig = m.matrix_audit
excluded = {}
def patched(root, effects, op_bits):
    keep = [e for e in effects if not (int(e.syndrome) == 0 and int(e.pauli) != 0)]
    excluded["zero_op_syndrome_nonidentity_classes"] = len(effects) - len(keep)
    r = orig(root, keep, op_bits); r.update(excluded); return r
m.matrix_audit = patched
print("C0 constants", tot, tot - 168, loc, flush=True)
sys.argv = [str(p), "--search-root", str(src), "--opt3b-result-dir", str(rdir), "--out-dir", str(out)]
m.main()
