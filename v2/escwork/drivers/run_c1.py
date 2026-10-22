"""Driver: run OPT3C1 (practical decoder, fresh finite-p) on a variant with overridden constants.
usage: run_c1.py SRC_ROOT OPT3B_DIR OPT3C0_DIR OUT_DIR [extra OPT3C1 args...]"""
import sys, json, importlib.util
from pathlib import Path
import stim
src, bdir, c0dir, out = [Path(a).resolve() for a in sys.argv[1:5]]
extra = sys.argv[5:]
p = src / "DIRECT_CS_HIGHSTAT_VALIDATION_WITH_DEPENDENCIES_2026-09-09/direct_cs_s21opt3c1_practical_decoder_finitep_validation.py"
spec = importlib.util.spec_from_file_location("opt3c1", p); m = importlib.util.module_from_spec(spec)
sys.modules["opt3c1"] = m; spec.loader.exec_module(m)
s = json.load(open(bdir / "s21opt3b_summary.json"))
tot = stim.Circuit.from_file(str(bdir / "s21opt3b_X_direct_d5_d13_noisy.stim")).num_detectors
loc = int(s["schedule"]["physical_locations"])
m.OP_DETECTORS = tot - 168; m.TRUTH_BITS = 168; m.TOTAL_DETECTORS = tot
m.DOWNSTREAM_LOCATIONS_PER_PATCH = loc; m.FULL_FACTORY_POTENTIAL_LOCATIONS = m.PREDOWNSTREAM_LOCATIONS + 2 * loc
print("C1 constants", m.OP_DETECTORS, tot, loc, m.FULL_FACTORY_POTENTIAL_LOCATIONS, flush=True)
sys.argv = [str(p), "--search-root", str(src), "--opt3b-result-dir", str(bdir), "--opt3c0-result-dir", str(c0dir), "--out-dir", str(out)] + extra
m.main()
