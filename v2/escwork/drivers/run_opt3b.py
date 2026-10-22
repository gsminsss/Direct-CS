"""Driver: run OPT3B with a different number of d5 wait rounds (constants overridden, code unchanged).
usage: run_opt3b.py SRC_ROOT OUT_DIR WAIT LIVE_DET HIRANO_LOC"""
import sys, importlib.util
from pathlib import Path
src, out, wait, live, hloc = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
p = src / "stim_pipeline/interface/direct_cs_s21opt3b_live_direct_d5_d13_exact_certificate.py"
spec = importlib.util.spec_from_file_location("opt3b", p); m = importlib.util.module_from_spec(spec)
sys.modules["opt3b"] = m; spec.loader.exec_module(m)
m.WAIT_ROUNDS = wait; m.EXPECTED_HIRANO_LIVE_DETECTORS = live; m.EXPECTED_HIRANO_LOCATIONS = hloc
sys.argv = [str(p), "--search-root", str(src), "--out-dir", out]
m.main()
