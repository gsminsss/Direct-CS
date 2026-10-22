"""usage: run_general.py SRC_ROOT OUT_DIR D_ESC WAIT LIVE_DET HIRANO_LOC"""
import sys, importlib.util
from pathlib import Path
src, out = Path(sys.argv[1]).resolve(), sys.argv[2]
d, wait, live, hloc = map(int, sys.argv[3:7])
p = src / "stim_pipeline/interface/zz_opt3b_general.py"
spec = importlib.util.spec_from_file_location("opt3bg", p); m = importlib.util.module_from_spec(spec)
sys.modules["opt3bg"] = m; spec.loader.exec_module(m)
m.D_ESC = d; m.WAIT_ROUNDS = wait; m.EXPECTED_HIRANO_LIVE_DETECTORS = live; m.EXPECTED_HIRANO_LOCATIONS = hloc
sys.argv = [str(p), "--search-root", str(src), "--out-dir", out]
m.main()
