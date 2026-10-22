"""Run OPT3C1 with the four-coset candidate lists augmented by exact ILP solutions.
Same seeds, data, gap and hyperparameters as the BP+OSD-only run.
usage: run_c1_ilp.py SRC OPT3B OPT3C0 OUT [extra args]"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, stim
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ilp_coset import ILPSolver
src, bdir, c0dir, out = [Path(a).resolve() for a in sys.argv[1:5]]; extra = sys.argv[5:]
p = src / "DIRECT_CS_HIGHSTAT_VALIDATION_WITH_DEPENDENCIES_2026-09-09/direct_cs_s21opt3c1_practical_decoder_finitep_validation.py"
spec = importlib.util.spec_from_file_location("opt3c1", p); m = importlib.util.module_from_spec(spec); sys.modules["opt3c1"] = m; spec.loader.exec_module(m)
s = json.load(open(bdir / "s21opt3b_summary.json")); loc = int(s["schedule"]["physical_locations"])
tot = stim.Circuit.from_file(str(bdir / "s21opt3b_X_direct_d5_d13_noisy.stim")).num_detectors
m.OP_DETECTORS = tot - 168; m.TRUTH_BITS = 168; m.TOTAL_DETECTORS = tot
m.DOWNSTREAM_LOCATIONS_PER_PATCH = loc; m.FULL_FACTORY_POTENTIAL_LOCATIONS = m.PREDOWNSTREAM_LOCATIONS + 2 * loc
REG = {}; SOLVERS = []
orig_imp = m.imp
def imp(name, path):
    mod = orig_imp(name, path)
    if name.endswith("_d1"):
        mk = mod._make_bposd
        def _make_bposd(B, h, pri, *a, **k):
            dec = mk(B, h, pri, *a, **k)
            sol = ILPSolver(h, pri, extra_solutions=int(__import__("os").environ.get("ILP_EXTRA", "2")))
            REG[id(dec)] = sol; SOLVERS.append(sol); return dec
        mod._make_bposd = _make_bposd
    if name.endswith("_d2"):
        cv = mod._collect_decoder_vectors
        def _collect(decoder, target, ncols):
            outv = cv(decoder, target, ncols)
            sol = REG.get(id(decoder))
            if sol is not None:
                for i, x in enumerate(sol.solve(target)):
                    if x.size == ncols: outv.append((f"ilp{i}", x))
            return outv
        mod._collect_decoder_vectors = _collect
    return mod
m.imp = imp
sys.argv = [str(p), "--search-root", str(src), "--opt3b-result-dir", str(bdir), "--opt3c0-result-dir", str(c0dir), "--out-dir", str(out)] + extra
try:
    m.main()
finally:
    json.dump([s.stats for s in SOLVERS], open(out / "ilp_stats.json", "w"), indent=1)
    print("ILP stats", [s.stats for s in SOLVERS], flush=True)
