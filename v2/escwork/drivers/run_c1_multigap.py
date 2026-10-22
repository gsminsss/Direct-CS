"""OPT3C1 with additional rescoring of the same trajectories at several gap thresholds.
The frozen gap-10 run and its outputs are unchanged; extra results go to multigap.json.
usage: run_c1_multigap.py SRC OPT3B OPT3C0 OUT [extra OPT3C1 args]"""
import sys, json, importlib.util
from pathlib import Path
import stim
GAPS = [4, 5, 6, 7, 8, 9, 10, 12]
src, bdir, c0dir, out = [Path(a).resolve() for a in sys.argv[1:5]]; extra = sys.argv[5:]
p = src / "DIRECT_CS_HIGHSTAT_VALIDATION_WITH_DEPENDENCIES_2026-09-09/direct_cs_s21opt3c1_practical_decoder_finitep_validation.py"
spec = importlib.util.spec_from_file_location("opt3c1", p); m = importlib.util.module_from_spec(spec); sys.modules["opt3c1"] = m; spec.loader.exec_module(m)
s = json.load(open(bdir / "s21opt3b_summary.json")); loc = int(s["schedule"]["physical_locations"])
tot = stim.Circuit.from_file(str(bdir / "s21opt3b_X_direct_d5_d13_noisy.stim")).num_detectors
m.OP_DETECTORS = tot - 168; m.TRUTH_BITS = 168; m.TOTAL_DETECTORS = tot
m.DOWNSTREAM_LOCATIONS_PER_PATCH = loc; m.FULL_FACTORY_POTENTIAL_LOCATIONS = m.PREDOWNSTREAM_LOCATIONS + 2 * loc
RES = []
orig_imp = m.imp
def imp(name, path):
    mod = orig_imp(name, path)
    if name.endswith("_b3"):
        sc = mod._score_b3
        def _score_b3(sampled, fw_exact, fw_amb, ce, ca, high, b1, s10, gap, me, mx):
            for g in GAPS:
                r = sc(sampled, fw_exact, fw_amb, ce, ca, high, b1, s10, float(g), me, mx)
                RES.append({k: r.get(k) for k in ("p", "gap", "shots", "operational_accepted", "operational_acceptance",
                    "stage_level_active_location_proxy_per_operational_accept", "stage_level_expected_active_locations_per_attempt",
                    "truth_closed_accepted", "closed_boundary_harmful", "detectable_residual_accepted",
                    "sampled_closed_boundary_harmful_with_total_fault_locations_le2")})
                json.dump(RES, open(out / "multigap.json", "w"), indent=1)
            return sc(sampled, fw_exact, fw_amb, ce, ca, high, b1, s10, gap, me, mx)
        mod._score_b3 = _score_b3
    return mod
m.imp = imp
out.mkdir(parents=True, exist_ok=True)
sys.argv = [str(p), "--search-root", str(src), "--opt3b-result-dir", str(bdir), "--opt3c0-result-dir", str(c0dir), "--out-dir", str(out)] + extra
m.main()
