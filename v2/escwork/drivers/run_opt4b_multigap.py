"""Resumable OPT4B with additional rescoring of every stratum at several gap thresholds.
usage: run_opt4b_multigap.py SRC OPT3B OPT3C0 OPT3C1 OUT [extra]"""
import sys, json, os, importlib.util
from pathlib import Path
import stim
GAPS = [4, 5, 6, 7, 8, 9, 10, 12]
src, bdir, c0, c1, out = [Path(a).resolve() for a in sys.argv[1:6]]; extra = sys.argv[6:]
p = src / "stim_pipeline/interface/zz_opt4b_resumable.py"
spec = importlib.util.spec_from_file_location("opt4bg", p); m = importlib.util.module_from_spec(spec); sys.modules["opt4bg"] = m; spec.loader.exec_module(m)
s = json.load(open(bdir / "s21opt3b_summary.json")); loc = int(s["schedule"]["physical_locations"])
tot = stim.Circuit.from_file(str(bdir / "s21opt3b_X_direct_d5_d13_noisy.stim")).num_detectors
am = os.environ.get("ALLOC_MAP")
if am: m.ALLOC_MAP = json.load(open(am))
m.OP_DETECTORS = tot - 168; m.DOWNSTREAM_LOCATIONS_PER_PATCH = loc; m.TOTAL_POTENTIAL_LOCATIONS = 890 + 2 * loc
RES = []
orig_imp = m.imp
def imp(name, path):
    mod = orig_imp(name, path)
    if name.endswith("_a2"):
        sc = mod._score_samples
        def _score_samples(samples, b3, b1, s10, fe, fa, ce, ca, high, me, mx, meta):
            g0 = mod.FROZEN_GAP
            for g in GAPS:
                mod.FROZEN_GAP = float(g)
                r = sc(samples, b3, b1, s10, fe, fa, ce, ca, high, me, mx, meta)
                RES.append({"key": meta["key"], "gap": g, **{k: r.get(k) for k in (
                    "shots", "location_triples", "closed_boundary_harmful_samples",
                    "A3_closed_boundary_binary_harmful_estimate", "A3_closed_boundary_binary_harmful_se",
                    "A3_closed_boundary_infidelity_estimate", "A3_closed_boundary_infidelity_se",
                    "R3_operational_rejection_estimate", "D3_detectable_residual_accept_estimate")}})
            mod.FROZEN_GAP = g0
            json.dump(RES, open(out / "multigap_order3.json", "w"), indent=1)
            return sc(samples, b3, b1, s10, fe, fa, ce, ca, high, me, mx, meta)
        mod._score_samples = _score_samples
    return mod
m.imp = imp
out.mkdir(parents=True, exist_ok=True)
sys.argv = [str(p), "--search-root", str(src), "--opt3b-result-dir", str(bdir), "--opt3c0-result-dir", str(c0), "--opt3c1-result-dir", str(c1), "--out-dir", str(out)] + extra
m.main()
