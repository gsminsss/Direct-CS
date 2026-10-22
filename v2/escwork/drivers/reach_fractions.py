"""Front-end and core reach fractions per attempt (independent of downstream design).
usage: reach_fractions.py SRC SHOTS SEED p1 p2 ..."""
import sys, importlib.util, json, numpy as np
from pathlib import Path
src = Path(sys.argv[1]).resolve(); shots = int(sys.argv[2]); seed = int(sys.argv[3]); ps = [float(x) for x in sys.argv[4:]]
p = src / "DIRECT_CS_HIGHSTAT_VALIDATION_WITH_DEPENDENCIES_2026-09-09/direct_cs_s10_complete_physical_factory_mc.py"
spec = importlib.util.spec_from_file_location("s10", p); s10 = importlib.util.module_from_spec(spec); sys.modules["s10"] = s10; spec.loader.exec_module(s10)
out = {}
for i, pp in enumerate(ps):
    rng = np.random.default_rng(seed + i); front = core = 0
    for _ in range(shots):
        fa = s10._front_block(pp, rng); fb = s10._front_block(pp, rng)
        if not fa["accepted"] or not fb["accepted"]: continue
        front += 1
        ok = s10._sample_core_fullfront(pp, rng, fa, fb)[0]
        if ok: core += 1
    out[pp] = dict(front=front / shots, core=core / shots, shots=shots)
    print(pp, out[pp], flush=True)
json.dump({str(k): v for k, v in out.items()}, open("reach_fractions.json", "w"), indent=1)
