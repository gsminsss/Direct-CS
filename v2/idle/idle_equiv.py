"""Check that every idle-fault symptom (detectors + observable) in the LP-scheduled downstream
circuit coincides with the symptom of some existing (non-idle) single fault location."""
import sys, stim, collections
sys.path.insert(0, ".")
import retime3t
from lp_retime import retime
def with_idle(path, p, sched):
    L = open(path).read().splitlines()
    ops, tickend, T = retime3t.parse(L)
    ops = [o for o in ops if o[2] <= T]
    t = retime([(k, tuple(f"q{q}" for q in qs)) for k, qs, _ in ops])[0] if sched == "lp" else [o[2] for o in ops]
    seq = collections.defaultdict(list)
    for i, (k, qs, tk) in enumerate(ops):
        for q in qs: seq[q].append(i)
    ins = collections.defaultdict(list); n = 0
    for q, s in seq.items():
        for a, b in zip(s, s[1:]):
            g = int(t[b] - t[a] - 1)
            if g > 0:
                n += g; ins[tickend[ops[a][2]]].append(f"DEPOLARIZE1[idle]({p}) {q}")
    out = []
    for i, l in enumerate(L):
        out += ins.get(i, []); out.append(l)
    return stim.Circuit("\n".join(out)), n
for path in sys.argv[1:]:
    c, n = with_idle(path, 1e-3, "lp")
    ex = c.explain_detector_error_model_errors()
    bad = []; idle_terms = 0
    for e in ex:
        tags = [l.noise_tag == "idle" for l in e.circuit_error_locations]
        if any(tags):
            idle_terms += 1
            if all(tags): bad.append(str(e.dem_error_terms))
    print(path.split("/")[-1], "idle slots", n, "| DEM terms reachable by an idle fault", idle_terms, "| of which ONLY by idle faults", len(bad))
    for b in bad[:5]: print("   ", b)
