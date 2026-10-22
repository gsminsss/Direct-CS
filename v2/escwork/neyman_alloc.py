"""Neyman allocation for the exact-three stratified estimator.
Var(A3_hat) = sum_s T_s^2 p_s (1-p_s) / n_s; minimise subject to sum n_s = B, n_s >= floor.
p_s from the pilot with a Jeffreys pseudo-count. Only the allocation uses the pilot;
the main sample is fresh, so the estimator stays unbiased."""
import json, sys, math
rows = json.load(open(sys.argv[1]))["0.0005"]; B = int(sys.argv[2]); floor = int(sys.argv[3])
w = {}
for r in rows:
    h, n = int(r["closed_boundary_harmful_samples"]), int(r["shots"])
    ps = (h + 0.5) / (n + 1.0)
    w[r["key"]] = int(r["location_triples"]) * math.sqrt(ps * (1 - ps))
free = set(w); alloc = {}
while True:   # iterative Neyman with floor
    Bf = B - floor * (len(w) - len(free)); W = sum(w[k] for k in free)
    trial = {k: Bf * w[k] / W for k in free}
    low = {k for k, v in trial.items() if v < floor}
    if not low: break
    free -= low
for k in w: alloc[k] = floor if k not in free else int(round(trial[k]))
json.dump(alloc, open(sys.argv[4], "w"), indent=1)
top = sorted(alloc.items(), key=lambda kv: -kv[1])[:6]
print("total", sum(alloc.values()), top)
