"""Expected qubit-ticks per 3T attempt with early abort, idle-free circuit (convention as abort_cost.py)."""
import sys, json, numpy as np, stim
sys.argv = ['x', 'snap/cc/direct_cs_review/source/reference/fd3_baseline_ckt.stim', '1']
src = open('abort_cost.py').read().split('# native cumulative')[0]
exec(src)
base = stim.Circuit.from_file(SRC); co = base.get_detector_coordinates()
tick = 0; dt = []; first = {}; last = {}
for inst in base.flattened():
    if inst.name == "TICK": tick += 1; continue
    if inst.name == "DETECTOR": dt.append(tick); continue
    if inst.name in ("OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS") or inst.name.startswith(("DEPOLARIZE", "X_ERROR", "Z_ERROR")): continue
    for t in inst.targets_copy():
        if t.is_qubit_target: first.setdefault(t.value, tick); last[t.value] = tick
dt = np.array(dt)
qt = np.array([sum(max(0, min(last[q], t) - first[q] + 1) for q in first if first[q] <= t) for t in range(90)])
ps_idx = [i for i in np.argsort(dt) if dt[i] <= 52 and tuple(co[i][2:]) != (0.0, 1.0)]
res = {}
for p in (5e-4, 8e-4, 9e-4, 1e-3):
    c = variant(p, False)
    d = c.compile_detector_sampler(seed=11).sample(200000)[:, ps_idx]
    fired = d.any(1); ft = np.where(fired, dt[ps_idx][np.argmax(d, 1)], 89)
    res[str(p)] = dict(hard_discard=float(fired.mean()), qticks_per_shot=float(qt[ft].mean()), qticks_full=int(qt[89]))
    print(p, res[str(p)])
json.dump(res, open('spacetime_abort.json', 'w'), indent=1)
