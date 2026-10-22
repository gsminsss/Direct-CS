"""Early-abort cost and idle-free hard-discard rate for the public 3T circuit.
Usage: python3 abort_cost.py circuit.stim shots"""
import stim, numpy as np, sys, json, re
sys.path.insert(0, '.')
import count_conv as cc

SRC = sys.argv[1]; N = int(sys.argv[2])
lines = open(SRC).read().splitlines()

def variant(p, idle):
    out = []; ops = set()
    for l in lines:
        s = l.strip(); name = re.split(r"[\s(]", s, maxsplit=1)[0] if s else ""
        if name == "TICK":
            ops = set(); out.append(l); continue
        if name in cc.ONE_Q | cc.TWO_Q | cc.RESET | cc.MEAS | {"MR"}:
            ops |= {int(t.lstrip('!')) for t in s.split(')')[-1].split()[0 if '(' not in s.split()[0] else 0:] if t.lstrip('!').isdigit()}
        m = re.match(r"^(\w+)\(([0-9.e-]+)\)(.*)$", s)
        if m and m.group(1) in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "M", "MX", "MR"):
            nm, val, rest = m.groups(); val = float(val) * p / 1e-3
            if nm == "DEPOLARIZE1" and not idle:
                keep = [t for t in rest.split() if int(t) in ops]
                if not keep: continue
                rest = " " + " ".join(keep)
            out.append(f"{nm}({val}){rest}"); continue
        out.append(l)
    return stim.Circuit("\n".join(out))

# native cumulative cost by tick (Direct convention)
ev, pr, nt = cc.parse(SRC)
_, nat = cc.count(ev, pr, lambda t: t)
cum = np.cumsum([nat.get(t, 0) for t in range(nt + 1)])
FULL = 1400                       # noisy part, native (236+816+348)
noisy_last = 89
print("native through tick 89:", cum[noisy_last], " (check vs 1400)")

base = stim.Circuit.from_file(SRC)
tick = 0; dt = []
co = base.get_detector_coordinates()
for inst in base.flattened():
    if inst.name == "TICK": tick += 1
    elif inst.name == "DETECTOR": dt.append(tick)
dt = np.array(dt)
ps = np.array([(dt[i] <= 52) and tuple(co[i][2:]) != (0.0, 1.0) for i in range(len(dt))])
order = np.argsort(dt); ps_idx = [i for i in order if ps[i]]
res = {}
import os
PS = [float(x) for x in os.environ.get("PLIST", "5e-4,1e-3").split(",")]
for p in PS:
    for idle in (True, False):
        c = variant(p, idle)
        d = c.compile_detector_sampler(seed=7).sample(N)[:, ps_idx]
        fired = d.any(1)
        first = np.where(fired, dt[ps_idx][np.argmax(d, 1)], noisy_last)
        cost = cum[np.minimum(first, noisy_last)].astype(float)
        cost[~fired] = FULL
        # abort detected at the detector's tick: charge locations executed up to it
        r = dict(hard_discard=float(fired.mean()), rho=float(cost.mean() / FULL),
                 mean_ticks=float(first.mean()), mean_abort_tick=float(first[fired].mean()))
        res[f"{p:.0e}_{'idle' if idle else 'noidle'}"] = r
        print(p, "idle" if idle else "no-idle", {k: round(v, 4) for k, v in r.items()})
old = json.load(open("abort_cost.json"))["res"] if os.path.exists("abort_cost.json") else {}
old.update(res)
json.dump(dict(cum=cum.tolist(), res=old), open("abort_cost.json", "w"), indent=1)
