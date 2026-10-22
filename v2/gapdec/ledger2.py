"""3T ledger from own complementary-gap runs.  Direct frozen.
Conventions for the 3T acceptance/quality:
  S : Sahay/public  - gap from all detectors incl. ideal d13 tail; failure = logical error
  D : Direct-like   - gap from operational detectors only (ideal round withheld);
                      failure = closed logical failure (residual undetectable by the
                      withheld round); detectable residuals counted as accepted outputs
  P : physical      - gap from operational detectors only; failure = logical error of a
                      continuation-aware decode (later rounds fix detectable residuals)
Path variants: tail (Sahay, 1,400), notail0 (1,400, no extra round), notail1 (+960 noisy d13 round)."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import json, numpy as np, sys
SC = str(_V2) + "/"
AB = json.load(open(SC + "abort_cost.json"))["res"]
A3 = {5e-4: 2.365e6, 8e-4: 2.236e6, 9e-4: 2.107e6, 1e-3: 2.107e6}
D = {5e-4: 6651.0, 8e-4: 8068.1, 9e-4: 8941.4, 1e-3: 9836.3}
PATH, EXTRA, SYN, ROUND, TPR, FULLT = 1400, 960, 676, 960, 8, 89
rng = np.random.default_rng(5)

def load(p, idle, mode):
    try: return json.load(open(f"runs/r_{p:.0e}_i{int(idle)}_{mode}.json".replace("e-04", "e-4").replace("e-03", "e-3")))
    except FileNotFoundError: return None

def curve(r, which):
    I = np.array(r["I"]); F = np.array(r[which]); out = []
    for g in range(len(I)):
        k = I[g:].sum()
        if k == 0: break
        out.append((g, k, F[g:].sum(), r["shots"] / k, I[g:].sum() and np.array(r.get("R", [0]*len(I)))[g:].sum() / k))
    return out

def storage_sync3(att, hd, abort_tick, full_t, M=300000):
    s = 1 / att; pf = min(1.0, hd / (1 - s))
    G = rng.geometric(s, size=(M, 3)) - 1; ab = rng.binomial(G, pf)
    t = ab * abort_tick + (G - ab) * full_t + full_t
    return float((t.max(1)[:, None] - t).sum(1).mean()) / TPR * ROUND, float(t.mean())

rows = []
for p in (5e-4, 8e-4, 9e-4, 1e-3):
    for idle in (1, 0):
        ab = AB[f"{p:.0e}_{'idle' if idle else 'noidle'}"]
        for mode, conv, which in (("tail", "S", "X"), ("notail1", "D", "XC"), ("notail1", "P", "XF"),
                                  ("notail0", "D", "XC"), ("notail0", "P", "XF")):
            r = load(p, idle, mode)
            if r is None: continue
            hd = r["discards"] / r["shots"]
            extra = EXTRA if mode == "notail1" else 0
            cshot = ab["rho"] * PATH + (1 - hd) * extra          # early-abort cost per shot
            target = A3[p] * p ** 3
            cand = [(3 * cshot * att + SYN, g, e / k, att, res) for g, k, e, att, res in curve(r, which)
                    if e / k * 3 <= target and k > 0 and e >= 0]
            # require statistical support: at least 10 errors at the chosen gap, else mark
            best = None
            for C, g, L, att, res in sorted(cand):
                best = (C, g, L, att, res); break
            if best is None:
                rows.append(dict(p=p, idle=idle, mode=mode, conv=conv, note="target not reached")); continue
            C, g, L, att, res = best
            k = np.array(r["I"])[g:].sum(); e = np.array(r[which])[g:].sum()
            st, ET = storage_sync3(att, hd, ab["mean_abort_tick"], FULLT + (8 if extra else 0))
            rows.append(dict(p=p, idle=idle, mode=mode, conv=conv, gap=g, errors=int(e), LER_T=e / k, attempts=att,
                             resid_frac=float(res), C3T=C, direct_vs_3T=D[p] / C - 1,
                             storage_sync3=st, C3T_sync3=C + st, direct_vs_3T_sync3=D[p] / (C + st) - 1,
                             breakeven_storage=D[p] - C, ET_ticks=ET, pipeline_storage_k1=3 * ET / TPR * ROUND, breakeven_k=(3 * ET / TPR * ROUND) / max(D[p] - C, 1e-9)))
json.dump(rows, open("ledger2.json", "w"), indent=1)
print(f"{'p':>6} {'idle':>4} {'mode':>8} {'conv':>4} {'gap':>4} {'err':>6} {'LER_T':>9} {'att':>6} {'resid':>6} {'C3T':>6} {'D/3T':>7} {'+sync3':>7} {'D/3T':>7} {'BE':>6} {'BE_k':>5}")
for x in rows:
    if "note" in x: print(x); continue
    print(f"{x['p']:6.0e} {x['idle']:4d} {x['mode']:>8} {x['conv']:>4} {x['gap']:4d} {x['errors']:6d} {x['LER_T']:9.2e} {x['attempts']:6.3f} {x['resid_frac']:6.3f} "
          f"{x['C3T']:6.0f} {100*x['direct_vs_3T']:+6.1f}% {x['C3T_sync3']:7.0f} {100*x['direct_vs_3T_sync3']:+6.1f}% {x['breakeven_storage']:6.0f} {x['breakeven_k']:5.1f}")
