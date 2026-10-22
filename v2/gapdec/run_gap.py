"""usage: run_gap.py p idle(0/1) mode(tail|notail0|notail1) shots seed out.json
Bins by complementary gap (dB, floor). For withheld modes, each kept shot is
classified as closed-correct / closed-logical-failure / detectable-residual."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import sys, json, time, math
sys.path.insert(0, str(_V2) + "/gapdec")
from common import *
import graph, notail
p, idle, mode, shots, seed, outp = float(sys.argv[1]), sys.argv[2] == "1", sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
L = variant_text(p, idle); nwith = 0
if mode in ("notail0", "notail1"):
    L, nwith = notail.transform(L, p, idle, mode == "notail1")
full = stim.Circuit("\n".join(L))
dt, ps = detector_info(full)
vis = notail.visible(len(dt), nwith)
m, nodes, wnodes, obs_node, st = graph.build(stim.Circuit("\n".join(obs_as_detector(L))), ps, vis)
nn = obs_node + 1
if nwith:   # second graph: everything visible, for the eventual (continuation-aware) logical decode
    mF, nodesF, _, obsF, _ = graph.build(stim.Circuit("\n".join(obs_as_detector(L))), ps, np.ones(len(dt), bool))
    nF = obsF + 1
sampler = full.compile_detector_sampler(seed=seed)
K = 200
I = np.zeros(K, np.int64); X = np.zeros(K, np.int64)      # strict: any observable error
R = np.zeros(K, np.int64); XC = np.zeros(K, np.int64)     # detectable residual; closed logical failure
XF = np.zeros(K, np.int64)                                # failure of the continuation-aware decode
disc = 0; done = 0; t0 = time.time()
while done < shots:
    n = min(200000, shots - done)
    det, obs = sampler.sample(n, separate_observables=True)
    keep = ~det[:, ps].any(1); disc += int((~keep).sum())
    dk = det[keep]
    syn = np.zeros((dk.shape[0], nn), np.uint8); syn[:, :nn - 1] = dk[:, nodes]
    f0, w0 = m.decode_batch(syn, return_weights=True)
    syn[:, nn - 1] = 1
    f1, w1 = m.decode_batch(syn, return_weights=True)
    pred = w1 < w0; o = obs[keep][:, 0]
    b = np.minimum(np.floor(np.abs(w1 - w0) * 10 / math.log(10)).astype(int), K - 1)
    err = pred != o
    np.add.at(I, b, 1); np.add.at(X, b[err], 1)
    if nwith:
        fw = np.where(pred[:, None], f1, f0)[:, :len(wnodes)] if f0.shape[1] else np.zeros((len(pred), len(wnodes)), np.uint8)
        if fw.shape[1] < len(wnodes):
            fw = np.pad(fw, ((0, 0), (0, len(wnodes) - fw.shape[1])))
        resid = (dk[:, wnodes] ^ fw.astype(bool)).any(1)
        np.add.at(R, b[resid], 1); np.add.at(XC, b[err & ~resid], 1)
        sF = np.zeros((dk.shape[0], nF), np.uint8); sF[:, :nF - 1] = dk[:, nodesF]
        _, v0 = mF.decode_batch(sF, return_weights=True); sF[:, nF - 1] = 1
        _, v1 = mF.decode_batch(sF, return_weights=True)
        np.add.at(XF, b[(v1 < v0) != o], 1)
    done += n
json.dump(dict(p=p, idle=idle, mode=mode, shots=shots, discards=disc, I=I.tolist(), X=X.tolist(),
               R=R.tolist(), XC=XC.tolist(), XF=XF.tolist(), stats=st, secs=time.time() - t0), open(outp, "w"))
k = I.sum()
print(outp, st, "disc", round(disc / shots, 4), f"strict LER {X.sum()/k:.2e}",
      f"closed LER {XC.sum()/k:.2e} resid {R.sum()/k:.3f} full-info LER {XF.sum()/k:.2e}" if nwith else "", f"{time.time()-t0:.0f}s")
