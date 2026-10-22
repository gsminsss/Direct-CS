"""Matching graph for kept shots.
* every fault mechanism that fires a postselected detector is dropped (it never
  occurs alone in a kept shot)
* the observable is an extra detector node (complementary-gap decoding)
* hyperedges are decomposed into existing graphlike mechanisms
* withheld detectors (Direct-convention future-validation bits) are not graph
  nodes; each edge carries them as fault_ids so the correction's predicted
  withheld syndrome can be compared with the truth afterwards."""
import stim, numpy as np, pymatching, math
from collections import defaultdict

def build(circ_with_obsdet, ps, visible):
    dem = circ_with_obsdet.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    nfull = len(ps); OBS = nfull
    vis_nodes = [i for i in range(nfull) if visible[i] and not ps[i]]
    wh_nodes = [i for i in range(nfull) if not visible[i]]
    idx = {d: k for k, d in enumerate(vis_nodes)}; idx[OBS] = len(vis_nodes)
    widx = {d: k for k, d in enumerate(wh_nodes)}
    mechs = []; st = defaultdict(int)
    for inst in dem.flattened():
        if inst.type != "error": continue
        pr = inst.args_copy()[0]
        ds = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        if any(d < nfull and ps[d] for d in ds):
            st["dropped_ps"] += 1; continue
        s = frozenset(idx[d] for d in ds if d in idx)
        h = frozenset(widx[d] for d in ds if d in widx)
        if not s:
            st["invisible"] += 1; continue
        if s == {idx[OBS]} and wh_nodes:
            st["obs_only_visible_dropped"] += 1; continue
        mechs.append((s, h, pr))
    E = {s for s, _, _ in mechs if len(s) <= 2}
    prob = defaultdict(float); lab = {}; best = defaultdict(float)
    def add(e, p, h):
        q = prob[e]; prob[e] = q * (1 - p) + p * (1 - q)
        if p > best[e]: best[e] = p; lab[e] = h
    def part(S):
        if not S: return []
        x = min(S)
        for y in sorted(S - {x}):
            if frozenset((x, y)) in E:
                r = part(S - {x, y})
                if r is not None: return [frozenset((x, y))] + r
        if frozenset((x,)) in E:
            r = part(S - {x})
            if r is not None: return [frozenset((x,))] + r
        return None
    for s, h, p in mechs:
        if len(s) <= 2: add(s, p, h); continue
        P = part(set(s))
        if P is None: st["undecomposable"] += 1; continue
        st["decomposed"] += 1
        for j, e in enumerate(P): add(e, p, h if j == 0 else frozenset())
    m = pymatching.Matching()
    obs_node = idx[OBS]
    for e, p in prob.items():
        p = min(p, 0.5 - 1e-9); w = math.log((1 - p) / p); e = sorted(e)
        f = set(lab[frozenset(e)])
        if len(e) == 1:
            if e[0] == obs_node: st["obs_boundary_edges"] += 1
            m.add_boundary_edge(e[0], weight=w, error_probability=p, fault_ids=f)
        else:
            m.add_edge(e[0], e[1], weight=w, error_probability=p, fault_ids=f)
    return m, np.array(vis_nodes), np.array(wh_nodes, int), obs_node, dict(st)
