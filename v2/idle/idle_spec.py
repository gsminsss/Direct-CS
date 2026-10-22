"""Build the explicit idle-location specification for the Direct front end + core
from a certificate-preserving schedule (per-qubit operation order fixed).

schedule = "lp"   : idle-minimising retiming (difference-constraint LP, integral)
schedule = "asap" : as-soon-as-possible layering of the same circuit
Output: JSON spec consumed by the patched simulator (src_idle), plus a census."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import sys, json, collections, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lp_retime import retime

SRC = Path(str(_V2) + "/escwork/src_base/src_current")
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
s10 = _load("spec_s10", SRC / "direct_cs_step10_physical_frontend.py")
m, s9 = s10.m, s10.s9ed

ops = []      # (kind, qubit-names tuple, tag)
def D(b, j): return f"{b}{j}"
def front(b):
    for li, loc in enumerate(s10.ENCODER_LOCATIONS):
        ops.append((loc.kind, tuple(D(b, q) for q in loc.qubits), ("ENC", b, li)))
def ed(b, inst):
    for oi, op in enumerate(s9.ED_OPS):
        qs = tuple(D(b, q) if q < 7 else f"{b}{inst}_e{q}" for q in op.qs)
        ops.append((op.typ, qs, ("ED", f"{b}{inst}", oi)))
for b in "AB": front(b); ed(b, "init")
for ci, ck in enumerate(m.sequence):
    cat = lambda q: f"c{ci}_{q}"
    for li, loc in enumerate(s10.CAT7_LOCATIONS):
        ops.append((loc.kind, tuple(cat(q) for q in loc.qubits), ("CAT", ci, li)))
    for j in range(7):
        for gi, (typ, q, _g) in enumerate(m.seqs[ck]):
            if typ == "1Q": qs = (D("A" if q == 0 else "B", j),)
            elif typ == "CX": qs = (D("A", j), D("B", j))
            else: qs = (D("A", j), D("B", j), cat(j))
            ops.append((typ, qs, ("BODY", ci, j, gi)))
    for j in range(7): ops.append(("MX", (cat(j),), ("CATRO", ci, j)))
    if ci == 2:
        for b in "AB": ed(b, "mid")
assert len(ops) == 890, len(ops)

def asap(ops):
    ready = collections.defaultdict(int); t = []
    for k, qs, _ in ops:
        s = max(ready[q] for q in qs); t.append(s)
        for q in qs: ready[q] = s + 1
    return t

def build(schedule):
    t = retime([(k, qs) for k, qs, _ in ops])[0] if schedule == "lp" else asap(ops)
    seq = collections.defaultdict(list)
    for i, (k, qs, tag) in enumerate(ops):
        for q in qs: seq[q].append(i)
    enc = collections.defaultdict(lambda: collections.defaultdict(int))     # (q,start) -> per-instance count
    edd = collections.defaultdict(lambda: collections.defaultdict(int))     # (oi,label) -> per-instance
    body = collections.defaultdict(lambda: collections.defaultdict(int))    # (ci,gi,role) -> per-pair
    catd = collections.defaultdict(lambda: collections.defaultdict(int))    # (q,start) -> per-check
    catro = collections.defaultdict(int)                                    # ci -> total slots (sum over j)
    raw = 0
    for qn, s in seq.items():
        for a, bb in zip(s, s[1:]):
            g = int(t[bb] - t[a] - 1)
            if g <= 0: continue
            raw += g
            tag = ops[a][2]
            if tag[0] == "ENC":
                _, blk, li = tag; loc = s10.ENCODER_LOCATIONS[li]
                start = 0 if loc.kind == "PREP" else (li - 7) + 1
                enc[(int(qn[1:]), start)][blk] += g
            elif tag[0] == "ED":
                _, inst, oi = tag
                lab = int(qn[1:]) if qn[0] in "AB" and "_" not in qn else int(qn.split("_e")[1])
                edd[(oi, lab)][inst] += g
            elif tag[0] == "BODY":
                _, ci, j, gi = tag
                if qn.startswith("c"):            # cat qubit after CZZ, waiting for X readout
                    catro[ci] += g
                else:
                    role = 0 if qn[0] == "A" else 1
                    body[(ci, gi, role)][j] += g
            elif tag[0] == "CAT":
                _, ci, li = tag; loc = s10.CAT7_LOCATIONS[li]
                q = int(qn.split("_")[1])
                start = 0 if loc.kind == "PREP" else (li - 11) + 1 if loc.kind == "CX" else None
                if start is None: continue            # after a flag measurement: qubit is done
                catd[(q, start)][ci] += g
            else:
                pass                                   # after a readout: qubit is done
    mx = lambda d: {k: max(v.values()) for k, v in d.items()}
    spec = dict(schedule=schedule,
                enc=[[q, st, g] for (q, st), g in mx(enc).items()],
                ed=[[oi, lab, g] for (oi, lab), g in mx(edd).items()],
                body=[[ci, gi, role, g] for (ci, gi, role), g in mx(body).items()],
                cat=[[q, st, g] for (q, st), g in mx(catd).items()],
                catro=[catro.get(ci, 0) for ci in range(6)])
    modelled = (2 * sum(g for *_, g in spec["enc"]) + 4 * sum(g for *_, g in spec["ed"])
                + 7 * sum(g for *_, g in spec["body"]) + 6 * sum(g for *_, g in spec["cat"]) + sum(spec["catro"]))
    spec["census"] = dict(schedule_idle_slots=raw, modelled_idle_locations=modelled,
                          note="per-position maximum over blocks/instances/pairs/checks (conservative)")
    return spec

if __name__ == "__main__":
    for s in ("asap", "lp"):
        sp = build(s); json.dump(sp, open(Path(__file__).parent / f"idle_spec_{s}.json", "w"), indent=1)
        print(s, sp["census"], "ed entries", len(sp["ed"]), "body", len(sp["body"]), "cat", len(sp["cat"]), "catro", sp["catro"])

def retime_uniform(ops):
    """LP retiming with identical internal timing for every instance of a gadget
    (encoder per block, ED per block/instance, each logical-check body per pair, CAT per check)."""
    import numpy as np, scipy.sparse as sp
    from scipy.optimize import milp, LinearConstraint, Bounds
    n = len(ops); seq = collections.defaultdict(list)
    for i, (k, qs, _) in enumerate(ops):
        for q in qs: seq[q].append(i)
    R = []; C = []; V = []; lo = []; hi = []; r = 0
    for q, s in seq.items():
        for a, c in zip(s, s[1:]):
            R += [r, r]; C += [c, a]; V += [1, -1]; lo.append(1); hi.append(np.inf); r += 1
    groups = collections.defaultdict(dict)    # gadget-type -> instance -> {pos: op index}
    for i, (_, _, tag) in enumerate(ops):
        if tag[0] == "ENC": groups["ENC"].setdefault(tag[1], {})[tag[2]] = i
        elif tag[0] == "ED": groups["ED"].setdefault(tag[1], {})[tag[2]] = i
        elif tag[0] == "BODY": groups[("BODY", tag[1])].setdefault(tag[2], {})[tag[3]] = i
        elif tag[0] == "CAT": groups["CAT"].setdefault(tag[1], {})[tag[2]] = i
    for g, insts in groups.items():
        keys = sorted(insts); ref = insts[keys[0]]; pos = sorted(ref)
        for kk in keys[1:]:
            ins = insts[kk]
            for p_ in pos[1:]:
                # t[ins p] - t[ins p0] - t[ref p] + t[ref p0] = 0
                R += [r] * 4; C += [ins[p_], ins[pos[0]], ref[p_], ref[pos[0]]]; V += [1, -1, -1, 1]
                lo.append(0); hi.append(0); r += 1
    A = sp.csr_matrix((V, (R, C)), shape=(r, n))
    c = np.zeros(n)
    for q, s in seq.items(): c[s[-1]] += 1; c[s[0]] -= 1
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(n), bounds=Bounds(0, 5000))
    assert res.success, res.message
    t = np.round(res.x).astype(int); return t - t.min()

if __name__ == "__main__":
    import lp_retime
    t = retime_uniform(ops)
    _orig = lp_retime.retime
    globals()["retime"] = lambda o: (t, None)
    sp_ = build("lp"); sp_["schedule"] = "lp_uniform_gadgets"
    json.dump(sp_, open(Path(__file__).parent / "idle_spec_lpu.json", "w"), indent=1)
    print("lp_uniform", sp_["census"], "ed", len(sp_["ed"]), "body", sp_["body"], "cat", sp_["cat"], "enc", sp_["enc"], "catro", sp_["catro"])
