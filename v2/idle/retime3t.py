"""3T circuit with idle noise placed according to a given schedule.
Idle depolarizing within one per-qubit gap commutes with everything else on that qubit's
timeline, so g idle slots in the gap between ops k and k+1 are modelled by one composed
DEPOLARIZE1 inserted at the end of op k's tick in the original circuit."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import sys, re, collections
sys.path.insert(0, str(_V2) + "/gapdec")
sys.path.insert(0, str(_V2) + "/idle")
import common, count_conv as cc
from lp_retime import retime, free_gap
OPS = cc.ONE_Q | cc.TWO_Q | cc.RESET | cc.MEAS | {"MR", "MX", "MY", "MRX"}
NOISE = re.compile(r"^(DEPOLARIZE\d|X_ERROR|Y_ERROR|Z_ERROR)\(")
def parse(L):
    tick = 0; ops = []; tickend = {}; last_noise = 0
    for i, l in enumerate(L):
        s = l.strip(); name = re.split(r"[\s(]", s, maxsplit=1)[0] if s else ""
        if name == "TICK": tickend[tick] = i; tick += 1; continue
        if NOISE.match(s): last_noise = tick
        if name in ("M", "MX", "MR") and "(" in s: last_noise = tick
        if name in OPS:
            tg = [t for t in s.split(")")[-1].split()]
            tg = [int(t.lstrip("!")) for t in tg if t.lstrip("!").isdigit()]
            if name in cc.TWO_Q:
                for a, b in zip(tg[::2], tg[1::2]): ops.append((name, (a, b), tick))
            else:
                for a in tg: ops.append((name, (a,), tick))
    tickend[tick] = len(L)
    return ops, tickend, last_noise
def lines(p, schedule="lp"):
    L = common.variant_text(p, False)
    ops, tickend, T = parse(L)
    ops = [o for o in ops if o[2] <= T]
    if schedule == "lp":
        t, _ = retime([(k, tuple(f"q{q}" for q in qs)) for k, qs, _ in ops])
    else:
        t = [o[2] for o in ops]
    seq = collections.defaultdict(list)
    for i, (k, qs, tk) in enumerate(ops):
        for q in qs: seq[q].append(i)
    ins = collections.defaultdict(list); nidle = 0
    for q, s in seq.items():
        for a, b in zip(s, s[1:]):
            g = int(t[b] - t[a] - 1)
            if g > 0 and not free_gap(ops[a][0], ops[b][0]):
                nidle += g; pg = 0.75 * (1 - (1 - 4 * p / 3) ** g)
                ins[tickend[ops[a][2]]].append(f"DEPOLARIZE1({pg}) {q}")
    out = []
    for i, l in enumerate(L):
        if i in ins: out += ins[i]
        out.append(l)
    if len(L) in ins: out += ins[len(L)]
    return out, nidle
if __name__ == "__main__":
    for s in ("orig", "lp"):
        L, n = lines(1e-3, s); print(s, "idle slots", n)
