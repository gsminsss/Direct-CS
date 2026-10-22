#!/usr/bin/env python3
"""Count active locations of a Stim circuit under two primitive conventions.

explicit : every gate/reset/measurement instruction target is one location
           (2Q gates one per pair, MR = M + R, classically controlled
           feedback and I excluded)  -- the S13-E convention.
native   : the Direct-CS convention.
           * a reset followed (on that qubit, with no intervening op) by a
             run of 1Q unitaries is one preparation location (RX, RY, ...)
           * a run of 1Q unitaries followed by a measurement is absorbed
             into one measurement location (MX, MY, ...)
           * any other run of consecutive 1Q unitaries is one composite
             1Q location
           * 2Q gates one per pair; M and R remain separate locations
Stage boundaries are given as TICK indices.
"""
import re, sys, json
from collections import defaultdict, Counter

ONE_Q = {"H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG",
         "X", "Y", "Z", "H_XY", "H_YZ", "C_XYZ", "C_ZYX"}
TWO_Q = {"CX", "CNOT", "CY", "CZ", "XCZ", "YCZ", "XCX", "SWAP", "ISWAP"}
RESET = {"R", "RX", "RY", "RZ"}
MEAS = {"M", "MX", "MY", "MZ"}
SKIP = {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS",
        "TICK", "I"}
NOISE = re.compile(r"^(DEPOLARIZE\d|X_ERROR|Y_ERROR|Z_ERROR|PAULI_CHANNEL_\d|E|ELSE_CORRELATED_ERROR|HERALDED.*)$")


def parse(path):
    tick = 0
    events = defaultdict(list)      # qubit -> [(tick, kind, name)]
    pairs = []                      # (tick, name)
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\s(]", line, 1)[0]
        if name == "TICK":
            tick += 1
            continue
        if name in SKIP or NOISE.match(name):
            continue
        targs = line.split(")", 1)[-1] if "(" in line.split()[0] else line[len(name):]
        targs = targs.split()
        if name in TWO_Q:
            for a, b in zip(targs[0::2], targs[1::2]):
                if a.startswith("rec") or b.startswith("rec") or a.startswith("sweep"):
                    continue   # classically controlled feedback: excluded
                pairs.append((tick, name))
                events[int(a)].append((tick, "2Q", name))
                events[int(b)].append((tick, "2Q", name))
            continue
        for t in targs:
            if not t.lstrip("!").isdigit():
                continue
            q = int(t.lstrip("!"))
            if name == "MR":
                events[q].append((tick, "M", "M"))
                events[q].append((tick, "R", "R"))
            elif name in RESET:
                events[q].append((tick, "R", name))
            elif name in MEAS:
                events[q].append((tick, "M", name))
            elif name in ONE_Q:
                events[q].append((tick, "1Q", name))
            else:
                raise ValueError(f"unhandled instruction {name}")
    return events, pairs, tick


def count(events, pairs, stage_of):
    exp = Counter(); nat = Counter()
    for t, _ in pairs:
        exp[stage_of(t)] += 1
        nat[stage_of(t)] += 1
    for q, ev in events.items():
        for t, k, _ in ev:
            if k != "2Q":
                exp[stage_of(t)] += 1
        i = 0
        while i < len(ev):
            t, k, _ = ev[i]
            if k == "2Q":
                i += 1
                continue
            if k == "R":
                j = i + 1
                while j < len(ev) and ev[j][1] == "1Q":
                    j += 1
                nat[stage_of(t)] += 1
                i = j
                continue
            if k == "M":
                nat[stage_of(t)] += 1
                i += 1
                continue
            # run of 1Q unitaries
            j = i
            while j < len(ev) and ev[j][1] == "1Q":
                j += 1
            if j < len(ev) and ev[j][1] == "M":
                nat[stage_of(ev[j][0])] += 1      # absorbed into the measurement
                i = j + 1
            else:
                nat[stage_of(t)] += 1             # composite 1Q
                i = j
    return exp, nat


if __name__ == "__main__":
    path = sys.argv[1]
    bounds = json.loads(sys.argv[2])            # [[name, start_tick, end_tick], ...]
    def stage_of(t):
        for n, a, b in bounds:
            if a <= t < b:
                return n
        return "after"
    ev, pr, nt = parse(path)
    exp, nat = count(ev, pr, stage_of)
    print(f"ticks={nt}")
    for n, _, _ in bounds + [["after", 0, 0]]:
        print(f"  {n:28s} explicit {exp[n]:6d}   native {nat[n]:6d}")
    print(json.dumps({"explicit": dict(exp), "native": dict(nat)}))
