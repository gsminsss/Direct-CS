import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import stim, numpy as np, re, sys
sys.path.insert(0, str(_V2) + "")
import count_conv as cc
SRC = str(_V2) + "/snap/cc/direct_cs_review/source/reference/fd3_baseline_ckt.stim"
LINES = open(SRC).read().splitlines()

def variant_text(p, idle):
    out = []; ops = set()
    for l in LINES:
        s = l.strip(); name = re.split(r"[\s(]", s, maxsplit=1)[0] if s else ""
        if name == "TICK":
            ops = set(); out.append(l); continue
        if name in cc.ONE_Q | cc.TWO_Q | cc.RESET | cc.MEAS | {"MR"}:
            ops |= {int(t.lstrip('!')) for t in s.split(')')[-1].split() if t.lstrip('!').isdigit()}
        m = re.match(r"^(\w+)\(([0-9.e-]+)\)(.*)$", s)
        if m and m.group(1) in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "M", "MX", "MR"):
            nm, val, rest = m.groups(); val = float(val) * p / 1e-3
            if nm == "DEPOLARIZE1" and not idle:
                keep = [t for t in rest.split() if int(t) in ops]
                if not keep: continue
                rest = " " + " ".join(keep)
            out.append(f"{nm}({val}){rest}"); continue
        out.append(l)
    return out

def detector_info(circ):
    co = circ.get_detector_coordinates(); tick = 0; dt = []
    for inst in circ.flattened():
        if inst.name == "TICK": tick += 1
        elif inst.name == "DETECTOR": dt.append(tick)
    dt = np.array(dt)
    ps = np.array([(dt[i] <= 52) and tuple(co[i][2:]) != (0.0, 1.0) for i in range(len(dt))])
    return dt, ps

def strip_detectors(lines, drop_mask):
    """Remove DETECTOR lines whose global index is in drop_mask (keeps measurement records)."""
    out = []; k = 0
    for l in lines:
        if l.strip().startswith("DETECTOR"):
            if not drop_mask[k]: out.append(l)
            k += 1
        else:
            out.append(l)
    return out

def obs_as_detector(lines):
    """Append a DETECTOR identical to OBSERVABLE_INCLUDE(0) (Gidney-style gap decoding)."""
    out = list(lines)
    for l in lines:
        if l.strip().startswith("OBSERVABLE_INCLUDE(0)"):
            out.append("DETECTOR(-1, -1, -1, -1) " + l.strip().split(")", 1)[1])
    return out
