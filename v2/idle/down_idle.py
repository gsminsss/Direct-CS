"""Downstream (escape + growth) Stim circuits with explicit idle DEPOLARIZE1 locations placed
per an LP-retimed schedule (computed on the Z probe, applied at identical ticks in both probes)."""
import os as _os
from pathlib import Path as _P
_V2 = _P(_os.environ.get("DIRECT_CS_WORK", _P(__file__).resolve().parents[1]))  # repository v2/ root
import sys, json, shutil, collections, re
from pathlib import Path
import stim
sys.path.insert(0, str(Path(__file__).parent))
import retime3t
from lp_retime import retime, free_gap
G = Path(str(_V2) + "/escwork/gen")
src, dst, sched = G / sys.argv[1], G / sys.argv[2], sys.argv[3]
shutil.copytree(src, dst, dirs_exist_ok=True)
Z = (src / "s21opt3b_Z_direct_d5_d13_noisy.stim").read_text().splitlines()
ops, tickend, T = retime3t.parse(Z); ops = [o for o in ops if o[2] <= T]
t = retime([(k, tuple(f"q{q}" for q in qs)) for k, qs, _ in ops])[0] if sched == "lp" else [o[2] for o in ops]
seq = collections.defaultdict(list)
for i, (k, qs, tk) in enumerate(ops):
    for q in qs: seq[q].append(i)
place = collections.defaultdict(list); n = 0       # tick -> [qubit,...] (one entry per idle slot)
for q, s in seq.items():
    for a, b in zip(s, s[1:]):
        g = int(t[b] - t[a] - 1)
        if g > 0 and not free_gap(ops[a][0], ops[b][0]): n += g; place[ops[a][2]] += [q] * g
P = float(re.search(r"DEPOLARIZE2\(([0-9.e-]+)\)", "\n".join(Z)).group(1))
for probe in ("Z", "X"):
    for kind in ("noisy",):
        f = src / f"s21opt3b_{probe}_direct_d5_d13_{kind}.stim"; L = f.read_text().splitlines()
        tick = 0; out = []
        for l in L:
            if l.strip() == "TICK":
                # one instruction per multiplicity rank; distinct args stop Stim from fusing
                # these with neighbouring DEPOLARIZE1 instructions (keeps instruction offsets aligned)
                cnt = collections.Counter(); ranks = collections.defaultdict(list)
                for q in place.get(tick, []): ranks[cnt[q]].append(q); cnt[q] += 1
                for r in sorted(ranks): out.append(f"DEPOLARIZE1({P * (1 + 1e-6 * (r + 1)):.12g}) " + " ".join(map(str, ranks[r])))
                tick += 1
            out.append(l)
        (dst / f.name).write_text("\n".join(out) + "\n")
s = json.loads((src / "s21opt3b_summary.json").read_text())
s["schedule"]["physical_locations_active"] = s["schedule"]["physical_locations"]
s["schedule"]["physical_locations"] = s["schedule"]["physical_locations"] + n
s["schedule"]["idle_locations"] = n
(dst / "s21opt3b_summary.json").write_text(json.dumps(s, indent=1))
print(dst.name, "idle locations per patch", n, "noise p", P)
