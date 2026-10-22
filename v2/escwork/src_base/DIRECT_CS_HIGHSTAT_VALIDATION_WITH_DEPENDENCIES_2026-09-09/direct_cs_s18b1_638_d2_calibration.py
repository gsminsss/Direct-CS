#!/usr/bin/env python3
"""S18-B1: calibration-only practical decoder on the physicalized 638-bit d13 boundary.

Purpose
-------
S18-M2 closed the *structural* noisy-d13 output contract.  S18-B1 is deliberately
separate: it calibrates a practical higher-order operational decoder without
changing that theorem gate.

Operational decoder input:
    exactly 638 detector bits per patch.
Scoring-only information:
    84 X-probe + 84 Z-probe terminal stabilizer truth bits.  These bits are
    NEVER supplied to D2 and are used only to distinguish a closed-boundary
    logical event from a delivered d13 block carrying a detectable residual.

Decoder policy
--------------
* exact/cheap tier: sector-only and sector(weight<=1)+one downstream effect
  records when they have a unique raw terminal Pauli label;
* ambiguous cheap-tier records: abstain;
* all remaining nonzero records: the existing D2 four-logical-coset BP+OSD
  list decoder, now on the 638-bit operational effect matrix;
* record 0: fixed identity frame for the structural output convention.

This script is CALIBRATION ONLY.  It sweeps gap thresholds on calibration seeds
and writes a recommended threshold.  S18-B2 must freeze that value and evaluate
it once on fresh seeds.  Do not quote B1 numbers as final performance.

Important scoring convention
----------------------------
A nonzero terminal truth syndrome means the delivered d13 block carries a
*detectable residual*.  Such a sample is not silently counted as a logical
failure, and it is not silently counted as a clean logical success either.
B1 therefore reports separately:
  - operational acceptance,
  - closed-boundary logical error among truth-closed accepted samples,
  - detectable-residual fraction among accepted samples,
  - raw-terminal proxy error (diagnostic only; do not quote as logical error).

The exact S18 structural d_f>=3 statement remains S18-M2's theorem gate and does
not depend on this empirical decoder.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OP_DETECTORS = 638
TRUTH_PER_PROBE = 84
TRUTH_BITS = 168
TOTAL_LOCATIONS = 3535
OLD_LOCATIONS = 2407
TOL = 1e-12
PAULI_NAMES = ["I", "X", "Z", "Y"]


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find(root: Path, names: list[str], prefer: tuple[str, ...] = ()) -> Path:
    hits: list[Path] = []
    for n in names:
        p = Path(n)
        if p.is_absolute() and p.exists():
            hits.append(p)
        elif "/" in n or "\\" in n:
            q = root / p
            if q.exists():
                hits.append(q)
        else:
            hits.extend(root.rglob(n))
    if not hits:
        raise FileNotFoundError((str(root), names))
    hits = list(dict.fromkeys(x.resolve() for x in hits))
    hits.sort(key=lambda p: (
        tuple(0 if token in str(p) else 1 for token in prefer),
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)), str(p),
    ))
    return hits[0]


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    ph = k / n
    den = 1 + z*z/n
    cen = (ph + z*z/(2*n))/den
    rad = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/den
    return [max(0.0, cen-rad), min(1.0, cen+rad)]


def _zero_upper95(n: int) -> float | None:
    return None if n <= 0 else 1.0 - 0.05 ** (1.0/n)


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@dataclass
class Choice:
    op: int
    truth: int
    pauli: int
    weight: float


@dataclass
class Trajectory:
    record_a: int
    record_b: int
    actual_pauli_a: int
    actual_pauli_b: int
    truth_a: int
    truth_b: int
    lv: np.ndarray
    core_record: tuple[int, ...]
    total_fault_locations: int
    pre_downstream_fault_locations: int
    downstream_fault_locations_a: int
    downstream_fault_locations_b: int
    intrinsic_oracle_infidelity: float


def _stage_from_tick(tick: int) -> str:
    if tick < 44:
        return "hirano_escape_plus_d5_rounds"
    if tick < 71:
        return "sahay_d5_to_d7"
    if tick < 77:
        return "sahay_d7_to_d13"
    return "new_noisy_first_d13_close"


def _build_truth_operational_model(cert, x_path: Path, z_path: Path):
    """Recover all 3535 physical locations from M2 truth-closed circuits.

    Returns D2 operational effect classes (638-bit + raw terminal Pauli), plus
    concrete per-location choices carrying the scoring-only 168-bit truth mask.
    """
    import stim

    xc = stim.Circuit(x_path.read_text(encoding="utf-8"))
    zc = stim.Circuit(z_path.read_text(encoding="utf-8"))
    if int(xc.num_detectors) != OP_DETECTORS + TRUTH_PER_PROBE:
        raise RuntimeError(("X detector count", xc.num_detectors))
    if int(zc.num_detectors) != OP_DETECTORS + TRUTH_PER_PROBE:
        raise RuntimeError(("Z detector count", zc.num_detectors))

    cert.EXPECTED_LOCATIONS = TOTAL_LOCATIONS
    cert._stage_from_tick = _stage_from_tick
    xl, xn, xm = cert._noise_locations_from_text(x_path)
    zl, zn, zm = cert._noise_locations_from_text(z_path)
    if len(xl) != TOTAL_LOCATIONS or len(zl) != TOTAL_LOCATIONS:
        raise RuntimeError((len(xl), len(zl), TOTAL_LOCATIONS))
    xp_sched = [(xm[i]["tick"], xm[i]["gate"], tuple(xm[i]["qubits"])) for i in xl]
    zp_sched = [(zm[i]["tick"], zm[i]["gate"], tuple(zm[i]["qubits"])) for i in zl]
    if xp_sched != zp_sched:
        raise RuntimeError("X/Z physical schedules differ")
    cert._LOCATION_META.clear(); cert._LOCATION_META.update(xm)

    print("S18-B1: recovering X truth-closed physical mechanisms...", flush=True)
    xp, xr = cert._recover_mechanism_map(xc, cert._raw_dem(xc), "X", xn)
    print("S18-B1: recovering Z truth-closed physical mechanisms...", flush=True)
    zp, zr = cert._recover_mechanism_map(zc, cert._raw_dem(zc), "Z", zn)
    universe = cert._all_mechanism_keys(xl)
    if set(xp) - universe or set(zp) - universe:
        raise RuntimeError("Recovered mechanism outside physical universe")

    mask_op = (1 << OP_DETECTORS) - 1
    mask_truth = (1 << TRUTH_PER_PROBE) - 1
    xop: dict[Any, tuple[int,int]] = {}
    zop: dict[Any, tuple[int,int]] = {}
    loc_choices: list[list[Choice]] = [[] for _ in range(TOTAL_LOCATIONS)]
    truth_nonzero_mechs = 0
    stage_counts = collections.Counter()
    mech_count = 0

    for m in sorted(universe, key=str):
        mx, lx = xp.get(m, (0, 0))
        mz, lz = zp.get(m, (0, 0))
        opx = int(mx) & mask_op
        opz = int(mz) & mask_op
        if opx != opz:
            raise RuntimeError(("probe-dependent operational syndrome", cert._mechanism_json(m), hex(opx), hex(opz)))
        tx = (int(mx) >> OP_DETECTORS) & mask_truth
        tz = (int(mz) >> OP_DETECTORS) & mask_truth
        truth = int(tx) | (int(tz) << TRUTH_PER_PROBE)
        pauli = int(lz) | (int(lx) << 1)
        xop[m] = (opx, int(lx))
        zop[m] = (opz, int(lz))
        loc = int(m[0])
        w = float(cert._mechanism_weight(cert._location_gate(loc)))
        loc_choices[loc].append(Choice(opx, truth, pauli, w))
        truth_nonzero_mechs += int(truth != 0)
        stage_counts[_stage_from_tick(int(xm[loc]["tick"]))] += 1
        mech_count += 1

    # Conditional mechanism weights at each physical location must sum to 1.
    bad = []
    hist = collections.Counter()
    for i, cs in enumerate(loc_choices):
        hist[len(cs)] += 1
        sw = sum(c.weight for c in cs)
        if abs(sw - 1.0) > 1e-10:
            bad.append((i, sw, len(cs)))
    if bad:
        raise RuntimeError(("location conditional weights", bad[:10]))

    # D2 sees operational detector bits only.  Truth bits do not enter the matrix.
    effects, emeta = cert._build_joint_effects(xop, zop)
    return {
        "effects": effects,
        "loc_choices": loc_choices,
        "stage_by_loc": [_stage_from_tick(int(xm[i]["tick"])) for i in range(TOTAL_LOCATIONS)],
        "meta": {
            "operational_detectors": OP_DETECTORS,
            "truth_bits_per_probe": TRUTH_PER_PROBE,
            "truth_bits_decoder_visible": False,
            "physical_locations_per_patch": TOTAL_LOCATIONS,
            "concrete_pauli_mechanisms_per_patch": mech_count,
            "mechanism_count_per_location_histogram": {str(k): int(v) for k,v in sorted(hist.items())},
            "mechanisms_with_nonzero_terminal_truth": int(truth_nonzero_mechs),
            "stage_mechanism_counts": dict(sorted(stage_counts.items())),
            "joint_effect_meta": emeta,
            "X_recovery_meta": xr,
            "Z_recovery_meta": zr,
        },
    }


def _build_sector_truth_map(root: Path, s18m, m2, ca, s8b, v7, v11, s18a,
                            hirano: Path, sahay: Path, existing_sector_path: Path,
                            signature_p: float) -> tuple[list[dict[str,Any]], dict[str,Any]]:
    """Rebuild 64 Steane sectors with the 168 scoring-only terminal truth bits."""
    existing = _load(existing_sector_path)
    by_existing = {int(r["steane_syndrome"]): r for r in existing}
    if set(by_existing) != set(range(64)):
        raise RuntimeError("bad existing S18-M sector map")

    print("S18-B1: rebuilding clean X/Z channels for sector truth signatures...", flush=True)
    _xn, xclean, _xo, xm = s18m._build_extended_channel(v7, v11, s18a, hirano, sahay, "X", signature_p)
    _zn, zclean, _zo, zm = s18m._build_extended_channel(v7, v11, s18a, hirano, sahay, "Z", signature_p)
    gx, qx, _ = m2._reconstruct_geo_qmap(v7, hirano, sahay, "X")
    gz, qz, _ = m2._reconstruct_geo_qmap(v7, hirano, sahay, "Z")
    if gx["data"] != gz["data"] or qx != qz:
        raise RuntimeError("X/Z geometry drift")
    m2._append_truth_detectors(xclean, "X", gx, qx)
    m2._append_truth_detectors(zclean, "Z", gz, qz)

    pos, ids = s8b._steane_qubits(hirano)
    if len(ids) != 7:
        raise RuntimeError(ids)
    mask_op = (1 << OP_DETECTORS) - 1
    mask_t = (1 << TRUTH_PER_PROBE) - 1
    singles: dict[tuple[int,str], tuple[int,int,int]] = {}
    for coord, q in enumerate(ids):
        for p in ("X","Y","Z"):
            dx, ox = ca._signature(ca._insert_single_error(xclean, int(xm["boundary_instruction_index"]), int(q), p, signature_p))
            dz, oz = ca._signature(ca._insert_single_error(zclean, int(zm["boundary_instruction_index"]), int(q), p, signature_p))
            opx, opz = int(dx) & mask_op, int(dz) & mask_op
            if opx != opz:
                raise RuntimeError(("sector operational mismatch", coord, p, hex(opx), hex(opz)))
            tx = (int(dx) >> OP_DETECTORS) & mask_t
            tz = (int(dz) >> OP_DETECTORS) & mask_t
            truth = int(tx) | (int(tz) << TRUTH_PER_PROBE)
            raw = int(oz) | (int(ox) << 1)
            singles[(coord,p)] = (opx, truth, raw)

    rows = []
    hist = collections.Counter()
    mismatch = []
    for syn in range(64):
        rep = s8b.s8live.RECOVERY_TABLE[syn]
        op = truth = raw = 0
        for coord, p in enumerate(rep.pauli_string):
            if p == "I":
                continue
            a,b,c = singles[(coord,p)]
            op ^= a; truth ^= b; raw ^= c
        old = by_existing[syn]
        if op != int(old["detector_638_hex"], 16) or raw != int(old["raw_terminal_pauli_int"]):
            mismatch.append((syn, hex(op), old["detector_638_hex"], raw, old["raw_terminal_pauli_int"]))
        hist[int(rep.weight)] += 1
        rows.append({
            **old,
            "terminal_truth_168_hex": hex(int(truth)),
            "terminal_truth_168_weight": int(truth.bit_count()),
            # Alias needed by unchanged D2 ranking/decoder implementation.
            "detector_470_hex": old["detector_638_hex"],
        })
    if mismatch:
        raise RuntimeError(("sector map drift after truth closure", mismatch[:10]))
    if dict(hist) != {0:1,1:21,2:42}:
        raise RuntimeError(dict(hist))
    return rows, {
        "sector_count": 64,
        "canonical_weight_histogram": {str(k):int(v) for k,v in sorted(hist.items())},
        "unique_operational_signatures": len({r["detector_638_hex"] for r in rows}),
        "nonzero_truth_sector_count": sum(int(r["terminal_truth_168_hex"],16) != 0 for r in rows),
        "truth_bits_decoder_visible": False,
        "existing_S18M_operational_and_raw_labels_reproduced": True,
    }


def _sample_choice(rng: np.random.Generator, cs: list[Choice]) -> Choice:
    if len(cs) == 1:
        return cs[0]
    # Most physical gates have uniform Pauli mechanisms; keep general weighted sampling.
    u = float(rng.random())
    acc = 0.0
    for c in cs:
        acc += c.weight
        if u <= acc + 1e-15:
            return c
    return cs[-1]


def _sample_downstream(p: float, rng: np.random.Generator, model: dict[str,Any]):
    k = int(rng.binomial(TOTAL_LOCATIONS, p))
    if k == 0:
        return 0,0,0,0,{}
    op = truth = pauli = 0
    stages = collections.Counter()
    locs = np.asarray(rng.choice(TOTAL_LOCATIONS, size=k, replace=False), dtype=np.int64)
    for z in locs:
        loc = int(z)
        c = _sample_choice(rng, model["loc_choices"][loc])
        op ^= int(c.op); truth ^= int(c.truth); pauli ^= int(c.pauli)
        stages[model["stage_by_loc"][loc]] += 1
    return int(op), int(truth), int(pauli), int(k), dict(stages)


def _oracle_infidelity(s10mc, lv: np.ndarray, rec: tuple[int,...]) -> float:
    target = s10mc._target(rec)
    f = float(abs(np.vdot(target, np.asarray(lv, complex)))**2)
    return max(0.0, min(1.0, 1.0-f))


def _raw_corrected_infidelity(s10mc, tr: Trajectory, pred_a: int, pred_b: int) -> float:
    residual_a = int(pred_a) ^ int(tr.actual_pauli_a)
    residual_b = int(pred_b) ^ int(tr.actual_pauli_b)
    corr = s10mc._apply(np.asarray(tr.lv, complex), residual_a, residual_b)
    target = s10mc._target(tr.core_record)
    f = float(abs(np.vdot(target, corr))**2)
    return max(0.0, min(1.0, 1.0-f))


def _sample_one_p(p: float, shots: int, seed: int, s10mc, sector_by_s: dict[int,dict[str,Any]], model, max_examples: int):
    rng = np.random.default_rng(int(seed))
    traj: list[Trajectory] = []
    reasons = collections.Counter(); fh = collections.Counter(); sa = collections.Counter(); sb = collections.Counter()
    front_acc = core_acc = 0
    sum_front = sum_core = sum_da = sum_db = 0
    max_faults = 0; examples=[]
    for shot in range(int(shots)):
        fa = s10mc._front_block(p, rng); fb = s10mc._front_block(p, rng)
        nfa = int(fa["encoder_fault_count"] + fa["initial_ed_fault_count"])
        nfb = int(fb["encoder_fault_count"] + fb["initial_ed_fault_count"])
        sum_front += nfa+nfb
        if not fa["accepted"] or not fb["accepted"]:
            reasons["initial_frontend_ED"] += 1; nt=nfa+nfb; fh[nt]+=1; max_faults=max(max_faults,nt); continue
        front_acc += 1
        ok, why, rec, ss, lv, nfc, nma, nmb, ncat, nro, sc, fast = s10mc._sample_core_fullfront(p, rng, fa, fb)
        nc = int(nfc+nma+nmb+ncat+nro); sum_core += nc
        if not ok:
            reasons[f"core_{why}"] += 1; nt=nfa+nfb+nc; fh[nt]+=1; max_faults=max(max_faults,nt); continue
        core_acc += 1
        sA,sB = map(int,ss)
        opa,ta,pa,ka,sta = _sample_downstream(p,rng,model)
        opb,tb,pb,kb,stb = _sample_downstream(p,rng,model)
        sum_da += ka; sum_db += kb; sa.update(sta); sb.update(stb)
        ra=sector_by_s[sA]; rb=sector_by_s[sB]
        rec_a=int(ra["detector_638_hex"],16)^opa; rec_b=int(rb["detector_638_hex"],16)^opb
        truth_a=int(ra["terminal_truth_168_hex"],16)^ta; truth_b=int(rb["terminal_truth_168_hex"],16)^tb
        act_a=int(ra["raw_terminal_pauli_int"])^pa; act_b=int(rb["raw_terminal_pauli_int"])^pb
        nt=nfa+nfb+nc+ka+kb; fh[nt]+=1; max_faults=max(max_faults,nt)
        T=Trajectory(int(rec_a),int(rec_b),int(act_a),int(act_b),int(truth_a),int(truth_b),
                     np.asarray(lv,complex).copy(),tuple(int(x) for x in rec),int(nt),int(nfa+nfb+nc),
                     int(ka),int(kb),float(_oracle_infidelity(s10mc,np.asarray(lv,complex),tuple(int(x) for x in rec))))
        traj.append(T)
        if len(examples)<max_examples and (sA or sB or ka or kb):
            examples.append({"shot":shot,"sector_pair":[sA,sB],"record_A":hex(rec_a),"record_B":hex(rec_b),
                             "truth_A":hex(truth_a),"truth_B":hex(truth_b),"raw_pauli_A":PAULI_NAMES[act_a],
                             "raw_pauli_B":PAULI_NAMES[act_b],"total_fault_locations":nt})
    queries={t.record_a for t in traj}|{t.record_b for t in traj}
    return {"p":float(p),"shots":int(shots),"seed":int(seed),"trajectories":traj,"queries":queries,
            "predecode":{"frontend_accepted":front_acc,"frontend_acceptance":front_acc/shots,
                         "core_accepted":core_acc,"core_acceptance_per_attempt":core_acc/shots,
                         "core_acceptance_given_frontend":core_acc/front_acc if front_acc else None,
                         "mean_frontend_fault_locations_per_attempt":sum_front/shots,
                         "mean_coremodule_fault_locations_per_attempt":sum_core/shots,
                         "mean_downstream_fault_locations_A_given_core_accept":sum_da/max(core_acc,1),
                         "mean_downstream_fault_locations_B_given_core_accept":sum_db/max(core_acc,1),
                         "downstream_stage_faults_A":dict(sorted(sa.items())),"downstream_stage_faults_B":dict(sorted(sb.items())),
                         "max_total_fault_locations_seen":max_faults,"total_fault_location_histogram":{str(k):v for k,v in sorted(fh.items())},
                         "rejection_reasons_before_downstream_decoder":dict(sorted(reasons.items())),
                         "unique_observed_patch_records":len(queries),"examples":examples}}


def _build_fast_catalogue(sectors: list[dict[str,Any]], effects: list[Any]):
    """Cheap exact accelerator only; not the S18 theorem certificate.

    Includes sector-only and canonical boundary weight<=1 plus one downstream
    effect.  Ambiguous raw-frame records are marked for abstention.  Clean
    downstream order-2 is intentionally left to D2 in B1; S18-M2 remains the
    independent exact structural theorem.
    """
    states: dict[int,int] = {}
    def add(q:int,p:int):
        states[q]=states.get(q,0)|(1<<int(p))
    for r in sectors:
        rs=int(r["detector_638_hex"],16); ps=int(r["raw_terminal_pauli_int"]); bw=int(r["canonical_weight"])
        add(rs,ps)
        if bw<=1:
            for e in effects:
                add(rs^int(e.syndrome), ps^int(e.pauli))
    exact={q:(b.bit_length()-1) for q,b in states.items() if b.bit_count()==1}
    amb={q for q,b in states.items() if b.bit_count()>1}
    # The all-zero record is fixed to identity by S18-M2's structural output
    # convention; raw-parity residual aliases are not allowed to turn this into
    # an empirical catalogue rejection.
    exact[0]=0; amb.discard(0)
    return exact, amb, {"catalogue_records":len(states),"unique_raw_frame_records":len(exact),"ambiguous_raw_frame_records":len(amb),
                        "scope":"sector-only + boundary weight<=1 plus one downstream physical effect; acceleration only"}


def _decode_result_for_record(q:int, exact:dict[int,int], amb:set[int], high:dict[int,Any], gap:float, max_effects:int):
    if q==0:
        return True,0,"zero_structural"
    if q in amb:
        return False,None,"cheap_ambiguous"
    if q in exact:
        return True,int(exact[q]),"cheap_exact"
    d=high.get(q)
    if d is None or not bool(d.success) or d.pauli is None:
        return False,None,"d2_failure"
    if not math.isfinite(float(d.gap)) or float(d.gap)<float(gap):
        return False,None,"d2_gap_abstain"
    k = int(d.best.correction_effect_count) if d.best is not None else 10**9
    if k>int(max_effects):
        return False,None,"d2_effect_count_abstain"
    return True,int(d.pauli),"d2_high"


def _score_threshold(sampled, exact, amb, high, s10mc, gap:float, max_effects:int, max_examples:int):
    shots=int(sampled["shots"]); accepted=closed_acc=residual_acc=0
    closed_harm=closed_harm_le2=0; raw_harm=0; inf_closed=0.0; intrinsic_closed=0.0
    tiers=collections.Counter(); residual_fault_hist=collections.Counter(); accepted_fault_hist=collections.Counter()
    harmful_ex=[]; residual_ex=[]
    for tr in sampled["trajectories"]:
        oka,pa,ta=_decode_result_for_record(tr.record_a,exact,amb,high,gap,max_effects)
        okb,pb,tb=_decode_result_for_record(tr.record_b,exact,amb,high,gap,max_effects)
        if not oka or not okb:
            tiers[f"reject_A:{ta}" if not oka else f"reject_B:{tb}"]+=1; continue
        tiers[f"A:{ta}"]+=1; tiers[f"B:{tb}"]+=1
        accepted+=1; accepted_fault_hist[tr.total_fault_locations]+=1
        inf=float(_raw_corrected_infidelity(s10mc,tr,int(pa),int(pb)))
        raw_harm += int(inf>TOL)
        closed = (int(tr.truth_a)==0 and int(tr.truth_b)==0)
        if closed:
            closed_acc+=1; inf_closed+=inf; intrinsic_closed+=float(tr.intrinsic_oracle_infidelity)
            if inf>TOL:
                closed_harm+=1; closed_harm_le2 += int(tr.total_fault_locations<=2)
                if len(harmful_ex)<max_examples:
                    harmful_ex.append({"record_A":hex(tr.record_a),"record_B":hex(tr.record_b),"pred_A":PAULI_NAMES[int(pa)],
                                       "pred_B":PAULI_NAMES[int(pb)],"actual_A":PAULI_NAMES[tr.actual_pauli_a],"actual_B":PAULI_NAMES[tr.actual_pauli_b],
                                       "truth_A":hex(tr.truth_a),"truth_B":hex(tr.truth_b),"infidelity":inf,
                                       "total_fault_locations":tr.total_fault_locations,"tier_A":ta,"tier_B":tb})
        else:
            residual_acc+=1; residual_fault_hist[tr.total_fault_locations]+=1
            if len(residual_ex)<max_examples:
                residual_ex.append({"record_A":hex(tr.record_a),"record_B":hex(tr.record_b),"truth_A":hex(tr.truth_a),"truth_B":hex(tr.truth_b),
                                    "raw_proxy_infidelity":inf,"total_fault_locations":tr.total_fault_locations,"tier_A":ta,"tier_B":tb})
    front=int(sampled["predecode"]["frontend_accepted"]); core=int(sampled["predecode"]["core_accepted"])
    executed=194.0+(front/shots)*696.0+(core/shots)*(2*TOTAL_LOCATIONS)
    return {"p":float(sampled["p"]),"shots":shots,"calibration_seed":int(sampled["seed"]),"gap":float(gap),
            "operational_accepted":accepted,"operational_acceptance":accepted/shots,
            "operational_acceptance_wilson95":_wilson(accepted,shots),"attempts_per_operational_accept":shots/accepted if accepted else None,
            "truth_closed_accepted":closed_acc,"truth_closed_fraction_among_accepted":closed_acc/accepted if accepted else None,
            "detectable_residual_accepted":residual_acc,"detectable_residual_fraction_among_accepted":residual_acc/accepted if accepted else None,
            "closed_boundary_harmful":closed_harm,
            "closed_boundary_harmful_fraction_among_truth_closed":closed_harm/closed_acc if closed_acc else None,
            "closed_boundary_harmful_wilson95":_wilson(closed_harm,closed_acc) if closed_acc else None,
            "closed_boundary_zero_harmful_upper95_if_applicable":_zero_upper95(closed_acc) if closed_acc and closed_harm==0 else None,
            "sampled_closed_boundary_harmful_with_total_fault_locations_le2":closed_harm_le2,
            "sampled_le2_regression_gate_pass":closed_harm_le2==0,
            "closed_boundary_conditional_raw_frame_infidelity":inf_closed/closed_acc if closed_acc else None,
            "closed_boundary_intrinsic_oracle_infidelity":intrinsic_closed/closed_acc if closed_acc else None,
            "raw_terminal_proxy_harmful_all_accepted_DO_NOT_QUOTE_AS_LOGICAL":raw_harm,
            "tier_counts":dict(sorted(tiers.items())),
            "accepted_fault_histogram":{str(k):int(v) for k,v in sorted(accepted_fault_hist.items())},
            "residual_fault_histogram":{str(k):int(v) for k,v in sorted(residual_fault_hist.items())},
            "stage_level_expected_active_locations_per_attempt":executed,
            "stage_level_active_location_proxy_per_operational_accept":executed/(accepted/shots) if accepted else None,
            "examples":{"closed_boundary_harmful":harmful_ex,"detectable_residual":residual_ex}}


def _cache_paths(out:Path,p:float):
    tag=f"{p:.12g}".replace(".","p").replace("-","m")
    return out/f"s18b1_cache_{tag}.jsonl", out/f"s18b1_cache_{tag}_meta.json"


def _deserialize_d2(d2,obj):
    def cand(c):
        if c is None: return None
        return d2.CosetCandidate(sector=int(c["sector"]),final_pauli=int(c["final_pauli"]),downstream_pauli=int(c["downstream_pauli"]),
            score=float(c["score"]),correction_effect_count=int(c["correction_effect_count"]),assigned_locations=tuple(map(int,c["assigned_locations"])),
            selected_columns=tuple(map(int,c["selected_columns"])),prefix_distance=int(c["prefix_distance"]),source=str(c["source"]))
    return d2.CosetDecodeResult(success=bool(obj["success"]),pauli=None if obj["pauli"] is None else int(obj["pauli"]),gap=float(obj["gap"]),
        coverage_paulis=tuple(map(int,obj.get("coverage_paulis",[]))),candidate_count=int(obj["candidate_count"]),invalid_candidate_count=int(obj["invalid_candidate_count"]),
        best=cand(obj.get("best")),best_different=cand(obj.get("best_different")),
        pauli_logsumexp={int(k):float(v) for k,v in obj.get("pauli_logsumexp",{}).items()},
        pauli_best_score={int(k):float(v) for k,v in obj.get("pauli_best_score",{}).items()},
        pauli_candidate_counts={int(k):int(v) for k,v in obj.get("pauli_candidate_counts",{}).items()},
        sectors_tested=tuple(map(int,obj.get("sectors_tested",[]))))


def _load_cache(d2,path:Path,meta:Path,wanted:dict[str,Any],resume:bool):
    if not resume or not path.exists(): return {}
    if not meta.exists() or _load(meta)!=wanted:
        raise RuntimeError("Stale S18-B1 D2 cache. Use a fresh --out-dir or --no-resume-cache.")
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o=json.loads(line); out[int(o["record"],16)]=_deserialize_d2(d2,o["decode"])
    return out


def _append_cache(path:Path,q:int,d2,obj):
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps({"record":hex(int(q)),"decode":d2._serialize(obj)},sort_keys=True)+"\n")


def _recommend(rows:list[dict[str,Any]], p_values:list[float], thresholds:list[float], acceptance_floor:float):
    stress=max(map(float,p_values)); by_gap={float(g):[] for g in thresholds}
    for r in rows: by_gap[float(r["gap"])].append(r)
    valid=[]
    for g,rs in by_gap.items():
        if len(rs)!=len(p_values): continue
        stress_row=min(rs,key=lambda x:abs(float(x["p"])-stress))
        if all(bool(x["sampled_le2_regression_gate_pass"]) for x in rs) and all(int(x["closed_boundary_harmful"])==0 for x in rs) and float(stress_row["operational_acceptance"])>=acceptance_floor:
            valid.append((g,stress_row))
    if valid:
        g,st=min(valid,key=lambda x:x[0])
        return {"recommended_gap":float(g),"rule":"smallest calibrated gap with zero observed closed-boundary harmful at every calibration p, sampled <=2 regression PASS, and stress-point operational acceptance above floor",
                "acceptance_floor_at_max_p":float(acceptance_floor),"stress_p":stress,"stress_acceptance":st["operational_acceptance"],"fallback_used":False}
    # fallback: minimize stress closed-harmful fraction, then maximize acceptance, then smallest gap
    opts=[]
    for g,rs in by_gap.items():
        if not rs: continue
        st=min(rs,key=lambda x:abs(float(x["p"])-stress)); rate=st["closed_boundary_harmful_fraction_among_truth_closed"]
        rate=1.0 if rate is None else float(rate)
        opts.append((rate,-float(st["operational_acceptance"]),float(g),st))
    if not opts: return {"recommended_gap":None,"rule":"no calibration rows","fallback_used":True}
    _,_,g,st=min(opts)
    return {"recommended_gap":float(g),"rule":"fallback: minimum stress-point closed-boundary harmful fraction, then maximum operational acceptance, then smallest gap",
            "acceptance_floor_at_max_p":float(acceptance_floor),"stress_p":stress,"stress_acceptance":st["operational_acceptance"],"fallback_used":True}


def _selftest():
    # Operational and truth records are intentionally separate.
    a=Choice(5,1,2,1.0); b=Choice(5,0,0,1.0)
    assert (a.op^b.op)==0 and (a.truth^b.truth)!=0 and (a.pauli^b.pauli)!=0
    return {"passed":True,"operational_detector_bits":OP_DETECTORS,"truth_bits_total":TRUTH_BITS,"truth_bits_decoder_visible":False,
            "stage":"S18-B1 calibration-only practical 638-bit D2"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--search-root",type=Path,default=Path("."))
    ap.add_argument("--s18m-result-dir",type=Path,default=Path("stim_pipeline/interface/s18m_results"))
    ap.add_argument("--s18m2-result-dir",type=Path,default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--p-values",type=float,nargs="+",default=[5e-4,1e-3])
    ap.add_argument("--shots",type=int,default=1500)
    ap.add_argument("--shots-per-p",type=int,nargs="+",default=None)
    ap.add_argument("--thresholds",type=float,nargs="+",default=[8,10,12,14,16])
    ap.add_argument("--prior-calibration-shots",type=int,default=20000)
    ap.add_argument("--prior-seed",type=int,default=20261001)
    ap.add_argument("--calibration-seed",type=int,default=20261011)
    ap.add_argument("--signature-p",type=float,default=0.001)
    ap.add_argument("--sector-candidates",type=int,default=8)
    ap.add_argument("--bp-max-iter",type=int,default=30)
    ap.add_argument("--bp-method",default="minimum_sum")
    ap.add_argument("--ms-scaling-factor",type=float,default=0.625)
    ap.add_argument("--osd-method",default="osd_cs")
    ap.add_argument("--osd-order",type=int,default=2)
    ap.add_argument("--max-correction-effects",type=int,default=8)
    ap.add_argument("--acceptance-floor-at-max-p",type=float,default=0.20)
    ap.add_argument("--max-examples",type=int,default=10)
    ap.add_argument("--no-resume-cache",action="store_true")
    ap.add_argument("--self-test-only",action="store_true")
    ap.add_argument("--out-dir",type=Path,default=Path("stim_pipeline/interface/s18b1_calibration_results"))
    args=ap.parse_args()
    if args.self_test_only:
        print(json.dumps(_selftest(),indent=2)); return
    if args.prior_seed==args.calibration_seed:
        raise SystemExit("prior and calibration seeds must differ")
    root=args.search_root.resolve(); out=(root/args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    s18mr=(root/args.s18m_result_dir).resolve() if not args.s18m_result_dir.is_absolute() else args.s18m_result_dir.resolve()
    s18m2r=(root/args.s18m2_result_dir).resolve() if not args.s18m2_result_dir.is_absolute() else args.s18m2_result_dir.resolve()

    # Locate/import preserved modules.
    d2=_import("s18b1_d2",_find(root,["direct_cs_s17c_d2_logical_coset_list_decoder.py"],("stim_pipeline/interface",)))
    d1=_import("s18b1_d1",_find(root,["direct_cs_s17c_d_hybrid_bposd_abstention.py"],("stim_pipeline/interface",)))
    s10mc=_import("s18b1_s10",_find(root,["direct_cs_s10_complete_physical_factory_mc.py"],("stim_pipeline/interface",)))
    cert=_import("s18b1_v13d",_find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"],("stim_pipeline/interface",)))
    s18m=_import("s18b1_s18m",_find(root,["direct_cs_s18m_modular_noisy_d13_certificate.py"],("stim_pipeline/interface",)))
    m2=_import("s18b1_m2",_find(root,["direct_cs_s18m2_terminal_truth_closure.py"],("stim_pipeline/interface",)))
    ca=_import("s18b1_ca",_find(root,["direct_cs_s17c_a_sector_to_470_map.py"],("stim_pipeline/interface",)))
    s8b=_import("s18b1_s8b",_find(root,["zxz_core_boundary_signature_map.py"],("stim_pipeline/interface",)))
    v7=_import("s18b1_v7",_find(root,["direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"],("stim_pipeline/interface",)))
    v11=_import("s18b1_v11",_find(root,["direct_cs_s17b_v11_complete_missing_basis.py"],("stim_pipeline/interface",)))
    s18a=_import("s18b1_s18a",_find(root,["direct_cs_s18a_noisy_d13_boundary_certificate.py"],("stim_pipeline/interface",)))

    # Patch unchanged D2/D1 routines to the extended operational detector count.
    d1.EXPECTED_DETECTORS=OP_DETECTORS; d2.EXPECTED_DETECTORS=OP_DETECTORS
    BpOsdDecoder=d1._require_ldpc()

    xtruth=s18m2r/"s18m2_X_truth_closed.stim"; ztruth=s18m2r/"s18m2_Z_truth_closed.stim"
    sector_op=s18mr/"s18m_sector_map_638.json"
    m2_summary_path=s18m2r/"s18m2_summary.json"
    for p in (xtruth,ztruth,sector_op,m2_summary_path):
        if not p.exists(): raise FileNotFoundError(p)
    m2_summary=_load(m2_summary_path)
    if not bool(m2_summary.get("main_gate",{}).get("combined_modular_boundary_output_contract_pass",False)):
        raise RuntimeError("S18-M2 structural output-contract gate is not PASS; B1 must not run")
    if bool(m2_summary.get("policy",{}).get("truth_bits_decoder_visible",True)):
        raise RuntimeError("S18-M2 truth bits unexpectedly decoder-visible")
    hirano=_find(root,["msc-ls/src"]); sahay=_find(root,["reference/fd3_baseline_ckt.stim"])

    model=_build_truth_operational_model(cert,xtruth,ztruth)
    sectors,smeta=_build_sector_truth_map(root,s18m,m2,ca,s8b,v7,v11,s18a,hirano,sahay,sector_op,float(args.signature_p))
    sector_by={int(r["steane_syndrome"]):r for r in sectors}
    _dump(out/"s18b1_sector_map_638_truth.json",sectors)

    print("S18-B1: building 638-bit D2 operational matrix...",flush=True)
    cols=d1._effect_columns(model["effects"])
    h=d1._build_sparse_check_matrix(cols)
    h_aug=d2._build_augmented_matrix(h,cols)
    rank_h=d2._gf2_row_rank_sparse(h); rank_aug=d2._gf2_row_rank_sparse(h_aug)
    exact,amb,cmeta=_build_fast_catalogue(sectors,model["effects"])
    matrix_meta={"H_shape":list(map(int,h.shape)),"H_aug_shape":list(map(int,h_aug.shape)),"H_rank":int(rank_h),"H_aug_rank":int(rank_aug),
                 "logical_rows_independent":bool(rank_aug>=rank_h+2),"effect_columns":len(cols),"raw_effect_classes":len(model["effects"]),
                 "fast_catalogue":cmeta,"downstream":model["meta"],"sector":smeta}

    # Four-coset smoke.
    pri_smoke=d1._effect_parity_priors(cols,min(map(float,args.p_values)))
    dec_smoke=d1._make_bposd(BpOsdDecoder,h_aug,pri_smoke,args.bp_max_iter,args.bp_method,args.ms_scaling_factor,args.osd_method,args.osd_order)
    smoke={}
    for q in range(4):
        target=np.zeros(OP_DETECTORS+2,dtype=np.uint8); target[OP_DETECTORS]=q&1; target[OP_DETECTORS+1]=(q>>1)&1
        ok=False
        try:
            for _,vec in d2._collect_decoder_vectors(dec_smoke,target,len(cols)):
                chk=np.asarray(h_aug@vec,dtype=np.uint8).reshape(-1)&1
                if np.array_equal(chk,target): ok=True; break
        except Exception: ok=False
        smoke[PAULI_NAMES[q]]=ok
    selftest={**_selftest(),"matrix":matrix_meta,"zero_record_four_raw_cosets_solver_smoke":smoke,
              "passed":bool(all(smoke.values()) and rank_aug>=rank_h+2 and smeta["existing_S18M_operational_and_raw_labels_reproduced"])}
    _dump(out/"s18b1_self_test.json",selftest)
    if not selftest["passed"]: raise RuntimeError(("S18-B1 selftest",selftest))

    shots_list=[int(args.shots)]*len(args.p_values) if args.shots_per_p is None else list(map(int,args.shots_per_p))
    if len(shots_list)!=len(args.p_values): raise SystemExit("--shots-per-p length must match p-values")
    rows=[]; audits=[]; prior_rows=[]; t0=time.time()

    for ip,(p,shots) in enumerate(zip(map(float,args.p_values),shots_list)):
        print(f"S18-B1: sector-prior calibration p={p:g}, seed={args.prior_seed+ip} ...",flush=True)
        prior,pmeta=d1._calibrate_sector_prior(p,int(args.prior_calibration_shots),int(args.prior_seed+ip),s10mc); pmeta["seed"]=int(args.prior_seed+ip); prior_rows.append(pmeta)
        print(f"S18-B1: calibration sampling p={p:g}, shots={shots}, seed={args.calibration_seed+ip} ...",flush=True)
        sampled=_sample_one_p(p,shots,int(args.calibration_seed+ip),s10mc,sector_by,model,int(args.max_examples))
        queries=set(sampled["queries"])
        unknown=sorted(q for q in queries if q not in exact and q not in amb and q!=0)
        pri=d1._effect_parity_priors(cols,p)
        decoder=d1._make_bposd(BpOsdDecoder,h_aug,pri,args.bp_max_iter,args.bp_method,args.ms_scaling_factor,args.osd_method,args.osd_order)

        cache,meta=_cache_paths(out,p)
        wanted={"stage":"S18-B1","p":p,"matrix_hash":_hash_obj({k:matrix_meta[k] for k in ("H_shape","H_aug_shape","H_rank","H_aug_rank","effect_columns")}),
                "prior_hash":hashlib.sha256(np.asarray(prior,dtype=np.float64).tobytes()).hexdigest(),"sector_candidates":args.sector_candidates,
                "bp_max_iter":args.bp_max_iter,"bp_method":args.bp_method,"ms_scaling_factor":args.ms_scaling_factor,"osd_method":args.osd_method,"osd_order":args.osd_order}
        if args.no_resume_cache:
            if cache.exists(): cache.unlink()
            _dump(meta,wanted); high={}
        else:
            if not meta.exists(): _dump(meta,wanted)
            high=_load_cache(d2,cache,meta,wanted,True)
        reused=sum(q in high for q in unknown); todo=[q for q in unknown if q not in high]
        print(f"S18-B1: records={len(queries)}, cheap_exact={sum(q in exact for q in queries)}, cheap_ambiguous={sum(q in amb for q in queries)}, D2 unknown={len(unknown)}, reused={reused}, to_decode={len(todo)} ...",flush=True)
        td=time.time()
        for ii,q in enumerate(todo):
            d=d2._decode_unknown_coset(d1,q,sectors,prior,p,decoder,h_aug,cols,int(args.sector_candidates),require_all_paulis=True)
            high[q]=d; _append_cache(cache,q,d2,d)
            if (ii+1)%25==0 or ii+1==len(todo): print(f"  decoded {ii+1}/{len(todo)} new records",flush=True)
        gaps=[float(d.gap) for q,d in high.items() if q in unknown and d.success and math.isfinite(float(d.gap))]
        cover=collections.Counter(len(d.coverage_paulis) for q,d in high.items() if q in unknown)
        audit={"p":p,"shots":shots,"calibration_seed":int(args.calibration_seed+ip),"observed_records":len(queries),"cheap_exact_observed":sum(q in exact for q in queries),
               "cheap_ambiguous_observed":sum(q in amb for q in queries),"d2_unknown":len(unknown),"cache_reused":reused,"new_decodes":len(todo),
               "d2_success":sum(bool(high[q].success) for q in unknown),"d2_failure":sum(not bool(high[q].success) for q in unknown),
               "coverage_histogram":{str(k):int(v) for k,v in sorted(cover.items())},"finite_gap_count":len(gaps),
               "gap_median":float(np.median(gaps)) if gaps else None,"gap_p10":float(np.quantile(gaps,.1)) if gaps else None,"gap_p90":float(np.quantile(gaps,.9)) if gaps else None,
               "new_decode_seconds":time.time()-td}
        audits.append(audit)
        for g in map(float,args.thresholds):
            r=_score_threshold(sampled,exact,amb,high,s10mc,g,int(args.max_correction_effects),int(args.max_examples)); rows.append(r)
            print(json.dumps({"p":p,"gap":g,"acc":r["operational_acceptance"],"closed":r["truth_closed_accepted"],"residual_frac":r["detectable_residual_fraction_among_accepted"],
                              "closed_harmful":r["closed_boundary_harmful"],"le2_gate":r["sampled_le2_regression_gate_pass"]},indent=2),flush=True)

    recommendation=_recommend(rows,list(map(float,args.p_values)),list(map(float,args.thresholds)),float(args.acceptance_floor_at_max_p))
    summary={"stage":"S18-B1 638-bit practical D2 calibration","calibration_only":True,"paper_final_performance":False,
             "structural_theorem_source":"S18-M2; B1 does not alter or replace the structural gate",
             "decoder_input":{"operational_detector_bits":638,"terminal_truth_bits_decoder_visible":False,"truth_bits_scoring_only":168},
             "decoder":{"algorithm":"D2 four-logical-coset BP+OSD list decoder on 638-bit operational effect matrix","sector_candidates":int(args.sector_candidates),
                        "osd_method":args.osd_method,"osd_order":int(args.osd_order),"thresholds_swept_on_calibration_only":list(map(float,args.thresholds)),
                        "cheap_tier":"sector-only + canonical boundary weight<=1 plus one downstream effect; ambiguous -> abstain","zero_record_frame":"I"},
             "scoring":{"closed_boundary":"both 168-bit scoring-only terminal truth syndromes are zero","detectable_residual":"at least one terminal truth syndrome nonzero",
                        "warning":"Residual outputs are neither silently called logical failures nor silently called clean successes. Raw-terminal proxy is diagnostic only."},
             "independence_plan":{"B1_prior_seed_base":int(args.prior_seed),"B1_calibration_seed_base":int(args.calibration_seed),"B2_requirement":"freeze recommended gap, then use fresh prior/evaluation seeds and no threshold sweep"},
             "matrix":matrix_meta,"self_test":selftest,"sector_priors":prior_rows,"decoder_audit":audits,"calibration_rows":rows,"recommendation":recommendation,
             "timing_seconds":time.time()-t0}
    _dump(out/"s18b1_summary.json",summary); _dump(out/"s18b1_calibration_rows.json",rows); _dump(out/"s18b1_decoder_audit.json",audits); _dump(out/"s18b1_sector_priors.json",prior_rows)
    print("S18-B1 CALIBRATION COMPLETE",flush=True); print(json.dumps(recommendation,indent=2),flush=True)


if __name__=="__main__":
    main()
