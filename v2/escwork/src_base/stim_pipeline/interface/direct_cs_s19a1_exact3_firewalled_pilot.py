#!/usr/bin/env python3
"""S19-A1: exact-three stratified pilot for the frozen S18-B3 direct-|CS> decoder.

This stage does NOT tune the S18 decoder.  It conditions on exactly three
failed physical locations in the complete physicalized S18 factory and
estimates the leading order-3 logical-infidelity numerator coefficient

    P_L,closed(p) = A3 p^3 + O(p^4)

under the already-frozen B3 policy

    zero -> S18-B2R exact<=2 firewall -> cheap exact -> D2(gap=10).

Structural <=2 claims remain sourced from S18-M2/B2R.  S19-A1 is a statistical
order-3 pilot used to discover active strata and allocate S19-A2 statistics.

Location model
--------------
Pre-downstream factory (890 potential locations):
  FRONT, CAT_PRE, CORE_PRE, MIDED, CAT_POST, CORE_POST.

Physicalized downstream (3535 locations per patch, 7070 total):
  DOWN_HIRANO_D5, DOWN_D5_D7, DOWN_D7_D13, DOWN_D13_CLOSE.

The downstream groups contain BOTH logical patches.  Patch identity is retained
in each sampled location/witness but is not split into separate pilot strata.
This yields 10 groups and C(10+3-1,3)=220 exact-three strata while preserving
exact combinatorial weights over all 7960 potential physical locations.

Decoder prior
-------------
S19-A1 uses one already-generated B3 sector prior at --decoder-p (default
2e-4) and the corresponding D2 parity prior at the same p.  No prior is fitted
on S19 samples.  S19-A2 should repeat active strata across multiple decoder-p
values to test prior stability before paper-final A3 use.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_GAP = 10.0
OP_DETECTORS = 638
TRUTH_BITS = 168
DOWNSTREAM_LOCATIONS_PER_PATCH = 3535
PREDOWNSTREAM_LOCATIONS = 890
TOTAL_POTENTIAL_LOCATIONS = PREDOWNSTREAM_LOCATIONS + 2 * DOWNSTREAM_LOCATIONS_PER_PATCH
TOL = 1e-12
PAULI_NAMES = ("I", "X", "Z", "Y")

GROUPS = (
    "FRONT",
    "CAT_PRE",
    "CORE_PRE",
    "MIDED",
    "CAT_POST",
    "CORE_POST",
    "DOWN_HIRANO_D5",
    "DOWN_D5_D7",
    "DOWN_D7_D13",
    "DOWN_D13_CLOSE",
)

STAGE_TO_GROUP = {
    "hirano_escape_plus_d5_rounds": "DOWN_HIRANO_D5",
    "sahay_d5_to_d7": "DOWN_D5_D7",
    "sahay_d7_to_d13": "DOWN_D7_D13",
    "new_noisy_first_d13_close": "DOWN_D13_CLOSE",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    # Some replay/debug traces contain amplitudes from Clifford-state helpers.
    # They are diagnostics only, but Python's json module cannot serialize
    # built-in/numpy complex scalars.  Preserve both components explicitly
    # instead of dropping the phase or coercing to a real number.
    if isinstance(x, (complex, np.complexfloating)):
        z = complex(x)
        return {"__complex__": [float(z.real), float(z.imag)]}
    raise TypeError(type(x).__name__)


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _hash_obj(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class S19Loc:
    group: str
    kind: str                    # pre | downstream
    pre_loc: Any | None = None   # s11a1.Loc
    patch: int = -1              # 0=A, 1=B
    downstream_loc: int = -1
    stage: str = ""


def _build_groups(a1, model: dict[str, Any]) -> dict[str, list[S19Loc]]:
    out: dict[str, list[S19Loc]] = {g: [] for g in GROUPS}

    # Reuse the already-audited S11 pre-downstream location construction, with
    # escape length 0.  These six groups sum to 890 locations.
    pre = a1._build_groups(0)
    for g in GROUPS[:6]:
        for loc in pre[g]:
            out[g].append(S19Loc(group=g, kind="pre", pre_loc=loc))

    # Physicalized d13 downstream from S18-B1/M2.  Put both patches in the same
    # coarse stage group while retaining patch identity per S19Loc.
    stage_by_loc = list(model["stage_by_loc"])
    if len(stage_by_loc) != DOWNSTREAM_LOCATIONS_PER_PATCH:
        raise RuntimeError(("downstream location count", len(stage_by_loc)))
    for patch in (0, 1):
        for li, stage in enumerate(stage_by_loc):
            if stage not in STAGE_TO_GROUP:
                raise RuntimeError(("unknown downstream stage", stage, li))
            g = STAGE_TO_GROUP[stage]
            out[g].append(S19Loc(group=g, kind="downstream", patch=patch,
                                 downstream_loc=int(li), stage=str(stage)))

    counts = {g: len(out[g]) for g in GROUPS}
    if sum(counts[g] for g in GROUPS[:6]) != PREDOWNSTREAM_LOCATIONS:
        raise RuntimeError(("predownstream count", counts))
    if sum(counts[g] for g in GROUPS[6:]) != 2 * DOWNSTREAM_LOCATIONS_PER_PATCH:
        raise RuntimeError(("downstream count", counts))
    if sum(counts.values()) != TOTAL_POTENTIAL_LOCATIONS:
        raise RuntimeError(("total potential locations", counts, sum(counts.values())))
    return out


def _all_strata(groups: dict[str, list[S19Loc]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for triple in itertools.combinations_with_replacement(range(len(GROUPS)), 3):
        cnt = [0] * len(GROUPS)
        for i in triple:
            cnt[i] += 1
        ways = 1
        for i, k in enumerate(cnt):
            ways *= math.comb(len(groups[GROUPS[i]]), int(k))
        key = "+".join(f"{GROUPS[i]}{k}" for i, k in enumerate(cnt) if k)
        rows.append({"key": key, "counts": tuple(cnt), "location_triples": int(ways)})
    expected_n = math.comb(len(GROUPS) + 2, 3)
    if len(rows) != expected_n:
        raise RuntimeError(("stratum count", len(rows), expected_n))
    if sum(int(r["location_triples"]) for r in rows) != math.comb(TOTAL_POTENTIAL_LOCATIONS, 3):
        raise RuntimeError("S19 exact-three strata do not partition C(N,3)")
    return rows


def _sample_locations(meta: dict[str, Any], groups: dict[str, list[S19Loc]], rng: np.random.Generator) -> list[S19Loc]:
    out: list[S19Loc] = []
    for i, k in enumerate(meta["counts"]):
        k = int(k)
        if not k:
            continue
        arr = groups[GROUPS[i]]
        ids = np.asarray(rng.choice(len(arr), size=k, replace=False), dtype=np.int64)
        out.extend(arr[int(j)] for j in ids)
    if len(out) != 3:
        raise RuntimeError(out)
    return out


class _RA:
    def __init__(self, rng): self.rng = rng
    def random(self): return float(self.rng.random())


def _simulate_predownstream_forced(a1, pre_locs: list[S19Loc], rng: np.random.Generator, want_trace: bool):
    """Run the S10 frontend/check/mid-ED exactly with 0..3 forced pre faults."""
    raw = [x.pre_loc for x in pre_locs]
    # No escape locations are present, so the bridge dict is never dereferenced.
    enc, ied, catfaults, cat_ro, core, mid, _efull, _eframe, trace = a1._instantiate_faults(raw, {}, rng)
    fa = a1._front_block_forced(enc[0], ied[0], rng)
    fb = a1._front_block_forced(enc[1], ied[1], rng)
    if not fa["accepted"] or not fb["accepted"]:
        return {"accepted": False, "reason": "initialED", "trace": trace if want_trace else None}

    ce = [a1.f.cat_effect(catfaults[i]) for i in range(6)]
    for i, x in enumerate(ce):
        if not x.accepted:
            return {"accepted": False, "reason": f"CAT{i}", "trace": trace if want_trace else None}

    fast = (
        a1.s10mc._is_plus_block(fa) and a1.s10mc._is_plus_block(fb)
        and not any(core[i] for i in range(6))
        and not mid[0] and not mid[1]
        and all(x.x_mask == 0 for x in ce)
    )
    if fast:
        u = float(rng.random())
        if u < 5/8: a, b = 0, 0
        elif u < 6/8: a, b = 0, 1
        elif u < 7/8: a, b = 1, 0
        else: a, b = 1, 1
        ideal = a1.s10mc._valid_rec(a, b)
        rec = [int(ideal[i]) ^ int(ce[i].phase_flip) ^ int(cat_ro[i]) for i in range(6)]
        if not a1.m.valid(rec):
            return {"accepted": False, "reason": "record", "trace": trace if want_trace else None}
        sa = sb = 0
        lv = a1.s10mc._branch_vec(a, b)
    else:
        va = a1.f.block_state_from_sector(int(fa["syndrome"]), np.asarray(fa["logical_vector"], complex))
        vb = a1.f.block_state_from_sector(int(fb["syndrome"]), np.asarray(fb["logical_vector"], complex))
        st = a1.f.pair_block_states(va, vb)
        rr = _RA(rng); rec = []
        for ci, ck in enumerate(a1.m.sequence):
            st, meas = a1.f.check_with_physical_cat(st, ck, core[ci], ce[ci], cat_ro[ci], rr)
            rec.append(int(meas))
            if ci == 2:
                st, pr, _ = a1.ed.apply_two_block_midpoint_compact(st, mid[0], mid[1])
                if pr <= 1e-20 or float(rng.random()) >= pr:
                    return {"accepted": False, "reason": "midED", "trace": trace if want_trace else None}
        if not a1.m.valid(rec):
            return {"accepted": False, "reason": "record", "trace": trace if want_trace else None}
        C, psum, recon, _ = a1.live.exact_sector_tensor(st)
        if abs(psum - 1) > 1e-9 or recon > 1e-9:
            raise RuntimeError((psum, recon))
        sa, sb, lv, _ = a1._sample_sector(C, rng)

    return {
        "accepted": True,
        "reason": "acc",
        "sector_a": int(sa),
        "sector_b": int(sb),
        "logical_vector": np.asarray(lv, complex).copy(),
        "core_record": tuple(int(x) for x in rec),
        "trace": trace if want_trace else None,
    }


def _sample_downstream_choice(b1mod, rng: np.random.Generator, model: dict[str, Any], loc: int):
    cs = model["loc_choices"][int(loc)]
    return b1mod._sample_choice(rng, cs)


def _simulate_exact3(a1, b1mod, sector_by: dict[int, dict[str, Any]], model: dict[str, Any],
                     locs: list[S19Loc], rng: np.random.Generator, want_trace: bool = False):
    pre = [x for x in locs if x.kind == "pre"]
    down = [x for x in locs if x.kind == "downstream"]
    base_trace = []
    rpre = _simulate_predownstream_forced(a1, pre, rng, want_trace)
    if not rpre["accepted"]:
        return {"accepted_predecode": False, "reason": rpre["reason"], "trace": rpre.get("trace") if want_trace else None}

    op = [0, 0]; truth = [0, 0]; pauli = [0, 0]
    dtrace = []
    for x in down:
        c = _sample_downstream_choice(b1mod, rng, model, x.downstream_loc)
        patch = int(x.patch)
        op[patch] ^= int(c.op); truth[patch] ^= int(c.truth); pauli[patch] ^= int(c.pauli)
        if want_trace:
            dtrace.append({
                "group": x.group, "patch": "A" if patch == 0 else "B",
                "downstream_location": int(x.downstream_loc), "stage": x.stage,
                "op_hex": hex(int(c.op)), "truth_hex": hex(int(c.truth)),
                "pauli": PAULI_NAMES[int(c.pauli)], "conditional_weight": float(c.weight),
            })

    sA, sB = int(rpre["sector_a"]), int(rpre["sector_b"])
    ra, rb = sector_by[sA], sector_by[sB]
    rec_a = int(ra["detector_638_hex"], 16) ^ int(op[0])
    rec_b = int(rb["detector_638_hex"], 16) ^ int(op[1])
    truth_a = int(ra["terminal_truth_168_hex"], 16) ^ int(truth[0])
    truth_b = int(rb["terminal_truth_168_hex"], 16) ^ int(truth[1])
    act_a = int(ra["raw_terminal_pauli_int"]) ^ int(pauli[0])
    act_b = int(rb["raw_terminal_pauli_int"]) ^ int(pauli[1])
    lv = np.asarray(rpre["logical_vector"], complex)
    core_record = tuple(rpre["core_record"])
    intrinsic = float(b1mod._oracle_infidelity(a1.s10mc, lv, core_record))

    trace = None
    if want_trace:
        trace = {"predownstream": rpre.get("trace") or [], "downstream": dtrace,
                 "sampled_groups": [x.group for x in locs]}
    return {
        "accepted_predecode": True,
        "reason": "acc",
        "record_a": int(rec_a), "record_b": int(rec_b),
        "truth_a": int(truth_a), "truth_b": int(truth_b),
        "actual_pauli_a": int(act_a), "actual_pauli_b": int(act_b),
        "logical_vector": lv, "core_record": core_record,
        "intrinsic_oracle_infidelity": intrinsic,
        "trace": trace,
    }


def _mean_se(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=float)
    m = float(a.mean())
    if len(a) <= 1:
        return m, 0.0
    return m, float(a.std(ddof=1) / math.sqrt(len(a)))


def _load_b3_prior(path: Path, p: float) -> tuple[np.ndarray, dict[str, Any]]:
    rows = _load(path)
    hit = None
    for r in rows:
        if abs(float(r["p"]) - float(p)) <= max(1e-15, 1e-10 * abs(float(p))):
            hit = r; break
    if hit is None:
        raise RuntimeError(("decoder-p missing from B3 sector priors", p, [r.get("p") for r in rows]))
    sp = hit["sector_probability"]
    prior = np.asarray([float(sp[str(i)]) for i in range(64)], dtype=float)
    if abs(float(prior.sum()) - 1.0) > 1e-10:
        raise RuntimeError(("bad B3 prior normalization", prior.sum()))
    return prior, hit


def _load_fw_cache(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o = json.loads(line)
        out[int(o["record"], 16)] = o["resolution"]
    return out


def _append_fw_cache(path: Path, q: int, res: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"record": hex(int(q)), "resolution": res}, sort_keys=True) + "\n")


def _load_d2_cache(b3mod, d2, path: Path) -> dict[int, Any]:
    out: dict[int, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o = json.loads(line)
        out[int(o["record"], 16)] = b3mod._deserialize_d2(d2, o["decode"])
    return out


def _append_d2_cache(path: Path, q: int, d2, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"record": hex(int(q)), "decode": d2._serialize(obj)}, sort_keys=True) + "\n")


def _score_samples(samples: list[dict[str, Any]], b3mod, b1mod, s10mc,
                   fw_exact: dict[int, int], fw_amb: set[int], cheap_exact: dict[int, int], cheap_amb: set[int],
                   high: dict[int, Any], max_effects: int, max_examples: int, meta: dict[str, Any]):
    vals_inf: list[float] = []
    vals_bin: list[float] = []
    vals_oracle: list[float] = []
    vals_reject: list[float] = []
    vals_resid: list[float] = []
    pre_reject = dec_reject = op_accept = closed_accept = residual_accept = harmful = 0
    tier_counts = collections.Counter(); reasons = collections.Counter(); examples = []

    for i, r in enumerate(samples):
        if not r["accepted_predecode"]:
            pre_reject += 1; reasons[r["reason"]] += 1
            vals_inf.append(0.0); vals_bin.append(0.0); vals_oracle.append(0.0)
            vals_reject.append(1.0); vals_resid.append(0.0)
            continue
        qa, qb = int(r["record_a"]), int(r["record_b"])
        oka, pa, ta = b3mod._decode_b3(qa, fw_exact, fw_amb, cheap_exact, cheap_amb, high, FROZEN_GAP, max_effects)
        okb, pb, tb = b3mod._decode_b3(qb, fw_exact, fw_amb, cheap_exact, cheap_amb, high, FROZEN_GAP, max_effects)
        if not oka or not okb:
            dec_reject += 1
            tier_counts[f"reject_A:{ta}" if not oka else f"reject_B:{tb}"] += 1
            vals_inf.append(0.0); vals_bin.append(0.0); vals_oracle.append(0.0)
            vals_reject.append(1.0); vals_resid.append(0.0)
            continue
        tier_counts[f"A:{ta}"] += 1; tier_counts[f"B:{tb}"] += 1
        op_accept += 1; vals_reject.append(0.0)
        closed = int(r["truth_a"]) == 0 and int(r["truth_b"]) == 0
        if not closed:
            residual_accept += 1
            vals_inf.append(0.0); vals_bin.append(0.0); vals_oracle.append(0.0); vals_resid.append(1.0)
            continue
        vals_resid.append(0.0); closed_accept += 1
        tr = b1mod.Trajectory(
            qa, qb, int(r["actual_pauli_a"]), int(r["actual_pauli_b"]),
            int(r["truth_a"]), int(r["truth_b"]), np.asarray(r["logical_vector"], complex),
            tuple(int(x) for x in r["core_record"]), 3, 0, 0, 0, float(r["intrinsic_oracle_infidelity"])
        )
        inf = float(b1mod._raw_corrected_infidelity(s10mc, tr, int(pa), int(pb)))
        h = inf > TOL
        harmful += int(h)
        vals_inf.append(inf); vals_bin.append(1.0 if h else 0.0)
        vals_oracle.append(float(r["intrinsic_oracle_infidelity"]))
        if h and len(examples) < max_examples:
            examples.append({
                "sample_index": i, "stratum": meta["key"], "infidelity": inf,
                "record_A": hex(qa), "record_B": hex(qb),
                "truth_A": hex(int(r["truth_a"])), "truth_B": hex(int(r["truth_b"])),
                "actual_A": PAULI_NAMES[int(r["actual_pauli_a"])],
                "actual_B": PAULI_NAMES[int(r["actual_pauli_b"])],
                "pred_A": PAULI_NAMES[int(pa)], "pred_B": PAULI_NAMES[int(pb)],
                "tier_A": ta, "tier_B": tb, "trace": r.get("trace"),
            })

    ways = int(meta["location_triples"])
    mi, sei = _mean_se(vals_inf); mb, seb = _mean_se(vals_bin); mo, seo = _mean_se(vals_oracle)
    mr, ser = _mean_se(vals_reject); md, sed = _mean_se(vals_resid)
    return {
        "key": meta["key"],
        "counts": {GROUPS[i]: int(k) for i, k in enumerate(meta["counts"]) if int(k)},
        "location_triples": ways, "shots": len(samples),
        "predownstream_reject": pre_reject, "decoder_reject": dec_reject,
        "operational_accept": op_accept, "truth_closed_operational_accept": closed_accept,
        "detectable_residual_operational_accept": residual_accept,
        "closed_boundary_harmful_samples": harmful,
        "A3_closed_boundary_infidelity_estimate": ways * mi,
        "A3_closed_boundary_infidelity_se": ways * sei,
        "A3_closed_boundary_binary_harmful_estimate": ways * mb,
        "A3_closed_boundary_binary_harmful_se": ways * seb,
        "A3_intrinsic_oracle_estimate": ways * mo,
        "A3_intrinsic_oracle_se": ways * seo,
        "R3_operational_rejection_estimate": ways * mr,
        "R3_operational_rejection_se": ways * ser,
        "D3_detectable_residual_accept_estimate": ways * md,
        "D3_detectable_residual_accept_se": ways * sed,
        "tier_counts": dict(sorted(tier_counts.items())),
        "predownstream_rejection_reasons": dict(sorted(reasons.items())),
        "harmful_examples": examples,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(est: str, se: str):
        v = sum(float(r[est]) for r in rows)
        s = math.sqrt(sum(float(r[se]) ** 2 for r in rows))
        return v, s
    a3, a3se = agg("A3_closed_boundary_infidelity_estimate", "A3_closed_boundary_infidelity_se")
    b3, b3se = agg("A3_closed_boundary_binary_harmful_estimate", "A3_closed_boundary_binary_harmful_se")
    o3, o3se = agg("A3_intrinsic_oracle_estimate", "A3_intrinsic_oracle_se")
    r3, r3se = agg("R3_operational_rejection_estimate", "R3_operational_rejection_se")
    d3, d3se = agg("D3_detectable_residual_accept_estimate", "D3_detectable_residual_accept_se")
    return {
        "A3_closed_boundary_infidelity_estimate": a3,
        "A3_closed_boundary_infidelity_naive_se": a3se,
        "A3_closed_boundary_infidelity_naive_95pct": [max(0.0, a3 - 1.95996398454*a3se), a3 + 1.95996398454*a3se],
        "A3_closed_boundary_binary_harmful_estimate": b3,
        "A3_closed_boundary_binary_harmful_naive_se": b3se,
        "A3_intrinsic_oracle_estimate": o3,
        "A3_intrinsic_oracle_naive_se": o3se,
        "R3_operational_rejection_estimate": r3,
        "R3_operational_rejection_naive_se": r3se,
        "D3_detectable_residual_accept_estimate": d3,
        "D3_detectable_residual_accept_naive_se": d3se,
        "total_conditional_exact3_shots": sum(int(r["shots"]) for r in rows),
        "closed_boundary_harmful_samples": sum(int(r["closed_boundary_harmful_samples"]) for r in rows),
        "p3_predictions": {str(p): float(a3 * p**3) for p in (1e-4, 2e-4, 5e-4, 1e-3)},
    }


def _self_test() -> dict[str, Any]:
    # Pure-Python combinatorial test.  Full model/stage counts are checked at runtime.
    toy_sizes = [19, 10, 16, 16, 10, 16, 31, 17, 13, 11]
    n = sum(toy_sizes)
    total = 0; count = 0
    for triple in itertools.combinations_with_replacement(range(10), 3):
        cnt = [0]*10
        for i in triple: cnt[i] += 1
        ways = 1
        for i,k in enumerate(cnt): ways *= math.comb(toy_sizes[i], k)
        total += ways; count += 1
    passed = count == 220 and total == math.comb(n, 3)
    return {
        "passed": bool(passed), "stage": "S19-A1 exact-three firewalled pilot",
        "group_count": 10, "expected_strata": 220, "toy_partition_exact": bool(total == math.comb(n,3)),
        "frozen_gap": FROZEN_GAP, "truth_bits_decoder_visible": False,
        "decoder_policy": "zero -> B2R exact<=2 firewall -> cheap exact -> D2",
        "threshold_sweep_on_S19": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s18m-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m_results"))
    ap.add_argument("--s18m2-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--b2r-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b2r_firewall_results"))
    ap.add_argument("--b3-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b3_validation_results"))
    ap.add_argument("--decoder-p", type=float, default=2e-4)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--shots-per-stratum", type=int, default=20, help="discovery floor for every stratum")
    ap.add_argument("--legacy-priority-shots", type=int, default=300, help="shots for predeclared legacy-S11-mapped priority strata")
    ap.add_argument("--seed", type=int, default=20270111)
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd_cs")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--max-examples-per-stratum", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s19a1_results"))
    args = ap.parse_args()

    st = _self_test()
    if args.self_test_only:
        print(json.dumps(st, indent=2))
        if not st["passed"]: raise RuntimeError(st)
        return
    if args.shots_per_stratum <= 0 or args.legacy_priority_shots < args.shots_per_stratum:
        raise SystemExit("require --shots-per-stratum >0 and --legacy-priority-shots >= floor")
    if int(args.seed) in {20261201,20261202,20261203,20261221,20261222,20261223}:
        raise SystemExit("S19 seed overlaps B3 prior/evaluation seeds")

    root = args.search_root.resolve()
    out = (root/args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _dump(out/"s19a1_self_test.json", st)

    # Source modules.
    a1 = _import("s19_a1_old_exact3", _find(root, ["direct_cs_s11a1_exact3_stratified_pilot.py"], ("stim_pipeline/interface",)))
    b1mod = _import("s19_b1", _find(root, ["direct_cs_s18b1_638_d2_calibration.py"], ("stim_pipeline/interface",)))
    b3mod = _import("s19_b3", _find(root, ["direct_cs_s18b3_firewalled_fresh_validation.py"], ("stim_pipeline/interface",)))
    b2rmod = _import("s19_b2r", _find(root, ["direct_cs_s18b2r_low_order_firewall.py"], ("stim_pipeline/interface",)))
    d2 = _import("s19_d2", _find(root, ["direct_cs_s17c_d2_logical_coset_list_decoder.py"], ("stim_pipeline/interface",)))
    d1 = _import("s19_d1", _find(root, ["direct_cs_s17c_d_hybrid_bposd_abstention.py"], ("stim_pipeline/interface",)))
    s10mc = _import("s19_s10", _find(root, ["direct_cs_s10_complete_physical_factory_mc.py"], ("stim_pipeline/interface",)))
    cert = _import("s19_cert", _find(root, ["direct_cs_s17b_v13d_joint_postselection_certificate.py"], ("stim_pipeline/interface",)))
    s18m = _import("s19_s18m", _find(root, ["direct_cs_s18m_modular_noisy_d13_certificate.py"], ("stim_pipeline/interface",)))
    m2 = _import("s19_m2", _find(root, ["direct_cs_s18m2_terminal_truth_closure.py"], ("stim_pipeline/interface",)))
    ca = _import("s19_ca", _find(root, ["direct_cs_s17c_a_sector_to_470_map.py"], ("stim_pipeline/interface",)))
    s8b = _import("s19_s8b", _find(root, ["zxz_core_boundary_signature_map.py"], ("stim_pipeline/interface",)))
    v7 = _import("s19_v7", _find(root, ["direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"], ("stim_pipeline/interface",)))
    v11 = _import("s19_v11", _find(root, ["direct_cs_s17b_v11_complete_missing_basis.py"], ("stim_pipeline/interface",)))
    s18a = _import("s19_s18a", _find(root, ["direct_cs_s18a_noisy_d13_boundary_certificate.py"], ("stim_pipeline/interface",)))
    d1.EXPECTED_DETECTORS = OP_DETECTORS; d2.EXPECTED_DETECTORS = OP_DETECTORS

    b3dir = (root/args.b3_result_dir).resolve() if not args.b3_result_dir.is_absolute() else args.b3_result_dir.resolve()
    b2rdir = (root/args.b2r_result_dir).resolve() if not args.b2r_result_dir.is_absolute() else args.b2r_result_dir.resolve()
    b3sum_path = b3dir/"s18b3_summary.json"; b3pri_path = b3dir/"s18b3_sector_priors.json"; b2rsum_path = b2rdir/"s18b2r_summary.json"
    for p in (b3sum_path, b3pri_path, b2rsum_path):
        if not p.exists(): raise FileNotFoundError(p)
    b3sum = _load(b3sum_path); b2rsum = _load(b2rsum_path)
    vg = b3sum.get("validation_gate", {})
    if not bool(vg.get("passed", False)) or int(vg.get("sampled_closed_boundary_harmful_le2_total", -1)) != 0:
        raise RuntimeError("S19 requires a clean S18-B3 PASS with zero sampled <=2 harmful")
    if abs(float(b3sum.get("frozen_gap", -1)) - FROZEN_GAP) > 1e-12:
        raise RuntimeError("S18-B3 frozen gap drift")
    if bool(b3sum.get("decoder_input", {}).get("terminal_truth_bits_decoder_visible", True)):
        raise RuntimeError("B3 truth bits unexpectedly decoder-visible")
    if not bool(b2rsum.get("validation_gate", {}).get("passed", False)):
        raise RuntimeError("B2R is not PASS")

    # Build the same S18 physicalized boundary model and sector map.
    s18mr = (root/args.s18m_result_dir).resolve() if not args.s18m_result_dir.is_absolute() else args.s18m_result_dir.resolve()
    s18m2r = (root/args.s18m2_result_dir).resolve() if not args.s18m2_result_dir.is_absolute() else args.s18m2_result_dir.resolve()
    xtruth=s18m2r/"s18m2_X_truth_closed.stim"; ztruth=s18m2r/"s18m2_Z_truth_closed.stim"
    sector_op=s18mr/"s18m_sector_map_638.json"; m2sum=s18m2r/"s18m2_summary.json"
    for p in (xtruth,ztruth,sector_op,m2sum):
        if not p.exists(): raise FileNotFoundError(p)
    ms=_load(m2sum)
    if not bool(ms.get("main_gate",{}).get("combined_modular_boundary_output_contract_pass",False)):
        raise RuntimeError("S18-M2 structural gate is not PASS")

    hirano=_find(root,["msc-ls/src"]); sahay=_find(root,["reference/fd3_baseline_ckt.stim"])
    print("S19-A1: rebuilding physicalized 638-bit truth-closed model...", flush=True)
    model=b1mod._build_truth_operational_model(cert,xtruth,ztruth)
    sectors,smeta=b1mod._build_sector_truth_map(root,s18m,m2,ca,s8b,v7,v11,s18a,hirano,sahay,sector_op,float(args.signature_p))
    sector_by={int(r["steane_syndrome"]):r for r in sectors}

    groups=_build_groups(a1,model); strata=_all_strata(groups)
    loc_inventory={g:len(groups[g]) for g in GROUPS}; loc_inventory["total"]=sum(loc_inventory.values())
    runtime_selftest = {
        "passed": loc_inventory["total"] == TOTAL_POTENTIAL_LOCATIONS and len(strata)==220 and
                  sum(int(r["location_triples"]) for r in strata)==math.comb(TOTAL_POTENTIAL_LOCATIONS,3),
        "location_inventory": loc_inventory, "stratum_count":len(strata),
        "total_location_triples":sum(int(r["location_triples"]) for r in strata),
        "C_7960_3":math.comb(TOTAL_POTENTIAL_LOCATIONS,3),
        "partition_exact":sum(int(r["location_triples"]) for r in strata)==math.comb(TOTAL_POTENTIAL_LOCATIONS,3),
    }
    _dump(out/"s19a1_runtime_self_test.json",runtime_selftest)
    # Legacy S11 evidence is used only to fix a higher-statistics discovery plan BEFORE
    # any S19 sample is drawn.  Old ESC_A/B strata map conservatively into the
    # physicalized DOWN_HIRANO_D5 union group.  New d5->d7/d7->d13/d13-close
    # strata still receive the nonzero global discovery floor.
    priority_keys={
        "DOWN_HIRANO_D53",
        "CORE_POST1+DOWN_HIRANO_D52",
        "CORE_POST2+DOWN_HIRANO_D51",
        "MIDED1+CORE_POST1+DOWN_HIRANO_D51",
        "CAT_POST1+DOWN_HIRANO_D52",
        "CAT_POST1+CORE_POST1+DOWN_HIRANO_D51",
        "MIDED2+DOWN_HIRANO_D51",
    }
    plan=[]
    for i,r in enumerate(strata):
        nshot=int(args.legacy_priority_shots if r["key"] in priority_keys else args.shots_per_stratum)
        plan.append({"index":i,"key":r["key"],"counts":{GROUPS[j]:int(k) for j,k in enumerate(r["counts"]) if int(k)},
                     "location_triples":int(r["location_triples"]),"shots":nshot,
                     "legacy_priority":bool(r["key"] in priority_keys)})
    planned_total=sum(int(r["shots"]) for r in plan)
    _dump(out/"s19a1_plan.json",{"decoder_p":float(args.decoder_p),"seed":int(args.seed),"rows":plan,
                                "total_conditional_shots":planned_total,"runtime_selftest":runtime_selftest,
                                "allocation_fixed_before_S19_sampling":True,
                                "legacy_priority_source":"S11-A6 old-escape active strata mapped to DOWN_HIRANO_D5; no S19 outcomes used",
                                "priority_keys":sorted(priority_keys)})
    print(json.dumps({"stage":"S19-A1 plan","groups":loc_inventory,"strata":len(strata),
                      "floor_shots_per_stratum":args.shots_per_stratum,"legacy_priority_shots":args.legacy_priority_shots,
                      "priority_strata":len(priority_keys),"total_conditional_shots":planned_total,
                      "C_N_3":math.comb(TOTAL_POTENTIAL_LOCATIONS,3)},indent=2),flush=True)
    if not runtime_selftest["passed"]: raise RuntimeError(runtime_selftest)
    if args.plan_only: return
    plan_by={r["key"]:r for r in plan}

    # Frozen B3 decoder components.
    prior, prior_meta = _load_b3_prior(b3pri_path,float(args.decoder_p))
    print(f"S19-A1: using frozen B3 sector prior at decoder-p={args.decoder_p:g}, seed={prior_meta.get('seed')}",flush=True)
    cols=d1._effect_columns(model["effects"]); h=d1._build_sparse_check_matrix(cols); h_aug=d2._build_augmented_matrix(h,cols)
    rank_h=d2._gf2_row_rank_sparse(h); rank_aug=d2._gf2_row_rank_sparse(h_aug)
    if int(rank_h)!=638 or int(rank_aug)<640: raise RuntimeError((rank_h,rank_aug))
    cheap_exact,cheap_amb,cheap_meta=b1mod._build_fast_catalogue(sectors,model["effects"])
    print("S19-A1: building frozen B2R firewall index...",flush=True)
    fw_idx=b2rmod.build_firewall_index(sectors,model)
    fw_cache_path=out/"s19a1_firewall_cache.jsonl"
    fw_cache=_load_fw_cache(fw_cache_path) if args.resume else {}
    if 0 not in fw_cache:
        fw_cache[0]=b2rmod.resolve_query(0,fw_idx,max_examples=1); _append_fw_cache(fw_cache_path,0,fw_cache[0])
    if fw_cache[0]["status"]!="exact" or fw_cache[0]["pauli"]!="I": raise RuntimeError("firewall zero contract")

    BpOsdDecoder=d1._require_ldpc()
    parity_priors=d1._effect_parity_priors(cols,float(args.decoder_p))
    decoder=d1._make_bposd(BpOsdDecoder,h_aug,parity_priors,args.bp_max_iter,args.bp_method,args.ms_scaling_factor,args.osd_method,args.osd_order)
    d2_cache_path=out/"s19a1_d2_cache.jsonl"
    cache_meta_path=out/"s19a1_cache_meta.json"
    cache_meta={
        "stage":"S19-A1", "decoder_p":float(args.decoder_p), "frozen_gap":FROZEN_GAP,
        "seed":int(args.seed), "floor_shots":int(args.shots_per_stratum),
        "legacy_priority_shots":int(args.legacy_priority_shots),
        "prior_hash":hashlib.sha256(np.asarray(prior,dtype=np.float64).tobytes()).hexdigest(),
        "matrix_hash":_hash_obj({"rank_h":int(rank_h),"rank_aug":int(rank_aug),"cols":len(cols)}),
        "firewall_meta_hash":_hash_obj(fw_idx.get("meta",{})),
        "sector_candidates":int(args.sector_candidates), "bp_max_iter":int(args.bp_max_iter),
        "bp_method":str(args.bp_method), "ms_scaling_factor":float(args.ms_scaling_factor),
        "osd_method":str(args.osd_method), "osd_order":int(args.osd_order),
        "max_correction_effects":int(args.max_correction_effects),
    }
    if args.resume:
        if not cache_meta_path.exists() or _load(cache_meta_path) != cache_meta:
            raise RuntimeError("S19 resume metadata mismatch; use a fresh --out-dir or rerun without --resume")
    else:
        # A non-resume run is intentionally fresh and must not silently append
        # to a previous statistical sample/cache.
        for stale in (fw_cache_path,d2_cache_path,out/"s19a1_strata.json",out/"s19a1_harmful_witnesses.json"):
            if stale.exists(): stale.unlink()
        fw_cache={0:b2rmod.resolve_query(0,fw_idx,max_examples=1)}
        _append_fw_cache(fw_cache_path,0,fw_cache[0])
        _dump(cache_meta_path,cache_meta)
    d2_cache=_load_d2_cache(b3mod,d2,d2_cache_path) if args.resume else {}

    # Checkpoint rows allow exact resume by completed stratum key.  Because each
    # stratum gets a deterministic independent RNG stream seed+index, resumed
    # rows are statistically identical to a fresh full run.
    rows_path=out/"s19a1_strata.json"
    rows=[]
    if args.resume and rows_path.exists():
        rows=_load(rows_path)
    done={r["key"] for r in rows}
    witnesses=[]
    wit_path=out/"s19a1_harmful_witnesses.json"
    if args.resume and wit_path.exists(): witnesses=_load(wit_path)

    t0=time.time()
    for si,meta in enumerate(strata):
        if meta["key"] in done:
            continue
        rng=np.random.default_rng(int(args.seed)+si*1000003)
        samples=[]
        stratum_shots=int(plan_by[meta["key"]]["shots"])
        for j in range(stratum_shots):
            locs=_sample_locations(meta,groups,rng)
            # Keep traces only for a tiny second-pass witness population would
            # be ideal, but exact branch randomness prevents deterministic trace
            # recovery.  Store trace on all pilot shots; with 50/stratum this is
            # still manageable and makes malignant examples fully replayable at
            # the sampled mechanism level.
            samples.append(_simulate_exact3(a1,b1mod,sector_by,model,locs,rng,want_trace=True))

        queries={int(r["record_a"]) for r in samples if r["accepted_predecode"]} | {int(r["record_b"]) for r in samples if r["accepted_predecode"]}
        new_fw=sorted(q for q in queries if q not in fw_cache)
        for q in new_fw:
            fw_cache[q]=b2rmod.resolve_query(q,fw_idx,max_examples=1)
            _append_fw_cache(fw_cache_path,q,fw_cache[q])
        fw_exact={q:int(fw_cache[q]["pauli_int"]) for q in queries if fw_cache[q]["status"]=="exact"}
        fw_amb={q for q in queries if fw_cache[q]["status"]=="ambiguous_reject"}
        fw_high={q for q in queries if fw_cache[q]["status"]=="higher_order_only"}
        unknown=sorted(q for q in queries if q!=0 and q not in fw_exact and q not in fw_amb and q not in cheap_exact and q not in cheap_amb)
        bad=[q for q in unknown if q not in fw_high]
        if bad: raise RuntimeError(("low-order query leaked to D2",[hex(q) for q in bad[:8]]))
        todo=[q for q in unknown if q not in d2_cache]
        for q in todo:
            dd=d2._decode_unknown_coset(d1,q,sectors,prior,float(args.decoder_p),decoder,h_aug,cols,int(args.sector_candidates),require_all_paulis=True)
            d2_cache[q]=dd; _append_d2_cache(d2_cache_path,q,d2,dd)
        if any((not bool(d2_cache[q].success)) or len(d2_cache[q].coverage_paulis)!=4 for q in unknown):
            raise RuntimeError(("S19 D2 completeness failure",meta["key"]))

        row=_score_samples(samples,b3mod,b1mod,s10mc,fw_exact,fw_amb,cheap_exact,cheap_amb,d2_cache,
                           int(args.max_correction_effects),int(args.max_examples_per_stratum),meta)
        row["index"]=si; row["seed"]=int(args.seed)+si*1000003
        row["firewall_status_counts"]={
            "exact":sum(q in fw_exact for q in queries),
            "ambiguous_reject":sum(q in fw_amb for q in queries),
            "higher_order_only":sum(q in fw_high for q in queries),
        }
        row["d2_unknown_records"]=len(unknown); row["new_d2_decodes"]=len(todo); row["new_firewall_joins"]=len(new_fw)
        rows.append(row); rows.sort(key=lambda r:int(r["index"])); _dump(rows_path,rows)
        if row["harmful_examples"]:
            witnesses.extend(row["harmful_examples"]); _dump(wit_path,witnesses)
        if (si+1)%10==0 or row["closed_boundary_harmful_samples"]:
            print(f"S19-A1 {si+1:3d}/{len(strata)} {meta['key']}: harmful={row['closed_boundary_harmful_samples']}/{row['shots']} "
                  f"A3={row['A3_closed_boundary_infidelity_estimate']:.6g} D2={len(unknown)}",flush=True)

    agg=_aggregate(rows)
    active=[r for r in rows if int(r["closed_boundary_harmful_samples"])>0 or float(r["A3_closed_boundary_infidelity_estimate"])>0]
    active.sort(key=lambda r:float(r["A3_closed_boundary_infidelity_estimate"]),reverse=True)
    summary={
        "stage":"S19-A1 exact-three stratified pilot with frozen S18-B3 decoder",
        "status":"PILOT_NOT_PAPER_FINAL",
        "structural_source":"S18-M2 + S18-B2R; S19 does not replace <=2 structural certificate",
        "decoder_policy":"zero -> B2R exact<=2 firewall -> cheap exact -> D2",
        "frozen_gap":FROZEN_GAP,"decoder_p":float(args.decoder_p),"decoder_sector_prior_source":"S18-B3 fresh sector prior",
        "decoder_prior_seed":prior_meta.get("seed"),"threshold_sweep_on_S19":False,"truth_bits_decoder_visible":False,
        "location_inventory":loc_inventory,"stratum_count":len(strata),
        "discovery_floor_shots_per_stratum":int(args.shots_per_stratum),"legacy_priority_shots":int(args.legacy_priority_shots),
        "planned_total_conditional_shots":sum(int(x["shots"]) for x in plan),
        "allocation_fixed_before_S19_sampling":True,
        "seed":int(args.seed),"aggregate":agg,"active_strata_count":len(active),
        "top_active_strata":[{k:r[k] for k in ("key","counts","location_triples","shots","closed_boundary_harmful_samples","A3_closed_boundary_infidelity_estimate","A3_closed_boundary_infidelity_se","d2_unknown_records")} for r in active[:30]],
        "matrix":{"H_rank":int(rank_h),"H_aug_rank":int(rank_aug),"effect_columns":len(cols)},
        "firewall_meta":fw_idx.get("meta",{}),"cheap_catalogue_meta":cheap_meta,"sector_meta":smeta,
        "next_stage":"S19-A2 fixed-plan independent high-statistics resampling of all active strata plus nonzero discovery floor on inactive strata; repeat active strata across B3 decoder priors to test A3 stability.",
        "interpretation":[
            "Exact-three location strata are reweighted by their exact combinatorial number of location triples.",
            "Zero-event strata are not certified zero by this pilot.",
            "A3 here is the truth-closed operationally accepted logical-infidelity numerator under the frozen B3 practical decoder.",
            "Because P_acc(0)=1 and the clean boundary is ideal at p=0, this numerator has the same leading p^3 coefficient as conditional clean-boundary logical infidelity.",
            "Paper-final A3 requires S19-A2 independent fixed allocation and prior-stability checks."
        ],
        "elapsed_seconds":time.time()-t0,
    }
    _dump(out/"s19a1_summary.json",summary); _dump(out/"s19a1_active_strata.json",active); _dump(wit_path,witnesses)
    print("S19-A1 PILOT COMPLETE",flush=True); print(json.dumps({"aggregate":agg,"active_strata_count":len(active),"top":[r["key"] for r in active[:10]]},indent=2),flush=True)


if __name__ == "__main__":
    main()
