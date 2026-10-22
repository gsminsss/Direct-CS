#!/usr/bin/env python3
"""S17-C/D1: certified hybrid BP+OSD decoder with physical-feasibility abstention.

Purpose
=======
S17-C/C established a conservative finite-p baseline:
  * frozen 14,197 certified ambiguous 470-bit records are rejected;
  * exact sector-aware total-budget<=2 records with a unique Pauli are corrected;
  * all other records are rejected as out-of-certificate.

At p=5e-4 and 1e-3 this becomes overly conservative because higher-order records
are common. This stage adds a PRACTICAL higher-order decoder only to the
out-of-certificate tier while keeping the exact low-order certificate frozen.

Decoder architecture
====================
Tier 0: certified ambiguous record -> unconditional reject.
Tier 1: exact <=2 unique catalogue record -> exact Pauli frame.
Tier 2: otherwise -> sector-list BP+OSD candidate decoder with abstention.

The BP+OSD matrix uses the 6,918 unique joint downstream effects from v13d.
Each column is an undecomposed detector hyperedge and carries a final logical
Pauli label. Its channel probability is the exact ODD-PARITY MARGINAL of that
effect under the true 2,407-location categorical stochastic model:

    q_e = [1 - product_l (1 - 2 p w_{e,l})] / 2,

where w_{e,l} is the conditional probability that a failed physical location l
produces effect e. Correlations between DIFFERENT effect columns caused by a
shared physical location are therefore not represented inside binary BP+OSD.
To prevent an unphysical correction from being accepted, every returned effect
set is post-validated by solving a distinct-physical-location assignment. A
candidate with no such assignment is rejected.

Incoming Steane sector
======================
The 64 S17-C/A sector signatures are treated as latent hypotheses. For each
unknown record, a small sector list is chosen using the 171-bit live-prefix
Hamming distance. A separate core-only calibration sample estimates the sector
prior at each p. For each listed sector s:

    residual = observed_470 XOR sector_signature[s]

is decoded by the SAME downstream BP+OSD instance. Candidate likelihoods are
scored under a physically feasible minimal distinct-location realization. The
best final Pauli is accepted only when its score advantage over the best
DIFFERENT-Pauli candidate exceeds a user-selected log-gap threshold. If only a
single valid candidate is found, its gap is conservatively set to zero.

This is deliberately a PILOT practical decoder, not a new structural proof.
The low-order proof remains the exact v13d/S17-C/B certificate. This stage asks
how much higher-order acceptance can be recovered without damaging the exact
<=2 regression gate.

Dependencies
============
  pip install -U ldpc

The current LDPCv2 package supports Python>=3.10. scipy is also required.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_DETECTORS = 470
EXPECTED_LOCATIONS = 2407
EXPECTED_EFFECTS = 6918
PREFIX_BITS = 171
PAULI_NAMES = {0: "I", 1: "X", 2: "Z", 3: "Y"}
TOL = 1e-10


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find(root: Path, name: str, prefer: tuple[str, ...] = ()) -> Path:
    xs = list(root.rglob(name))
    if not xs:
        raise FileNotFoundError(f"Could not find {name!r} under {root.resolve()}")
    xs.sort(key=lambda p: (
        tuple(0 if token in str(p) else 1 for token in prefer),
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)), str(p),
    ))
    return xs[0]


def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Can't import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_ldpc():
    try:
        from ldpc.bposd_decoder import BpOsdDecoder
        return BpOsdDecoder
    except Exception as e1:
        try:
            from ldpc import bposd_decoder
            return bposd_decoder
        except Exception as e2:
            raise RuntimeError(
                "S17-C/D requires the current 'ldpc' package. Install with:\n"
                "  py -3.11 -m pip install -U ldpc\n"
                f"Import errors: {e1!r}; {e2!r}"
            )


def _bits_to_u8(x: int, n: int = EXPECTED_DETECTORS) -> np.ndarray:
    out = np.zeros(n, dtype=np.uint8)
    y = int(x)
    while y:
        b = y & -y
        i = b.bit_length() - 1
        if i >= n:
            raise ValueError(f"bit {i} exceeds detector count {n}")
        out[i] = 1
        y ^= b
    return out


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    ph = k / n
    den = 1 + z*z/n
    cen = (ph + z*z/(2*n))/den
    rad = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/den
    return [max(0.0, cen-rad), min(1.0, cen+rad)]


def _zero_event_upper95(n: int) -> float | None:
    if n <= 0:
        return None
    return 1.0 - 0.05 ** (1.0 / n)


@dataclass
class EffectColumn:
    syndrome: int
    pauli: int
    loc_weight: dict[int, float]
    total_weight: float


@dataclass
class Candidate:
    sector: int
    pauli: int
    score: float
    correction_effect_count: int
    assigned_locations: tuple[int, ...]
    selected_columns: tuple[int, ...]
    prefix_distance: int


@dataclass
class DecodeResult:
    success: bool
    pauli: int | None
    gap: float
    candidate_count: int
    best: Candidate | None
    best_different: Candidate | None
    invalid_candidate_count: int
    all_valid_candidates_same_pauli: bool


def _sector_rows(path: Path):
    rows = _load_json(path)
    if not isinstance(rows, list) or len(rows) != 64:
        raise RuntimeError("Expected 64 S17-C/A sector rows")
    rows = sorted(rows, key=lambda r: int(r["steane_syndrome"]))
    if [int(r["steane_syndrome"]) for r in rows] != list(range(64)):
        raise RuntimeError("Sector rows must be exactly 0..63")
    if int(rows[0]["detector_470_hex"], 16) != 0 or int(rows[0]["raw_terminal_pauli_int"]) != 0:
        raise RuntimeError("Sector 0 is not zero-record identity")
    return rows


def _effect_columns(effects: list[Any]) -> list[EffectColumn]:
    cols = []
    for e in effects:
        s = int(e.syndrome)
        p = int(e.pauli)
        # First-order v13d proved that zero syndrome has identity only. Such a
        # column cannot help explain a record and has no logical action.
        if s == 0:
            if p != 0:
                raise RuntimeError("zero-syndrome nonidentity downstream effect found")
            continue
        lw = {int(k): float(v) for k, v in e.loc_weight.items() if float(v) > 0}
        cols.append(EffectColumn(s, p, lw, float(sum(lw.values()))))
    return cols


def _build_sparse_check_matrix(cols: list[EffectColumn]):
    from scipy.sparse import coo_matrix
    rr = []
    cc = []
    for j, c in enumerate(cols):
        x = int(c.syndrome)
        while x:
            b = x & -x
            rr.append(b.bit_length()-1)
            cc.append(j)
            x ^= b
    data = np.ones(len(rr), dtype=np.uint8)
    h = coo_matrix((data, (np.asarray(rr), np.asarray(cc))), shape=(EXPECTED_DETECTORS, len(cols)), dtype=np.uint8).tocsr()
    return h


def _effect_parity_priors(cols: list[EffectColumn], p: float) -> np.ndarray:
    """Exact marginal P[odd number of occurrences of each effect]."""
    out = np.zeros(len(cols), dtype=float)
    for j, c in enumerate(cols):
        prod = 1.0
        for w in c.loc_weight.values():
            prod *= (1.0 - 2.0*float(p)*float(w))
        q = 0.5*(1.0-prod)
        # BP packages do not like exact 0/1 probabilities.
        out[j] = min(0.499999999, max(1e-15, q))
    return out


def _make_bposd(BpOsdDecoder, h, priors: np.ndarray, max_iter: int, bp_method: str,
                ms_scaling: float, osd_method: str, osd_order: int):
    kwargs = dict(
        max_iter=int(max_iter),
        bp_method=str(bp_method),
        ms_scaling_factor=float(ms_scaling),
        schedule="parallel",
        osd_method=str(osd_method),
        osd_order=int(osd_order),
    )
    # Current LDPCv2 uses error_channel. Older compatibility layers use
    # channel_probs. Try the modern API first.
    try:
        return BpOsdDecoder(h, error_channel=np.asarray(priors, dtype=float), **kwargs)
    except TypeError as e1:
        try:
            return BpOsdDecoder(h, channel_probs=np.asarray(priors, dtype=float), **kwargs)
        except TypeError:
            # Last fallback: error_rate + channel_probs keyword in old bposd.
            return BpOsdDecoder(h, error_rate=float(np.mean(priors)), channel_probs=np.asarray(priors, dtype=float), **kwargs)


def _decode_vector(decoder, syndrome: np.ndarray, ncols: int) -> np.ndarray:
    raw = decoder.decode(np.asarray(syndrome, dtype=np.uint8))
    arr = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if arr.size != ncols:
        for name in ("osdw_decoding", "decoding", "bp_decoding"):
            if hasattr(decoder, name):
                a = np.asarray(getattr(decoder, name), dtype=np.uint8).reshape(-1)
                if a.size == ncols:
                    arr = a
                    break
    if arr.size != ncols:
        raise RuntimeError(f"BP+OSD returned correction length {arr.size}, expected {ncols}")
    return arr & 1


def _syndrome_from_selected(cols: list[EffectColumn], sel: np.ndarray) -> int:
    s = 0
    for j in np.flatnonzero(sel):
        s ^= int(cols[int(j)].syndrome)
    return int(s)


def _assign_distinct_locations(cols: list[EffectColumn], selected: tuple[int, ...]):
    """Maximum-weight distinct-location realization of selected effects.

    Returns (locations, log_conditional_weight) or None. The assignment is a
    physical-feasibility guard against shared-location correlations omitted by
    the binary BP approximation.
    """
    if not selected:
        return tuple(), 0.0
    from scipy.optimize import linear_sum_assignment

    supports = [cols[j].loc_weight for j in selected]
    union = sorted(set().union(*(d.keys() for d in supports)))
    if len(union) < len(selected):
        return None
    pos = {loc:i for i,loc in enumerate(union)}
    big = 1e12
    cost = np.full((len(selected), len(union)), big, dtype=float)
    for r, d in enumerate(supports):
        for loc, w in d.items():
            cost[r, pos[loc]] = -math.log(max(float(w), 1e-300))
    ri, ci = linear_sum_assignment(cost)
    if len(ri) != len(selected):
        return None
    vals = cost[ri, ci]
    if np.any(vals >= big/2):
        return None
    locs = tuple(int(union[int(c)]) for c in ci)
    return locs, -float(np.sum(vals))


def _calibrate_sector_prior(p: float, shots: int, seed: int, s10mc, alpha: float = 0.5):
    rng = np.random.default_rng(int(seed))
    counts = np.full(64, float(alpha), dtype=float)
    accepted = 0
    attempts = 0
    reasons = collections.Counter()
    for _ in range(int(shots)):
        attempts += 1
        fa = s10mc._front_block(p, rng)
        fb = s10mc._front_block(p, rng)
        if not fa["accepted"] or not fb["accepted"]:
            reasons["initial_frontend_ED"] += 1
            continue
        ok, why, rec, ss, lv, nfc, nma, nmb, ncat, nro, sc, fast = s10mc._sample_core_fullfront(p, rng, fa, fb)
        if not ok:
            reasons[f"core_{why}"] += 1
            continue
        accepted += 1
        sa, sb = map(int, ss)
        counts[sa] += 1.0
        counts[sb] += 1.0
    prior = counts / counts.sum()
    return prior, {
        "p": float(p),
        "calibration_attempts": int(attempts),
        "core_accepted": int(accepted),
        "laplace_alpha_per_sector": float(alpha),
        "sector_probability": {str(i): float(prior[i]) for i in range(64)},
        "sector_count_smoothed": {str(i): float(counts[i]) for i in range(64)},
        "rejection_reasons": dict(sorted(reasons.items())),
    }


def _rank_sector_candidates(record: int, sectors: list[dict[str, Any]], prior: np.ndarray, k: int):
    mask = (1 << PREFIX_BITS) - 1
    scored = []
    for r in sectors:
        s = int(r["steane_syndrome"])
        rs = int(r["detector_470_hex"], 16)
        d = ((int(record) ^ rs) & mask).bit_count()
        # Primary key is live-prefix distance; prior only breaks ties. This keeps
        # the shortlist detector-driven instead of prior-dominated.
        scored.append((int(d), -math.log(max(float(prior[s]), 1e-300)), s))
    scored.sort()
    out = [s for _,_,s in scored[:max(1,int(k))]]
    if 0 not in out:
        out[-1] = 0
    # preserve rank order after injecting zero
    out = list(dict.fromkeys(out))
    return out


def _candidate_for_sector(record: int, sector: int, sectors: list[dict[str, Any]], prior: np.ndarray,
                          p: float, decoder, h, cols: list[EffectColumn]) -> Candidate | None:
    r = sectors[int(sector)]
    rs = int(r["detector_470_hex"], 16)
    residual = int(record) ^ rs
    prefix_distance = ((int(record) ^ rs) & ((1 << PREFIX_BITS)-1)).bit_count()

    if residual == 0:
        selected = tuple()
        assn = (tuple(), 0.0)
    else:
        syn = _bits_to_u8(residual)
        corr = _decode_vector(decoder, syn, len(cols))
        if _syndrome_from_selected(cols, corr) != residual:
            return None
        selected = tuple(int(x) for x in np.flatnonzero(corr))
        assn = _assign_distinct_locations(cols, selected)
        if assn is None:
            return None

    locs, log_cond_weight = assn
    k = len(locs)
    # Probability ratio against no downstream faults. Each assigned location
    # contributes p*w/(1-p); common (1-p)^N cancels across candidates.
    logscore = math.log(max(float(prior[int(sector)]), 1e-300))
    if k:
        logscore += k * math.log(max(float(p), 1e-300) / max(1.0-float(p), 1e-300))
        logscore += float(log_cond_weight)
    pauli = int(r["raw_terminal_pauli_int"])
    for j in selected:
        pauli ^= int(cols[j].pauli)
    return Candidate(
        sector=int(sector), pauli=int(pauli), score=float(logscore),
        correction_effect_count=int(k), assigned_locations=tuple(locs),
        selected_columns=tuple(selected), prefix_distance=int(prefix_distance),
    )


def _decode_unknown_record(record: int, sectors: list[dict[str, Any]], prior: np.ndarray, p: float,
                           decoder, h, cols: list[EffectColumn], sector_candidates: int) -> DecodeResult:
    valid: list[Candidate] = []
    invalid = 0
    for s in _rank_sector_candidates(record, sectors, prior, sector_candidates):
        try:
            c = _candidate_for_sector(record, s, sectors, prior, p, decoder, h, cols)
        except Exception:
            c = None
        if c is None:
            invalid += 1
        else:
            valid.append(c)
    if not valid:
        return DecodeResult(False, None, 0.0, 0, None, None, invalid, False)
    valid.sort(key=lambda c: c.score, reverse=True)
    best = valid[0]
    diff = next((c for c in valid[1:] if c.pauli != best.pauli), None)
    if len(valid) == 1:
        gap = 0.0
        same = False
    elif diff is None:
        gap = float("inf")
        same = True
    else:
        gap = float(best.score - diff.score)
        same = False
    return DecodeResult(True, int(best.pauli), gap, len(valid), best, diff, invalid, same)


def _finish_threshold(sampled: dict[str, Any], exact_decoder: dict[int, int], reject_set: set[int],
                      high_decoder: dict[int, DecodeResult], s10mc, min_gap: float,
                      max_correction_effects: int, max_examples: int):
    shots = int(sampled["shots"])
    trajectories = sampled["trajectories"]
    accepted = 0
    explicit_reject = exact_accept = high_accept = high_abstain = 0
    harmful = harmful_le2 = 0
    inf_sum = oracle_sum = 0.0
    high_harmful = 0
    examples = {"higher_order_harmful": [], "higher_order_abstain": [], "higher_order_accept": []}

    def one_patch(r: int):
        if r in reject_set:
            return "reject", None, None
        if r in exact_decoder:
            return "exact", int(exact_decoder[r]), None
        d = high_decoder.get(r)
        if d is None or not d.success or d.best is None:
            return "abstain", None, d
        if d.best.correction_effect_count > int(max_correction_effects):
            return "abstain", None, d
        if float(d.gap) < float(min_gap):
            return "abstain", None, d
        return "high", int(d.pauli), d

    for tr in trajectories:
        ka, pa, da = one_patch(int(tr.record_a))
        kb, pb, db = one_patch(int(tr.record_b))
        if ka == "reject" or kb == "reject":
            explicit_reject += 1
            continue
        if ka == "abstain" or kb == "abstain":
            high_abstain += 1
            if len(examples["higher_order_abstain"]) < max_examples:
                examples["higher_order_abstain"].append({
                    "record_A": hex(int(tr.record_a)), "record_B": hex(int(tr.record_b)),
                    "A_tier": ka, "B_tier": kb,
                    "A_gap": None if da is None else da.gap,
                    "B_gap": None if db is None else db.gap,
                    "total_fault_locations": int(tr.total_fault_locations),
                })
            continue

        exact_accept += int(ka == "exact" and kb == "exact")
        high_accept += int(ka == "high" or kb == "high")
        # Reuse the audited S17-C/C fidelity definition.
        residual_a = int(pa) ^ int(tr.actual_pauli_a)
        residual_b = int(pb) ^ int(tr.actual_pauli_b)
        corr = s10mc._apply(np.asarray(tr.lv, complex), residual_a, residual_b)
        target = s10mc._target(tr.core_record)
        f = float(abs(np.vdot(target, corr))**2)
        inf = max(0.0, min(1.0, 1.0-f))
        accepted += 1
        inf_sum += inf
        oracle_sum += float(tr.intrinsic_oracle_infidelity)
        if inf > TOL:
            harmful += 1
            harmful_le2 += int(tr.total_fault_locations <= 2)
            high_harmful += int(ka == "high" or kb == "high")
            if len(examples["higher_order_harmful"]) < max_examples:
                examples["higher_order_harmful"].append({
                    "record_A": hex(int(tr.record_a)), "record_B": hex(int(tr.record_b)),
                    "A_tier": ka, "B_tier": kb,
                    "pred_A": PAULI_NAMES[int(pa)], "pred_B": PAULI_NAMES[int(pb)],
                    "actual_A": PAULI_NAMES[int(tr.actual_pauli_a)], "actual_B": PAULI_NAMES[int(tr.actual_pauli_b)],
                    "A_gap": None if da is None else da.gap,
                    "B_gap": None if db is None else db.gap,
                    "infidelity": inf, "intrinsic_oracle_infidelity": float(tr.intrinsic_oracle_infidelity),
                    "total_fault_locations": int(tr.total_fault_locations),
                })
        elif (ka == "high" or kb == "high") and len(examples["higher_order_accept"]) < max_examples:
            examples["higher_order_accept"].append({
                "record_A": hex(int(tr.record_a)), "record_B": hex(int(tr.record_b)),
                "A_tier": ka, "B_tier": kb,
                "A_gap": None if da is None else da.gap,
                "B_gap": None if db is None else db.gap,
                "total_fault_locations": int(tr.total_fault_locations),
            })

    front_acc = int(sampled["predecode"]["frontend_accepted"])
    core_acc = int(sampled["predecode"]["core_accepted"])
    executed_proxy = 194.0 + (front_acc/shots)*696.0 + (core_acc/shots)*(2*EXPECTED_LOCATIONS)
    return {
        "min_log_score_gap": float(min_gap),
        "max_correction_effects": int(max_correction_effects),
        "delivered_accepted": int(accepted),
        "end_to_end_acceptance": accepted/shots,
        "end_to_end_acceptance_wilson95": _wilson(accepted, shots),
        "attempts_per_delivered_output": shots/accepted if accepted else None,
        "explicit_certified_reject": int(explicit_reject),
        "higher_order_abstain": int(high_abstain),
        "accepted_with_any_higher_order_decode": int(high_accept),
        "accepted_exact_both_patches": int(exact_accept),
        "harmful_accepted_samples": int(harmful),
        "harmful_using_higher_order_tier": int(high_harmful),
        "accepted_harmful_samples_with_total_fault_locations_le2": int(harmful_le2),
        "global_le2_regression_gate_pass": bool(harmful_le2 == 0),
        "conditional_logical_infidelity": inf_sum/accepted if accepted else None,
        "logical_infidelity_per_attempt": inf_sum/shots,
        "conditional_intrinsic_oracle_infidelity": oracle_sum/accepted if accepted else None,
        "decoder_excess_infidelity_conditional": (inf_sum-oracle_sum)/accepted if accepted else None,
        "harmful_fraction_among_accepted": harmful/accepted if accepted else None,
        "harmful_fraction_wilson95": _wilson(harmful, accepted) if accepted else None,
        "zero_harmful_one_sided_95_upper_if_applicable": _zero_event_upper95(accepted) if accepted and harmful == 0 else None,
        "stage_level_active_location_proxy_per_delivered_output": executed_proxy/(accepted/shots) if accepted else None,
        "examples": examples,
    }


def _serialize_decode_result(d: DecodeResult):
    def cobj(c):
        if c is None:
            return None
        return {
            "sector": int(c.sector), "pauli": PAULI_NAMES[int(c.pauli)], "score": float(c.score),
            "correction_effect_count": int(c.correction_effect_count),
            "assigned_locations": list(map(int,c.assigned_locations)),
            "selected_columns": list(map(int,c.selected_columns)),
            "prefix_distance": int(c.prefix_distance),
        }
    return {
        "success": bool(d.success), "pauli": None if d.pauli is None else PAULI_NAMES[int(d.pauli)],
        "gap": None if not math.isfinite(float(d.gap)) else float(d.gap),
        "gap_is_infinite_consensus": bool(math.isinf(float(d.gap))),
        "candidate_count": int(d.candidate_count), "invalid_candidate_count": int(d.invalid_candidate_count),
        "all_valid_candidates_same_pauli": bool(d.all_valid_candidates_same_pauli),
        "best": cobj(d.best), "best_different_pauli": cobj(d.best_different),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s17c-c-script", type=Path, default=None)
    ap.add_argument("--s10-script", type=Path, default=None)
    ap.add_argument("--v13d-script", type=Path, default=None)
    ap.add_argument("--sector-map", type=Path, default=None)
    ap.add_argument("--sector-aware-reject-policy", type=Path, default=None)
    ap.add_argument("--x-circuit", type=Path, default=None)
    ap.add_argument("--z-circuit", type=Path, default=None)
    ap.add_argument("--p-values", type=float, nargs="+", default=[5e-4, 1e-3])
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=None)
    ap.add_argument("--prior-calibration-shots", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--sector-candidates", type=int, default=4)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd0")
    ap.add_argument("--osd-order", type=int, default=0)
    ap.add_argument("--gap-thresholds", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0, 8.0])
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.search_root.resolve()

    c_path = (args.s17c_c_script or _find(root, "direct_cs_s17c_c_finite_p_e2e_certified_lookup.py", ("stim_pipeline/interface",))).resolve()
    s10_path = (args.s10_script or _find(root, "direct_cs_s10_complete_physical_factory_mc.py", ("stim_pipeline/interface",))).resolve()
    v13d_path = (args.v13d_script or _find(root, "direct_cs_s17b_v13d_joint_postselection_certificate.py", ("stim_pipeline/interface",))).resolve()
    sector_path = (args.sector_map or _find(root, "s17c_a_sector_map.json", ("s17c_a_results",))).resolve()
    reject_path = (args.sector_aware_reject_policy or _find(root, "s17c_b_sector_aware_reject_policy.json", ("s17c_b_results",))).resolve()
    x_path = (args.x_circuit or _find(root, "s17b_v11_X_transition_circuit.stim", ("s17b_v11_order2_results",))).resolve()
    z_path = (args.z_circuit or _find(root, "s17b_v11_Z_transition_circuit.stim", ("s17b_v11_order2_results",))).resolve()

    print(f"S17-C/D S17-C/C script: {c_path}", flush=True)
    print(f"S17-C/D sector map: {sector_path}", flush=True)
    print(f"S17-C/D reject policy: {reject_path}", flush=True)

    BpOsdDecoder = _require_ldpc()
    s17cc = _import_module("s17cd_s17cc", c_path)
    s10mc = _import_module("s17cd_s10mc", s10_path)
    v13d = _import_module("s17cd_v13d", v13d_path)
    sectors = _sector_rows(sector_path)
    reject_obj = _load_json(reject_path)
    reject_set = {int(x,16) for x in reject_obj["total_sector_aware_rejected_syndromes_hex"]}
    if len(reject_set) != 14197 or 0 in reject_set:
        raise RuntimeError(f"Expected frozen 14,197 nonzero reject records, got {len(reject_set)}")

    downstream = s17cc._build_downstream_model(v13d, x_path, z_path)
    if len(downstream["effects"]) != EXPECTED_EFFECTS:
        raise RuntimeError((len(downstream["effects"]), EXPECTED_EFFECTS))
    cols = _effect_columns(downstream["effects"])
    h = _build_sparse_check_matrix(cols)
    if h.shape[0] != EXPECTED_DETECTORS:
        raise RuntimeError(h.shape)

    # Static matrix guards independent of ldpc decode quality.
    matrix_meta = {
        "raw_joint_unique_effects": len(downstream["effects"]),
        "nonzero_syndrome_effect_columns": len(cols),
        "check_matrix_shape": list(map(int,h.shape)),
        "check_matrix_nnz": int(h.nnz),
        "frozen_reject_records": len(reject_set),
        "sector_count": len(sectors),
        "physical_locations_per_patch": downstream["meta"]["physical_locations_per_patch"],
        "same_mechanism_detector_syndromes_match": downstream["meta"]["joint_meta"]["all_same_mechanism_detector_syndromes_match_across_probes"],
    }

    # Tiny decoder smoke test: zero syndrome must decode to a vector satisfying zero.
    p_smoke = max(1e-6, min(args.p_values))
    pri_smoke = _effect_parity_priors(cols, p_smoke)
    dec_smoke = _make_bposd(BpOsdDecoder, h, pri_smoke, args.bp_max_iter, args.bp_method,
                            args.ms_scaling_factor, args.osd_method, args.osd_order)
    zcorr = _decode_vector(dec_smoke, np.zeros(EXPECTED_DETECTORS, dtype=np.uint8), len(cols))
    smoke_ok = (_syndrome_from_selected(cols, zcorr) == 0)
    st = {
        "passed": bool(smoke_ok and len(reject_set)==14197 and 0 not in reject_set and len(sectors)==64),
        "ldpc_decoder_class": getattr(BpOsdDecoder, "__name__", str(BpOsdDecoder)),
        "matrix": matrix_meta,
        "zero_syndrome_decode_satisfies_checks": bool(smoke_ok),
        "scope_warning": "BP+OSD effect columns use exact single-effect parity marginals but ignore cross-effect correlations from shared locations; every accepted correction is therefore post-validated by distinct physical-location assignment and remains an empirical higher-order decoder, not a structural certificate.",
    }
    _dump(args.out_dir/"s17c_d_self_test.json", st)
    if not st["passed"]:
        raise RuntimeError(st)
    if args.self_test_only:
        print(json.dumps(st, indent=2), flush=True)
        print("S17-C/D SELF-TEST PASS", flush=True)
        return

    if args.shots_per_p is not None:
        if len(args.shots_per_p) != len(args.p_values):
            raise SystemExit("--shots-per-p length must match --p-values")
        shot_list = [int(x) for x in args.shots_per_p]
    else:
        shot_list = [int(args.shots)]*len(args.p_values)

    all_rows = []
    prior_rows = []
    decoder_audits = []
    t_all = time.time()

    for ip, (p, shots) in enumerate(zip(args.p_values, shot_list)):
        p = float(p)
        print(f"S17-C/D: calibrating sector prior p={p:g}, attempts={args.prior_calibration_shots} ...", flush=True)
        sector_prior, prior_meta = _calibrate_sector_prior(p, args.prior_calibration_shots,
                                                           args.seed + 10000 + ip, s10mc)
        prior_rows.append(prior_meta)

        print(f"S17-C/D: finite-p sample p={p:g}, shots={shots} ...", flush=True)
        sampled = s17cc._sample_one_p(p, int(shots), int(args.seed+ip), s10mc,
                                      {int(r["steane_syndrome"]):r for r in sectors}, downstream,
                                      args.max_examples)
        queries = set(sampled["queries"])
        print(f"S17-C/D: exact <=2 resolving {len(queries)} observed patch records ...", flush=True)
        exact_decoder, exact_meta = s17cc._resolve_observed_records(queries, sectors, downstream["effects"], v13d, reject_set)
        unknown = sorted(q for q in queries if q not in reject_set and q not in exact_decoder)

        pri = _effect_parity_priors(cols, p)
        decoder = _make_bposd(BpOsdDecoder, h, pri, args.bp_max_iter, args.bp_method,
                              args.ms_scaling_factor, args.osd_method, args.osd_order)
        high: dict[int, DecodeResult] = {}
        tdec = time.time()
        print(f"S17-C/D: BP+OSD decoding {len(unknown)} out-of-certificate unique records with top-{args.sector_candidates} sectors ...", flush=True)
        for qi, q in enumerate(unknown):
            high[q] = _decode_unknown_record(q, sectors, sector_prior, p, decoder, h, cols, args.sector_candidates)
            if (qi+1) % 500 == 0:
                print(f"  decoded {qi+1}/{len(unknown)}", flush=True)
        dec_seconds = time.time()-tdec

        success = sum(d.success for d in high.values())
        physical_invalid = sum(d.invalid_candidate_count for d in high.values())
        infinite_consensus = sum(d.success and math.isinf(d.gap) for d in high.values())
        gaps = [float(d.gap) for d in high.values() if d.success and math.isfinite(float(d.gap))]
        decoder_audits.append({
            "p": p,
            "observed_patch_records": len(queries),
            "exact_unique_records": len(exact_decoder),
            "frozen_reject_records_observed": sum(q in reject_set for q in queries),
            "out_of_certificate_unique_records": len(unknown),
            "higher_order_decode_success_records": int(success),
            "higher_order_decode_failure_records": int(len(unknown)-success),
            "candidate_physical_assignment_failures_total": int(physical_invalid),
            "infinite_gap_same_pauli_consensus_records": int(infinite_consensus),
            "finite_gap_summary": {
                "count": len(gaps),
                "median": None if not gaps else float(np.median(gaps)),
                "p10": None if not gaps else float(np.quantile(gaps,0.1)),
                "p90": None if not gaps else float(np.quantile(gaps,0.9)),
            },
            "bp_osd_seconds": dec_seconds,
            "exact_catalogue_meta": exact_meta,
            "effect_prior_min": float(np.min(pri)),
            "effect_prior_max": float(np.max(pri)),
        })

        sweeps = []
        for gap in args.gap_thresholds:
            row = _finish_threshold(sampled, exact_decoder, reject_set, high, s10mc,
                                    float(gap), args.max_correction_effects, args.max_examples)
            sweeps.append(row)
            print(json.dumps({
                "p": p, "gap": gap,
                "acceptance": row["end_to_end_acceptance"],
                "cond_infidelity": row["conditional_logical_infidelity"],
                "harmful": row["harmful_accepted_samples"],
                "le2_gate": row["global_le2_regression_gate_pass"],
                "higher_order_accept": row["accepted_with_any_higher_order_decode"],
                "higher_order_abstain": row["higher_order_abstain"],
            }, indent=2), flush=True)

        # Remove heavy trajectory list before storing compact row.
        front = {k:v for k,v in sampled["predecode"].items() if k != "examples"}
        all_rows.append({
            "p": p, "shots": int(shots), "seed": int(args.seed+ip),
            "frontend": front,
            "conservative_s17c_c_reference": {
                "core_accepted": int(front["core_accepted"]),
                "exact_unique_query_records": len(exact_decoder),
                "out_of_certificate_unique_query_records": len(unknown),
            },
            "threshold_sweep": sweeps,
        })

        # Save a bounded sample of high-order decoder decisions for audit.
        sample_keys = unknown[:min(200, len(unknown))]
        _dump(args.out_dir/f"s17c_d_high_decoder_examples_p{p:g}.json",
              [{"record":hex(q), **_serialize_decode_result(high[q])} for q in sample_keys])

    global_le2 = all(
        x["global_le2_regression_gate_pass"]
        for row in all_rows for x in row["threshold_sweep"]
    )
    summary = {
        "stage": "S17-C/D1 certified hybrid BP+OSD + abstention pilot",
        "inputs": {
            "s17c_c_script": str(c_path), "s10_core_script": str(s10_path), "v13d_script": str(v13d_path),
            "sector_map": str(sector_path), "sector_aware_reject_policy": str(reject_path),
            "X_circuit": str(x_path), "Z_circuit": str(z_path),
        },
        "decoder": {
            "tier0": "frozen 14,197 certified ambiguous records -> reject",
            "tier1": "exact total-budget<=2 unique catalogue -> exact frame",
            "tier2": "out-of-certificate only: top-K latent-sector BP+OSD over 6,918 undecomposed joint-effect hyperedges, followed by distinct-location physical-feasibility validation and score-gap abstention",
            "bp_max_iter": int(args.bp_max_iter), "bp_method": args.bp_method,
            "ms_scaling_factor": float(args.ms_scaling_factor), "osd_method": args.osd_method,
            "osd_order": int(args.osd_order), "sector_candidates": int(args.sector_candidates),
            "gap_thresholds": list(map(float,args.gap_thresholds)),
            "max_correction_effects": int(args.max_correction_effects),
            "sector_prior": f"independent core-only Monte Carlo calibration, {args.prior_calibration_shots} attempts per p with Laplace smoothing",
        },
        "matrix": matrix_meta,
        "self_test": st,
        "sector_prior_calibration": prior_rows,
        "decoder_audit": decoder_audits,
        "finite_p_rows": all_rows,
        "main_gate": {
            "frozen_low_order_reject_policy_unchanged": True,
            "global_no_sampled_accepted_harmful_event_with_total_physical_fault_locations_le2": bool(global_le2),
            "higher_order_decoder_is_empirical_not_structural": True,
        },
        "scope_warning": "The BP+OSD binary effect model preserves each effect's exact odd-parity marginal but not correlations between different effect columns sharing physical locations. Physical-feasibility assignment removes impossible returned corrections, but finite-p performance remains empirical and must be threshold-selected on independent calibration/evaluation data before paper-final use.",
        "ideal_d13_close_warning": "The 470-bit boundary still uses the ideal downstream d13 close structurally; standalone noisy delivered-d13 performance remains unresolved.",
        "next": "Choose a confidence threshold from the acceptance-vs-infidelity sweep. If recovery is strong without damaging logical error, rerun with independent calibration/evaluation seeds and then add the noisy first d13 QEC round + error-matched 3T resource baseline. If recovery is weak, compare BP+LSD or a categorical-location decoder before considering ML.",
        "timing_seconds": time.time()-t_all,
    }
    _dump(args.out_dir/"s17c_d_summary.json", summary)
    _dump(args.out_dir/"s17c_d_finite_p_rows.json", all_rows)
    _dump(args.out_dir/"s17c_d_decoder_audit.json", decoder_audits)
    _dump(args.out_dir/"s17c_d_sector_priors.json", prior_rows)

    print(json.dumps({
        "global_le2_regression_gate_pass": bool(global_le2),
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
