#!/usr/bin/env python3
"""S17-C/C: finite-p E2E direct-|CS> Monte Carlo with certified 470-bit lookup.

Scientific purpose
==================
S17-C/A mapped all 64 canonical Steane boundary sectors into the 470-bit
continuous operational record used by S17-B v13d. S17-C/B then proved that,
under the total low-order budget

    canonical_boundary_weight + downstream_fault_locations <= 2,

every covered configuration is either assigned a unique final Pauli frame or
mapped to the inherited 14,197-record nonzero reject set. No additional
sector-aware reject records were needed.

This stage integrates finite-p probabilities from the full non-Clifford direct-
|CS> physical front/core with TWO noisy 2407-location continuous handoff/growth
patches. The downstream physical sampling is concrete-location aware and
samples exactly one local Pauli mechanism conditional on a failed location.

Decoder policy
==============
This first finite-p E2E baseline deliberately uses a conservative certified
lookup decoder:

  * if an observed 470-bit record is in the S17-C/B reject set -> reject;
  * else, if the record has exactly one Pauli label in the exact total-budget
    <=2 catalogue -> accept and apply that Pauli frame;
  * else (no <=2 explanation) -> reject as OUT-OF-CERTIFICATE.

The <=2 decoder labels for the ACTUALLY OBSERVED finite-p records are recovered
exactly. The expensive clean-boundary order-2 catalogue is streamed once over
all query records (the same 23.9M location-compatible effect pairs audited in
v13d/S17-C/B), avoiding a >20M-entry Python dictionary.

This policy is intentionally conservative at p where weight>=3 events are
common. It gives a decoder-independent certified finite-p baseline and tells us
whether a higher-order practical decoder (S17-C/D, e.g. BP/OSD) is actually
needed for competitive acceptance.

Important scope
===============
* Full physical front/core uses the existing S10-C active-location model:
  physical encoded preparation + initial ED + verified CAT7/readout + 322-loc
  non-Clifford six-check core + physical midpoint ED.
* Each output patch then uses the S17-B v11/v13d 2407-location noisy continuous
  handoff: Hirano ZXZ + two d5 rounds + Sahay d5->7->13.
* The ideal downstream d13 close remains a STRUCTURAL scoring boundary, exactly
  as in S17-B. A paper-final standalone factory must either model the first
  noisy downstream d13 QEC round or explicitly charge it to the consumer.
* Noise model is the stated independent active-location stochastic Pauli model;
  idle/leakage/coherent/correlated multi-location errors remain outside scope.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_DETECTORS = 470
EXPECTED_DOWNSTREAM_LOCATIONS = 2407
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
        len(str(p)),
        str(p),
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


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    ph = k / n
    den = 1 + z*z/n
    cen = (ph + z*z/(2*n))/den
    rad = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/den
    return [max(0.0, cen-rad), min(1.0, cen+rad)]


def _zero_event_upper95(n: int) -> float | None:
    # Exact one-sided Clopper-Pearson upper bound for k=0: 1-alpha^(1/n).
    if n <= 0:
        return None
    return 1.0 - 0.05 ** (1.0 / n)


def _sector_rows(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = _load_json(path)
    if not isinstance(rows, list) or len(rows) != 64:
        raise RuntimeError(f"Expected 64 sector rows, got {type(rows).__name__}/{len(rows) if isinstance(rows, list) else 'n/a'}")
    by_s = {int(r["steane_syndrome"]): r for r in rows}
    if set(by_s) != set(range(64)):
        raise RuntimeError("Sector map must contain syndromes 0..63 exactly once")
    if len({int(r["detector_470_hex"], 16) for r in rows}) != 64:
        raise RuntimeError("Sector 470-bit signatures are not injective")
    z = by_s[0]
    if int(z["detector_470_hex"], 16) != 0 or int(z["raw_terminal_pauli_int"]) != 0:
        raise RuntimeError("Zero sector is not zero-record/identity")
    if any(int(r["steane_syndrome"]) != 0 and int(r["detector_470_hex"], 16) == 0 for r in rows):
        raise RuntimeError("Nontrivial sector maps to zero record")
    if any(int(r["detector_470_hex"], 16).bit_length() > EXPECTED_DETECTORS for r in rows):
        raise RuntimeError("Sector signature exceeds 470 bits")
    return rows, by_s


@dataclass
class DownstreamConcreteChoice:
    syndrome: int
    pauli: int


@dataclass
class StoredTrajectory:
    record_a: int
    record_b: int
    actual_pauli_a: int
    actual_pauli_b: int
    lv: np.ndarray
    core_record: tuple[int, ...]
    total_fault_locations: int
    pre_downstream_fault_locations: int
    downstream_fault_locations_a: int
    downstream_fault_locations_b: int
    intrinsic_oracle_infidelity: float


def _build_downstream_model(v13d, x_path: Path, z_path: Path):
    import stim

    x_det = v13d._detector_lines(x_path)
    z_det = v13d._detector_lines(z_path)
    if x_det != z_det or len(x_det) != EXPECTED_DETECTORS:
        raise RuntimeError("X/Z detector definitions differ or detector count !=470")

    x_locs, x_native_to_ord, x_meta = v13d._noise_locations_from_text(x_path)
    z_locs, z_native_to_ord, z_meta = v13d._noise_locations_from_text(z_path)
    if len(x_locs) != EXPECTED_DOWNSTREAM_LOCATIONS or len(z_locs) != EXPECTED_DOWNSTREAM_LOCATIONS:
        raise RuntimeError((len(x_locs), len(z_locs), EXPECTED_DOWNSTREAM_LOCATIONS))
    x_phys = [(x_meta[i]["tick"], x_meta[i]["gate"], tuple(x_meta[i]["qubits"])) for i in x_locs]
    z_phys = [(z_meta[i]["tick"], z_meta[i]["gate"], tuple(z_meta[i]["qubits"])) for i in z_locs]
    if x_phys != z_phys:
        raise RuntimeError("X/Z physical stochastic schedule differs")

    v13d._LOCATION_META.clear()
    v13d._LOCATION_META.update(x_meta)
    xc = stim.Circuit(x_path.read_text(encoding="utf-8"))
    zc = stim.Circuit(z_path.read_text(encoding="utf-8"))

    print("S17-C/C: recovering X-probe concrete downstream mechanisms...", flush=True)
    xpartial, xmeta_out = v13d._recover_mechanism_map(xc, v13d._raw_dem(xc), "X", x_native_to_ord)
    print("S17-C/C: recovering Z-probe concrete downstream mechanisms...", flush=True)
    zpartial, zmeta_out = v13d._recover_mechanism_map(zc, v13d._raw_dem(zc), "Z", z_native_to_ord)

    universe = v13d._all_mechanism_keys(x_locs)
    if set(xpartial) - universe or set(zpartial) - universe:
        raise RuntimeError("Explained mechanism outside complete physical universe")
    xmap = {m: xpartial.get(m, (0, 0)) for m in universe}
    zmap = {m: zpartial.get(m, (0, 0)) for m in universe}

    effects, joint_meta = v13d._build_joint_effects(xmap, zmap)
    first_states, _first_wit, first_meta = v13d._joint_first_order(effects)
    if not first_meta["joint_first_order_information_gate_pass"]:
        raise RuntimeError(f"Inherited v13d first-order gate failed: {first_meta}")

    loc_choices: list[list[DownstreamConcreteChoice]] = [[] for _ in range(EXPECTED_DOWNSTREAM_LOCATIONS)]
    for mkey in sorted(universe, key=str):
        sx, lx_probe = xmap[mkey]
        sz, lz_probe = zmap[mkey]
        if sx != sz:
            raise RuntimeError("Same physical mechanism has probe-dependent detector syndrome")
        # X probe flip = Z component; Z probe flip = X component.
        p = int(lz_probe) | (int(lx_probe) << 1)
        loc = int(mkey[0])
        loc_choices[loc].append(DownstreamConcreteChoice(int(sx), int(p)))

    hist = collections.Counter(len(x) for x in loc_choices)
    if dict(hist) != {1: 837, 3: 215, 15: 1355}:
        raise RuntimeError(f"Unexpected per-location concrete mechanism histogram: {dict(hist)}")
    if sum(map(len, loc_choices)) != 21807:
        raise RuntimeError(sum(map(len, loc_choices)))

    stage_by_loc = [v13d._stage_from_tick(int(x_meta[i]["tick"])) for i in range(EXPECTED_DOWNSTREAM_LOCATIONS)]
    return {
        "choices": loc_choices,
        "stage_by_loc": stage_by_loc,
        "effects": effects,
        "meta": {
            "stim_version": getattr(stim, "__version__", "unknown"),
            "detectors": EXPECTED_DETECTORS,
            "physical_locations_per_patch": EXPECTED_DOWNSTREAM_LOCATIONS,
            "concrete_pauli_mechanisms_per_patch": 21807,
            "joint_unique_effects": len(effects),
            "mechanism_count_per_location_histogram": dict(sorted(hist.items())),
            "x_probe_explained_mechanisms": len(xpartial),
            "z_probe_explained_mechanisms": len(zpartial),
            "joint_meta": joint_meta,
            "joint_first_order": first_meta,
        },
    }


def _sample_downstream(p: float, rng: np.random.Generator, model: dict[str, Any]):
    n = EXPECTED_DOWNSTREAM_LOCATIONS
    k = int(rng.binomial(n, p))
    if k == 0:
        return 0, 0, 0, {}
    syndrome = 0
    pauli = 0
    stages = collections.Counter()
    locs = np.asarray(rng.choice(n, size=k, replace=False), dtype=np.int64)
    for q in locs:
        loc = int(q)
        opts = model["choices"][loc]
        oi = 0 if len(opts) == 1 else int(rng.integers(0, len(opts)))
        c = opts[oi]
        syndrome ^= int(c.syndrome)
        pauli ^= int(c.pauli)
        stages[model["stage_by_loc"][loc]] += 1
    return int(syndrome), int(pauli), int(k), dict(stages)


def _add_query_label(states: dict[int, int], q: int, p: int) -> None:
    if q in states:
        states[q] |= 1 << int(p)


def _resolve_observed_records(
    queries: set[int],
    sectors: list[dict[str, Any]],
    effects: list[Any],
    v13d,
    reject_set: set[int],
) -> tuple[dict[int, int], dict[str, Any]]:
    """Exact <=2 catalogue labels only for observed finite-p query records."""
    states = {int(q): 0 for q in queries}
    t0 = time.time()

    # Boundary order0 and boundary w<=1 + downstream order1.
    foreground_hits = 0
    for r in sectors:
        bw = int(r["canonical_weight"])
        rs = int(r["detector_470_hex"], 16)
        ps = int(r["raw_terminal_pauli_int"])
        if rs in states:
            old = states[rs]
            _add_query_label(states, rs, ps)
            foreground_hits += states[rs] != old
        if bw <= 1:
            for e in effects:
                q = rs ^ int(e.syndrome)
                if q not in states:
                    continue
                old = states[q]
                _add_query_label(states, q, ps ^ int(e.pauli))
                foreground_hits += states[q] != old

    # Clean boundary downstream order2. Stream all location-compatible pairs,
    # but only touch the observed query set.
    pair_count = pair_query_hits = new_labels = 0
    tscan = time.time()
    n = len(effects)
    for i in range(n):
        a = effects[i]
        for j in range(i, n):
            b = effects[j]
            if v13d._pair_weight_distinct_locations(a, b, i == j) <= 1e-15:
                continue
            pair_count += 1
            q = int(a.syndrome) ^ int(b.syndrome)
            if q not in states:
                continue
            pair_query_hits += 1
            old = states[q]
            _add_query_label(states, q, int(a.pauli) ^ int(b.pauli))
            new_labels += states[q] != old

    ambiguous_not_rejected = [q for q, bits in states.items() if bits.bit_count() > 1 and q not in reject_set]
    if ambiguous_not_rejected:
        raise RuntimeError(
            "Observed <=2 ambiguity absent from inherited S17-C/B reject policy. "
            f"Examples: {[hex(x) for x in ambiguous_not_rejected[:10]]}"
        )
    zero_bits = states.get(0, 0)
    if 0 in queries and zero_bits != 1:
        raise RuntimeError(f"Observed zero record is not uniquely I in <=2 catalogue: bits={zero_bits:b}")

    resolved = {q: (bits.bit_length()-1) for q, bits in states.items() if bits.bit_count() == 1 and q not in reject_set}
    ambiguous = {q for q, bits in states.items() if bits.bit_count() > 1 or q in reject_set}
    unknown = {q for q, bits in states.items() if bits == 0 and q not in reject_set}
    return resolved, {
        "observed_query_records": len(queries),
        "resolved_unique_records": len(resolved),
        "explicit_or_catalogue_ambiguous_records": len(ambiguous),
        "out_of_certificate_records": len(unknown),
        "foreground_label_insertions": int(foreground_hits),
        "clean_order2_location_compatible_effect_pairs_scanned": int(pair_count),
        "clean_order2_pair_query_hits": int(pair_query_hits),
        "clean_order2_new_labels_on_queries": int(new_labels),
        "clean_order2_scan_seconds": time.time()-tscan,
        "total_decoder_build_seconds": time.time()-t0,
    }


def _oracle_infidelity(s10mc, lv: np.ndarray, rec: tuple[int, ...]) -> float:
    target = s10mc._target(rec)
    f = float(abs(np.vdot(target, np.asarray(lv, complex)))**2)
    return max(0.0, min(1.0, 1.0-f))


def _corrected_infidelity(s10mc, tr: StoredTrajectory, pred_a: int, pred_b: int) -> float:
    residual_a = int(pred_a) ^ int(tr.actual_pauli_a)
    residual_b = int(pred_b) ^ int(tr.actual_pauli_b)
    corr = s10mc._apply(np.asarray(tr.lv, complex), residual_a, residual_b)
    target = s10mc._target(tr.core_record)
    f = float(abs(np.vdot(target, corr))**2)
    return max(0.0, min(1.0, 1.0-f))


def _sample_one_p(
    p: float,
    shots: int,
    seed: int,
    s10mc,
    sector_by_s: dict[int, dict[str, Any]],
    downstream: dict[str, Any],
    max_examples: int,
):
    rng = np.random.default_rng(seed)
    trajectories: list[StoredTrajectory] = []
    reasons = collections.Counter()
    fault_hist = collections.Counter()
    downstream_stage_faults_a = collections.Counter()
    downstream_stage_faults_b = collections.Counter()
    front_accepted = 0
    core_accepted = 0
    sum_front_faults = 0
    sum_coremodule_faults = 0
    sum_down_a = 0
    sum_down_b = 0
    max_faults = 0
    examples = []

    for shot in range(int(shots)):
        fa = s10mc._front_block(p, rng)
        fb = s10mc._front_block(p, rng)
        nfa = int(fa["encoder_fault_count"] + fa["initial_ed_fault_count"])
        nfb = int(fb["encoder_fault_count"] + fb["initial_ed_fault_count"])
        sum_front_faults += nfa+nfb
        if not fa["accepted"] or not fb["accepted"]:
            reasons["initial_frontend_ED"] += 1
            nt = nfa+nfb
            fault_hist[nt] += 1
            max_faults = max(max_faults, nt)
            continue
        front_accepted += 1

        ok, why, rec, ss, lv, nfc, nma, nmb, ncat, nro, sc, fast = s10mc._sample_core_fullfront(p, rng, fa, fb)
        ncoremods = int(nfc+nma+nmb+ncat+nro)
        sum_coremodule_faults += ncoremods
        if not ok:
            reasons[f"core_{why}"] += 1
            nt = nfa+nfb+ncoremods
            fault_hist[nt] += 1
            max_faults = max(max_faults, nt)
            continue
        core_accepted += 1
        sa, sb = map(int, ss)

        dsa, dpa, nda, sta = _sample_downstream(p, rng, downstream)
        dsb, dpb, ndb, stb = _sample_downstream(p, rng, downstream)
        sum_down_a += nda
        sum_down_b += ndb
        downstream_stage_faults_a.update(sta)
        downstream_stage_faults_b.update(stb)

        ra = sector_by_s[sa]
        rb = sector_by_s[sb]
        rec_a = int(ra["detector_470_hex"], 16) ^ int(dsa)
        rec_b = int(rb["detector_470_hex"], 16) ^ int(dsb)
        actual_a = int(ra["raw_terminal_pauli_int"]) ^ int(dpa)
        actual_b = int(rb["raw_terminal_pauli_int"]) ^ int(dpb)
        nt = nfa+nfb+ncoremods+nda+ndb
        fault_hist[nt] += 1
        max_faults = max(max_faults, nt)

        tr = StoredTrajectory(
            record_a=int(rec_a), record_b=int(rec_b),
            actual_pauli_a=int(actual_a), actual_pauli_b=int(actual_b),
            lv=np.asarray(lv, complex).copy(), core_record=tuple(int(x) for x in rec),
            total_fault_locations=int(nt), pre_downstream_fault_locations=int(nfa+nfb+ncoremods),
            downstream_fault_locations_a=int(nda), downstream_fault_locations_b=int(ndb),
            intrinsic_oracle_infidelity=float(_oracle_infidelity(s10mc, np.asarray(lv, complex), tuple(int(x) for x in rec))),
        )
        trajectories.append(tr)
        if len(examples) < max_examples and (sa or sb or nda or ndb):
            examples.append({
                "shot": int(shot), "steane_sector_pair": [sa, sb],
                "record_A": hex(rec_a), "record_B": hex(rec_b),
                "actual_pauli_A": PAULI_NAMES[actual_a], "actual_pauli_B": PAULI_NAMES[actual_b],
                "fault_locations": {"pre_downstream": nfa+nfb+ncoremods, "A": nda, "B": ndb, "total": nt},
                "intrinsic_oracle_infidelity": tr.intrinsic_oracle_infidelity,
            })

    queries = {tr.record_a for tr in trajectories} | {tr.record_b for tr in trajectories}
    return {
        "p": float(p), "shots": int(shots), "seed": int(seed),
        "trajectories": trajectories, "queries": queries,
        "predecode": {
            "frontend_accepted": int(front_accepted),
            "frontend_acceptance": front_accepted/shots,
            "core_accepted": int(core_accepted),
            "core_acceptance_per_attempt": core_accepted/shots,
            "core_acceptance_given_frontend": core_accepted/front_accepted if front_accepted else None,
            "mean_frontend_fault_locations_per_attempt": sum_front_faults/shots,
            "mean_coremodule_fault_locations_per_attempt": sum_coremodule_faults/shots,
            "mean_downstream_fault_locations_A_given_core_accept": sum_down_a/max(core_accepted, 1),
            "mean_downstream_fault_locations_B_given_core_accept": sum_down_b/max(core_accepted, 1),
            "downstream_stage_faults_A": dict(sorted(downstream_stage_faults_a.items())),
            "downstream_stage_faults_B": dict(sorted(downstream_stage_faults_b.items())),
            "max_total_fault_locations_seen": int(max_faults),
            "total_fault_location_histogram": {str(k): int(v) for k, v in sorted(fault_hist.items())},
            "rejection_reasons_before_downstream_decoder": dict(sorted(reasons.items())),
            "unique_observed_patch_records": len(queries),
            "examples": examples,
        },
    }


def _finish_one_p(sampled: dict[str, Any], decoder: dict[int, int], reject_set: set[int], s10mc, max_examples: int):
    shots = int(sampled["shots"])
    trajectories: list[StoredTrajectory] = sampled["trajectories"]
    accepted = 0
    explicit_reject = unknown_reject = 0
    inf_sum = oracle_inf_sum = 0.0
    harmful = 0
    harmful_le2 = 0
    harmful_examples = []
    accepted_fault_hist = collections.Counter()
    unknown_examples = []
    reject_examples = []

    for tr in trajectories:
        ra, rb = int(tr.record_a), int(tr.record_b)
        exp = (ra in reject_set) or (rb in reject_set)
        if exp:
            explicit_reject += 1
            if len(reject_examples) < max_examples:
                reject_examples.append({"record_A": hex(ra), "record_B": hex(rb), "A_reject": ra in reject_set, "B_reject": rb in reject_set, "total_fault_locations": tr.total_fault_locations})
            continue
        if ra not in decoder or rb not in decoder:
            unknown_reject += 1
            if len(unknown_examples) < max_examples:
                unknown_examples.append({"record_A": hex(ra), "record_B": hex(rb), "A_known": ra in decoder, "B_known": rb in decoder, "total_fault_locations": tr.total_fault_locations})
            continue

        pa = int(decoder[ra]); pb = int(decoder[rb])
        inf = float(_corrected_infidelity(s10mc, tr, pa, pb))
        accepted += 1
        inf_sum += inf
        oracle_inf_sum += float(tr.intrinsic_oracle_infidelity)
        accepted_fault_hist[tr.total_fault_locations] += 1
        if inf > TOL:
            harmful += 1
            harmful_le2 += int(tr.total_fault_locations <= 2)
            if len(harmful_examples) < max_examples:
                harmful_examples.append({
                    "record_A": hex(ra), "record_B": hex(rb),
                    "predicted_pauli_A": PAULI_NAMES[pa], "predicted_pauli_B": PAULI_NAMES[pb],
                    "actual_pauli_A": PAULI_NAMES[tr.actual_pauli_a], "actual_pauli_B": PAULI_NAMES[tr.actual_pauli_b],
                    "infidelity": inf, "intrinsic_oracle_infidelity": tr.intrinsic_oracle_infidelity,
                    "total_fault_locations": tr.total_fault_locations,
                    "pre_downstream_fault_locations": tr.pre_downstream_fault_locations,
                    "downstream_fault_locations_A": tr.downstream_fault_locations_a,
                    "downstream_fault_locations_B": tr.downstream_fault_locations_b,
                })

    core_acc = int(sampled["predecode"]["core_accepted"])
    # Stage-level active-location proxy under this driver: both physical frontends
    # (194) always sampled; the 696 CAT/core/midED locations are sampled after
    # frontend acceptance; both 2407-location downstream patches after core accept.
    front_acc = int(sampled["predecode"]["frontend_accepted"])
    executed_location_proxy = 194.0 + (front_acc/shots)*696.0 + (core_acc/shots)*(2*EXPECTED_DOWNSTREAM_LOCATIONS)

    return {
        "p": float(sampled["p"]), "shots": shots, "seed": int(sampled["seed"]),
        "frontend": {k:v for k,v in sampled["predecode"].items() if k not in {"examples"}},
        "certified_lookup_policy": {
            "core_accepted_before_downstream_policy": core_acc,
            "explicit_ambiguous_record_rejected": int(explicit_reject),
            "out_of_certificate_record_rejected": int(unknown_reject),
            "delivered_accepted": int(accepted),
            "end_to_end_acceptance": accepted/shots,
            "end_to_end_acceptance_wilson95": _wilson(accepted, shots),
            "attempts_per_delivered_output": shots/accepted if accepted else None,
            "conditional_logical_infidelity": inf_sum/accepted if accepted else None,
            "logical_infidelity_per_attempt": inf_sum/shots,
            "harmful_accepted_samples": int(harmful),
            "harmful_fraction_among_accepted": harmful/accepted if accepted else None,
            "harmful_fraction_wilson95": _wilson(harmful, accepted) if accepted else None,
            "zero_harmful_one_sided_95_upper_if_applicable": _zero_event_upper95(accepted) if accepted and harmful == 0 else None,
            "accepted_harmful_samples_with_total_fault_locations_le2": int(harmful_le2),
            "global_le2_regression_gate_pass": harmful_le2 == 0,
            "conditional_intrinsic_oracle_infidelity": oracle_inf_sum/accepted if accepted else None,
            "decoder_excess_infidelity_conditional": (inf_sum-oracle_inf_sum)/accepted if accepted else None,
            "accepted_total_fault_location_histogram": {str(k):int(v) for k,v in sorted(accepted_fault_hist.items())},
        },
        "resource_proxy": {
            "stage_level_expected_active_locations_executed_per_attempt": executed_location_proxy,
            "stage_level_active_location_proxy_per_delivered_output": executed_location_proxy/(accepted/shots) if accepted else None,
            "warning": "This is an active-location stage proxy, not the paper-final spacetime/gate resource accounting. CAT/internal early-abort timing and the noisy downstream d13 consumer round must be handled separately.",
        },
        "examples": {
            "harmful_accepted": harmful_examples,
            "out_of_certificate_rejections": unknown_examples,
            "explicit_ambiguity_rejections": reject_examples,
        },
    }


def _selftest_static(sectors: list[dict[str, Any]], reject_set: set[int], downstream_meta: dict[str, Any]) -> dict[str, Any]:
    by = {int(r["steane_syndrome"]):r for r in sectors}
    return {
        "passed": bool(
            len(sectors) == 64 and
            int(by[0]["detector_470_hex"],16) == 0 and
            int(by[0]["raw_terminal_pauli_int"]) == 0 and
            0 not in reject_set and
            downstream_meta["physical_locations_per_patch"] == EXPECTED_DOWNSTREAM_LOCATIONS and
            downstream_meta["detectors"] == EXPECTED_DETECTORS
        ),
        "sector_count": len(sectors),
        "reject_count": len(reject_set),
        "zero_record_rejected": 0 in reject_set,
        "downstream_locations_per_patch": downstream_meta["physical_locations_per_patch"],
        "detectors": downstream_meta["detectors"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s10-script", type=Path, default=None)
    ap.add_argument("--v13d-script", type=Path, default=None)
    ap.add_argument("--sector-map", type=Path, default=None)
    ap.add_argument("--sector-aware-reject-policy", type=Path, default=None)
    ap.add_argument("--x-circuit", type=Path, default=None)
    ap.add_argument("--z-circuit", type=Path, default=None)
    ap.add_argument("--p-values", type=float, nargs="+", default=[1e-4, 2e-4, 5e-4, 1e-3])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--max-examples-per-p", type=int, default=12)
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.search_root.resolve()

    s10_path = (args.s10_script or _find(root, "direct_cs_s10_complete_physical_factory_mc.py", ("stim_pipeline/interface",))).resolve()
    v13d_path = (args.v13d_script or _find(root, "direct_cs_s17b_v13d_joint_postselection_certificate.py", ("stim_pipeline/interface",))).resolve()
    sector_path = (args.sector_map or _find(root, "s17c_a_sector_map.json", ("s17c_a_results",))).resolve()
    reject_path = (args.sector_aware_reject_policy or _find(root, "s17c_b_sector_aware_reject_policy.json", ("s17c_b_results",))).resolve()
    x_path = (args.x_circuit or _find(root, "s17b_v11_X_transition_circuit.stim", ("s17b_v11_order2_results",))).resolve()
    z_path = (args.z_circuit or _find(root, "s17b_v11_Z_transition_circuit.stim", ("s17b_v11_order2_results",))).resolve()

    print(f"S17-C/C S10 core: {s10_path}", flush=True)
    print(f"S17-C/C sector map: {sector_path}", flush=True)
    print(f"S17-C/C sector-aware reject policy: {reject_path}", flush=True)
    print(f"S17-C/C X circuit: {x_path}", flush=True)
    print(f"S17-C/C Z circuit: {z_path}", flush=True)

    s10mc = _import_module("s17cc_s10mc", s10_path)
    v13d = _import_module("s17cc_v13d", v13d_path)
    sectors, sector_by_s = _sector_rows(sector_path)
    reject_obj = _load_json(reject_path)
    reject_set = {int(x,16) for x in reject_obj["total_sector_aware_rejected_syndromes_hex"]}
    if len(reject_set) != int(reject_obj["total_reject_count"]):
        raise RuntimeError("Duplicate/malformed sector-aware reject records")
    if int(reject_obj.get("added_sector_aware_reject_count", -1)) != 0:
        print("WARNING: sector-aware policy has added rejects; S17-C/C will honor them.", flush=True)
    if 0 in reject_set:
        raise RuntimeError("Sector-aware policy rejects zero record")

    downstream = _build_downstream_model(v13d, x_path, z_path)
    st = _selftest_static(sectors, reject_set, downstream["meta"])
    # Reuse S10's executable p=0/module self-test as an additional guard.
    s10st = s10mc._selftest()
    st["s10_physical_front_core_selftest_pass"] = bool(s10st["passed"])
    st["s10_modeled_locations_before_old_escape"] = int(s10st["modeled_locations_before_escape"])
    st["new_full_path_active_locations_if_all_stages_execute"] = int(s10st["modeled_locations_before_escape"] + 2*EXPECTED_DOWNSTREAM_LOCATIONS)
    st["passed"] = bool(st["passed"] and st["s10_physical_front_core_selftest_pass"] and st["new_full_path_active_locations_if_all_stages_execute"] == 5704)
    _dump(args.out_dir/"s17c_c_self_test.json", st)
    if not st["passed"]:
        raise RuntimeError(f"S17-C/C self-test failed: {st}")
    if args.self_test_only:
        print(json.dumps(st, indent=2), flush=True)
        print("S17-C/C SELF-TEST PASS", flush=True)
        return

    if args.shots_per_p is not None:
        if len(args.shots_per_p) != len(args.p_values):
            raise SystemExit("--shots-per-p length must match --p-values")
        shot_list = [int(x) for x in args.shots_per_p]
    else:
        shot_list = [int(args.shots)] * len(args.p_values)

    sampled_rows = []
    all_queries: set[int] = set()
    t_mc = time.time()
    for i, (p, shots) in enumerate(zip(args.p_values, shot_list)):
        print(f"S17-C/C: sampling p={p:g}, shots={shots} ...", flush=True)
        sr = _sample_one_p(float(p), int(shots), int(args.seed+i), s10mc, sector_by_s, downstream, args.max_examples_per_p)
        sampled_rows.append(sr)
        all_queries.update(sr["queries"])
        print(json.dumps({
            "p": p, "shots": shots,
            "frontend_acceptance": sr["predecode"]["frontend_acceptance"],
            "core_acceptance_per_attempt": sr["predecode"]["core_acceptance_per_attempt"],
            "unique_patch_records_so_far": len(all_queries),
        }, indent=2), flush=True)

    print(f"S17-C/C: resolving {len(all_queries)} unique observed patch records against exact <=2 sector-aware catalogue...", flush=True)
    decoder, decoder_meta = _resolve_observed_records(all_queries, sectors, downstream["effects"], v13d, reject_set)

    rows = []
    global_le2_ok = True
    for sr in sampled_rows:
        row = _finish_one_p(sr, decoder, reject_set, s10mc, args.max_examples_per_p)
        rows.append(row)
        global_le2_ok &= bool(row["certified_lookup_policy"]["global_le2_regression_gate_pass"])
        print(json.dumps({
            "p": row["p"],
            "acceptance": row["certified_lookup_policy"]["end_to_end_acceptance"],
            "conditional_logical_infidelity": row["certified_lookup_policy"]["conditional_logical_infidelity"],
            "harmful_accepted": row["certified_lookup_policy"]["harmful_accepted_samples"],
            "explicit_reject": row["certified_lookup_policy"]["explicit_ambiguous_record_rejected"],
            "unknown_reject": row["certified_lookup_policy"]["out_of_certificate_record_rejected"],
        }, indent=2), flush=True)

    # Remove heavy in-memory trajectories before serialization.
    for sr in sampled_rows:
        sr.pop("trajectories", None)
        sr["queries"] = len(sr["queries"])

    summary = {
        "stage": "S17-C/C finite-p E2E with certified conservative <=2 lookup decoder",
        "inputs": {
            "s10_core_script": str(s10_path),
            "v13d_script": str(v13d_path),
            "sector_map": str(sector_path),
            "sector_aware_reject_policy": str(reject_path),
            "X_circuit": str(x_path),
            "Z_circuit": str(z_path),
        },
        "scope": {
            "physical_front_core": "S10-C full active-location physical front/core through post-midpoint Steane sector",
            "downstream_per_patch": "2407 active stochastic locations: Hirano ZXZ + two d5 QEC rounds + Sahay d5->7->13",
            "two_patch_correlations": "preserved exactly through the sampled non-Clifford 4-dimensional logical vector and joint Steane sector pair",
            "decoder": "accept only records with unique exact total-budget<=2 sector-aware Pauli label; reject inherited ambiguity and all out-of-certificate records",
            "ideal_d13_close_warning": "The current 470-bit continuous boundary uses an ideal downstream d13 syndrome close structurally. This is not yet a standalone noisy delivered-d13 factory performance claim.",
            "noise_model": "independent active-location stochastic Pauli model; no idle/leakage/coherent/calibration/correlated multi-location noise",
        },
        "self_test": st,
        "downstream_model": downstream["meta"],
        "decoder_query_audit": decoder_meta,
        "finite_p_rows": rows,
        "main_gate": {
            "global_no_sampled_accepted_harmful_event_with_total_physical_fault_locations_le2": bool(global_le2_ok),
            "certified_conservative_lookup_decoder_constructed": True,
            "ready_to_decide_if_higher_order_decoder_is_needed": True,
        },
        "timing": {"mc_plus_decoder_seconds": time.time()-t_mc},
        "next_if_acceptance_is_too_conservative": "S17-C/D: add a practical higher-order hypergraph decoder (BP/OSD preferred) only for records outside the exact <=2 lookup, while keeping the 14,197 certified reject records frozen.",
        "next_if_acceptance_is_already_competitive": "Proceed to noisy downstream d13-close convention + error-matched 3T baseline/resource comparison before paper-final claims.",
    }
    _dump(args.out_dir/"s17c_c_summary.json", summary)
    _dump(args.out_dir/"s17c_c_finite_p_rows.json", rows)
    _dump(args.out_dir/"s17c_c_decoder_query_audit.json", decoder_meta)
    _dump(args.out_dir/"s17c_c_predecode_sampling_summary.json", sampled_rows)

    print(json.dumps({
        "global_le2_regression_gate_pass": global_le2_ok,
        "observed_query_records": decoder_meta["observed_query_records"],
        "resolved_unique_records": decoder_meta["resolved_unique_records"],
        "out_of_certificate_records": decoder_meta["out_of_certificate_records"],
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
