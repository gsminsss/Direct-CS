#!/usr/bin/env python3
"""S11-A1: exact-three-fault stratified pilot for the complete S10 direct-|CS> factory.

Goal
----
Estimate the leading accepted logical-infidelity coefficient

    P_L(p) = A3 p^3 + O(p^4)

for the *complete S10 active-location stochastic-Pauli model* by conditioning on
exactly three failed physical locations.  The 2946 modeled locations are
partitioned into eight coarse circuit strata.  Every unordered allocation of
three failed locations among those strata is sampled separately, with exact
combinatorial reweighting.

This is a PILOT / stratum-discovery stage.  It is unbiased for the chosen
operational decoder but low-statistics strata with zero observed failures are
not a proof that their A3 contribution is zero.  S11-A2 should allocate larger
statistics using this pilot's per-stratum results.

Decoder conventions reported simultaneously
-------------------------------------------
1. certified_lookup_lower:
   Existing <=2 exact lookup.  Only uniquely resolved accepted shots contribute
   their measured infidelity; unresolved accepted shots contribute zero.
2. certified_lookup_upper:
   Same, but every unresolved accepted shot is pessimistically assigned
   infidelity one.
3. identity_fallback (primary operational estimator):
   Existing exact lookup where unique; any accepted unknown/ambiguous syndrome
   uses logical identity as the fallback frame.  This is a complete,
   implementable decoder and therefore defines a finite A3 for every shot.
4. intrinsic_pauli_orbit:
   Best fidelity over the 16 two-logical-qubit Pauli images of the post-core
   logical state.  This diagnoses non-Pauli state distortion independently of
   a particular Pauli-frame decoder.

Because P_acc(0)=1, the p^3 coefficient of conditional logical infidelity is
identical to the p^3 coefficient of the accepted logical-infidelity numerator.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / 'src_current'


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError((name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s10mc = _load('s11_s10mc', HERE / 'direct_cs_s10_complete_physical_factory_mc.py')
f = s10mc.s10
m = s10mc.m
ed = s10mc.s9ed
live = s10mc.s8live

ACTUAL_BITS = 171
ACTUAL_MASK = (1 << ACTUAL_BITS) - 1
TOL = 1e-10

# Coarse stratification.  The ordering is physical time order except that the
# two escapes are kept separate because their logical roles are distinct.
GROUPS = (
    'FRONT',       # two encoded preparations + initial ED: 2 * 97
    'CAT_PRE',     # CAT checks 0,1,2: 3 * 35
    'CORE_PRE',    # core check locations in checks 0,1,2
    'MIDED',       # two midpoint Steane ED blocks: 2 * 82
    'CAT_POST',    # CAT checks 3,4,5: 3 * 35
    'CORE_POST',   # core check locations in checks 3,4,5
    'ESC_A',       # ZXZ A
    'ESC_B',       # ZXZ B
)


@dataclass(frozen=True)
class Loc:
    group: str
    kind: str
    a: int
    b: int = -1


def _build_groups(escape_n: int = 1028) -> dict[str, list[Loc]]:
    g: dict[str, list[Loc]] = {x: [] for x in GROUPS}
    # FRONT: block, local 0..96.  0..14 encoder/prep, 15..96 initial ED.
    for block in range(2):
        for li in range(97):
            g['FRONT'].append(Loc('FRONT', 'front', block, li))
    # Six CAT gadgets. local 0..27 prep/verification, 28..34 physical X readout.
    for ci in range(6):
        gn = 'CAT_PRE' if ci < 3 else 'CAT_POST'
        for li in range(35):
            g[gn].append(Loc(gn, 'cat', ci, li))
    # Core gate locations already carry check index.
    for lid, (ci, _ck, _j, _gi) in enumerate(m.locations):
        gn = 'CORE_PRE' if int(ci) < 3 else 'CORE_POST'
        g[gn].append(Loc(gn, 'core', int(lid), -1))
    # Midpoint ED: block, local 0..81.
    for block in range(2):
        for li in range(len(ed.ED_OPS)):
            g['MIDED'].append(Loc('MIDED', 'mided', block, li))
    for li in range(escape_n):
        g['ESC_A'].append(Loc('ESC_A', 'escapeA', li, -1))
        g['ESC_B'].append(Loc('ESC_B', 'escapeB', li, -1))
    counts = {k: len(v) for k, v in g.items()}
    expected = {
        'FRONT': 194, 'CAT_PRE': 105, 'CORE_PRE': 161, 'MIDED': 164,
        'CAT_POST': 105, 'CORE_POST': 161, 'ESC_A': escape_n, 'ESC_B': escape_n,
    }
    if counts != expected:
        raise AssertionError((counts, expected))
    return g


def _all_strata(group_locs: dict[str, list[Loc]]) -> list[dict[str, Any]]:
    out = []
    for triple in itertools.combinations_with_replacement(range(len(GROUPS)), 3):
        cnt = [0] * len(GROUPS)
        for i in triple:
            cnt[i] += 1
        ways = 1
        for i, k in enumerate(cnt):
            ways *= math.comb(len(group_locs[GROUPS[i]]), k)
        key = '+'.join(f'{GROUPS[i]}{k}' for i, k in enumerate(cnt) if k)
        out.append({'key': key, 'counts': cnt, 'location_triples': int(ways)})
    total_n = sum(len(group_locs[x]) for x in GROUPS)
    if len(out) != math.comb(len(GROUPS) + 3 - 1, 3):
        raise AssertionError(len(out))
    if sum(x['location_triples'] for x in out) != math.comb(total_n, 3):
        raise AssertionError('strata do not partition C(N,3)')
    return out


def _sample_locations(stratum: dict[str, Any], group_locs: dict[str, list[Loc]], rng: np.random.Generator) -> list[Loc]:
    rows: list[Loc] = []
    for i, k in enumerate(stratum['counts']):
        if not k:
            continue
        arr = group_locs[GROUPS[i]]
        ids = np.asarray(rng.choice(len(arr), size=int(k), replace=False), dtype=np.int64)
        rows.extend(arr[int(j)] for j in ids)
    if len(rows) != 3 or len(set(rows)) != 3:
        raise AssertionError(rows)
    return rows


def _random_opt_count(loc: Loc, bridge: dict[str, Any]) -> int:
    if loc.kind == 'front':
        li = loc.b
        if li < f.ENCODER_LOCATION_COUNT:
            return len(f.ENCODER_LOCATIONS[li].options)
        return len(ed.ED_OPTIONS[li - f.ENCODER_LOCATION_COUNT])
    if loc.kind == 'cat':
        li = loc.b
        return len(f.CAT7_LOCATIONS[li].options) if li < f.CAT7_LOCATION_COUNT else 1
    if loc.kind == 'core':
        ci, ck, _j, gi = m.locations[loc.a]
        return len(m.opt_cache[(ck, gi)])
    if loc.kind == 'mided':
        return len(ed.ED_OPTIONS[loc.b])
    if loc.kind in ('escapeA', 'escapeB'):
        return len(bridge['sigs'][loc.a])
    raise AssertionError(loc)


def _instantiate_faults(locs: list[Loc], bridge: dict[str, Any], rng: np.random.Generator):
    # Physical-fault containers.
    enc = [dict(), dict()]
    ied = [dict(), dict()]
    cats = [dict() for _ in range(6)]
    cat_ro = [0] * 6
    core = [collections.defaultdict(list) for _ in m.sequence]
    mid = [dict(), dict()]
    esc_full = [0, 0]
    esc_frame = [0, 0]
    trace = []

    for loc in locs:
        nopt = _random_opt_count(loc, bridge)
        oi = 0 if nopt == 1 else int(rng.integers(0, nopt))
        row = {'group': loc.group, 'kind': loc.kind, 'a': int(loc.a), 'b': int(loc.b), 'option_index': oi, 'option_count': nopt}
        if loc.kind == 'front':
            block, li = loc.a, loc.b
            if li < f.ENCODER_LOCATION_COUNT:
                enc[block][li] = oi
                row['fault_class'] = f.ENCODER_LOCATIONS[li].options[oi][2]
            else:
                ei = li - f.ENCODER_LOCATION_COUNT
                ied[block][ei] = oi
                row['fault_class'] = ed.ED_OPTIONS[ei][oi][0]
        elif loc.kind == 'cat':
            ci, li = loc.a, loc.b
            if li < f.CAT7_LOCATION_COUNT:
                cats[ci][li] = oi
                row['fault_class'] = f.CAT7_LOCATIONS[li].options[oi][3]
            else:
                # A physical X-basis readout fault flips that bit; only parity matters.
                cat_ro[ci] ^= 1
                row['fault_class'] = 'READOUT_FLIP'
        elif loc.kind == 'core':
            lid = loc.a
            ci, ck, j, gi = m.locations[lid]
            core[int(ci)][int(j)].append((int(gi), oi))
            row.update({'check_index': int(ci), 'check': ck, 'coordinate': int(j), 'gate_index': int(gi), 'fault_class': m.opt_cache[(ck, gi)][oi]})
        elif loc.kind == 'mided':
            block, li = loc.a, loc.b
            mid[block][li] = oi
            row['fault_class'] = ed.ED_OPTIONS[li][oi][0]
        elif loc.kind in ('escapeA', 'escapeB'):
            bi = 0 if loc.kind == 'escapeA' else 1
            esc_full[bi] ^= int(bridge['sigs'][loc.a][oi])
            esc_frame[bi] ^= int(bridge['frames'][loc.a][oi])
            row.update({'fault_signature_full_hex': hex(int(bridge['sigs'][loc.a][oi])), 'fault_frame': int(bridge['frames'][loc.a][oi])})
        else:
            raise AssertionError(loc)
        trace.append(row)
    return enc, ied, cats, cat_ro, core, mid, esc_full, esc_frame, trace


def _front_block_forced(enc_faults, ed_faults, rng):
    x, z = f.encoder_error(enc_faults)
    ok, s, lv, pr = f.apply_initial_ed_compact(x, z, ed_faults)
    if not ok or pr <= 1e-20:
        return {'accepted': False}
    if pr < 1 - 1e-12 and float(rng.random()) >= pr:
        return {'accepted': False}
    return {'accepted': True, 'syndrome': int(s), 'logical_vector': np.asarray(lv, complex), 'accept_probability': float(pr)}


class _RA:
    def __init__(self, rng): self.rng = rng
    def random(self): return float(self.rng.random())


def _sample_sector(C, rng):
    probs = np.sum(np.abs(C) ** 2, axis=(1, 3))
    flat = probs.reshape(-1)
    total = float(flat.sum())
    if abs(total - 1) > 1e-9:
        raise AssertionError(total)
    idx = int(rng.choice(flat.size, p=flat / total))
    sa, sb = divmod(idx, 64)
    pr = float(probs[sa, sb])
    v = C[sa, :, sb, :].reshape(4) / math.sqrt(pr)
    return int(sa), int(sb), v, int(np.count_nonzero(probs > 1e-12))


def _target(rec):
    return s10mc._target(rec)


def _phase_inf(v, t):
    nv = float(np.linalg.norm(v)); nt = float(np.linalg.norm(t))
    if nv <= 1e-20 or nt <= 1e-20:
        return 1.0
    fid = float(abs(np.vdot(t / nt, v / nv)) ** 2)
    return 1 - min(1.0, max(0.0, fid))


def _pauli_orbit_inf(v, t):
    best = 0.0
    for a in range(4):
        for b in range(4):
            w = s10mc._apply(v, a, b)
            best = max(best, float(abs(np.vdot(t, w)) ** 2))
    return 1 - min(1.0, max(0.0, best))


def _simulate_exact3(locs: list[Loc], bridge: dict[str, Any], rng: np.random.Generator, want_trace=False, want_state=False):
    enc, ied, catfaults, cat_ro, core, mid, efull, eframe, trace = _instantiate_faults(locs, bridge, rng)
    fa = _front_block_forced(enc[0], ied[0], rng)
    fb = _front_block_forced(enc[1], ied[1], rng)
    base = {'accepted': False, 'reason': 'initialED', 'legacy_resolved': False, 'identity_infidelity': 0.0, 'legacy_lower_infidelity': 0.0, 'legacy_upper_infidelity': 0.0, 'intrinsic_pauli_orbit_infidelity': 0.0, 'identity_binary_failure': 0}
    if not fa['accepted'] or not fb['accepted']:
        return base | ({'trace': trace} if want_trace else {})

    ce = [f.cat_effect(catfaults[i]) for i in range(6)]
    for i, x in enumerate(ce):
        if not x.accepted:
            return (base | {'reason': f'CAT{i}'}) | ({'trace': trace} if want_trace else {})

    # Analytic quantum fast path when all actual quantum-state-changing modules are ideal.
    fast = (
        s10mc._is_plus_block(fa) and s10mc._is_plus_block(fb)
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
        ideal = s10mc._valid_rec(a, b)
        rec = [int(ideal[i]) ^ int(ce[i].phase_flip) ^ int(cat_ro[i]) for i in range(6)]
        if not m.valid(rec):
            return (base | {'reason': 'record'}) | ({'trace': trace} if want_trace else {})
        sa = sb = 0
        lv = s10mc._branch_vec(a, b)
        sector_count = 1
    else:
        va = f.block_state_from_sector(int(fa['syndrome']), np.asarray(fa['logical_vector'], complex))
        vb = f.block_state_from_sector(int(fb['syndrome']), np.asarray(fb['logical_vector'], complex))
        st = f.pair_block_states(va, vb)
        rr = _RA(rng); rec = []
        for ci, ck in enumerate(m.sequence):
            st, meas = f.check_with_physical_cat(st, ck, core[ci], ce[ci], cat_ro[ci], rr)
            rec.append(int(meas))
            if ci == 2:
                st, pr, _ = ed.apply_two_block_midpoint_compact(st, mid[0], mid[1])
                if pr <= 1e-20 or float(rng.random()) >= pr:
                    return (base | {'reason': 'midED'}) | ({'trace': trace} if want_trace else {})
        if not m.valid(rec):
            return (base | {'reason': 'record'}) | ({'trace': trace} if want_trace else {})
        C, psum, recon, _ = live.exact_sector_tensor(st)
        if abs(psum - 1) > 1e-9 or recon > 1e-9:
            raise AssertionError((psum, recon))
        sa, sb, lv, sector_count = _sample_sector(C, rng)

    sra = bridge['sector'][int(sa)]
    srb = bridge['sector'][int(sb)]
    fulla = int(sra.full) ^ int(efull[0])
    fullb = int(srb.full) ^ int(efull[1])
    nowa = fulla & ACTUAL_MASK
    nowb = fullb & ACTUAL_MASK
    if nowa in bridge['unsafe'] or nowb in bridge['unsafe']:
        return (base | {'reason': 'handoff_policy'}) | ({'trace': trace} if want_trace else {})

    target = _target(rec)
    aa = int(sra.frame) ^ int(eframe[0])
    ab = int(srb.frame) ^ int(eframe[1])
    pa = s10mc._unique(bridge['table'].get(fulla))
    pb = s10mc._unique(bridge['table'].get(fullb))
    legacy_resolved = pa is not None and pb is not None

    if legacy_resolved:
        corr_legacy = s10mc._apply(lv, int(pa) ^ aa, int(pb) ^ ab)
        legacy_inf = _phase_inf(corr_legacy, target)
        lower = legacy_inf
        upper = legacy_inf
    else:
        lower = 0.0
        upper = 1.0

    # Complete operational decoder: identity fallback for each non-unique block.
    pfa = int(pa) if pa is not None else 0
    pfb = int(pb) if pb is not None else 0
    corr_id = s10mc._apply(lv, pfa ^ aa, pfb ^ ab)
    id_inf = _phase_inf(corr_id, target)
    intrinsic = _pauli_orbit_inf(lv, target)

    out = {
        'accepted': True,
        'reason': 'acc',
        'legacy_resolved': bool(legacy_resolved),
        'identity_infidelity': float(id_inf),
        'legacy_lower_infidelity': float(lower),
        'legacy_upper_infidelity': float(upper),
        'intrinsic_pauli_orbit_infidelity': float(intrinsic),
        'identity_binary_failure': int(id_inf > TOL),
        'record': [int(x) for x in rec],
        'sector_pair': [int(sa), int(sb)],
        'sector_count': int(sector_count),
        'full_syndrome_A_hex': hex(fulla),
        'full_syndrome_B_hex': hex(fullb),
        'predicted_frame_A': None if pa is None else int(pa),
        'predicted_frame_B': None if pb is None else int(pb),
        'actual_frame_A': int(aa),
        'actual_frame_B': int(ab),
    }
    if want_trace:
        out['trace'] = trace
    if want_state:
        # Internal-only exact logical vector for later decoder studies. The
        # default remains False so S11-A1 JSON outputs are unchanged.
        out['_logical_vector'] = np.asarray(lv, complex).copy()
    return out


@dataclass
class Stat:
    n: int = 0
    accepted: int = 0
    legacy_resolved: int = 0
    unresolved_accepted: int = 0
    id_fail: int = 0
    sum_id: float = 0.0
    sum_id2: float = 0.0
    sum_low: float = 0.0
    sum_up: float = 0.0
    sum_intr: float = 0.0
    sum_intr2: float = 0.0
    reasons: dict[str, int] | None = None

    def update(self, r):
        self.n += 1
        if self.reasons is None: self.reasons = {}
        self.reasons[r['reason']] = self.reasons.get(r['reason'], 0) + 1
        if not r['accepted']:
            return
        self.accepted += 1
        if r['legacy_resolved']: self.legacy_resolved += 1
        else: self.unresolved_accepted += 1
        self.id_fail += int(r['identity_binary_failure'])
        x = float(r['identity_infidelity']); self.sum_id += x; self.sum_id2 += x*x
        self.sum_low += float(r['legacy_lower_infidelity']); self.sum_up += float(r['legacy_upper_infidelity'])
        y = float(r['intrinsic_pauli_orbit_infidelity']); self.sum_intr += y; self.sum_intr2 += y*y


def _mean_se(sumx, sumx2, n):
    if n <= 0: return 0.0, None
    mean = sumx / n
    if n <= 1: return mean, None
    var = max(0.0, (sumx2 - n * mean * mean) / (n - 1))
    return mean, math.sqrt(var / n)


def _stratum_row(meta, st: Stat):
    ways = int(meta['location_triples'])
    mi, sei = _mean_se(st.sum_id, st.sum_id2, st.n)
    mint, seint = _mean_se(st.sum_intr, st.sum_intr2, st.n)
    return {
        'key': meta['key'],
        'counts': {GROUPS[i]: int(k) for i, k in enumerate(meta['counts']) if k},
        'location_triples': ways,
        'samples': st.n,
        'accepted': st.accepted,
        'acceptance_given_exact3': st.accepted / st.n if st.n else None,
        'legacy_resolved_accepted': st.legacy_resolved,
        'legacy_unresolved_accepted': st.unresolved_accepted,
        'identity_binary_failures': st.id_fail,
        'identity_mean_infidelity_given_random_exact3_location_triple': mi,
        'identity_mean_infidelity_se': sei,
        'A3_identity_contribution': ways * mi,
        'A3_identity_contribution_se': None if sei is None else ways * sei,
        'A3_legacy_lower_contribution': ways * (st.sum_low / st.n if st.n else 0.0),
        'A3_legacy_upper_contribution': ways * (st.sum_up / st.n if st.n else 0.0),
        'intrinsic_mean_infidelity': mint,
        'intrinsic_mean_infidelity_se': seint,
        'A3_intrinsic_contribution': ways * mint,
        'A3_intrinsic_contribution_se': None if seint is None else ways * seint,
        'rejection_reasons': dict(sorted((st.reasons or {}).items())),
    }


def _aggregate(rows, n_total):
    a3 = sum(float(r['A3_identity_contribution']) for r in rows)
    var = sum((float(r['A3_identity_contribution_se']) ** 2) for r in rows if r['A3_identity_contribution_se'] is not None)
    sei = math.sqrt(var)
    low = sum(float(r['A3_legacy_lower_contribution']) for r in rows)
    up = sum(float(r['A3_legacy_upper_contribution']) for r in rows)
    intr = sum(float(r['A3_intrinsic_contribution']) for r in rows)
    vari = sum((float(r['A3_intrinsic_contribution_se']) ** 2) for r in rows if r['A3_intrinsic_contribution_se'] is not None)
    sei_intr = math.sqrt(vari)
    total_comb = math.comb(n_total, 3)
    binary_num = sum(int(r['location_triples']) * (int(r['identity_binary_failures']) / int(r['samples'])) for r in rows if r['samples'])
    return {
        'A3_identity_fallback_infidelity_estimate': a3,
        'A3_identity_fallback_standard_error_naive': sei,
        'A3_identity_fallback_approx_95pct_interval': [max(0.0, a3 - 1.96 * sei), a3 + 1.96 * sei],
        'A3_identity_binary_failure_estimate': binary_num,
        'A3_certified_lookup_resolved_lower_estimate': low,
        'A3_certified_lookup_unresolved_as_failure_upper_estimate': up,
        'A3_intrinsic_pauli_orbit_infidelity_estimate': intr,
        'A3_intrinsic_pauli_orbit_standard_error_naive': sei_intr,
        'total_location_triples': total_comb,
        'conditional_exact3_identity_mean_infidelity': a3 / total_comb,
        'note_on_zero_event_strata': 'Naive normal SE is not a rigorous rare-event upper bound when a stratum observes zero failures; S11-A1 is a pilot used to allocate S11-A2 statistics.',
    }


def _self_test():
    gl = _build_groups(1028)
    strata = _all_strata(gl)
    n = sum(len(x) for x in gl.values())
    return {
        'passed': True,
        'modeled_location_count': n,
        'expected_modeled_location_count': 2946,
        'group_counts': {k: len(v) for k, v in gl.items()},
        'stratum_count': len(strata),
        'expected_stratum_count': 120,
        'sum_stratum_location_triples': sum(x['location_triples'] for x in strata),
        'C_2946_3': math.comb(2946, 3),
        'partition_exact': sum(x['location_triples'] for x in strata) == math.comb(2946, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hirano-src', type=Path)
    ap.add_argument('--signature-p', type=float, default=1e-3)
    ap.add_argument('--shots-per-stratum', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260820)
    ap.add_argument('--max-witnesses', type=int, default=100)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--out-dir', type=Path, default=HERE/'s11a1_results')
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    st = _self_test(); (a.out_dir/'s11a1_self_test.json').write_text(json.dumps(st, indent=2), encoding='utf-8')
    if not st['passed'] or not st['partition_exact']:
        raise AssertionError(st)
    if a.self_test:
        print(json.dumps(st, indent=2)); print('S11-A1 SELF-TEST PASS'); return
    if a.hirano_src is None:
        raise SystemExit('--hirano-src required unless --self-test')
    if a.shots_per_stratum <= 0:
        raise SystemExit('--shots-per-stratum must be >0')

    t0 = time.time()
    bridge = s10mc._bridge(a.hirano_src.resolve(), float(a.signature_p))
    esc_n = int(bridge['meta']['physical_locations'])
    groups = _build_groups(esc_n)
    strata = _all_strata(groups)
    rng = np.random.default_rng(a.seed)
    rows = []; witnesses = []

    print(f'S11-A1 exact-three pilot: {len(strata)} strata x {a.shots_per_stratum} shots = {len(strata)*a.shots_per_stratum} conditional triple shots', flush=True)
    for si, meta in enumerate(strata):
        stat = Stat()
        for _ in range(a.shots_per_stratum):
            locs = _sample_locations(meta, groups, rng)
            r = _simulate_exact3(locs, bridge, rng, want_trace=False)
            stat.update(r)
            if r['accepted'] and r['identity_infidelity'] > TOL and len(witnesses) < a.max_witnesses:
                # Deterministically re-run the sampled physical fault choices would require storing RNG/class choices;
                # instead capture a fresh fully traced malignant event only when encountered below.
                # We therefore rerun no branch here; summary counts remain unbiased.
                witnesses.append({'stratum': meta['key'], 'sample_index_within_stratum': stat.n-1, 'identity_infidelity': r['identity_infidelity'], 'record': r.get('record'), 'sector_pair': r.get('sector_pair'), 'full_syndrome_A_hex': r.get('full_syndrome_A_hex'), 'full_syndrome_B_hex': r.get('full_syndrome_B_hex')})
        row = _stratum_row(meta, stat); rows.append(row)
        if (si + 1) % 10 == 0 or stat.id_fail or stat.unresolved_accepted:
            print(f"S11-A1 stratum {si+1:3d}/{len(strata)} {meta['key']}: acc={stat.accepted}/{stat.n}, id_fail={stat.id_fail}, unresolved={stat.unresolved_accepted}, A3_contrib={row['A3_identity_contribution']:.6g}", flush=True)

    agg = _aggregate(rows, sum(len(x) for x in groups.values()))
    summary = {
        'stage': 'S11-A1',
        'scope': 'complete S10 active-location factory, conditioned on exactly three distinct failed physical locations',
        'status': 'PILOT_NOT_PAPER_FINAL',
        'decoder_primary': 'certified <=2 lookup with identity fallback for unknown/ambiguous accepted syndromes',
        'location_inventory': {k: len(v) for k, v in groups.items()} | {'total': sum(len(v) for v in groups.values())},
        'stratum_count': len(strata),
        'shots_per_stratum': int(a.shots_per_stratum),
        'total_conditional_triple_shots': int(len(strata) * a.shots_per_stratum),
        'seed': int(a.seed),
        'signature_p': float(a.signature_p),
        'aggregate': agg,
        'suggested_p3_predictions': {str(p): agg['A3_identity_fallback_infidelity_estimate'] * float(p) ** 3 for p in (2e-4, 5e-4, 1e-3)},
        'important_interpretation': [
            'The estimator conditions on exactly three failed physical locations and reweights each stratum by its exact number of location triples.',
            'Fault class at each failed location is sampled uniformly from that location\'s nonidentity classes, matching the S10 stochastic-Pauli convention.',
            'This is a pilot. Zero observed malignant shots in a stratum do not prove zero A3 contribution.',
            'S11-A2 should use the per-stratum pilot to allocate larger statistics, and explicit malignant witnesses should be replayed/minimized separately.',
        ],
        'elapsed_seconds': time.time() - t0,
    }
    (a.out_dir/'s11a1_strata.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (a.out_dir/'s11a1_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (a.out_dir/'s11a1_malignant_examples.json').write_text(json.dumps(witnesses, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('S11-A1 PILOT COMPLETE: upload s11a1_summary.json and s11a1_strata.json for adaptive S11-A2 allocation.', flush=True)


if __name__ == '__main__':
    main()
