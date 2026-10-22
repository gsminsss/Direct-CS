#!/usr/bin/env python3
"""S11-A5 v2: certified live-171 PREFIX whitelist + residual exact-three A3 audit.

Critical semantics
------------------
The 171 detector bits available at live handoff are an ACCEPT/REJECT record,
not a complete logical-frame decoder.  Different 195-bit full recovery
syndromes sharing one safe 171-bit prefix can legitimately require different
Pauli frames; the future noiseless 24-detector audit tail resolves those
frames.  Therefore A5 must never require ``171 prefix -> unique frame``.

The exact certified policy is instead:

    accept a live 171-bit prefix iff it occurs among exact accepted <=2
    S8-C boundary states after the existing unsafe-prefix policy;
    reject every other 171-bit prefix.

This prefix-only whitelist preserves every exact accepted <=2 boundary state
by construction and introduces no new <=2 rejection.  Frame recovery remains
the existing 195-bit <=2-priority / conservative order-3 decoder.

A5 v2 replays the S11-A3 active exact-three strata.  Exact-three trajectories
whose 171-bit prefix is absent from the certified <=2-safe prefix set are
rejected.  Retained trajectories are scored with the existing full-195 decoder,
so any residual unresolved interval is a genuine safe-prefix collision /
full-syndrome ambiguity problem, not an invalid 171-bit frame assignment.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TOL = 1e-10
NUMERICAL_A3_TOL = 1e-6
FRAME_LABELS = ('I', 'X', 'Z', 'Y')


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError((name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a1 = _load('s11a5_a1', HERE / 'direct_cs_s11a1_exact3_stratified_pilot.py')
a3 = _load('s11a5_a3', HERE / 'direct_cs_s11a3_priority_decoder_a3.py')


def _json_safe(x):
    if isinstance(x, np.ndarray):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return _json_safe(x.item())
    if isinstance(x, complex):
        return {'real': float(x.real), 'imag': float(x.imag)}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in x]
    return x


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2), encoding='utf-8')


def _singleton(flags: int) -> int | None:
    f = int(flags)
    return f.bit_length() - 1 if f and f.bit_count() == 1 else None


def _mean_se(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=float)
    m = float(a.mean())
    if len(a) <= 1:
        return m, 0.0
    return m, float(a.std(ddof=1) / math.sqrt(len(a)))


def _save_whitelist_npz(path: Path, whitelist: set[int]) -> None:
    """Compact storage for the certified safe 171-bit ACCEPT prefixes."""
    items = sorted(int(k) for k in whitelist)
    n = len(items)
    lo = np.empty(n, dtype=np.uint64)
    mid = np.empty(n, dtype=np.uint64)
    hi = np.empty(n, dtype=np.uint64)
    mask64 = (1 << 64) - 1
    for i, x in enumerate(items):
        lo[i] = x & mask64
        mid[i] = (x >> 64) & mask64
        hi[i] = (x >> 128) & mask64
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        prefix_word0=lo, prefix_word1=mid, prefix_word2=hi,
        actual_bits=np.asarray([a1.ACTUAL_BITS], dtype=np.uint16),
    )


def _build_live171_whitelist(bridge: dict[str, Any]) -> tuple[set[int], dict[str, Any]]:
    """Build exact accepted-<=2 SAFE PREFIX set.

    Important: a safe 171-bit prefix is *not* required to have one frame.
    Multiple full-195 tails / frames behind one present-time prefix are normal;
    only full-syndrome ambiguity projects the prefix into bridge['unsafe'].
    """
    mask = (1 << a1.ACTUAL_BITS) - 1
    unsafe = set(int(x) for x in bridge['unsafe'])
    safe_prefixes: set[int] = set()
    frames_by_safe_prefix: dict[int, set[int]] = collections.defaultdict(set)
    reachable_full = safe_full = unsafe_full = ambiguous_full = 0

    for full, flags in bridge['table'].items():
        reachable_full += 1
        pref = int(full) & mask
        if pref in unsafe:
            unsafe_full += 1
            continue
        p = _singleton(int(flags))
        if p is None:
            ambiguous_full += 1
            raise AssertionError(('non-unsafe <=2 full syndrome is not singleton', hex(int(full)), int(flags), hex(pref)))
        safe_full += 1
        safe_prefixes.add(pref)
        frames_by_safe_prefix[pref].add(int(p))

    if 0 not in safe_prefixes:
        raise AssertionError('fault-free 171-bit prefix missing from safe-prefix whitelist')
    if 0 in unsafe:
        raise AssertionError('fault-free prefix unexpectedly unsafe')
    if safe_prefixes & unsafe:
        raise AssertionError(('safe/unsafe prefix overlap', len(safe_prefixes & unsafe)))

    # Exact no-new-<=2-rejection replay at table level.
    preserved = 0
    missed = 0
    for full, flags in bridge['table'].items():
        pref = int(full) & mask
        if pref in unsafe:
            continue
        if _singleton(int(flags)) is None:
            raise AssertionError(('safe table ambiguity', full, flags))
        preserved += 1
        if pref not in safe_prefixes:
            missed += 1
    if missed:
        raise AssertionError(('safe-prefix whitelist lost <=2 accepted full states', missed))

    frame_mult_hist = collections.Counter(len(v) for v in frames_by_safe_prefix.values())
    multi_frame_prefixes = sum(1 for v in frames_by_safe_prefix.values() if len(v) > 1)
    cert = {
        'actual_bits': int(a1.ACTUAL_BITS),
        'exact_boundary_full_states': int(reachable_full),
        'existing_unsafe_prefixes': int(len(unsafe)),
        'certified_safe_prefixes_whitelisted': int(len(safe_prefixes)),
        'certified_safe_full_states_preserved': int(preserved),
        'existing_unsafe_full_states_rejected': int(unsafe_full),
        'nonunsafe_ambiguous_full_states': int(ambiguous_full),
        'safe_prefixes_with_multiple_full195_frames': int(multi_frame_prefixes),
        'safe_prefix_frame_multiplicity_histogram': {str(k): int(v) for k,v in sorted(frame_mult_hist.items())},
        'safe_unsafe_prefix_overlap': 0,
        'safe_full_states_missed_by_whitelist': int(missed),
        'faultfree_prefix_hex': hex(0),
        'exact_no_new_le2_acceptance_penalty': True,
        'frame_decoding_uses_full195_not_live171': True,
    }
    return safe_prefixes, cert


def _active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        lo = float(r['A3_priority_resolved_lower_estimate'])
        un = int(r['priority_unresolved_accepted'])
        if lo > NUMERICAL_A3_TOL or un > 0:
            out.append(r)
    return out


def _audit_stratum_whitelist(meta, row_a3, seed: int, groups, bridge, whitelist: set[int],
                              clean_table, single_effects, sectors, max_examples: int):
    rng = np.random.default_rng(int(seed))
    ways = int(meta['location_triples'])
    shots = int(row_a3['shots'])
    mask = (1 << a1.ACTUAL_BITS) - 1

    lower_vals: list[float] = []
    upper_vals: list[float] = []
    oracle_vals: list[float] = []
    accepted_pre = accepted_post = rejected_new = 0
    resolved = unresolved = harmful = 0
    frame_alias_harmful = hidden_state_harmful = 0
    reasons = collections.Counter()
    examples = []

    boundary = bridge['table']
    for i in range(shots):
        locs = a1._sample_locations(meta, groups, rng)
        r = a1._simulate_exact3(locs, bridge, rng, want_trace=False, want_state=True)
        reasons[r['reason']] += 1
        if not r['accepted']:
            lower_vals.append(0.0); upper_vals.append(0.0); oracle_vals.append(0.0)
            continue

        accepted_pre += 1
        fa = int(r['full_syndrome_A_hex'], 16); fb = int(r['full_syndrome_B_hex'], 16)
        pa = fa & mask; pb = fb & mask
        if pa not in whitelist or pb not in whitelist:
            rejected_new += 1
            lower_vals.append(0.0); upper_vals.append(0.0); oracle_vals.append(0.0)
            if len(examples) < max_examples:
                examples.append({
                    'kind':'new_live171_prefix_whitelist_rejection', 'sample':i, 'stratum':meta['key'],
                    'prefix_A':hex(pa), 'prefix_B':hex(pb),
                    'A_whitelisted':pa in whitelist, 'B_whitelisted':pb in whitelist,
                    'full_A':hex(fa), 'full_B':hex(fb),
                })
            continue

        accepted_post += 1
        da = a3._decode_one(fa, boundary, clean_table, single_effects, sectors)
        db = a3._decode_one(fb, boundary, clean_table, single_effects, sectors)
        lv = np.asarray(r['_logical_vector'], dtype=complex)
        target = a1._target(r['record'])
        oracle = float(a1._pauli_orbit_inf(lv, target))
        oracle_vals.append(oracle)

        if da.predicted is None or db.predicted is None:
            unresolved += 1
            lower_vals.append(0.0); upper_vals.append(1.0)
            if len(examples) < max_examples:
                examples.append({
                    'kind':'retained_safe_prefix_full195_unresolved', 'sample':i, 'stratum':meta['key'],
                    'prefix_A':hex(pa), 'prefix_B':hex(pb), 'full_A':hex(fa), 'full_B':hex(fb),
                    'source_A':da.source, 'source_B':db.source,
                    'candidates_A':[FRAME_LABELS[x] for x in a3._frame_list(da.all_flags)],
                    'candidates_B':[FRAME_LABELS[x] for x in a3._frame_list(db.all_flags)],
                })
            continue

        resolved += 1
        aa = int(r['actual_frame_A']); ab = int(r['actual_frame_B'])
        corr = a1.s10mc._apply(lv, int(da.predicted) ^ aa, int(db.predicted) ^ ab)
        inf = float(a1._phase_inf(corr, target))
        lower_vals.append(inf); upper_vals.append(inf)
        if inf > TOL:
            harmful += 1
            alias = (int(da.predicted) != aa) or (int(db.predicted) != ab)
            if alias: frame_alias_harmful += 1
            else: hidden_state_harmful += 1
            if len(examples) < max_examples:
                examples.append({
                    'kind':'residual_harmful_safe_prefix_collision', 'sample':i, 'stratum':meta['key'],
                    'infidelity':inf, 'prefix_A':hex(pa), 'prefix_B':hex(pb),
                    'full_A':hex(fa), 'full_B':hex(fb),
                    'decoder_source_A':da.source, 'decoder_source_B':db.source,
                    'pred_A':FRAME_LABELS[int(da.predicted)], 'pred_B':FRAME_LABELS[int(db.predicted)],
                    'actual_A':FRAME_LABELS[aa], 'actual_B':FRAME_LABELS[ab],
                    'frame_alias':bool(alias), 'intrinsic_pauli_orbit_infidelity':oracle,
                })

    if accepted_pre != int(row_a3['accepted']):
        raise AssertionError(('S11-A3 accepted fingerprint mismatch', meta['key'], accepted_pre, row_a3['accepted']))

    lm,lse=_mean_se(lower_vals); um,use=_mean_se(upper_vals); om,ose=_mean_se(oracle_vals)
    return {
        'key':meta['key'], 'counts':{a1.GROUPS[i]:int(k) for i,k in enumerate(meta['counts']) if k},
        'location_triples':ways, 'shots':shots,
        'accepted_before_whitelist':int(accepted_pre), 'new_whitelist_rejections':int(rejected_new),
        'accepted_after_whitelist':int(accepted_post),
        'whitelist_acceptance_fraction_among_preaccepted':(accepted_post/accepted_pre if accepted_pre else None),
        'retained_resolved':int(resolved), 'retained_unresolved':int(unresolved),
        'residual_harmful':int(harmful), 'residual_frame_alias_harmful':int(frame_alias_harmful),
        'residual_hidden_state_harmful':int(hidden_state_harmful),
        'A3_whitelist_resolved_lower_estimate':float(ways*lm),
        'A3_whitelist_resolved_lower_naive_se':float(ways*lse),
        'A3_whitelist_unresolved_as_failure_upper_estimate':float(ways*um),
        'A3_whitelist_unresolved_as_failure_upper_naive_se':float(ways*use),
        'A3_whitelist_interval_width_estimate':float(ways*(um-lm)),
        'B3_additional_whitelist_rejection_estimate':float(ways*(rejected_new/shots)),
        'A3_best_pauli_orbit_after_whitelist_estimate':float(ways*om),
        'A3_best_pauli_orbit_after_whitelist_naive_se':float(ways*ose),
        'old_A3_priority_lower_same_samples':float(row_a3['A3_priority_resolved_lower_estimate']),
        'old_A3_priority_upper_same_samples':float(row_a3['A3_priority_unresolved_as_failure_upper_estimate']),
        'rejection_reasons_before_whitelist':dict(sorted(reasons.items())), 'examples':examples,
    }


def _aggregate(active_results: list[dict[str, Any]], rows_a3: list[dict[str, Any]], total_locations: int):
    low=sum(float(r['A3_whitelist_resolved_lower_estimate']) for r in active_results)
    vlow=sum(float(r['A3_whitelist_resolved_lower_naive_se'])**2 for r in active_results)
    up=sum(float(r['A3_whitelist_unresolved_as_failure_upper_estimate']) for r in active_results)
    vup=sum(float(r['A3_whitelist_unresolved_as_failure_upper_naive_se'])**2 for r in active_results)
    b3=sum(float(r['B3_additional_whitelist_rejection_estimate']) for r in active_results)
    oracle=sum(float(r['A3_best_pauli_orbit_after_whitelist_estimate']) for r in active_results)
    old_low=sum(float(r['A3_priority_resolved_lower_estimate']) for r in rows_a3)
    old_up=sum(float(r['A3_priority_unresolved_as_failure_upper_estimate']) for r in rows_a3)
    rejected=sum(int(r['new_whitelist_rejections']) for r in active_results)
    accepted_pre=sum(int(r['accepted_before_whitelist']) for r in active_results)
    accepted_post=sum(int(r['accepted_after_whitelist']) for r in active_results)
    unresolved=sum(int(r['retained_unresolved']) for r in active_results)
    harmful=sum(int(r['residual_harmful']) for r in active_results)
    alias=sum(int(r['residual_frame_alias_harmful']) for r in active_results)
    hidden=sum(int(r['residual_hidden_state_harmful']) for r in active_results)
    return {
        'A3_certified_live171_prefix_whitelist_resolved_lower_same_samples':float(low),
        'A3_certified_live171_prefix_whitelist_resolved_lower_naive_se':float(math.sqrt(vlow)),
        'A3_certified_live171_prefix_whitelist_unresolved_as_failure_upper_same_samples':float(up),
        'A3_certified_live171_prefix_whitelist_unresolved_as_failure_upper_naive_se':float(math.sqrt(vup)),
        'A3_certified_live171_prefix_whitelist_interval_width_same_samples':float(up-low),
        'B3_additional_live171_prefix_whitelist_rejection_same_samples_estimate':float(b3),
        'A3_best_pauli_orbit_after_whitelist_same_samples_estimate':float(oracle),
        'old_S11A3_A3_resolved_lower_same_samples':float(old_low),
        'old_S11A3_A3_unresolved_as_failure_upper_same_samples':float(old_up),
        'active_strata_preaccepted_samples':int(accepted_pre),
        'active_strata_newly_rejected_samples':int(rejected),
        'active_strata_postwhitelist_accepted_samples':int(accepted_post),
        'active_strata_new_rejection_fraction_among_preaccepted':(float(rejected/accepted_pre) if accepted_pre else None),
        'retained_unresolved_samples':int(unresolved), 'residual_harmful_samples':int(harmful),
        'residual_frame_alias_harmful_samples':int(alias), 'residual_hidden_state_harmful_samples':int(hidden),
        'total_location_triples':int(math.comb(int(total_locations),3)),
        'p3_predictions_lower_upper':{str(p):[float(low*p**3),float(up*p**3)] for p in (2e-4,5e-4,1e-3)},
        'p3_predictions_additional_rejection':{str(p):float(b3*p**3) for p in (2e-4,5e-4,1e-3)},
        'statistical_warning':'The prefix whitelist is exactly certified against all <=2 boundary states. Residual order-3 lower/upper coefficients are same-seed Monte Carlo estimates; retained full-195 ambiguity still requires exactification or high-statistics bounding before paper-final A3.',
    }


def _synthetic_self_test():
    high = 1 << (a1.ACTUAL_BITS + 3)
    bridge = {
        'table': {
            0x00: 1 << 0,
            0x01: 1 << 1,
            high | 0x01: 1 << 2,  # SAME 171 prefix, DIFFERENT frame: legal
            0x02: (1 << 0) | (1 << 2),
        },
        'unsafe': {0x02},
    }
    wl, cert = _build_live171_whitelist(bridge)
    assert wl == {0,1}
    assert cert['exact_no_new_le2_acceptance_penalty']
    assert cert['safe_prefixes_with_multiple_full195_frames'] == 1
    return {
        'passed':True, 'synthetic_safe_prefixes':len(wl),
        'synthetic_same_171_prefix_different_full195_frames_allowed':True,
        'synthetic_unsafe_prefix_rejected':True,
        'synthetic_exact_table_certificate_pass':True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hirano-src', type=Path)
    ap.add_argument('--signature-p', type=float, default=1e-3)
    ap.add_argument('--s11a3-strata', type=Path, default=Path('stim_pipeline/interface/s11a3_results/s11a3_strata.json'))
    ap.add_argument('--a3-seed', type=int, default=20260823)
    ap.add_argument('--max-examples-per-stratum', type=int, default=20)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--out-dir', type=Path, default=HERE / 's11a5_results')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    st = _synthetic_self_test()
    _dump(args.out_dir / 's11a5_self_test.json', st)
    if args.self_test:
        print(json.dumps(st, indent=2))
        print('S11-A5 SELF-TEST PASS')
        return

    if args.hirano_src is None:
        raise SystemExit('--hirano-src required unless --self-test')
    if not args.s11a3_strata.exists():
        raise SystemExit(f'--s11a3-strata not found: {args.s11a3_strata}')

    t0 = time.time()
    rows_a3 = json.loads(args.s11a3_strata.read_text(encoding='utf-8'))
    if len(rows_a3) != 120:
        raise AssertionError(('expected 120 S11-A3 strata', len(rows_a3)))

    print('S11-A5: building complete S10 boundary bridge...', flush=True)
    bridge = a1.s10mc._bridge(args.hirano_src.resolve(), float(args.signature_p))
    print('S11-A5: constructing exact accepted-<=2 live-171 whitelist...', flush=True)
    whitelist, cert = _build_live171_whitelist(bridge)
    _dump(args.out_dir / 's11a5_live171_whitelist_certificate.json', cert)
    _save_whitelist_npz(args.out_dir / 's11a5_live171_safe_prefixes.npz', whitelist)
    print(f"S11-A5 v2 prefix whitelist: safe_prefixes={cert['certified_safe_prefixes_whitelisted']}, preserved_safe_full={cert['certified_safe_full_states_preserved']}, multi_frame_prefixes={cert['safe_prefixes_with_multiple_full195_frames']}", flush=True)

    print('S11-A5 v2: rebuilding full-195 decoder inputs...', flush=True)
    paired, clean_analysis, _policy, _meta = a3.s8c._escape_catalogue(args.hirano_src.resolve(), float(args.signature_p))
    clean_table = {int(k):int(v) for k,v in clean_analysis['state_table'].items()}
    single_effects = a3.a2._single_effects_from_bridge(bridge)
    sectors = list(bridge['sector'].values())

    groups = a1._build_groups(int(bridge['meta']['physical_locations']))
    strata = a1._all_strata(groups)
    bymeta = {s['key']: s for s in strata}
    active = _active_rows(rows_a3)
    active_keys = [r['key'] for r in active]
    plan = {
        'stage': 'S11-A5-plan',
        'source_s11a3_strata': str(args.s11a3_strata),
        'active_strata_count': len(active),
        'active_strata': [
            {
                'key': r['key'],
                'shots': int(r['shots']),
                'old_lower': float(r['A3_priority_resolved_lower_estimate']),
                'old_unresolved': int(r['priority_unresolved_accepted']),
            }
            for r in active
        ],
        'all_non_negligible_S11A3_lower_and_unresolved_mass_covered': abs(
            sum(float(r['A3_priority_resolved_lower_estimate']) for r in active)
            - sum(float(r['A3_priority_resolved_lower_estimate']) for r in rows_a3)
        ) < 1e-5 and sum(int(r['priority_unresolved_accepted']) for r in active) == sum(int(r['priority_unresolved_accepted']) for r in rows_a3),
    }
    _dump(args.out_dir / 's11a5_plan.json', plan)
    if not plan['all_non_negligible_S11A3_lower_and_unresolved_mass_covered']:
        raise AssertionError('active-strata reduction lost S11-A3 mass')
    print(f'S11-A5: replaying {len(active)} active strata (all meaningful S11-A3 lower/unresolved mass)...', flush=True)

    results = []
    examples = []
    ckdir = args.out_dir / 'checkpoints'
    ckdir.mkdir(parents=True, exist_ok=True)
    strata_index = {s['key']: i for i, s in enumerate(strata)}

    for i, row_a3 in enumerate(active, 1):
        key = row_a3['key']
        meta = bymeta[key]
        cp = ckdir / f'{key}.json'
        rr = None
        if args.resume and cp.exists():
            old = json.loads(cp.read_text(encoding='utf-8'))
            if int(old.get('shots', -1)) == int(row_a3['shots']):
                rr = old
                print(f'S11-A5 resume {i}/{len(active)} {key}', flush=True)
        if rr is None:
            seed = int(args.a3_seed) + 1009 * (strata_index[key] + 1)
            rr = _audit_stratum_whitelist(meta, row_a3, seed, groups, bridge, whitelist, clean_table, single_effects, sectors, int(args.max_examples_per_stratum))
            _dump(cp, rr)
            print(
                f"S11-A5 {i:2d}/{len(active)} {key}: preacc={rr['accepted_before_whitelist']}/{rr['shots']}, "
                f"newreject={rr['new_whitelist_rejections']}, postacc={rr['accepted_after_whitelist']}, "
                f"harmful={rr['residual_harmful']}, unresolved={rr['retained_unresolved']}, A3=[{rr['A3_whitelist_resolved_lower_estimate']:.6g},{rr['A3_whitelist_unresolved_as_failure_upper_estimate']:.6g}]",
                flush=True,
            )
        results.append(rr)
        examples.extend(rr.get('examples', []))
        _dump(args.out_dir / 's11a5_partial_active_strata.json', results)

    agg = _aggregate(results, rows_a3, sum(len(v) for v in groups.values()))
    summary = {
        'stage': 'S11-A5',
        'status': 'CERTIFIED_LIVE171_PREFIX_WHITELIST_POLICY_RESIDUAL_A3_INTERVAL_NOT_PAPER_FINAL',
        'scope': 'complete S10 exact-three active-location model with exact <=2-derived live-171 ACCEPT/REJECT prefix whitelist; frame decoding remains full-195',
        'policy_semantics': [
            'A live 171-bit prefix is accepted iff it occurs among exact accepted <=2 S8-C boundary states after the existing unsafe-prefix policy.',
            'The 171-bit record controls ACCEPT/REJECT only; it does not assign a logical Pauli frame.',
            'Pauli-frame decoding remains based on the 195-bit full recovery syndrome (171 live bits plus the noiseless 24-bit audit tail used for decoder/scoring context).',
            'Multiple full-195 tails/frames sharing one safe 171-bit prefix are therefore allowed and expected.',
            'The prefix whitelist preserves every certified accepted <=2 boundary state exactly; additional rejection starts only at order >=3.',
            'Residual order-3 uncertainty comes only from exact-three events that alias a certified-safe <=2 prefix and remain ambiguous at full-195 decoder level.',
        ],
        'whitelist_certificate': cert,
        'active_strata_plan': plan,
        'aggregate': agg,
        'top_strata_by_residual_A3': [
            {
                'key': r['key'],
                'lower': r['A3_whitelist_resolved_lower_estimate'],
                'upper': r['A3_whitelist_unresolved_as_failure_upper_estimate'],
                'harmful': r['residual_harmful'],
                'new_rejections': r['new_whitelist_rejections'],
            }
            for r in sorted(results, key=lambda x: float(x['A3_whitelist_unresolved_as_failure_upper_estimate']), reverse=True)
        ],
        'top_strata_by_additional_rejection_B3': [
            {
                'key': r['key'],
                'B3': r['B3_additional_whitelist_rejection_estimate'],
                'new_rejections': r['new_whitelist_rejections'],
            }
            for r in sorted(results, key=lambda x: float(x['B3_additional_whitelist_rejection_estimate']), reverse=True)
        ],
        'elapsed_seconds': time.time() - t0,
        'next_gate': 'If the retained full-195 ambiguity interval is small, proceed to independent-seed/high-statistics A6. If not, exactify only the retained safe-prefix collision syndromes.',
    }
    _dump(args.out_dir / 's11a5_active_strata.json', results)
    _dump(args.out_dir / 's11a5_examples.json', examples)
    _dump(args.out_dir / 's11a5_summary.json', summary)
    print(json.dumps(_json_safe(summary), indent=2), flush=True)
    print('S11-A5 v2 COMPLETE: certified <=2-safe live-171 prefix policy + full-195 residual A3 interval audit.', flush=True)


if __name__ == '__main__':
    main()
