#!/usr/bin/env python3
"""S11-A5: certified live-171 whitelist decoder + residual exact-three A3 audit.

Purpose
-------
S11-A4 showed that almost every sampled order-3 ambiguous 171-bit prefix is
absent from the exact accepted <=2 boundary catalogue.  Instead of trying to
enumerate every possible order-3 ambiguous prefix, A5 uses the stronger and
simpler certified construction:

    accept a live 171-bit prefix iff it is an exact accepted <=2 prefix,
    and apply the unique <=2 frame attached to that prefix;
    reject every other prefix.

This "lower-order whitelist" has an exact guarantee by construction:
  * every exact accepted <=2 boundary state is preserved;
  * every prefix already unsafe at <=2 remains rejected;
  * no prefix absent from the <=2 accepted set can enter the live factory.

Therefore adding the whitelist cannot reduce the established fault distance
below three.  It only changes order >=3 acceptance.  The remaining leading
order-3 logical failures are exactly the physical three-fault events that
*alias a certified safe <=2 live prefix* or hide a non-Pauli logical-state
error behind such a prefix.

A5 replays the S11-A3 sample fingerprints only in the strata that carried all
non-negligible lower A3 or unresolved mass.  The other strata contributed zero
on the same sample set and remain zero under a stricter policy.  This yields a
same-sample residual A3 point estimate with no decoder-unresolved interval.
It is still a Monte-Carlo estimate of the order-3 coefficient; independent
high-statistics seeds are the next paper-final step.
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


def _save_whitelist_npz(path: Path, whitelist: dict[int, int]) -> None:
    """Compact reusable storage for 171-bit prefix -> frame map.

    Prefixes are split into three little-endian uint64 words to avoid a huge
    multi-million-entry JSON file.
    """
    items = sorted((int(k), int(v)) for k, v in whitelist.items())
    n = len(items)
    lo = np.empty(n, dtype=np.uint64)
    mid = np.empty(n, dtype=np.uint64)
    hi = np.empty(n, dtype=np.uint64)
    fr = np.empty(n, dtype=np.uint8)
    mask64 = (1 << 64) - 1
    for i, (x, f) in enumerate(items):
        lo[i] = x & mask64
        mid[i] = (x >> 64) & mask64
        hi[i] = (x >> 128) & mask64
        fr[i] = f
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prefix_word0=lo, prefix_word1=mid, prefix_word2=hi, frame=fr, actual_bits=np.asarray([a1.ACTUAL_BITS], dtype=np.uint16))


def _build_live171_whitelist(bridge: dict[str, Any]) -> tuple[dict[int, int], dict[str, Any]]:
    """Build exact accepted <=2 prefix->frame map.

    The bridge state table is the exact S8-C boundary <=2 full-recovery table.
    Prefixes in bridge['unsafe'] are the already-certified 171-bit reject set.
    Every remaining reachable <=2 prefix must carry one and only one frame.
    """
    mask = (1 << a1.ACTUAL_BITS) - 1
    unsafe = set(int(x) for x in bridge['unsafe'])
    frames_by_prefix: dict[int, set[int]] = collections.defaultdict(set)
    reachable_full = safe_full = unsafe_full = ambiguous_full = 0

    for full, flags in bridge['table'].items():
        reachable_full += 1
        p = _singleton(int(flags))
        pref = int(full) & mask
        if pref in unsafe:
            unsafe_full += 1
            continue
        if p is None:
            ambiguous_full += 1
            raise AssertionError(('non-unsafe <=2 full syndrome is not singleton', hex(int(full)), int(flags), hex(pref)))
        safe_full += 1
        frames_by_prefix[pref].add(int(p))

    conflicts = {p: sorted(v) for p, v in frames_by_prefix.items() if len(v) != 1}
    if conflicts:
        first = next(iter(conflicts.items()))
        raise AssertionError(('safe 171-bit prefix has multiple <=2 frames', first, len(conflicts)))

    whitelist = {int(p): int(next(iter(v))) for p, v in frames_by_prefix.items()}
    if 0 not in whitelist:
        raise AssertionError('fault-free 171-bit prefix missing from whitelist')
    if 0 in unsafe:
        raise AssertionError('fault-free prefix unexpectedly unsafe')

    # Exact no-<=2-penalty replay at the table level.
    preserved = 0
    frame_mismatch = 0
    for full, flags in bridge['table'].items():
        pref = int(full) & mask
        if pref in unsafe:
            continue
        p = _singleton(int(flags))
        if p is None:
            raise AssertionError(('safe table ambiguity', full, flags))
        preserved += 1
        if whitelist.get(pref) != int(p):
            frame_mismatch += 1
    if frame_mismatch:
        raise AssertionError(('whitelist <=2 frame mismatch', frame_mismatch))

    cert = {
        'actual_bits': int(a1.ACTUAL_BITS),
        'exact_boundary_full_states': int(reachable_full),
        'existing_unsafe_prefixes': int(len(unsafe)),
        'certified_safe_prefixes_whitelisted': int(len(whitelist)),
        'certified_safe_full_states_preserved': int(preserved),
        'existing_unsafe_full_states_rejected': int(unsafe_full),
        'nonunsafe_ambiguous_full_states': int(ambiguous_full),
        'safe_prefix_frame_conflicts': int(len(conflicts)),
        'safe_full_frame_mismatches': int(frame_mismatch),
        'faultfree_prefix_hex': hex(0),
        'faultfree_frame': FRAME_LABELS[int(whitelist[0])],
        'exact_no_le2_acceptance_penalty': True,
        'exact_no_le2_frame_change': True,
    }
    return whitelist, cert


def _active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        lo = float(r['A3_priority_resolved_lower_estimate'])
        un = int(r['priority_unresolved_accepted'])
        if lo > NUMERICAL_A3_TOL or un > 0:
            out.append(r)
    return out


def _audit_stratum_whitelist(meta, row_a3, seed: int, groups, bridge, whitelist, max_examples: int):
    rng = np.random.default_rng(int(seed))
    ways = int(meta['location_triples'])
    shots = int(row_a3['shots'])
    mask = (1 << a1.ACTUAL_BITS) - 1

    vals: list[float] = []
    oracle_vals: list[float] = []
    accepted_pre = 0
    accepted_post = 0
    rejected_new = 0
    harmful = 0
    frame_alias_harmful = 0
    hidden_state_harmful = 0
    harmful_prefix_collision = 0
    reasons = collections.Counter()
    examples = []

    for i in range(shots):
        locs = a1._sample_locations(meta, groups, rng)
        r = a1._simulate_exact3(locs, bridge, rng, want_trace=False, want_state=True)
        reasons[r['reason']] += 1
        if not r['accepted']:
            vals.append(0.0)
            oracle_vals.append(0.0)
            continue

        accepted_pre += 1
        fa = int(r['full_syndrome_A_hex'], 16)
        fb = int(r['full_syndrome_B_hex'], 16)
        pa = fa & mask
        pb = fb & mask
        wa = whitelist.get(pa)
        wb = whitelist.get(pb)
        if wa is None or wb is None:
            rejected_new += 1
            vals.append(0.0)
            oracle_vals.append(0.0)
            if len(examples) < max_examples:
                examples.append({
                    'kind': 'new_live171_whitelist_rejection',
                    'sample': i,
                    'stratum': meta['key'],
                    'prefix_A': hex(pa),
                    'prefix_B': hex(pb),
                    'A_whitelisted': wa is not None,
                    'B_whitelisted': wb is not None,
                    'full_A': hex(fa),
                    'full_B': hex(fb),
                })
            continue

        accepted_post += 1
        aa = int(r['actual_frame_A'])
        ab = int(r['actual_frame_B'])
        lv = np.asarray(r['_logical_vector'], dtype=complex)
        target = a1._target(r['record'])
        corr = a1.s10mc._apply(lv, int(wa) ^ aa, int(wb) ^ ab)
        inf = float(a1._phase_inf(corr, target))
        vals.append(inf)
        oracle = float(a1._pauli_orbit_inf(lv, target))
        oracle_vals.append(oracle)

        if inf > TOL:
            harmful += 1
            alias = (int(wa) != aa) or (int(wb) != ab)
            if alias:
                frame_alias_harmful += 1
            else:
                hidden_state_harmful += 1
            harmful_prefix_collision += 1
            if len(examples) < max_examples:
                examples.append({
                    'kind': 'residual_harmful_safe_prefix_collision',
                    'sample': i,
                    'stratum': meta['key'],
                    'infidelity': inf,
                    'record': r.get('record'),
                    'sector_pair': r.get('sector_pair'),
                    'prefix_A': hex(pa),
                    'prefix_B': hex(pb),
                    'full_A': hex(fa),
                    'full_B': hex(fb),
                    'whitelist_frame_A': FRAME_LABELS[int(wa)],
                    'whitelist_frame_B': FRAME_LABELS[int(wb)],
                    'actual_frame_A': FRAME_LABELS[aa],
                    'actual_frame_B': FRAME_LABELS[ab],
                    'frame_alias': bool(alias),
                    'intrinsic_pauli_orbit_infidelity': oracle,
                })

    # Same-seed pre-policy acceptance fingerprint must match S11-A3 exactly.
    if accepted_pre != int(row_a3['accepted']):
        raise AssertionError(('S11-A3 accepted fingerprint mismatch', meta['key'], accepted_pre, row_a3['accepted']))

    m, se = _mean_se(vals)
    om, ose = _mean_se(oracle_vals)
    reject_coeff = ways * (rejected_new / shots)
    return {
        'key': meta['key'],
        'counts': {a1.GROUPS[i]: int(k) for i, k in enumerate(meta['counts']) if k},
        'location_triples': ways,
        'shots': shots,
        'accepted_before_whitelist': int(accepted_pre),
        'new_whitelist_rejections': int(rejected_new),
        'accepted_after_whitelist': int(accepted_post),
        'whitelist_acceptance_fraction_among_preaccepted': (accepted_post / accepted_pre if accepted_pre else None),
        'residual_harmful': int(harmful),
        'residual_frame_alias_harmful': int(frame_alias_harmful),
        'residual_hidden_state_harmful': int(hidden_state_harmful),
        'residual_harmful_safe_prefix_collisions': int(harmful_prefix_collision),
        'A3_whitelist_decoder_estimate': float(ways * m),
        'A3_whitelist_decoder_naive_se': float(ways * se),
        'B3_additional_whitelist_rejection_estimate': float(reject_coeff),
        'A3_best_pauli_orbit_after_whitelist_estimate': float(ways * om),
        'A3_best_pauli_orbit_after_whitelist_naive_se': float(ways * ose),
        'old_A3_priority_lower_same_samples': float(row_a3['A3_priority_resolved_lower_estimate']),
        'old_A3_priority_upper_same_samples': float(row_a3['A3_priority_unresolved_as_failure_upper_estimate']),
        'rejection_reasons_before_whitelist': dict(sorted(reasons.items())),
        'examples': examples,
    }


def _aggregate(active_results: list[dict[str, Any]], rows_a3: list[dict[str, Any]], total_locations: int):
    a3 = sum(float(r['A3_whitelist_decoder_estimate']) for r in active_results)
    va3 = sum(float(r['A3_whitelist_decoder_naive_se']) ** 2 for r in active_results)
    b3 = sum(float(r['B3_additional_whitelist_rejection_estimate']) for r in active_results)
    oracle = sum(float(r['A3_best_pauli_orbit_after_whitelist_estimate']) for r in active_results)
    voracle = sum(float(r['A3_best_pauli_orbit_after_whitelist_naive_se']) ** 2 for r in active_results)
    old_low = sum(float(r['A3_priority_resolved_lower_estimate']) for r in rows_a3)
    old_up = sum(float(r['A3_priority_unresolved_as_failure_upper_estimate']) for r in rows_a3)
    total_comb = math.comb(int(total_locations), 3)
    rejected = sum(int(r['new_whitelist_rejections']) for r in active_results)
    accepted_pre = sum(int(r['accepted_before_whitelist']) for r in active_results)
    accepted_post = sum(int(r['accepted_after_whitelist']) for r in active_results)
    harmful = sum(int(r['residual_harmful']) for r in active_results)
    alias = sum(int(r['residual_frame_alias_harmful']) for r in active_results)
    hidden = sum(int(r['residual_hidden_state_harmful']) for r in active_results)

    return {
        'A3_certified_live171_whitelist_same_samples_estimate': float(a3),
        'A3_certified_live171_whitelist_same_samples_naive_se': float(math.sqrt(va3)),
        'B3_additional_live171_whitelist_rejection_same_samples_estimate': float(b3),
        'A3_best_pauli_orbit_after_whitelist_same_samples_estimate': float(oracle),
        'A3_best_pauli_orbit_after_whitelist_same_samples_naive_se': float(math.sqrt(voracle)),
        'old_S11A3_A3_resolved_lower_same_samples': float(old_low),
        'old_S11A3_A3_unresolved_as_failure_upper_same_samples': float(old_up),
        'A3_reduction_vs_old_resolved_lower': float(old_low - a3),
        'A3_fraction_of_old_resolved_lower_remaining': (float(a3 / old_low) if old_low else None),
        'active_strata_preaccepted_samples': int(accepted_pre),
        'active_strata_newly_rejected_samples': int(rejected),
        'active_strata_postwhitelist_accepted_samples': int(accepted_post),
        'active_strata_new_rejection_fraction_among_preaccepted': (float(rejected / accepted_pre) if accepted_pre else None),
        'residual_harmful_samples': int(harmful),
        'residual_frame_alias_harmful_samples': int(alias),
        'residual_hidden_state_harmful_samples': int(hidden),
        'total_location_triples': int(total_comb),
        'p3_predictions_residual_logical': {str(p): float(a3 * p ** 3) for p in (2e-4, 5e-4, 1e-3)},
        'p3_predictions_additional_rejection': {str(p): float(b3 * p ** 3) for p in (2e-4, 5e-4, 1e-3)},
        'statistical_warning': 'A5 certifies the live-171 whitelist policy against all exact <=2 boundary states. The residual A3 coefficient is still estimated from the S11-A3 exact-three Monte Carlo samples; independent seeds/high statistics are required for a paper-final confidence interval.',
    }


def _synthetic_self_test():
    high = 1 << (a1.ACTUAL_BITS + 3)
    bridge = {
        'table': {
            0x00: 1 << 0,
            0x01: 1 << 1,
            high | 0x01: 1 << 1,  # same 171-bit prefix, same frame
            0x02: (1 << 0) | (1 << 2),
        },
        'unsafe': {0x02},
    }
    wl, cert = _build_live171_whitelist(bridge)
    assert wl == {0: 0, 1: 1}
    assert cert['exact_no_le2_acceptance_penalty']
    assert cert['safe_prefix_frame_conflicts'] == 0
    return {
        'passed': True,
        'synthetic_safe_prefixes': int(len(wl)),
        'synthetic_unsafe_prefix_rejected': True,
        'synthetic_duplicate_full_same_prefix_frame_consistent': True,
        'synthetic_exact_table_certificate_pass': True,
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
    _save_whitelist_npz(args.out_dir / 's11a5_live171_whitelist.npz', whitelist)
    print(f"S11-A5 whitelist certificate: safe_prefixes={cert['certified_safe_prefixes_whitelisted']}, preserved_safe_full={cert['certified_safe_full_states_preserved']}, conflicts={cert['safe_prefix_frame_conflicts']}", flush=True)

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
            rr = _audit_stratum_whitelist(meta, row_a3, seed, groups, bridge, whitelist, int(args.max_examples_per_stratum))
            _dump(cp, rr)
            print(
                f"S11-A5 {i:2d}/{len(active)} {key}: preacc={rr['accepted_before_whitelist']}/{rr['shots']}, "
                f"newreject={rr['new_whitelist_rejections']}, postacc={rr['accepted_after_whitelist']}, "
                f"harmful={rr['residual_harmful']}, A3={rr['A3_whitelist_decoder_estimate']:.6g}",
                flush=True,
            )
        results.append(rr)
        examples.extend(rr.get('examples', []))
        _dump(args.out_dir / 's11a5_partial_active_strata.json', results)

    agg = _aggregate(results, rows_a3, sum(len(v) for v in groups.values()))
    summary = {
        'stage': 'S11-A5',
        'status': 'CERTIFIED_LIVE171_WHITELIST_POLICY_RESIDUAL_A3_SAME_SAMPLE_NOT_YET_PAPER_FINAL_STATS',
        'scope': 'complete S10 exact-three active-location model with exact <=2-derived live-171 whitelist rejection policy',
        'policy_semantics': [
            'A live 171-bit prefix is accepted iff it is reachable by an exact accepted <=2 S8-C boundary state.',
            'Each whitelisted prefix uses the unique exact <=2 logical Pauli frame; all other prefixes are rejected.',
            'The whitelist therefore preserves every certified accepted <=2 boundary state and changes no <=2 frame.',
            'Additional rejection starts only at order >=3 within the modeled active-location stochastic-Pauli scope.',
            'Residual order-3 failures are safe-prefix aliases or hidden logical-state failures that cannot be removed by rejecting only prefixes absent from the <=2 accepted set.',
        ],
        'whitelist_certificate': cert,
        'active_strata_plan': plan,
        'aggregate': agg,
        'top_strata_by_residual_A3': [
            {
                'key': r['key'],
                'A3': r['A3_whitelist_decoder_estimate'],
                'SE': r['A3_whitelist_decoder_naive_se'],
                'harmful': r['residual_harmful'],
                'new_rejections': r['new_whitelist_rejections'],
            }
            for r in sorted(results, key=lambda x: float(x['A3_whitelist_decoder_estimate']), reverse=True)
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
        'next_gate': 'If residual A3 is stable and substantially narrower, run independent seeds / targeted high statistics (S11-A6) for paper-final A3 confidence intervals and then proceed to resource-baseline comparison.',
    }
    _dump(args.out_dir / 's11a5_active_strata.json', results)
    _dump(args.out_dir / 's11a5_examples.json', examples)
    _dump(args.out_dir / 's11a5_summary.json', summary)
    print(json.dumps(_json_safe(summary), indent=2), flush=True)
    print('S11-A5 COMPLETE: certified <=2-safe live-171 whitelist policy + same-sample residual A3 audit.', flush=True)


if __name__ == '__main__':
    main()
