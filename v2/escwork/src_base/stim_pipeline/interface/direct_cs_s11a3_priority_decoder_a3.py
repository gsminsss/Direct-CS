#!/usr/bin/env python3
"""S11-A3: priority order-3 boundary decoder + adaptive exact-three A3 audit.

Purpose
-------
S11-A2 showed that a conservative order<=3 *set union* resolves ~93% of pure
escape triples, but it can create false ambiguity by allowing order-3 candidate
frames to compete with an already-certified <=2 frame.  For a leading-order
fault analysis this is too pessimistic: a decoder should preserve the exact
<=2 table first, and only consult order-3 candidates when the full recovery
syndrome has no <=2 explanation.

This stage defines one concrete, conservative decoder:

  1. PRIORITY-<=2: use the exact S8-C boundary <=2 full-syndrome table whenever
     it is logically unique.  This preserves the certified d_f=3 decoder.
  2. ORDER-3 EXTENSION: only if the syndrome is absent from the <=2 table,
     construct an over-approximate order-3 frame set from
       a) one escape single effect + any exact S8-C boundary <=2 state, and
       b) any canonical Steane sector + exact clean-escape <=2 state.
     The second term is required because one physical core fault can produce a
     canonical weight-2 sector (S8-E), so representative boundary weight alone
     is not a physical fault-order bound.
  3. A singleton order-3 set is decoded.  A multi-frame set remains unresolved;
     no optimistic tie breaking is used.

The order-3 extension intentionally ignores distinct-location overlap in the
single+<=2 decomposition.  This can only add candidates.  Therefore a singleton
is safe, while ambiguity can still be false ambiguity.  The resulting lower /
unresolved-as-failure upper A3 interval is conservative.

Unlike S11-A2, this decoder is applied to *all 120 complete-factory exact-three
strata* with adaptive oversampling of the dominant pilot strata.  It therefore
produces a global decoder-aware A3 interval, not just pure-escape diagnostics.
It is still marked NOT_PAPER_FINAL until the unresolved A3 interval is small or
remaining ambiguous syndromes are exactified in S11-A4.
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

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError((name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a1 = _load('s11a3_a1', HERE / 'direct_cs_s11a1_exact3_stratified_pilot.py')
a2 = _load('s11a3_a2', HERE / 'direct_cs_s11a2_decoder_aware_exact3.py')
s8c = _load('s11a3_s8c', HERE / 'zxz_core_sector_escape_composability.py')

TOL = 1e-10
FRAME_LABELS = ('I', 'X', 'Z', 'Y')

# Oversample every stratum that had non-negligible identity-fallback A3 in A1,
# plus the hidden CORE_POST3 mechanism found by A2.
DOMINANT_KEYS = (
    'ESC_A3', 'ESC_B3', 'CORE_POST3',
    'CORE_POST1+ESC_A2', 'CORE_POST1+ESC_B2',
    'CORE_POST2+ESC_A1', 'CORE_POST2+ESC_B1',
    'CAT_POST1+ESC_A2', 'CAT_POST1+ESC_B2',
)


def _frame_list(flags: int) -> list[int]:
    return [i for i in range(4) if int(flags) & (1 << i)]


def _singleton(flags: int) -> int | None:
    f = int(flags)
    return f.bit_length() - 1 if f and f.bit_count() == 1 else None


def _combine_flags(flags: int, frame: int) -> int:
    out = 0
    for f in _frame_list(flags):
        out |= 1 << (int(f) ^ int(frame))
    return out


def _order3_extension_flags(
    full: int,
    boundary_le2: dict[int, int],
    clean_escape_le2: dict[int, int],
    single_effects: list[tuple[int, int]],
    sectors: list[Any],
) -> int:
    """Conservative new order-3 frame candidates for one block.

    Candidate family A: one escape single + exact S8-C boundary<=2 remainder.
    Candidate family B: arbitrary canonical Steane sector + clean escape<=2.
    Family B closes the physical-vs-representative-weight gap discovered in S8-E.
    """
    target = int(full)
    flags = 0

    # A: escape1 + boundary<=2.  Duplicated syndrome/frame single classes need
    # only one query because this is a reachability set, not a probability model.
    for syn, fr in single_effects:
        rem = int(boundary_le2.get(target ^ int(syn), 0))
        if rem:
            flags |= _combine_flags(rem, int(fr))
            if flags == 0b1111:
                break

    # B: arbitrary incoming Steane syndrome sector + exact clean escape<=2.
    # Only 64 sectors, so this is cheap and deliberately over-complete.
    if flags != 0b1111:
        for s in sectors:
            rem = int(clean_escape_le2.get(target ^ int(s.full), 0))
            if rem:
                flags |= _combine_flags(rem, int(s.frame))
                if flags == 0b1111:
                    break
    return int(flags)


@dataclass
class DecodeOne:
    predicted: int | None
    source: str
    base_flags: int
    order3_flags: int
    all_flags: int


def _decode_one(full: int, boundary_le2, clean_escape_le2, single_effects, sectors) -> DecodeOne:
    full = int(full)
    base = int(boundary_le2.get(full, 0))
    if base:
        p = _singleton(base)
        # An ambiguous exact <=2 full syndrome should already have an unsafe
        # 171-bit prefix and therefore cannot be accepted by the S10 policy.
        return DecodeOne(p, 'le2_unique' if p is not None else 'le2_ambiguous', base, 0, base)
    ext = _order3_extension_flags(full, boundary_le2, clean_escape_le2, single_effects, sectors)
    p = _singleton(ext)
    return DecodeOne(p, 'le3_unique' if p is not None else ('le3_ambiguous' if ext else 'le3_empty'), 0, ext, ext)


def _all_candidate_flags(full: int, boundary_le2, clean_escape_le2, single_effects, sectors) -> int:
    base = int(boundary_le2.get(int(full), 0))
    ext = _order3_extension_flags(int(full), boundary_le2, clean_escape_le2, single_effects, sectors)
    return int(base | ext)


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


def _mean_se(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=float)
    m = float(a.mean())
    if len(a) <= 1:
        return m, 0.0
    return m, float(a.std(ddof=1) / math.sqrt(len(a)))


def _score_resolved(r: dict[str, Any], da: DecodeOne, db: DecodeOne) -> float:
    lv = np.asarray(r['_logical_vector'], dtype=complex)
    target = a1._target(r['record'])
    aa = int(r['actual_frame_A']); ab = int(r['actual_frame_B'])
    assert da.predicted is not None and db.predicted is not None
    corr = a1.s10mc._apply(lv, int(da.predicted) ^ aa, int(db.predicted) ^ ab)
    return float(a1._phase_inf(corr, target))


def _audit_stratum(meta, shots: int, seed: int, groups, bridge, clean_table, single_effects, sectors, max_examples: int):
    rng = np.random.default_rng(int(seed))
    lower_vals: list[float] = []
    upper_vals: list[float] = []
    oracle_vals: list[float] = []
    identity_vals: list[float] = []
    accepted = resolved = unresolved = 0
    resolved_harmful = frame_alias_harmful = hidden_state_harmful = 0
    actual_not_in_union = 0
    accepted_base_ambiguous = 0
    source_hist_a = collections.Counter(); source_hist_b = collections.Counter()
    cand_hist_a = collections.Counter(); cand_hist_b = collections.Counter()
    reasons = collections.Counter(); examples = []

    boundary = bridge['table']
    unsafe = bridge['unsafe']
    mask = (1 << a1.ACTUAL_BITS) - 1

    for i in range(int(shots)):
        locs = a1._sample_locations(meta, groups, rng)
        r = a1._simulate_exact3(locs, bridge, rng, want_trace=False, want_state=True)
        reasons[r['reason']] += 1
        if not r['accepted']:
            lower_vals.append(0.0); upper_vals.append(0.0); oracle_vals.append(0.0); identity_vals.append(0.0)
            continue

        accepted += 1
        fa = int(r['full_syndrome_A_hex'], 16); fb = int(r['full_syndrome_B_hex'], 16)
        da = _decode_one(fa, boundary, clean_table, single_effects, sectors)
        db = _decode_one(fb, boundary, clean_table, single_effects, sectors)
        source_hist_a[da.source] += 1; source_hist_b[db.source] += 1
        cand_hist_a[int(da.all_flags).bit_count()] += 1; cand_hist_b[int(db.all_flags).bit_count()] += 1

        # Base-table ambiguity is inconsistent with the accepted S8-C 171-bit
        # policy and is a hard integration failure, not a decoder ambiguity.
        if da.source == 'le2_ambiguous' or db.source == 'le2_ambiguous':
            accepted_base_ambiguous += 1

        # Verify the conservative total candidate union contains the explicit
        # sector+escape Pauli frame carried by the simulated trajectory.  For a
        # priority <=2 alias, the actual frame may intentionally differ from the
        # chosen <=2 prediction, but it must still exist in the total <=3 union.
        ua = _all_candidate_flags(fa, boundary, clean_table, single_effects, sectors)
        ub = _all_candidate_flags(fb, boundary, clean_table, single_effects, sectors)
        aa = int(r['actual_frame_A']); ab = int(r['actual_frame_B'])
        if not (ua & (1 << aa)) or not (ub & (1 << ab)):
            actual_not_in_union += 1
            if len(examples) < max_examples:
                examples.append({'kind':'actual_frame_not_in_union','sample':i,'stratum':meta['key'],'full_A':hex(fa),'full_B':hex(fb),'actual_A':FRAME_LABELS[aa],'actual_B':FRAME_LABELS[ab],'union_A':[FRAME_LABELS[x] for x in _frame_list(ua)],'union_B':[FRAME_LABELS[x] for x in _frame_list(ub)]})

        target = a1._target(r['record'])
        lv = np.asarray(r['_logical_vector'], complex)
        oracle = float(a1._pauli_orbit_inf(lv, target))
        oracle_vals.append(oracle)
        identity_vals.append(float(r['identity_infidelity']))

        if da.predicted is None or db.predicted is None:
            unresolved += 1
            lower_vals.append(0.0); upper_vals.append(1.0)
            if len(examples) < max_examples and (da.source != 'le3_empty' or db.source != 'le3_empty'):
                examples.append({'kind':'unresolved','sample':i,'stratum':meta['key'],'record':r.get('record'),'sector_pair':r.get('sector_pair'),'full_A':hex(fa),'full_B':hex(fb),'source_A':da.source,'source_B':db.source,'candidates_A':[FRAME_LABELS[x] for x in _frame_list(da.all_flags)],'candidates_B':[FRAME_LABELS[x] for x in _frame_list(db.all_flags)]})
            continue

        resolved += 1
        inf = _score_resolved(r, da, db)
        lower_vals.append(inf); upper_vals.append(inf)
        if inf > TOL:
            resolved_harmful += 1
            alias = (int(da.predicted) != aa) or (int(db.predicted) != ab)
            if alias: frame_alias_harmful += 1
            else: hidden_state_harmful += 1
            if len(examples) < max_examples:
                examples.append({'kind':'resolved_harmful','sample':i,'stratum':meta['key'],'infidelity':inf,'record':r.get('record'),'sector_pair':r.get('sector_pair'),'full_A':hex(fa),'full_B':hex(fb),'pred_A':FRAME_LABELS[int(da.predicted)],'pred_B':FRAME_LABELS[int(db.predicted)],'actual_A':FRAME_LABELS[aa],'actual_B':FRAME_LABELS[ab],'decoder_source_A':da.source,'decoder_source_B':db.source,'frame_alias':bool(alias)})

    ways = int(meta['location_triples'])
    ml, sel = _mean_se(lower_vals); mu, seu = _mean_se(upper_vals)
    mo, seo = _mean_se(oracle_vals); mi, sei = _mean_se(identity_vals)
    return {
        'key': meta['key'],
        'counts': {a1.GROUPS[i]: int(k) for i,k in enumerate(meta['counts']) if k},
        'location_triples': ways,
        'shots': int(shots),
        'accepted': accepted,
        'accepted_fraction': accepted/shots if shots else None,
        'priority_resolved_accepted': resolved,
        'priority_unresolved_accepted': unresolved,
        'priority_resolution_fraction_among_accepted': resolved/accepted if accepted else None,
        'priority_resolved_harmful': resolved_harmful,
        'priority_frame_alias_harmful': frame_alias_harmful,
        'priority_hidden_state_harmful': hidden_state_harmful,
        'actual_frame_not_in_conservative_union': actual_not_in_union,
        'accepted_exact_le2_ambiguous_policy_violation': accepted_base_ambiguous,
        'decoder_source_histogram_A': dict(sorted(source_hist_a.items())),
        'decoder_source_histogram_B': dict(sorted(source_hist_b.items())),
        'candidate_size_histogram_A': {str(k):int(v) for k,v in sorted(cand_hist_a.items())},
        'candidate_size_histogram_B': {str(k):int(v) for k,v in sorted(cand_hist_b.items())},
        'A3_priority_resolved_lower_estimate': ways * ml,
        'A3_priority_resolved_lower_se': ways * sel,
        'A3_priority_unresolved_as_failure_upper_estimate': ways * mu,
        'A3_priority_unresolved_as_failure_upper_se': ways * seu,
        'A3_priority_interval_width_estimate': ways * (mu - ml),
        'A3_identity_fallback_estimate_for_comparison': ways * mi,
        'A3_identity_fallback_se_for_comparison': ways * sei,
        'A3_best_pauli_orbit_estimate': ways * mo,
        'A3_best_pauli_orbit_se': ways * seo,
        'rejection_reasons': dict(sorted(reasons.items())),
        'examples': examples,
    }


def _aggregate(rows: list[dict[str, Any]], total_locations: int):
    low = sum(float(r['A3_priority_resolved_lower_estimate']) for r in rows)
    up = sum(float(r['A3_priority_unresolved_as_failure_upper_estimate']) for r in rows)
    vlow = sum(float(r['A3_priority_resolved_lower_se'])**2 for r in rows)
    vup = sum(float(r['A3_priority_unresolved_as_failure_upper_se'])**2 for r in rows)
    identity = sum(float(r['A3_identity_fallback_estimate_for_comparison']) for r in rows)
    oracle = sum(float(r['A3_best_pauli_orbit_estimate']) for r in rows)
    unresolved_accepted = sum(int(r['priority_unresolved_accepted']) for r in rows)
    accepted = sum(int(r['accepted']) for r in rows)
    resolved_harmful = sum(int(r['priority_resolved_harmful']) for r in rows)
    frame_alias = sum(int(r['priority_frame_alias_harmful']) for r in rows)
    hidden = sum(int(r['priority_hidden_state_harmful']) for r in rows)
    total_comb = math.comb(int(total_locations), 3)
    return {
        'A3_priority_decoder_resolved_lower_estimate': low,
        'A3_priority_decoder_resolved_lower_naive_se': math.sqrt(vlow),
        'A3_priority_decoder_unresolved_as_failure_upper_estimate': up,
        'A3_priority_decoder_unresolved_as_failure_upper_naive_se': math.sqrt(vup),
        'A3_interval_width_due_to_unresolved': up - low,
        'A3_identity_fallback_estimate_same_samples': identity,
        'A3_best_pauli_orbit_estimate_same_samples': oracle,
        'accepted_conditional_exact3_samples_total': accepted,
        'unresolved_accepted_samples_total': unresolved_accepted,
        'resolved_harmful_samples_total': resolved_harmful,
        'resolved_frame_alias_harmful_samples_total': frame_alias,
        'resolved_hidden_state_harmful_samples_total': hidden,
        'total_location_triples': total_comb,
        'p3_predictions_lower_upper': {
            str(p): [low * p**3, up * p**3] for p in (2e-4, 5e-4, 1e-3)
        },
        'statistical_warning': 'Normal SE is descriptive only; zero-event strata are not rigorous rare-event upper bounds. Remaining unresolved order-3 contribution must be exactified or further bounded before paper-final A3.',
    }


def _self_test():
    # Priority semantics: an exact <=2 frame wins even if an order-3 extension
    # would contain another frame.
    base = {5: 1 << 2}
    clean = {0:1<<0, 1:1<<1}
    singles = [(4,1)]
    class S:
        def __init__(self, full, frame): self.full=full; self.frame=frame
    sectors = [S(0,0)]
    d = _decode_one(5, base, clean, singles, sectors)
    assert d.predicted == 2 and d.source == 'le2_unique'

    # No base explanation: order3 extension can yield a singleton.
    base2 = {1:1<<1}
    d2 = _decode_one(5, base2, clean, singles, sectors)
    assert d2.predicted == 0  # single frame X + base X -> I
    assert d2.source == 'le3_unique'

    gl = a1._build_groups(1028); strata = a1._all_strata(gl)
    return {
        'passed': True,
        'priority_base_prediction': int(d.predicted),
        'priority_order3_prediction': int(d2.predicted),
        'modeled_locations': sum(len(v) for v in gl.values()),
        'strata': len(strata),
        'partition_exact': sum(int(x['location_triples']) for x in strata) == math.comb(2946,3),
        'dominant_keys_present': all(any(s['key']==k for s in strata) for k in DOMINANT_KEYS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hirano-src', type=Path)
    ap.add_argument('--signature-p', type=float, default=1e-3)
    ap.add_argument('--base-shots-per-stratum', type=int, default=200)
    ap.add_argument('--dominant-shots-per-stratum', type=int, default=10000)
    ap.add_argument('--core-post3-shots', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=20260823)
    ap.add_argument('--max-examples-per-stratum', type=int, default=8)
    ap.add_argument('--only-dominant', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--out-dir', type=Path, default=HERE/'s11a3_results')
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    st = _self_test(); _dump(args.out_dir/'s11a3_self_test.json', st)
    if args.self_test:
        print(json.dumps(st, indent=2)); print('S11-A3 SELF-TEST PASS'); return
    if args.hirano_src is None:
        raise SystemExit('--hirano-src required unless --self-test')

    t0 = time.time()
    print('S11-A3: building complete S10 boundary bridge...', flush=True)
    bridge = a1.s10mc._bridge(args.hirano_src.resolve(), float(args.signature_p))
    print('S11-A3: rebuilding exact clean escape <=2 table...', flush=True)
    paired, clean_analysis, _policy, _meta = s8c._escape_catalogue(args.hirano_src.resolve(), float(args.signature_p))
    clean_table = {int(k):int(v) for k,v in clean_analysis['state_table'].items()}
    single_effects = a2._single_effects_from_bridge(bridge)
    sectors = list(bridge['sector'].values())
    groups = a1._build_groups(int(bridge['meta']['physical_locations']))
    strata = a1._all_strata(groups)
    print(f"S11-A3 decoder inputs: boundary_le2={len(bridge['table'])}, clean_escape_le2={len(clean_table)}, unique_escape_single_effects={len(single_effects)}, sectors={len(sectors)}", flush=True)

    rows = []; all_examples = []
    ckdir = args.out_dir/'checkpoints'; ckdir.mkdir(parents=True, exist_ok=True)
    selected = [s for s in strata if (not args.only_dominant or s['key'] in DOMINANT_KEYS)]

    for si, meta in enumerate(selected):
        key = meta['key']
        shots = int(args.base_shots_per_stratum)
        if key in DOMINANT_KEYS: shots = int(args.dominant_shots_per_stratum)
        if key == 'CORE_POST3': shots = int(args.core_post3_shots)
        cp = ckdir / f'{key}.json'
        row = None
        if args.resume and cp.exists():
            old = json.loads(cp.read_text(encoding='utf-8'))
            if int(old.get('shots',-1)) == shots:
                row = old
                print(f'S11-A3 resume {si+1}/{len(selected)} {key}: shots={shots}', flush=True)
        if row is None:
            seed = int(args.seed) + 1009 * (a1._all_strata(groups).index(meta) + 1)
            row = _audit_stratum(meta, shots, seed, groups, bridge, clean_table, single_effects, sectors, int(args.max_examples_per_stratum))
            _dump(cp, row)
            print(f"S11-A3 {si+1:3d}/{len(selected)} {key}: acc={row['accepted']}/{shots}, resolved={row['priority_resolved_accepted']}, harmful={row['priority_resolved_harmful']}, unresolved={row['priority_unresolved_accepted']}, A3=[{row['A3_priority_resolved_lower_estimate']:.6g},{row['A3_priority_unresolved_as_failure_upper_estimate']:.6g}]", flush=True)
        if row['actual_frame_not_in_conservative_union']:
            raise AssertionError(('candidate union lost actual frame', key, row['actual_frame_not_in_conservative_union']))
        if row['accepted_exact_le2_ambiguous_policy_violation']:
            raise AssertionError(('accepted <=2 ambiguous syndrome bypassed policy', key, row['accepted_exact_le2_ambiguous_policy_violation']))
        rows.append(row); all_examples.extend(row.get('examples',[]))
        _dump(args.out_dir/'s11a3_partial_strata.json', rows)

    complete = len(selected) == 120
    agg = _aggregate(rows, sum(len(v) for v in groups.values()))
    summary = {
        'stage':'S11-A3',
        'status':'GLOBAL_PRIORITY_DECODER_A3_INTERVAL_NOT_PAPER_FINAL' if complete else 'DOMINANT_ONLY_DIAGNOSTIC',
        'scope':'complete S10 exact-three active-location model with <=2-priority / conservative order-3 full-recovery decoder',
        'decoder_semantics':[
            'Exact S8-C boundary <=2 full-syndrome singleton always has priority.',
            'Only syndromes absent from <=2 use the conservative order-3 extension.',
            'Order-3 extension = escape1 + S8-C boundary<=2, union arbitrary Steane sector + clean escape<=2.',
            'Distinct-location overlap is ignored only in the order-3 extension, which can create false ambiguity but not false singleton uniqueness.',
            '195-bit full recovery syndrome includes the future noiseless 24-detector audit round as decoder/scoring context; it is outside the factory fault budget.',
        ],
        'location_inventory': {k:len(v) for k,v in groups.items()} | {'total':sum(len(v) for v in groups.values())},
        'strata_total':120,
        'strata_evaluated':len(rows),
        'all_120_strata_evaluated':complete,
        'base_shots_per_stratum':int(args.base_shots_per_stratum),
        'dominant_shots_per_stratum':int(args.dominant_shots_per_stratum),
        'core_post3_shots':int(args.core_post3_shots),
        'dominant_keys':list(DOMINANT_KEYS),
        'aggregate':agg,
        'top_strata_by_lower_A3':[
            {'key':r['key'],'lower':r['A3_priority_resolved_lower_estimate'],'upper':r['A3_priority_unresolved_as_failure_upper_estimate'],'harmful':r['priority_resolved_harmful'],'unresolved':r['priority_unresolved_accepted']}
            for r in sorted(rows,key=lambda x:float(x['A3_priority_resolved_lower_estimate']),reverse=True)[:15]
        ],
        'top_strata_by_interval_width':[
            {'key':r['key'],'lower':r['A3_priority_resolved_lower_estimate'],'upper':r['A3_priority_unresolved_as_failure_upper_estimate'],'width':r['A3_priority_interval_width_estimate'],'unresolved':r['priority_unresolved_accepted']}
            for r in sorted(rows,key=lambda x:float(x['A3_priority_interval_width_estimate']),reverse=True)[:15]
        ],
        'elapsed_seconds':time.time()-t0,
        'next_gate':'If the global unresolved interval is still large, S11-A4 exactifies only the top ambiguity-width syndromes/strata. If narrow, use independent seeds and higher statistics for the paper-final A3 estimate.',
    }
    _dump(args.out_dir/'s11a3_strata.json', rows)
    _dump(args.out_dir/'s11a3_examples.json', all_examples)
    _dump(args.out_dir/'s11a3_summary.json', summary)
    print(json.dumps(_json_safe(summary), indent=2), flush=True)
    print('S11-A3 COMPLETE: upload s11a3_summary.json and s11a3_strata.json.', flush=True)


if __name__ == '__main__':
    main()
