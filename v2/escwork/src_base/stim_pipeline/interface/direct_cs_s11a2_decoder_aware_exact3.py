#!/usr/bin/env python3
"""S11-A2: decoder-aware exact-three-fault analysis for the complete S10 factory.

Scientific purpose
------------------
S11-A1 found that the leading exact-three coefficient obtained with an identity
fallback is dominated by ESC_A3/ESC_B3, while the post-core state remains in the
two-logical-qubit Pauli orbit to numerical precision.  This stage separates
three distinct questions:

1) Are pure escape triples frame-resolvable from the full recovery syndrome if
   the decoder is allowed to use one future noiseless audit round (the same
   195-bit recovery record already used for scoring in S7-B3/S7-B5)?
2) How much of the dominant ESC_A3/ESC_B3 identity-fallback coefficient survives
   a conservative order<=3 syndrome-frame resolver?
3) What is the hidden intrinsic/logical-Pauli contribution of CORE_POST3, which
   the 200-shot pilot can easily miss despite known deterministic malignant
   triple witnesses?

Decoder semantics
-----------------
For each escape block, build the exact clean <=2 full-recovery syndrome -> frame
set table from the validated k=2 live-handoff catalogue.  For a target full
syndrome S, conservatively extend the candidate frame set to order <=3 via

    union_g  frame(g) XOR Frames_le2[S XOR syndrome(g)]

where g ranges over all distinct single-fault syndrome/frame effects.  Because
location-overlap constraints are intentionally NOT used in this extension, the
candidate set is an over-approximation: a singleton is certifiably unique, while
an ambiguous set may be a false ambiguity.  Therefore this decoder never makes
an optimistic order-3 uniqueness claim.

This stage is still not the final paper A3.  It is a decoder-calibration and
adaptive-statistics stage.  A later S11-A3 can exactify any remaining ambiguous
dominant syndromes and combine the mixed CORE_POST+ESC strata.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError((name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a1 = _load('s11a2_a1', HERE / 'direct_cs_s11a1_exact3_stratified_pilot.py')
s8c = _load('s11a2_s8c', HERE / 'zxz_core_sector_escape_composability.py')

TOL = 1e-10
FRAME_LABELS = ('I', 'X', 'Z', 'Y')


def _frame_list(flags: int) -> list[int]:
    return [i for i in range(4) if int(flags) & (1 << i)]


def _singleton(flags: int) -> int | None:
    f = int(flags)
    if f and f.bit_count() == 1:
        return f.bit_length() - 1
    return None


def _combine_flags(base_flags: int, frame: int) -> int:
    out = 0
    for f in _frame_list(base_flags):
        out |= 1 << (int(f) ^ int(frame))
    return out


def _single_effects_from_bridge(bridge: dict[str, Any]) -> list[tuple[int, int]]:
    # Duplicate classes with the same full syndrome/frame need only be queried once.
    s = set()
    for lid in range(len(bridge['sigs'])):
        for syn, fr in zip(bridge['sigs'][lid], bridge['frames'][lid]):
            s.add((int(syn), int(fr)))
    return sorted(s)


def _le3_candidate_flags(full_syndrome: int, le2_table: dict[int, int], single_effects: list[tuple[int, int]]) -> int:
    """Conservative order<=3 frame set from full recovery syndrome.

    Ignoring distinct-location constraints can only ADD candidate frames, so a
    singleton result is safe.  This intentionally favors false ambiguity over
    false uniqueness.
    """
    target = int(full_syndrome)
    flags = int(le2_table.get(target, 0))
    for syn, fr in single_effects:
        rem = int(le2_table.get(target ^ int(syn), 0))
        if rem:
            flags |= _combine_flags(rem, int(fr))
            if flags == 0b1111:
                break
    return flags


def _find_stratum(groups, key: str):
    for st in a1._all_strata(groups):
        if st['key'] == key:
            return st
    raise KeyError(key)


def _residual_inf_one_block(record, actual_frame: int, predicted_frame: int, block: int) -> float:
    target = a1._target(record)
    fa = (int(actual_frame) ^ int(predicted_frame)) if block == 0 else 0
    fb = (int(actual_frame) ^ int(predicted_frame)) if block == 1 else 0
    corr = a1.s10mc._apply(target, fa, fb)
    return a1._phase_inf(corr, target)


def _mean_se(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    arr = np.asarray(xs, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, 0.0
    return mean, float(arr.std(ddof=1) / math.sqrt(len(arr)))


def _audit_escape3(key: str, shots: int, seed: int, groups, bridge, clean_table, single_effects):
    if key not in ('ESC_A3', 'ESC_B3'):
        raise ValueError(key)
    st = _find_stratum(groups, key)
    block = 0 if key == 'ESC_A3' else 1
    rng = np.random.default_rng(seed)
    accepted = 0
    unique = 0
    ambiguous = 0
    empty = 0
    unique_mismatch = 0
    actual_not_in_candidates = 0
    id_infs: list[float] = []
    dec_infs: list[float] = []
    lower_infs: list[float] = []
    upper_infs: list[float] = []
    cand_hist = collections.Counter()
    examples = []

    for i in range(shots):
        locs = a1._sample_locations(st, groups, rng)
        r = a1._simulate_exact3(locs, bridge, rng, want_trace=True)
        id_infs.append(float(r.get('identity_infidelity', 0.0)) if r['accepted'] else 0.0)
        if not r['accepted']:
            dec_infs.append(0.0); lower_infs.append(0.0); upper_infs.append(0.0)
            continue
        accepted += 1
        full = int(r['full_syndrome_A_hex'], 16) if block == 0 else int(r['full_syndrome_B_hex'], 16)
        actual = int(r['actual_frame_A']) if block == 0 else int(r['actual_frame_B'])
        flags = _le3_candidate_flags(full, clean_table, single_effects)
        cands = _frame_list(flags)
        cand_hist[len(cands)] += 1
        if not (flags & (1 << actual)):
            actual_not_in_candidates += 1
            if len(examples) < 20:
                examples.append({'kind':'actual_not_in_candidates','sample':i,'full_syndrome_hex':hex(full),'actual_frame':FRAME_LABELS[actual],'candidate_frames':[FRAME_LABELS[x] for x in cands],'trace':r.get('trace')})
        pred = _singleton(flags)
        if pred is None:
            if flags == 0: empty += 1
            else: ambiguous += 1
            # no optimistic assignment: lower=0, upper=1 for accepted unresolved
            dec_infs.append(0.0); lower_infs.append(0.0); upper_infs.append(1.0)
            continue
        unique += 1
        inf = _residual_inf_one_block(r['record'], actual, pred, block)
        dec_infs.append(inf); lower_infs.append(inf); upper_infs.append(inf)
        if inf > TOL:
            unique_mismatch += 1
            if len(examples) < 20:
                examples.append({'kind':'unique_mismatch','sample':i,'full_syndrome_hex':hex(full),'actual_frame':FRAME_LABELS[actual],'predicted_frame':FRAME_LABELS[pred],'candidate_frames':[FRAME_LABELS[x] for x in cands],'infidelity':inf,'trace':r.get('trace')})

    ways = int(st['location_triples'])
    mid, seid = _mean_se(id_infs)
    mdec, sedec = _mean_se(dec_infs)
    ml, sel = _mean_se(lower_infs)
    mu, seu = _mean_se(upper_infs)
    return {
        'stratum': key,
        'location_triples': ways,
        'shots': shots,
        'accepted': accepted,
        'accepted_fraction': accepted/shots if shots else None,
        'conservative_le3_unique_accepted': unique,
        'conservative_le3_ambiguous_accepted': ambiguous,
        'conservative_le3_empty_candidate_accepted': empty,
        'conservative_le3_resolution_fraction_among_accepted': unique/accepted if accepted else None,
        'conservative_le3_unique_frame_mismatches': unique_mismatch,
        'actual_frame_not_in_candidate_set': actual_not_in_candidates,
        'candidate_set_size_histogram': {str(k):int(v) for k,v in sorted(cand_hist.items())},
        'A3_identity_fallback_estimate': ways*mid,
        'A3_identity_fallback_se': ways*seid,
        'A3_conservative_le3_resolved_lower_estimate': ways*ml,
        'A3_conservative_le3_resolved_lower_se': ways*sel,
        'A3_conservative_le3_unresolved_as_failure_upper_estimate': ways*mu,
        'A3_conservative_le3_unresolved_as_failure_upper_se': ways*seu,
        'examples': examples,
    }


def _audit_core_post3(shots: int, seed: int, groups, bridge, max_examples: int):
    key = 'CORE_POST3'
    st = _find_stratum(groups, key)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    accepted = 0
    harmful = 0
    legacy_resolved = 0
    harmful_examples = []
    syndrome_hist = collections.Counter()
    sector_hist = collections.Counter()

    for i in range(shots):
        locs = a1._sample_locations(st, groups, rng)
        r = a1._simulate_exact3(locs, bridge, rng, want_trace=True)
        x = float(r.get('identity_infidelity', 0.0)) if r['accepted'] else 0.0
        vals.append(x)
        if not r['accepted']:
            continue
        accepted += 1
        legacy_resolved += int(bool(r.get('legacy_resolved')))
        syndrome_hist[(r.get('full_syndrome_A_hex'), r.get('full_syndrome_B_hex'))] += 1
        sector_hist[tuple(r.get('sector_pair') or [])] += 1
        if x > TOL:
            harmful += 1
            if len(harmful_examples) < max_examples:
                harmful_examples.append({
                    'sample': i,
                    'identity_infidelity': x,
                    'record': r.get('record'),
                    'sector_pair': r.get('sector_pair'),
                    'full_syndrome_A_hex': r.get('full_syndrome_A_hex'),
                    'full_syndrome_B_hex': r.get('full_syndrome_B_hex'),
                    'predicted_frame_A': r.get('predicted_frame_A'),
                    'predicted_frame_B': r.get('predicted_frame_B'),
                    'actual_frame_A': r.get('actual_frame_A'),
                    'actual_frame_B': r.get('actual_frame_B'),
                    'trace': r.get('trace'),
                })

    ways = int(st['location_triples'])
    mean, se = _mean_se(vals)
    return {
        'stratum': key,
        'location_triples': ways,
        'shots': shots,
        'accepted': accepted,
        'accepted_fraction': accepted/shots if shots else None,
        'legacy_resolved_accepted': legacy_resolved,
        'harmful_accepted': harmful,
        'harmful_fraction_per_random_location_triple': harmful/shots if shots else None,
        'mean_accepted_logical_infidelity_per_random_location_triple': mean,
        'A3_core_post3_identity_infidelity_estimate': ways*mean,
        'A3_core_post3_identity_infidelity_se': ways*se,
        'top_sector_pairs': [[list(k),int(v)] for k,v in sector_hist.most_common(12)],
        'harmful_examples': harmful_examples,
    }


def _self_test():
    # synthetic <=2 table and single effects; singleton uniqueness must be stable.
    le2 = {0:1<<0, 0b001:1<<1, 0b010:1<<2, 0b011:1<<3}
    singles = [(0b100,1)]
    # S=101 => single 100 + rem 001 => X xor X = I.
    f = _le3_candidate_flags(0b101, le2, singles)
    assert f == (1<<0), f
    # An explicitly ambiguous <=2 remainder remains ambiguous after extension.
    le2b = dict(le2); le2b[0b001] = (1<<0)|(1<<1)
    f2 = _le3_candidate_flags(0b101, le2b, singles)
    assert f2.bit_count() == 2, f2
    groups = a1._build_groups(1028)
    assert _find_stratum(groups,'ESC_A3')['location_triples'] == math.comb(1028,3)
    assert _find_stratum(groups,'CORE_POST3')['location_triples'] == math.comb(161,3)
    return {'passed':True,'synthetic_unique_flags':f,'synthetic_ambiguous_flags':f2,'ESC_A3_location_triples':math.comb(1028,3),'CORE_POST3_location_triples':math.comb(161,3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hirano-src', type=Path)
    ap.add_argument('--signature-p', type=float, default=1e-3)
    ap.add_argument('--escape-shots', type=int, default=5000)
    ap.add_argument('--core-post3-shots', type=int, default=50000)
    ap.add_argument('--seed', type=int, default=20260821)
    ap.add_argument('--max-examples', type=int, default=30)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--out-dir', type=Path, default=HERE/'s11a2_results')
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    st = _self_test()
    (args.out_dir/'s11a2_self_test.json').write_text(json.dumps(st,indent=2),encoding='utf-8')
    if args.self_test:
        print(json.dumps(st,indent=2)); print('S11-A2 SELF-TEST PASS'); return
    if args.hirano_src is None:
        raise SystemExit('--hirano-src required unless --self-test')

    t0=time.time()
    print('S11-A2: building S10 boundary bridge...', flush=True)
    bridge = a1.s10mc._bridge(args.hirano_src.resolve(), float(args.signature_p))
    print('S11-A2: rebuilding exact clean escape <=2 table for decoder calibration...', flush=True)
    _paired, clean_analysis, _clean_policy, _meta = s8c._escape_catalogue(args.hirano_src.resolve(), float(args.signature_p))
    clean_table = {int(k):int(v) for k,v in clean_analysis['state_table'].items()}
    single_effects = _single_effects_from_bridge(bridge)
    groups = a1._build_groups(int(bridge['meta']['physical_locations']))
    print(f'S11-A2 decoder catalogue: clean_le2_syndromes={len(clean_table)}, unique_single_effects={len(single_effects)}', flush=True)

    ea = _audit_escape3('ESC_A3', int(args.escape_shots), int(args.seed), groups, bridge, clean_table, single_effects)
    print(f"S11-A2 ESC_A3: accepted={ea['accepted']}, le3_resolved={ea['conservative_le3_unique_accepted']}, mismatch={ea['conservative_le3_unique_frame_mismatches']}", flush=True)
    eb = _audit_escape3('ESC_B3', int(args.escape_shots), int(args.seed)+1, groups, bridge, clean_table, single_effects)
    print(f"S11-A2 ESC_B3: accepted={eb['accepted']}, le3_resolved={eb['conservative_le3_unique_accepted']}, mismatch={eb['conservative_le3_unique_frame_mismatches']}", flush=True)
    cp = _audit_core_post3(int(args.core_post3_shots), int(args.seed)+2, groups, bridge, int(args.max_examples))
    print(f"S11-A2 CORE_POST3: accepted={cp['accepted']}, harmful={cp['harmful_accepted']}, A3={cp['A3_core_post3_identity_infidelity_estimate']:.6g}", flush=True)

    if ea['actual_frame_not_in_candidate_set'] or eb['actual_frame_not_in_candidate_set']:
        raise AssertionError(('candidate decoder lost an actually reachable frame', ea['actual_frame_not_in_candidate_set'], eb['actual_frame_not_in_candidate_set']))
    if ea['conservative_le3_unique_frame_mismatches'] or eb['conservative_le3_unique_frame_mismatches']:
        raise AssertionError(('conservative singleton decoder mismatch', ea['conservative_le3_unique_frame_mismatches'], eb['conservative_le3_unique_frame_mismatches']))

    summary = {
        'stage':'S11-A2',
        'status':'DECODER_CALIBRATION_NOT_PAPER_FINAL_A3',
        'scope':'complete S10 factory exact-three diagnostics; conservative full-recovery order<=3 frame resolver for pure escape triples + targeted CORE_POST3 statistics',
        'decoder_observation':'195-bit full recovery syndrome = 171 live-handoff bits + one future noiseless 24-detector audit round; future round is scoring/decoder context, not factory fault budget',
        'conservative_decoder_note':'distinct-location legality is ignored in the le3 extension; singleton => certified unique, ambiguity may be false ambiguity',
        'clean_le2_syndrome_count':len(clean_table),
        'unique_single_syndrome_frame_effects':len(single_effects),
        'escape_A3':ea,
        'escape_B3':eb,
        'core_post3':cp,
        'elapsed_seconds':time.time()-t0,
        'next_decision':[
            'If ESC_A3/B3 conservative le3 resolution is high, remove identity-fallback A3 as a decoder artifact and exactify only remaining ambiguous dominant syndromes.',
            'Use CORE_POST3 targeted estimate to quantify the genuinely undetectable logical-Pauli triple coefficient.',
            'Then extend the same decoder-aware analysis to CORE_POST1+ESC2 and CORE_POST2+ESC1 mixed strata before quoting paper-final A3.'
        ],
    }
    (args.out_dir/'s11a2_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (args.out_dir/'s11a2_escapeA3.json').write_text(json.dumps(ea,indent=2),encoding='utf-8')
    (args.out_dir/'s11a2_escapeB3.json').write_text(json.dumps(eb,indent=2),encoding='utf-8')
    (args.out_dir/'s11a2_core_post3.json').write_text(json.dumps(cp,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2), flush=True)
    print('S11-A2 COMPLETE: upload s11a2_summary.json first.', flush=True)

if __name__=='__main__':
    main()
