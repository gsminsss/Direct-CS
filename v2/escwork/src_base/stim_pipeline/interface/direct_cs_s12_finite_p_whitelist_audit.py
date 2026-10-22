#!/usr/bin/env python3
"""S12: paired finite-p audit of the complete direct-|CS> factory.

This stage answers the performance question left open by S11:
  * What is the *actual finite-p* acceptance after the exact live-171 whitelist?
  * How much additional yield does the whitelist cost relative to the frozen
    S7-B4/S8-C live policy on the very same physical trajectories?
  * What logical-error bracket is obtained with the frozen priority <=2 / 
    conservative <=3 full-195 frame decoder?
  * Where does the cubic A3 p^3 / B3 p^3 asymptotic description cease to be
    quantitatively reliable?

Important separation of roles
-----------------------------
171 live bits: accept/reject only.
195 full bits: deferred frame-decoder context (171 present + ideal/noiseless 24-bit
future audit tail), exactly as in S8-C/S11.  This stage does not turn the 195-bit
record into an immediate live observable.

The stochastic model is the same complete S10 active-location local-Pauli model:
2946 physical locations per factory attempt, excluding idles/leakage/coherent
calibration/correlated multi-location CZZ mechanisms.
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError((name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s10mc = _load('s12_s10mc', HERE / 'direct_cs_s10_complete_physical_factory_mc.py')
a3 = _load('s12_a3', HERE / 'direct_cs_s11a3_priority_decoder_a3.py')
a5 = _load('s12_a5', HERE / 'direct_cs_s11a5v2_certified_live171_prefix_whitelist.py')

ACTUAL_BITS = 171
ACTUAL_MASK = (1 << ACTUAL_BITS) - 1
TOL = 1e-10


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


def _wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    ph = k / n
    d = 1 + z*z/n
    c = (ph + z*z/(2*n))/d
    h = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/d
    return [max(0.0, c-h), min(1.0, c+h)]


def _mean_se(sumx: float, sumx2: float, n: int) -> tuple[float, float | None]:
    if n <= 0:
        return 0.0, None
    m = float(sumx) / n
    if n <= 1:
        return m, None
    var = max(0.0, (float(sumx2) - n*m*m)/(n-1))
    return m, math.sqrt(var/n)


def _score(lv, rec, actual_a: int, actual_b: int, da, db) -> float | None:
    if da.predicted is None or db.predicted is None:
        return None
    corr = s10mc._apply(np.asarray(lv, complex), int(da.predicted)^int(actual_a), int(db.predicted)^int(actual_b))
    fid = float(abs(np.vdot(s10mc._target(rec), corr))**2)
    fid = min(1.0, max(0.0, fid))
    return 1.0 - fid


def _oracle_inf(lv, rec) -> float:
    return float(a3.a1._pauli_orbit_inf(np.asarray(lv, complex), s10mc._target(rec)))


def _decoder_inputs(hirano_src: Path, signature_p: float, bridge: dict[str, Any]):
    _paired, clean_analysis, _policy, _meta = a3.s8c._escape_catalogue(hirano_src, float(signature_p))
    clean_table = {int(k): int(v) for k, v in clean_analysis['state_table'].items()}
    single_effects = a3.a2._single_effects_from_bridge(bridge)
    sectors = list(bridge['sector'].values())
    return clean_table, single_effects, sectors


def _decode(full: int, bridge, clean_table, single_effects, sectors):
    return a3._decode_one(int(full), bridge['table'], clean_table, single_effects, sectors)


def _all_flags(full: int, bridge, clean_table, single_effects, sectors) -> int:
    return int(a3._all_candidate_flags(int(full), bridge['table'], clean_table, single_effects, sectors))


def _new_accumulator():
    return {
        'accepted': 0,
        'resolved': 0,
        'unresolved': 0,
        'resolved_harmful': 0,
        'frame_alias_harmful': 0,
        'hidden_state_harmful': 0,
        'sum_lower_attempt': 0.0,
        'sum2_lower_attempt': 0.0,
        'sum_upper_attempt': 0.0,
        'sum2_upper_attempt': 0.0,
        'sum_oracle_attempt': 0.0,
        'sum2_oracle_attempt': 0.0,
        'decoder_source_A': collections.Counter(),
        'decoder_source_B': collections.Counter(),
        'candidate_count_A': collections.Counter(),
        'candidate_count_B': collections.Counter(),
        'actual_frame_outside_decoder_union': 0,
    }


def _update_policy(acc, accepted: bool, fulla: int, fullb: int, aa: int, ab: int, lv, rec,
                   bridge, clean_table, single_effects, sectors):
    if not accepted:
        return
    acc['accepted'] += 1
    da = _decode(fulla, bridge, clean_table, single_effects, sectors)
    db = _decode(fullb, bridge, clean_table, single_effects, sectors)
    acc['decoder_source_A'][da.source] += 1
    acc['decoder_source_B'][db.source] += 1
    acc['candidate_count_A'][int(da.all_flags).bit_count()] += 1
    acc['candidate_count_B'][int(db.all_flags).bit_count()] += 1

    ua = _all_flags(fulla, bridge, clean_table, single_effects, sectors)
    ub = _all_flags(fullb, bridge, clean_table, single_effects, sectors)
    if not (ua & (1 << int(aa))) or not (ub & (1 << int(ab))):
        # At finite p this is allowed: >3-fault trajectories can lie outside the
        # truncated <=3 decoder model.  Record it as decoder coverage diagnostic.
        acc['actual_frame_outside_decoder_union'] += 1

    oracle = _oracle_inf(lv, rec)
    acc['sum_oracle_attempt'] += oracle
    acc['sum2_oracle_attempt'] += oracle*oracle

    inf = _score(lv, rec, aa, ab, da, db)
    if inf is None:
        acc['unresolved'] += 1
        # lower=0, upper=1 for this accepted trajectory
        acc['sum_upper_attempt'] += 1.0
        acc['sum2_upper_attempt'] += 1.0
        return

    acc['resolved'] += 1
    acc['sum_lower_attempt'] += inf
    acc['sum2_lower_attempt'] += inf*inf
    acc['sum_upper_attempt'] += inf
    acc['sum2_upper_attempt'] += inf*inf
    if inf > TOL:
        acc['resolved_harmful'] += 1
        frame_alias = (int(da.predicted) != int(aa)) or (int(db.predicted) != int(ab))
        acc['frame_alias_harmful'] += int(frame_alias)
        acc['hidden_state_harmful'] += int(not frame_alias)


def _finish_policy(acc, shots: int) -> dict[str, Any]:
    nacc = int(acc['accepted'])
    lo, lose = _mean_se(acc['sum_lower_attempt'], acc['sum2_lower_attempt'], shots)
    up, upse = _mean_se(acc['sum_upper_attempt'], acc['sum2_upper_attempt'], shots)
    oi, oise = _mean_se(acc['sum_oracle_attempt'], acc['sum2_oracle_attempt'], shots)
    clo = (acc['sum_lower_attempt']/nacc) if nacc else None
    cup = (acc['sum_upper_attempt']/nacc) if nacc else None
    coi = (acc['sum_oracle_attempt']/nacc) if nacc else None
    return {
        'accepted': nacc,
        'acceptance': nacc/shots,
        'acceptance_wilson95': _wilson(nacc, shots),
        'attempts_per_accepted_output': (shots/nacc if nacc else None),
        'resolved_accepted': int(acc['resolved']),
        'unresolved_accepted': int(acc['unresolved']),
        'resolved_fraction_among_accepted': (acc['resolved']/nacc if nacc else None),
        'resolved_harmful_samples': int(acc['resolved_harmful']),
        'frame_alias_harmful_samples': int(acc['frame_alias_harmful']),
        'hidden_state_harmful_samples': int(acc['hidden_state_harmful']),
        'logical_infidelity_per_attempt_lower': float(lo),
        'logical_infidelity_per_attempt_lower_naive_se': lose,
        'logical_infidelity_per_attempt_upper': float(up),
        'logical_infidelity_per_attempt_upper_naive_se': upse,
        'conditional_logical_infidelity_lower': (float(clo) if clo is not None else None),
        'conditional_logical_infidelity_upper': (float(cup) if cup is not None else None),
        'intrinsic_best_pauli_orbit_infidelity_per_attempt': float(oi),
        'intrinsic_best_pauli_orbit_infidelity_per_attempt_naive_se': oise,
        'conditional_intrinsic_best_pauli_orbit_infidelity': (float(coi) if coi is not None else None),
        'decoder_source_A': dict(sorted(acc['decoder_source_A'].items())),
        'decoder_source_B': dict(sorted(acc['decoder_source_B'].items())),
        'candidate_count_A': {str(k): int(v) for k,v in sorted(acc['candidate_count_A'].items())},
        'candidate_count_B': {str(k): int(v) for k,v in sorted(acc['candidate_count_B'].items())},
        'actual_frame_outside_decoder_union': int(acc['actual_frame_outside_decoder_union']),
    }


def _run_one(p: float, shots: int, seed: int, bridge, whitelist: set[int], clean_table, single_effects, sectors,
             asymptotic: dict[str, float] | None, max_examples: int):
    rng = np.random.default_rng(int(seed))
    old = _new_accumulator(); new = _new_accumulator()
    reasons = collections.Counter()
    front_acc = core_acc = 0
    old_live_acc = new_live_acc = new_rej = 0
    examples = []
    total_fault_sum = 0
    total_fault_max = 0
    fault_hist = collections.Counter()

    for i in range(int(shots)):
        fa = s10mc._front_block(p, rng); fb = s10mc._front_block(p, rng)
        nfa = int(fa['encoder_fault_count'] + fa['initial_ed_fault_count'])
        nfb = int(fb['encoder_fault_count'] + fb['initial_ed_fault_count'])
        if not fa['accepted'] or not fb['accepted']:
            reasons['initialED'] += 1
            total = nfa+nfb
            total_fault_sum += total; total_fault_max=max(total_fault_max,total); fault_hist[total]+=1
            continue
        front_acc += 1

        ok, why, rec, ss, lv, nfc, nma, nmb, ncat, nro, sc, fast = s10mc._sample_core_fullfront(p, rng, fa, fb)
        if not ok:
            reasons[why] += 1
            total = nfa+nfb+nfc+nma+nmb+ncat+nro
            total_fault_sum += total; total_fault_max=max(total_fault_max,total); fault_hist[total]+=1
            continue
        core_acc += 1
        sa, sb = ss

        af, afr, nea = s10mc._sample_escape(p, rng, bridge['sigs'], bridge['frames'])
        bf, bfr, neb = s10mc._sample_escape(p, rng, bridge['sigs'], bridge['frames'])
        total = nfa+nfb+nfc+nma+nmb+ncat+nro+nea+neb
        total_fault_sum += total; total_fault_max=max(total_fault_max,total); fault_hist[total]+=1

        sra = bridge['sector'][int(sa)]; srb = bridge['sector'][int(sb)]
        fulla = int(sra.full) ^ int(af); fullb = int(srb.full) ^ int(bf)
        pa = fulla & ACTUAL_MASK; pb = fullb & ACTUAL_MASK
        old_accept = pa not in bridge['unsafe'] and pb not in bridge['unsafe']
        if not old_accept:
            reasons['existing_live_policy'] += 1
            continue
        old_live_acc += 1

        new_accept = pa in whitelist and pb in whitelist
        if not new_accept:
            new_rej += 1
            reasons['new_live171_whitelist'] += 1
            if len(examples) < int(max_examples):
                examples.append({
                    'kind':'new_live171_whitelist_rejection','sample':i,
                    'prefix_A':hex(pa),'prefix_B':hex(pb),
                    'A_whitelisted':pa in whitelist,'B_whitelisted':pb in whitelist,
                    'full_A':hex(fulla),'full_B':hex(fullb),'total_fault_locations':int(total),
                })
        else:
            new_live_acc += 1

        aa = int(sra.frame) ^ int(afr); ab = int(srb.frame) ^ int(bfr)
        _update_policy(old, True, fulla, fullb, aa, ab, lv, rec, bridge, clean_table, single_effects, sectors)
        _update_policy(new, new_accept, fulla, fullb, aa, ab, lv, rec, bridge, clean_table, single_effects, sectors)

    oldr = _finish_policy(old, shots); newr = _finish_policy(new, shots)
    add_abs = new_rej/shots
    add_cond = new_rej/old_live_acc if old_live_acc else None
    row = {
        'p': float(p), 'shots': int(shots), 'seed': int(seed),
        'frontend_accepted': int(front_acc), 'frontend_acceptance': front_acc/shots,
        'core_midED_accepted': int(core_acc), 'core_midED_acceptance': core_acc/shots,
        'old_policy': oldr,
        'live171_whitelist_policy': newr,
        'paired_whitelist_effect': {
            'old_live_accepted': int(old_live_acc),
            'new_live_accepted': int(new_live_acc),
            'newly_rejected': int(new_rej),
            'additional_rejection_probability_per_attempt': float(add_abs),
            'additional_rejection_probability_wilson95': _wilson(new_rej, shots),
            'additional_rejection_fraction_among_old_live_accepted': (float(add_cond) if add_cond is not None else None),
            'acceptance_ratio_new_over_old': (new_live_acc/old_live_acc if old_live_acc else None),
        },
        'mean_total_failed_locations_per_attempt': total_fault_sum/shots,
        'max_total_failed_locations_seen': int(total_fault_max),
        'total_failed_location_histogram': {str(k):int(v) for k,v in sorted(fault_hist.items())},
        'rejection_reasons': dict(sorted(reasons.items())),
        'examples': examples,
    }
    if asymptotic is not None:
        a3lo = asymptotic['A3_lower']; a3up = asymptotic['A3_upper']; b3 = asymptotic['B3']
        pred_lo = a3lo*p**3; pred_up = a3up*p**3; pred_b = b3*p**3
        row['cubic_asymptotic_comparison'] = {
            'A3_lower_p3': float(pred_lo), 'A3_upper_p3': float(pred_up), 'B3_p3': float(pred_b),
            'observed_whitelist_logical_per_attempt_lower_over_A3lower_p3': (newr['logical_infidelity_per_attempt_lower']/pred_lo if pred_lo>0 else None),
            'observed_whitelist_logical_per_attempt_upper_over_A3upper_p3': (newr['logical_infidelity_per_attempt_upper']/pred_up if pred_up>0 else None),
            'observed_additional_rejection_over_B3p3': (add_abs/pred_b if pred_b>0 else None),
            'warning':'Ratios are meaningful only when direct finite-p event statistics are adequate; at low p logical events may be too rare.'
        }
    return row


def _asymptotic_from_summary(path: Path | None):
    if path is None or not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    a = d.get('postclosure_aggregate', d.get('aggregate', {}))
    try:
        return {
            'A3_lower': float(a['A3_lower_estimate']),
            'A3_upper': float(a['A3_upper_estimate']),
            'B3': float(a['B3_additional_rejection_estimate']),
            'source': str(path),
        }
    except Exception:
        return None


def _selftest():
    st10 = s10mc._selftest()
    st5 = a5._synthetic_self_test()
    # Ratio/interval arithmetic smoke test.
    assert _wilson(0, 100) is not None and _wilson(100, 100) is not None
    return {
        'passed': True,
        's10_complete_factory_selftest_passed': bool(st10['passed']),
        'live171_prefix_whitelist_selftest_passed': bool(st5['passed']),
        'modeled_locations_complete_factory': int(st10['modeled_locations_with_dual_escape']),
        'expected_modeled_locations': 2946,
        'paired_policy_semantics': 'same physical trajectory scored under frozen old live policy and exact <=2-derived 171-prefix whitelist',
        'frame_decoder_semantics': 'priority exact <=2 full-195; conservative <=3 extension only when <=2 absent; multi-frame/empty remains unresolved',
        'future_24_bits_are_live': False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hirano-src', type=Path)
    ap.add_argument('--signature-p', type=float, default=1e-3)
    ap.add_argument('--p-values', type=float, nargs='+', default=[1e-4,2e-4,5e-4,1e-3,2e-3])
    ap.add_argument('--shots', type=int, default=20000)
    ap.add_argument('--shots-per-p', type=int, nargs='+')
    ap.add_argument('--seed', type=int, default=20260828)
    ap.add_argument('--a6b-summary', type=Path)
    ap.add_argument('--max-examples-per-p', type=int, default=12)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--out-dir', type=Path, default=HERE/'s12_results')
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    st = _selftest(); _dump(args.out_dir/'s12_self_test.json', st)
    if args.self_test:
        print(json.dumps(st, indent=2)); print('S12 SELF-TEST PASS'); return
    if args.hirano_src is None:
        raise SystemExit('--hirano-src required unless --self-test')
    if args.shots <= 0:
        raise SystemExit('--shots must be >0')
    if args.shots_per_p is not None and len(args.shots_per_p) != len(args.p_values):
        raise SystemExit('--shots-per-p must have exactly one entry per --p-values item')

    shots_list = list(args.shots_per_p) if args.shots_per_p is not None else [int(args.shots)]*len(args.p_values)
    if any(int(x)<=0 for x in shots_list):
        raise SystemExit('all shots-per-p values must be >0')

    t0=time.time(); hirano=args.hirano_src.resolve()
    print('S12: building complete S10 boundary bridge...', flush=True)
    bridge=s10mc._bridge(hirano,float(args.signature_p))
    if int(bridge['meta']['physical_locations']) != 1028:
        raise AssertionError(('escape physical location count changed', bridge['meta']['physical_locations']))
    print('S12: constructing exact <=2-derived live-171 safe-prefix whitelist...', flush=True)
    whitelist, cert = a5._build_live171_whitelist(bridge)
    _dump(args.out_dir/'s12_live171_whitelist_certificate.json', cert)
    print(f"S12 whitelist: safe_prefixes={len(whitelist)} preserved_le2={cert['certified_safe_full_states_preserved']}", flush=True)
    print('S12: building frozen priority full-195 decoder tables...', flush=True)
    clean_table,single_effects,sectors=_decoder_inputs(hirano,float(args.signature_p),bridge)
    print(f'S12 decoder: boundary_le2={len(bridge["table"])} clean_escape_le2={len(clean_table)} single_effects={len(single_effects)} sectors={len(sectors)}', flush=True)

    asym=_asymptotic_from_summary(args.a6b_summary)
    if asym:
        print(f"S12 asymptotic reference: A3=[{asym['A3_lower']:.6g},{asym['A3_upper']:.6g}] B3={asym['B3']:.6g}", flush=True)

    ckdir=args.out_dir/'checkpoints'; ckdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(p,shots) in enumerate(zip(args.p_values,shots_list)):
        ck=ckdir/f'p_{i:02d}.json'
        if args.resume and ck.exists():
            row=json.loads(ck.read_text(encoding='utf-8')); rows.append(row)
            print(f'S12 resume p={p:g}: loaded checkpoint',flush=True); continue
        row=_run_one(float(p),int(shots),int(args.seed)+i,bridge,whitelist,clean_table,single_effects,sectors,asym,int(args.max_examples_per_p))
        rows.append(row); _dump(ck,row)
        print(f"S12 p={p:g}: old_acc={row['old_policy']['acceptance']:.6f} new_acc={row['live171_whitelist_policy']['acceptance']:.6f} add_rej={row['paired_whitelist_effect']['additional_rejection_probability_per_attempt']:.6f} new_PL=[{row['live171_whitelist_policy']['logical_infidelity_per_attempt_lower']:.6g},{row['live171_whitelist_policy']['logical_infidelity_per_attempt_upper']:.6g}] unresolved={row['live171_whitelist_policy']['unresolved_accepted']}",flush=True)

    _dump(args.out_dir/'s12_sweep.json',rows)
    summary={
        'stage':'S12',
        'status':'FINITE_P_PAIRED_FACTORY_AUDIT',
        'scope':'complete S10 2946-location active-location stochastic-Pauli factory; frozen S7-B4/S8-C old live policy versus exact <=2-derived live-171 prefix whitelist on paired trajectories',
        'p_values':[float(x) for x in args.p_values],
        'shots_per_p':[int(x) for x in shots_list],
        'seed':int(args.seed),
        'signature_p':float(args.signature_p),
        'exact_whitelist_certificate':cert,
        'asymptotic_reference':asym,
        'rows_compact':[{
            'p':r['p'],'shots':r['shots'],
            'old_acceptance':r['old_policy']['acceptance'],
            'whitelist_acceptance':r['live171_whitelist_policy']['acceptance'],
            'additional_whitelist_rejection_per_attempt':r['paired_whitelist_effect']['additional_rejection_probability_per_attempt'],
            'whitelist_attempts_per_accepted_output':r['live171_whitelist_policy']['attempts_per_accepted_output'],
            'whitelist_logical_per_attempt_lower':r['live171_whitelist_policy']['logical_infidelity_per_attempt_lower'],
            'whitelist_logical_per_attempt_upper':r['live171_whitelist_policy']['logical_infidelity_per_attempt_upper'],
            'whitelist_conditional_logical_lower':r['live171_whitelist_policy']['conditional_logical_infidelity_lower'],
            'whitelist_conditional_logical_upper':r['live171_whitelist_policy']['conditional_logical_infidelity_upper'],
            'whitelist_unresolved':r['live171_whitelist_policy']['unresolved_accepted'],
            'old_unresolved':r['old_policy']['unresolved_accepted'],
            'cubic_comparison':r.get('cubic_asymptotic_comparison'),
        } for r in rows],
        'interpretation_rules':[
            'Acceptance/yield numbers are direct finite-p Monte Carlo observables under the modeled local active-location Pauli process.',
            'Logical lower/upper use the frozen full-195 priority <=2 / conservative <=3 decoder; unresolved accepted samples contribute 0 to lower and 1 to upper.',
            'The exact <=2 live-171 whitelist safety certificate is unchanged by finite-p sampling.',
            'A3 p^3 and B3 p^3 are asymptotic references, not probabilities guaranteed accurate at p where the cubic terms are no longer small.',
            'The final 24 bits of the 195-bit decoder signature remain ideal/deferred future-tail context and are not available to the present-time 171-bit accept/reject rule.',
        ],
        'noise_model_limits':['idle errors','leakage','coherent/calibration errors','correlated multi-location CZZ faults beyond local stochastic Pauli'],
        'elapsed_seconds':time.time()-t0,
    }
    _dump(args.out_dir/'s12_summary.json',summary)
    print('S12 COMPLETE: upload s12_summary.json and s12_sweep.json for finite-p interpretation and resource-baseline transition.',flush=True)


if __name__=='__main__':
    main()
