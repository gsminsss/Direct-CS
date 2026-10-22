#!/usr/bin/env python3
"""S11-A6: global independent-seed high-statistics A3/B3 audit.

Purpose
-------
S11-A5 v2 certified the *policy*: live 171-bit prefixes are ACCEPTed iff they
occur among exact accepted <=2 boundary states.  This preserves all certified
<=2 accepted states exactly, while frame recovery remains a 195-bit decoder.

A5's residual A3 interval was already narrow, but its Monte-Carlo standard
error was larger than the remaining decoder interval and its B3 rejection
estimate was computed only on the 12 A3-active pilot strata.  A6 therefore:

1. keeps the exact A5 live-171 prefix whitelist unchanged;
2. fixes a sampling plan *before* drawing A6 samples, using only old A5 pilot
   statistics, so the A6 estimator is independent of the pilot;
3. samples **all 120 exact-three location strata**, with a nonzero discovery
   floor on previously inactive strata;
4. uses Neyman-like allocation on the old A5-active strata to target a desired
   aggregate A3 standard error;
5. reports A3, the full-factory order-3 additional-rejection coefficient B3,
   unresolved full-195 width, best-Pauli-orbit coefficient, and naive normal
   confidence intervals.

This is a statistical performance audit, not an additional exact distance
certificate.  The exact <=2/no-new-rejection certificate comes from A5.
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
FRAME_LABELS = ('I','X','Z','Y')
EXPECTED_ESCAPE_LOCATIONS = 1028
EXPECTED_FACTORY_LOCATIONS = 2946


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError((name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a5 = _load('s11a6_a5', HERE / 'direct_cs_s11a5v2_certified_live171_prefix_whitelist.py')
a1 = a5.a1
a3 = a5.a3


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(a5._json_safe(obj), indent=2), encoding='utf-8')


def _mean_se(xs: list[float]) -> tuple[float,float]:
    if not xs:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=float)
    m = float(a.mean())
    if len(a) <= 1:
        return m, 0.0
    return m, float(a.std(ddof=1)/math.sqrt(len(a)))


def _normal95(mean: float, se: float, nonnegative: bool=True) -> list[float]:
    lo = float(mean - 1.959963984540054*se)
    hi = float(mean + 1.959963984540054*se)
    if nonnegative:
        lo = max(0.0, lo)
    return [lo, hi]


def _pilot_c(row: dict[str,Any]) -> float:
    n = max(1, int(row.get('shots', 0)))
    se = float(row.get('A3_whitelist_resolved_lower_naive_se', 0.0))
    return max(0.0, se*math.sqrt(n))


def _build_fixed_plan(
    strata: list[dict[str,Any]],
    pilot_active: list[dict[str,Any]],
    target_a3_se: float,
    inactive_floor: int,
    active_floor: int,
    max_shots: int,
) -> tuple[list[dict[str,Any]], dict[str,Any]]:
    """Neyman-like fixed plan determined solely from pre-A6 pilot data."""
    bypilot = {r['key']:r for r in pilot_active}
    active_keys = set(bypilot)
    c = {k:_pilot_c(r) for k,r in bypilot.items()}
    sumc = sum(c.values())
    ideal_active_total = int(math.ceil((sumc/max(float(target_a3_se),1e-12))**2)) if sumc else active_floor*len(active_keys)
    ideal_active_total = max(ideal_active_total, active_floor*len(active_keys))
    residual = max(0, ideal_active_total - active_floor*len(active_keys))

    alloc: dict[str,int] = {}
    for s in strata:
        k=s['key']
        if k in active_keys:
            extra = int(round(residual*c.get(k,0.0)/sumc)) if sumc else 0
            alloc[k] = min(max_shots, active_floor + extra)
        else:
            alloc[k] = int(inactive_floor)

    # The continuous Neyman total ignores the distortion introduced by per-stratum
    # floors.  Increase only the active allocations until the pilot-predicted SE
    # actually meets the requested target (or all active strata hit the cap).
    def _expected_var() -> float:
        return sum(cv*cv/max(1,alloc[k]) for k,cv in c.items())
    expected_var = _expected_var()
    for _ in range(8):
        expected_se = math.sqrt(expected_var)
        if expected_se <= float(target_a3_se)*(1+1e-12):
            break
        scale = (expected_se/max(float(target_a3_se),1e-12))**2
        changed = False
        for k in active_keys:
            newn = min(int(max_shots), max(int(active_floor), int(math.ceil(alloc[k]*scale))))
            if newn > alloc[k]:
                alloc[k] = newn; changed = True
        expected_var = _expected_var()
        if not changed:
            break
    plan_rows=[]
    for i,s in enumerate(strata):
        k=s['key']
        plan_rows.append({
            'index':i,
            'key':k,
            'counts':{a1.GROUPS[j]:int(v) for j,v in enumerate(s['counts']) if v},
            'location_triples':int(s['location_triples']),
            'shots':int(alloc[k]),
            'pilot_active':bool(k in active_keys),
            'pilot_A3':float(bypilot[k]['A3_whitelist_resolved_lower_estimate']) if k in bypilot else 0.0,
            'pilot_A3_se':float(bypilot[k]['A3_whitelist_resolved_lower_naive_se']) if k in bypilot else 0.0,
        })
    meta={
        'target_a3_se_from_pilot':float(target_a3_se),
        'inactive_discovery_floor_shots':int(inactive_floor),
        'active_floor_shots':int(active_floor),
        'max_shots_per_stratum':int(max_shots),
        'pilot_active_count':len(active_keys),
        'global_strata_count':len(strata),
        'planned_total_shots':sum(r['shots'] for r in plan_rows),
        'planned_inactive_total_shots':sum(r['shots'] for r in plan_rows if not r['pilot_active']),
        'planned_active_total_shots':sum(r['shots'] for r in plan_rows if r['pilot_active']),
        'pilot_predicted_A3_se_on_active_strata':float(math.sqrt(expected_var)),
        'allocation_fixed_before_A6_sampling':True,
        'A6_estimator_uses_only_A6_samples':True,
    }
    return plan_rows, meta


def _audit(meta: dict[str,Any], shots: int, seed: int, groups, bridge, whitelist: set[int],
           clean_table, single_effects, sectors, max_examples: int) -> dict[str,Any]:
    rng=np.random.default_rng(int(seed))
    ways=int(meta['location_triples'])
    mask=(1<<a1.ACTUAL_BITS)-1
    boundary=bridge['table']

    lower=[]; upper=[]; oracle=[]; newrej=[]
    accepted_pre=accepted_post=rejected_new=0
    resolved=unresolved=harmful=0
    alias=hidden=0
    reasons=collections.Counter()
    examples=[]; unresolved_examples=[]

    for i in range(int(shots)):
        locs=a1._sample_locations(meta, groups, rng)
        r=a1._simulate_exact3(locs, bridge, rng, want_trace=False, want_state=True)
        reasons[r['reason']]+=1
        if not r['accepted']:
            lower.append(0.0); upper.append(0.0); oracle.append(0.0); newrej.append(0.0)
            continue

        accepted_pre+=1
        fa=int(r['full_syndrome_A_hex'],16); fb=int(r['full_syndrome_B_hex'],16)
        pa=fa&mask; pb=fb&mask
        if pa not in whitelist or pb not in whitelist:
            rejected_new+=1
            newrej.append(1.0)
            lower.append(0.0); upper.append(0.0); oracle.append(0.0)
            if len(examples)<max_examples:
                examples.append({'kind':'new_live171_prefix_whitelist_rejection','sample':i,'stratum':meta['key'],
                                 'prefix_A':hex(pa),'prefix_B':hex(pb),'A_whitelisted':pa in whitelist,
                                 'B_whitelisted':pb in whitelist,'full_A':hex(fa),'full_B':hex(fb)})
            continue
        newrej.append(0.0)
        accepted_post+=1
        da=a3._decode_one(fa,boundary,clean_table,single_effects,sectors)
        db=a3._decode_one(fb,boundary,clean_table,single_effects,sectors)
        lv=np.asarray(r['_logical_vector'],dtype=complex)
        target=a1._target(r['record'])
        orc=float(a1._pauli_orbit_inf(lv,target))
        oracle.append(orc)
        if da.predicted is None or db.predicted is None:
            unresolved+=1
            lower.append(0.0); upper.append(1.0)
            ue={'kind':'retained_safe_prefix_full195_unresolved','sample':i,'stratum':meta['key'],
                'prefix_A':hex(pa),'prefix_B':hex(pb),'full_A':hex(fa),'full_B':hex(fb),
                'source_A':da.source,'source_B':db.source,
                'candidates_A':[FRAME_LABELS[x] for x in a3._frame_list(da.all_flags)],
                'candidates_B':[FRAME_LABELS[x] for x in a3._frame_list(db.all_flags)]}
            unresolved_examples.append(ue)
            if len(examples)<max_examples: examples.append(ue)
            continue

        resolved+=1
        aa=int(r['actual_frame_A']); ab=int(r['actual_frame_B'])
        corr=a1.s10mc._apply(lv,int(da.predicted)^aa,int(db.predicted)^ab)
        inf=float(a1._phase_inf(corr,target))
        lower.append(inf); upper.append(inf)
        if inf>TOL:
            harmful+=1
            is_alias=(int(da.predicted)!=aa) or (int(db.predicted)!=ab)
            if is_alias: alias+=1
            else: hidden+=1
            if len(examples)<max_examples:
                examples.append({'kind':'residual_harmful_safe_prefix_collision','sample':i,'stratum':meta['key'],
                                 'infidelity':inf,'prefix_A':hex(pa),'prefix_B':hex(pb),
                                 'full_A':hex(fa),'full_B':hex(fb),'decoder_source_A':da.source,
                                 'decoder_source_B':db.source,'pred_A':FRAME_LABELS[int(da.predicted)],
                                 'pred_B':FRAME_LABELS[int(db.predicted)],'actual_A':FRAME_LABELS[aa],
                                 'actual_B':FRAME_LABELS[ab],'frame_alias':bool(is_alias),
                                 'intrinsic_pauli_orbit_infidelity':orc})

    lm,lse=_mean_se(lower); um,use=_mean_se(upper); om,ose=_mean_se(oracle); rm,rse=_mean_se(newrej)
    return {
        'key':meta['key'],
        'counts':{a1.GROUPS[i]:int(k) for i,k in enumerate(meta['counts']) if k},
        'location_triples':ways,'shots':int(shots),'seed':int(seed),
        'accepted_before_whitelist':accepted_pre,'new_whitelist_rejections':rejected_new,
        'accepted_after_whitelist':accepted_post,
        'retained_resolved':resolved,'retained_unresolved':unresolved,
        'residual_harmful':harmful,'residual_frame_alias_harmful':alias,
        'residual_hidden_state_harmful':hidden,
        'A3_lower_estimate':float(ways*lm),'A3_lower_se':float(ways*lse),
        'A3_upper_estimate':float(ways*um),'A3_upper_se':float(ways*use),
        'A3_unresolved_width':float(ways*(um-lm)),
        'B3_additional_rejection_estimate':float(ways*rm),'B3_additional_rejection_se':float(ways*rse),
        'A3_best_pauli_orbit_estimate':float(ways*om),'A3_best_pauli_orbit_se':float(ways*ose),
        'rejection_reasons_before_whitelist':dict(sorted(reasons.items())),
        'examples':examples,'unresolved_examples':unresolved_examples,
    }


def _aggregate(rows: list[dict[str,Any]]) -> dict[str,Any]:
    low=sum(r['A3_lower_estimate'] for r in rows); sel=math.sqrt(sum(r['A3_lower_se']**2 for r in rows))
    up=sum(r['A3_upper_estimate'] for r in rows); seu=math.sqrt(sum(r['A3_upper_se']**2 for r in rows))
    b3=sum(r['B3_additional_rejection_estimate'] for r in rows); seb=math.sqrt(sum(r['B3_additional_rejection_se']**2 for r in rows))
    orb=sum(r['A3_best_pauli_orbit_estimate'] for r in rows); seo=math.sqrt(sum(r['A3_best_pauli_orbit_se']**2 for r in rows))
    return {
        'A3_lower_estimate':float(low),'A3_lower_naive_se':float(sel),'A3_lower_naive_95pct':_normal95(low,sel),
        'A3_upper_estimate':float(up),'A3_upper_naive_se':float(seu),'A3_upper_naive_95pct':_normal95(up,seu),
        'A3_unresolved_interval_width':float(up-low),
        'A3_combined_naive_95pct_including_unresolved':[max(0.0,low-1.959963984540054*sel), up+1.959963984540054*seu],
        'B3_additional_rejection_estimate':float(b3),'B3_additional_rejection_naive_se':float(seb),
        'B3_additional_rejection_naive_95pct':_normal95(b3,seb),
        'A3_best_pauli_orbit_estimate':float(orb),'A3_best_pauli_orbit_naive_se':float(seo),
        'A3_best_pauli_orbit_naive_95pct':_normal95(orb,seo),
        'total_shots':sum(int(r['shots']) for r in rows),
        'accepted_before_whitelist':sum(int(r['accepted_before_whitelist']) for r in rows),
        'new_whitelist_rejections':sum(int(r['new_whitelist_rejections']) for r in rows),
        'accepted_after_whitelist':sum(int(r['accepted_after_whitelist']) for r in rows),
        'retained_unresolved_samples':sum(int(r['retained_unresolved']) for r in rows),
        'residual_harmful_samples':sum(int(r['residual_harmful']) for r in rows),
        'residual_frame_alias_harmful_samples':sum(int(r['residual_frame_alias_harmful']) for r in rows),
        'residual_hidden_state_harmful_samples':sum(int(r['residual_hidden_state_harmful']) for r in rows),
        'p3_predictions_lower_upper':{str(p):[float(low*p**3),float(up*p**3)] for p in (2e-4,5e-4,1e-3)},
        'p3_predictions_additional_rejection':{str(p):float(b3*p**3) for p in (2e-4,5e-4,1e-3)},
    }


def _self_test(pilot_active_path: Path|None=None):
    synthetic=a5._synthetic_self_test()
    assert synthetic['passed']
    # allocation arithmetic with synthetic pilot rows
    fake_strata=[]
    for i in range(120):
        counts=[0]*len(a1.GROUPS); counts[0]=3
        fake_strata.append({'key':f'K{i}','counts':tuple(counts),'location_triples':1000+i})
    pilot=[]
    for i in range(12):
        pilot.append({'key':f'K{i}','shots':10000,'A3_whitelist_resolved_lower_estimate':1000+i,
                      'A3_whitelist_resolved_lower_naive_se':100.0+i})
    plan,meta=_build_fixed_plan(fake_strata,pilot,50000.0,2000,5000,100000)
    assert len(plan)==120 and all(r['shots']>=2000 for r in plan)
    assert all(r['shots']>=5000 for r in plan[:12])
    real_groups=a1._build_groups(EXPECTED_ESCAPE_LOCATIONS)
    real_strata=a1._all_strata(real_groups)
    real_locations=sum(len(v) for v in real_groups.values())
    real_triples=sum(int(s['location_triples']) for s in real_strata)
    assert real_locations == EXPECTED_FACTORY_LOCATIONS
    assert len(real_strata) == 120
    assert real_triples == math.comb(EXPECTED_FACTORY_LOCATIONS,3)
    return {'passed':True,'A5_synthetic_policy_selftest_pass':True,'allocation_strata':len(plan),
            'allocation_fixed_before_sampling':meta['allocation_fixed_before_A6_sampling'],
            'global_nonzero_discovery_floor':True,
            'escape_locations_per_block':EXPECTED_ESCAPE_LOCATIONS,
            'complete_factory_locations':real_locations,
            'complete_factory_exact3_location_triples':real_triples}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--hirano-src',type=Path)
    ap.add_argument('--signature-p',type=float,default=1e-3)
    ap.add_argument('--pilot-active',type=Path,default=Path('stim_pipeline/interface/s11a5v2_results/s11a5_active_strata.json'))
    ap.add_argument('--seed',type=int,default=20260826)
    ap.add_argument('--target-a3-se',type=float,default=50000.0)
    ap.add_argument('--inactive-floor-shots',type=int,default=2000)
    ap.add_argument('--active-floor-shots',type=int,default=5000)
    ap.add_argument('--max-shots-per-stratum',type=int,default=100000)
    ap.add_argument('--max-examples-per-stratum',type=int,default=12)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--plan-only',action='store_true')
    ap.add_argument('--out-dir',type=Path,default=HERE/'s11a6_results')
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)

    st=_self_test(args.pilot_active if args.pilot_active.exists() else None)
    _dump(args.out_dir/'s11a6_self_test.json',st)
    if args.self_test:
        print(json.dumps(st,indent=2)); print('S11-A6 SELF-TEST PASS'); return
    if not args.pilot_active.exists():
        raise SystemExit(f'--pilot-active not found: {args.pilot_active}')

    pilot=json.loads(args.pilot_active.read_text(encoding='utf-8'))
    # a1._build_groups() takes the per-escape-block location count, NOT the
    # complete-factory location count.  S10 has 1028 locations in each escape
    # block plus 890 non-escape locations, for 2946 total.
    groups_plan=a1._build_groups(EXPECTED_ESCAPE_LOCATIONS)
    plan_group_total=sum(len(v) for v in groups_plan.values())
    if plan_group_total != EXPECTED_FACTORY_LOCATIONS:
        raise AssertionError(('complete factory location count changed in plan', plan_group_total))
    strata_plan=a1._all_strata(groups_plan)
    plan_triple_total=sum(int(s['location_triples']) for s in strata_plan)
    expected_triples=math.comb(EXPECTED_FACTORY_LOCATIONS,3)
    if plan_triple_total != expected_triples:
        raise AssertionError(('exact-three partition changed in plan', plan_triple_total, expected_triples))
    plan_rows,plan_meta=_build_fixed_plan(strata_plan,pilot,float(args.target_a3_se),int(args.inactive_floor_shots),
                                          int(args.active_floor_shots),int(args.max_shots_per_stratum))
    plan={'stage':'S11-A6-plan','pilot_active':str(args.pilot_active),'independent_seed':int(args.seed),
          'plan_meta':plan_meta,'strata':plan_rows}
    _dump(args.out_dir/'s11a6_plan.json',plan)
    print(f"S11-A6 fixed plan: strata=120 total_shots={plan_meta['planned_total_shots']} active={plan_meta['planned_active_total_shots']} inactive_discovery={plan_meta['planned_inactive_total_shots']} pilot_predicted_active_SE={plan_meta['pilot_predicted_A3_se_on_active_strata']:.6g}",flush=True)
    if args.plan_only:
        print(json.dumps(plan_meta,indent=2)); return
    if args.hirano_src is None:
        raise SystemExit('--hirano-src required unless --self-test/--plan-only')

    t0=time.time()
    print('S11-A6: building complete S10 boundary bridge...',flush=True)
    bridge=a1.s10mc._bridge(args.hirano_src.resolve(),float(args.signature_p))
    # bridge['meta']['physical_locations'] is the size of ONE ZXZ escape
    # block.  The complete factory has additional FRONT/CAT/CORE/MIDED modules.
    if int(bridge['meta']['physical_locations']) != EXPECTED_ESCAPE_LOCATIONS:
        raise AssertionError(('escape physical location count changed', bridge['meta']['physical_locations']))
    print('S11-A6: rebuilding exact A5 live-171 safe-prefix whitelist...',flush=True)
    whitelist,cert=a5._build_live171_whitelist(bridge)
    _dump(args.out_dir/'s11a6_live171_whitelist_certificate.json',cert)
    if not cert['exact_no_new_le2_acceptance_penalty'] or cert['safe_unsafe_prefix_overlap']!=0 or cert['safe_full_states_missed_by_whitelist']!=0:
        raise AssertionError(('A5 exact whitelist certificate failed',cert))

    print('S11-A6: rebuilding full-195 decoder inputs...',flush=True)
    paired,clean_analysis,_policy,_meta=a3.s8c._escape_catalogue(args.hirano_src.resolve(),float(args.signature_p))
    clean_table={int(k):int(v) for k,v in clean_analysis['state_table'].items()}
    single_effects=a3.a2._single_effects_from_bridge(bridge)
    sectors=list(bridge['sector'].values())
    groups=a1._build_groups(int(bridge['meta']['physical_locations']))
    runtime_group_total=sum(len(v) for v in groups.values())
    if runtime_group_total != EXPECTED_FACTORY_LOCATIONS:
        raise AssertionError(('complete factory location count changed at runtime', runtime_group_total))
    strata=a1._all_strata(groups)
    runtime_triple_total=sum(int(s['location_triples']) for s in strata)
    if runtime_triple_total != math.comb(EXPECTED_FACTORY_LOCATIONS,3):
        raise AssertionError(('exact-three partition changed at runtime', runtime_triple_total))
    bymeta={s['key']:s for s in strata}
    if [s['key'] for s in strata] != [r['key'] for r in plan_rows]:
        raise AssertionError('stratum ordering changed between plan and runtime')

    results=[]; all_unresolved=[]; all_examples=[]
    ckdir=args.out_dir/'checkpoints'; ckdir.mkdir(parents=True,exist_ok=True)
    for i,pr in enumerate(plan_rows,1):
        key=pr['key']; meta=bymeta[key]; shots=int(pr['shots'])
        cp=ckdir/f'{key}.json'; rr=None
        if args.resume and cp.exists():
            old=json.loads(cp.read_text(encoding='utf-8'))
            if int(old.get('shots',-1))==shots and int(old.get('seed',-1))==int(args.seed)+1009*i:
                rr=old; print(f'S11-A6 resume {i:3d}/120 {key}',flush=True)
        if rr is None:
            seed=int(args.seed)+1009*i
            rr=_audit(meta,shots,seed,groups,bridge,whitelist,clean_table,single_effects,sectors,int(args.max_examples_per_stratum))
            _dump(cp,rr)
            print(f"S11-A6 {i:3d}/120 {key}: shots={shots} preacc={rr['accepted_before_whitelist']} newrej={rr['new_whitelist_rejections']} postacc={rr['accepted_after_whitelist']} harmful={rr['residual_harmful']} unresolved={rr['retained_unresolved']} A3=[{rr['A3_lower_estimate']:.6g},{rr['A3_upper_estimate']:.6g}]",flush=True)
        results.append(rr); all_unresolved.extend(rr.get('unresolved_examples',[])); all_examples.extend(rr.get('examples',[]))
        _dump(args.out_dir/'s11a6_partial_strata.json',results)

    agg=_aggregate(results)
    pilot_active_keys={r['key'] for r in pilot}
    newly_discovered=[r for r in results if r['key'] not in pilot_active_keys and (r['A3_upper_estimate']>1e-9 or r['retained_unresolved']>0)]
    topvar=sorted(results,key=lambda r:r['A3_lower_se']**2,reverse=True)
    summary={
        'stage':'S11-A6',
        'status':'GLOBAL_INDEPENDENT_HIGHSTAT_A3_B3_AUDIT',
        'scope':'all 120 exact-three location strata under the exact A5 live-171 prefix whitelist, using a fixed plan determined only from pre-A6 pilot data',
        'exact_policy_certificate':cert,
        'plan_meta':plan_meta,
        'aggregate':agg,
        'newly_discovered_nonpilot_strata_count':len(newly_discovered),
        'newly_discovered_nonpilot_strata':[{'key':r['key'],'A3_lower':r['A3_lower_estimate'],'A3_upper':r['A3_upper_estimate'],'harmful':r['residual_harmful'],'unresolved':r['retained_unresolved'],'B3':r['B3_additional_rejection_estimate']} for r in newly_discovered],
        'top_strata_by_A3_upper':[{'key':r['key'],'A3_lower':r['A3_lower_estimate'],'A3_upper':r['A3_upper_estimate'],'A3_se':r['A3_lower_se'],'harmful':r['residual_harmful'],'unresolved':r['retained_unresolved']} for r in sorted(results,key=lambda r:r['A3_upper_estimate'],reverse=True)[:20]],
        'top_strata_by_A3_variance':[{'key':r['key'],'A3_se':r['A3_lower_se'],'variance_fraction':(r['A3_lower_se']**2/sum(x['A3_lower_se']**2 for x in results) if sum(x['A3_lower_se']**2 for x in results)>0 else 0.0)} for r in topvar[:20]],
        'top_strata_by_B3':[{'key':r['key'],'B3':r['B3_additional_rejection_estimate'],'B3_se':r['B3_additional_rejection_se'],'new_rejections':r['new_whitelist_rejections']} for r in sorted(results,key=lambda r:r['B3_additional_rejection_estimate'],reverse=True)[:20]],
        'elapsed_seconds':time.time()-t0,
        'interpretation_gate':{
            'target_A3_SE':float(args.target_a3_se),
            'A3_SE_target_met':bool(agg['A3_lower_naive_se']<=float(args.target_a3_se)),
            'relative_A3_SE':(agg['A3_lower_naive_se']/agg['A3_lower_estimate'] if agg['A3_lower_estimate']>0 else None),
            'residual_unresolved_fraction_of_A3_lower':(agg['A3_unresolved_interval_width']/agg['A3_lower_estimate'] if agg['A3_lower_estimate']>0 else None),
            'next':'If no new dominant nonpilot stratum appears and A3 SE/unresolved width are acceptable, freeze A3/B3 performance numbers; otherwise promote only the variance-dominant/newly-discovered strata for A6b.'
        },
        'statistical_note':'Naive SE/95% intervals are stratified Monte-Carlo normal approximations for bounded infidelity/rejection variables. The exact result in A6 is the <=2 whitelist preservation certificate, not the order-3 coefficient itself.'
    }
    _dump(args.out_dir/'s11a6_strata.json',results)
    _dump(args.out_dir/'s11a6_retained_unresolved.json',all_unresolved)
    _dump(args.out_dir/'s11a6_examples.json',all_examples)
    _dump(args.out_dir/'s11a6_summary.json',summary)
    print(json.dumps(a5._json_safe(summary),indent=2),flush=True)
    print('S11-A6 COMPLETE: global independent-seed high-stat A3/B3 audit across all 120 strata.',flush=True)


if __name__=='__main__':
    main()
