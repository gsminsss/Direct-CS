#!/usr/bin/env python3
"""S11-A6b: targeted paper-final A3 closure after the global S11-A6 audit.

This stage deliberately separates two jobs:

1. STATISTICS CLOSURE
   Add a new independent sample only to variance-dominant exact-three strata,
   then pool those samples with the fixed-plan S11-A6 sample *within each
   stratum*.  The target set and shot allocation are determined entirely from
   the already-finished A6 results before any A6b sample is drawn.

2. RETAINED-UNRESOLVED CLOSURE
   Replay only the A6/A6b trajectories that remain accepted by the exact A5
   live-171 whitelist but ambiguous to the full-195 decoder.  For pure ESC3
   strata, use exact distinct-location three-escape reachability.  For
   CORE_POST1+ESC2, use a deliberately conservative ALL-64-sector + exact-E2
   candidate union.  A singleton from that over-approximation is safe to use;
   a multi-frame set remains unresolved.  The mixed-family closure is therefore
   conservative, not advertised as an exhaustive physical-core-single proof.

Exact certification remains the S10 <=2 / A5 whitelist certificate.  A6b is a
statistical performance closure plus an exact pure-escape ambiguity closure.
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
DEFAULT_A6_SEED = 20260826
DEFAULT_A6B_SEED = 20260827


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError((name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a6 = _load('s11a6b_a6', HERE / 'direct_cs_s11a6_global_independent_highstat_a3.py')
a4 = _load('s11a6b_a4', HERE / 'direct_cs_s11a4_exact_overlap_selective_rejection.py')
a5 = a6.a5
a3 = a6.a3
a1 = a6.a1


def _json_safe(x):
    return a5._json_safe(x)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2), encoding='utf-8')


def _normal95(mean: float, se: float) -> list[float]:
    z=1.959963984540054
    return [max(0.0,float(mean-z*se)), float(mean+z*se)]


def _select_variance_rows(rows: list[dict[str,Any]], cumulative: float) -> tuple[list[dict[str,Any]], float]:
    total=sum(float(r['A3_lower_se'])**2 for r in rows)
    ranked=sorted(rows,key=lambda r:float(r['A3_lower_se'])**2,reverse=True)
    out=[]; c=0.0
    for r in ranked:
        v=float(r['A3_lower_se'])**2
        if v<=0: continue
        out.append(r); c += v/total if total else 0.0
        if c >= float(cumulative): break
    return out,c


def _allocation_for_target(rows: list[dict[str,Any]], selected: list[dict[str,Any]], target_se: float,
                           max_total_shots_per_stratum: int) -> tuple[dict[str,int],dict[str,Any]]:
    """Minimum-total-shot continuous allocation with old-shot lower bounds.

    For each selected stratum h, c_h = SE_h*sqrt(n_h), so the projected
    variance after N_h total pooled samples is c_h^2/N_h.  Nonselected
    stratum variance is held fixed.  We solve N_h=max(n_h,c_h*t) by bisection.
    """
    selected_keys={r['key'] for r in selected}
    fixed_var=sum(float(r['A3_lower_se'])**2 for r in rows if r['key'] not in selected_keys)
    target2=float(target_se)**2
    if fixed_var >= target2:
        raise RuntimeError(('target SE impossible with selected variance set', math.sqrt(fixed_var), target_se))
    budget=target2-fixed_var
    c={r['key']:float(r['A3_lower_se'])*math.sqrt(max(1,int(r['shots']))) for r in selected}
    old={r['key']:int(r['shots']) for r in selected}

    def svar(t: float) -> float:
        s=0.0
        for k,cv in c.items():
            N=min(int(max_total_shots_per_stratum), max(old[k], cv*t))
            s += cv*cv/max(float(N),1.0)
        return s

    lo=0.0; hi=1.0
    while svar(hi) > budget:
        hi*=2.0
        if hi>1e12:
            raise RuntimeError('could not satisfy target SE under max-shot cap')
    for _ in range(100):
        mid=(lo+hi)/2
        if svar(mid)>budget: lo=mid
        else: hi=mid
    t=hi
    totalN={}
    add={}
    for k,cv in c.items():
        N=int(math.ceil(min(int(max_total_shots_per_stratum), max(old[k],cv*t))))
        totalN[k]=N; add[k]=max(0,N-old[k])
    pred=math.sqrt(fixed_var + sum(c[k]**2/max(totalN[k],1) for k in c))
    return add,{
        'target_pooled_A3_lower_se':float(target_se),
        'fixed_nonselected_A3_variance':float(fixed_var),
        'selected_variance_budget':float(budget),
        'predicted_pooled_A3_lower_se':float(pred),
        'selected_total_shots_after_pooling':totalN,
        'selected_additional_shots':add,
        'total_additional_shots':int(sum(add.values())),
    }


def _coeff_stats(row: dict[str,Any], estimate_key: str, se_key: str) -> tuple[int,float,float]:
    n=int(row['shots']); ways=float(row['location_triples'])
    mean=float(row[estimate_key])/ways if ways else 0.0
    sem=float(row[se_key])/ways if ways else 0.0
    var=float(sem*sem*n) if n>1 else 0.0
    return n,mean,var


def _pool_scalar(old: dict[str,Any], new: dict[str,Any], estimate_key: str, se_key: str) -> tuple[float,float]:
    n1,m1,v1=_coeff_stats(old,estimate_key,se_key)
    n2,m2,v2=_coeff_stats(new,estimate_key,se_key)
    N=n1+n2; ways=float(old['location_triples'])
    if int(old['location_triples']) != int(new['location_triples']):
        raise AssertionError(('ways mismatch',old['key']))
    if N<=0: return 0.0,0.0
    m=(n1*m1+n2*m2)/N
    M2=max(0,n1-1)*v1 + max(0,n2-1)*v2
    if n1 and n2:
        M2 += (m2-m1)**2*n1*n2/N
    var=M2/max(1,N-1) if N>1 else 0.0
    se=math.sqrt(var/N) if N>1 else 0.0
    return ways*m, ways*se


def _pool_rows(old: dict[str,Any], new: dict[str,Any]) -> dict[str,Any]:
    r=dict(old)
    r['shots']=int(old['shots'])+int(new['shots'])
    for ek,sk in (
        ('A3_lower_estimate','A3_lower_se'),('A3_upper_estimate','A3_upper_se'),
        ('B3_additional_rejection_estimate','B3_additional_rejection_se'),
        ('A3_best_pauli_orbit_estimate','A3_best_pauli_orbit_se')):
        e,s=_pool_scalar(old,new,ek,sk); r[ek]=e; r[sk]=s
    for k in ('accepted_before_whitelist','new_whitelist_rejections','accepted_after_whitelist',
              'retained_resolved','retained_unresolved','residual_harmful',
              'residual_frame_alias_harmful','residual_hidden_state_harmful'):
        r[k]=int(old.get(k,0))+int(new.get(k,0))
    r['A3_unresolved_width']=float(r['A3_upper_estimate']-r['A3_lower_estimate'])
    r['pooled_from']={'A6_shots':int(old['shots']),'A6b_additional_shots':int(new['shots'])}
    return r


def _aggregate(rows: list[dict[str,Any]]) -> dict[str,Any]:
    low=sum(float(r['A3_lower_estimate']) for r in rows); sel=math.sqrt(sum(float(r['A3_lower_se'])**2 for r in rows))
    up=sum(float(r['A3_upper_estimate']) for r in rows); seu=math.sqrt(sum(float(r['A3_upper_se'])**2 for r in rows))
    b3=sum(float(r['B3_additional_rejection_estimate']) for r in rows); seb=math.sqrt(sum(float(r['B3_additional_rejection_se'])**2 for r in rows))
    orb=sum(float(r['A3_best_pauli_orbit_estimate']) for r in rows); seo=math.sqrt(sum(float(r['A3_best_pauli_orbit_se'])**2 for r in rows))
    return {
        'A3_lower_estimate':low,'A3_lower_naive_se':sel,'A3_lower_naive_95pct':_normal95(low,sel),
        'A3_upper_estimate':up,'A3_upper_naive_se':seu,'A3_upper_naive_95pct':_normal95(up,seu),
        'A3_unresolved_interval_width':up-low,
        'A3_combined_naive_95pct_including_unresolved':[max(0.0,low-1.959963984540054*sel),up+1.959963984540054*seu],
        'B3_additional_rejection_estimate':b3,'B3_additional_rejection_naive_se':seb,'B3_additional_rejection_naive_95pct':_normal95(b3,seb),
        'A3_best_pauli_orbit_estimate':orb,'A3_best_pauli_orbit_naive_se':seo,
        'total_pooled_shots':sum(int(r['shots']) for r in rows),
        'retained_unresolved_samples':sum(int(r['retained_unresolved']) for r in rows),
        'residual_harmful_samples':sum(int(r['residual_harmful']) for r in rows),
        'p3_predictions_lower_upper':{str(p):[low*p**3,up*p**3] for p in (2e-4,5e-4,1e-3)},
        'p3_predictions_additional_rejection':{str(p):b3*p**3 for p in (2e-4,5e-4,1e-3)},
    }


def _single_groups_from_paired(paired):
    return a4._single_groups(paired)


def _all64_e2_flags(target: int, pair_flags: dict[int,int], sectors) -> int:
    out=0
    for s in sectors:
        pf=int(pair_flags.get(int(target)^int(s.full),0))
        if pf:
            out |= a4._combine_flags(pf,int(s.frame))
    return int(out)


def _replay_unresolved(meta, row, seed, groups, bridge, whitelist, clean_table, single_effects, sectors):
    """Replay one run and retain every post-whitelist full-195 ambiguous sample."""
    rng=np.random.default_rng(int(seed)); mask=(1<<a1.ACTUAL_BITS)-1; boundary=bridge['table']
    items=[]
    for i in range(int(row['shots'])):
        locs=a1._sample_locations(meta,groups,rng)
        r=a1._simulate_exact3(locs,bridge,rng,want_trace=True,want_state=True)
        if not r['accepted']: continue
        fa=int(r['full_syndrome_A_hex'],16); fb=int(r['full_syndrome_B_hex'],16)
        if (fa&mask) not in whitelist or (fb&mask) not in whitelist: continue
        da=a3._decode_one(fa,boundary,clean_table,single_effects,sectors)
        db=a3._decode_one(fb,boundary,clean_table,single_effects,sectors)
        if da.predicted is None or db.predicted is None:
            items.append((i,r,da,db))
    if len(items)!=int(row['retained_unresolved']):
        raise AssertionError(('unresolved replay mismatch',meta['key'],len(items),row['retained_unresolved']))
    return items


def _apply_closure_to_row(row: dict[str,Any], resolved_infs: list[float]) -> dict[str,Any]:
    """Replace baseline unresolved (lower=0, upper=1) by resolved infidelities."""
    if not resolved_infs: return dict(row)
    r=dict(row); N=int(r['shots']); ways=float(r['location_triples'])
    for ek,sk,oldval in (('A3_lower_estimate','A3_lower_se',0.0),('A3_upper_estimate','A3_upper_se',1.0)):
        mean=float(r[ek])/ways; sem=float(r[sk])/ways
        var=sem*sem*N if N>1 else 0.0
        S=N*mean
        SS=max(0,N-1)*var + N*mean*mean
        for inf in resolved_infs:
            S += float(inf)-oldval
            SS += float(inf)*float(inf)-oldval*oldval
        m=S/N
        v=max(0.0,(SS-N*m*m)/max(1,N-1)) if N>1 else 0.0
        r[ek]=ways*m; r[sk]=ways*math.sqrt(v/N) if N>1 else 0.0
    r['retained_unresolved']=max(0,int(r['retained_unresolved'])-len(resolved_infs))
    r['A3_unresolved_width']=float(r['A3_upper_estimate']-r['A3_lower_estimate'])
    r['closure_resolved_unresolved_samples']=len(resolved_infs)
    return r


def _synthetic_pool_test():
    old={'key':'K','shots':100,'location_triples':1000,'A3_lower_estimate':100.0,'A3_lower_se':10.0,
         'A3_upper_estimate':120.0,'A3_upper_se':11.0,'B3_additional_rejection_estimate':50.0,'B3_additional_rejection_se':5.0,
         'A3_best_pauli_orbit_estimate':2.0,'A3_best_pauli_orbit_se':1.0,
         'accepted_before_whitelist':1,'new_whitelist_rejections':1,'accepted_after_whitelist':0,'retained_resolved':0,'retained_unresolved':1,
         'residual_harmful':0,'residual_frame_alias_harmful':0,'residual_hidden_state_harmful':0}
    new=dict(old); new['shots']=100; new['A3_lower_estimate']=100.0
    p=_pool_rows(old,new)
    assert p['shots']==200 and abs(p['A3_lower_estimate']-100.0)<1e-12
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--hirano-src',type=Path)
    ap.add_argument('--signature-p',type=float,default=1e-3)
    ap.add_argument('--a6-strata',type=Path,default=Path('stim_pipeline/interface/s11a6v2_results/s11a6_strata.json'))
    ap.add_argument('--a6-summary',type=Path,default=Path('stim_pipeline/interface/s11a6v2_results/s11a6_summary.json'))
    ap.add_argument('--a6-seed',type=int,default=DEFAULT_A6_SEED)
    ap.add_argument('--seed',type=int,default=DEFAULT_A6B_SEED)
    ap.add_argument('--target-pooled-a3-se',type=float,default=40000.0)
    ap.add_argument('--variance-cumulative',type=float,default=0.85)
    ap.add_argument('--max-total-shots-per-stratum',type=int,default=500000)
    ap.add_argument('--max-examples-per-stratum',type=int,default=12)
    ap.add_argument('--pair-progress',type=int,default=500)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--plan-only',action='store_true')
    ap.add_argument('--skip-unresolved-closure',action='store_true')
    ap.add_argument('--out-dir',type=Path,default=HERE/'s11a6b_results')
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)

    if not _synthetic_pool_test(): raise AssertionError('pool test')
    st={'passed':True,'pooling_selftest':True,'pure_ESC3_closure_semantics':'exact distinct-location reachability',
        'CORE_POST1_ESC2_closure_semantics':'ALL64-sector + exact-E2 conservative over-approximation; singleton safe, multi-frame retained'}
    _dump(args.out_dir/'s11a6b_self_test.json',st)
    if args.self_test:
        print(json.dumps(st,indent=2)); print('S11-A6b SELF-TEST PASS'); return
    if not args.a6_strata.exists() or not args.a6_summary.exists():
        raise SystemExit('A6 strata/summary files not found')

    old_rows=json.loads(args.a6_strata.read_text(encoding='utf-8'))
    old_summary=json.loads(args.a6_summary.read_text(encoding='utf-8'))
    if len(old_rows)!=120: raise AssertionError(('expected 120 A6 strata',len(old_rows)))
    selected,cum=_select_variance_rows(old_rows,float(args.variance_cumulative))
    add,allocmeta=_allocation_for_target(old_rows,selected,float(args.target_pooled_a3_se),int(args.max_total_shots_per_stratum))
    plan={'stage':'S11-A6b-plan','source_A6_summary':str(args.a6_summary),'source_A6_strata':str(args.a6_strata),
          'new_independent_seed':int(args.seed),'variance_cumulative_requested':float(args.variance_cumulative),
          'variance_cumulative_selected':float(cum),'selected_keys':[r['key'] for r in selected],
          'allocation':allocmeta,'old_A6_lower':old_summary['aggregate']['A3_lower_estimate'],
          'old_A6_lower_se':old_summary['aggregate']['A3_lower_naive_se']}
    _dump(args.out_dir/'s11a6b_plan.json',plan)
    print(f"S11-A6b fixed augmentation: selected={len(selected)} add_shots={allocmeta['total_additional_shots']} predicted_pooled_SE={allocmeta['predicted_pooled_A3_lower_se']:.6g}",flush=True)
    for r in selected:
        print(f"  {r['key']}: old={r['shots']} add={add[r['key']]} total={r['shots']+add[r['key']]}",flush=True)
    if args.plan_only: return
    if args.hirano_src is None: raise SystemExit('--hirano-src required unless --self-test/--plan-only')

    t0=time.time()
    bridge=a1.s10mc._bridge(args.hirano_src.resolve(),float(args.signature_p))
    if int(bridge['meta']['physical_locations'])!=a6.EXPECTED_ESCAPE_LOCATIONS:
        raise AssertionError(('escape locations changed',bridge['meta']['physical_locations']))
    whitelist,cert=a5._build_live171_whitelist(bridge)
    if not cert['exact_no_new_le2_acceptance_penalty'] or cert['safe_unsafe_prefix_overlap']!=0 or cert['safe_full_states_missed_by_whitelist']!=0:
        raise AssertionError(('A5 whitelist certificate failed',cert))
    _dump(args.out_dir/'s11a6b_live171_whitelist_certificate.json',cert)

    paired,clean_analysis,_policy,_meta=a3.s8c._escape_catalogue(args.hirano_src.resolve(),float(args.signature_p))
    clean_table={int(k):int(v) for k,v in clean_analysis['state_table'].items()}
    single_effects=a3.a2._single_effects_from_bridge(bridge)
    sectors=list(bridge['sector'].values())
    groups=a1._build_groups(a6.EXPECTED_ESCAPE_LOCATIONS)
    strata=a1._all_strata(groups); bymeta={s['key']:s for s in strata}

    old_by={r['key']:r for r in old_rows}
    additional=[]; ckdir=args.out_dir/'checkpoints'; ckdir.mkdir(parents=True,exist_ok=True)
    selected_order=[r['key'] for r in selected]
    for i,key in enumerate(selected_order,1):
        nadd=int(add[key]); meta=bymeta[key]
        if nadd<=0: continue
        seed=int(args.seed)+1009*i
        cp=ckdir/f'{key}.json'; rr=None
        if args.resume and cp.exists():
            z=json.loads(cp.read_text(encoding='utf-8'))
            if int(z.get('shots',-1))==nadd and int(z.get('seed',-1))==seed: rr=z
        if rr is None:
            rr=a6._audit(meta,nadd,seed,groups,bridge,whitelist,clean_table,single_effects,sectors,int(args.max_examples_per_stratum))
            _dump(cp,rr)
        additional.append(rr)
        print(f"S11-A6b {i}/{len(selected_order)} {key}: add={nadd} harmful={rr['residual_harmful']} unresolved={rr['retained_unresolved']} A3=[{rr['A3_lower_estimate']:.6g},{rr['A3_upper_estimate']:.6g}]",flush=True)
    _dump(args.out_dir/'s11a6b_additional_strata.json',additional)

    new_by={r['key']:r for r in additional}
    pooled=[]
    for r in old_rows:
        pooled.append(_pool_rows(r,new_by[r['key']]) if r['key'] in new_by else dict(r))
    preagg=_aggregate(pooled)
    _dump(args.out_dir/'s11a6b_pooled_strata_preclosure.json',pooled)

    closure={'attempted':False,'pure_ESC3_exact':True,'mixed_CORE_POST1_ESC2_exact':False,'items':[],'resolved_count':0,'remaining_count':preagg['retained_unresolved_samples']}
    post=pooled
    if not args.skip_unresolved_closure and preagg['retained_unresolved_samples']>0:
        closure['attempted']=True
        # Replay unresolved trajectories from the old A6 run and the independent A6b additions.
        replay=[]
        affected={r['key'] for r in pooled if int(r['retained_unresolved'])>0}
        old_index={r['key']:i+1 for i,r in enumerate(old_rows)}
        selected_index={k:i+1 for i,k in enumerate(selected_order)}
        for key in sorted(affected):
            meta=bymeta[key]
            old=old_by[key]
            if int(old['retained_unresolved'])>0:
                seed=int(args.a6_seed)+1009*old_index[key]
                items=_replay_unresolved(meta,old,seed,groups,bridge,whitelist,clean_table,single_effects,sectors)
                for item in items: replay.append(('A6',key,int(old['shots']),item))
            if key in new_by and int(new_by[key]['retained_unresolved'])>0:
                seed=int(args.seed)+1009*selected_index[key]
                items=_replay_unresolved(meta,new_by[key],seed,groups,bridge,whitelist,clean_table,single_effects,sectors)
                for item in items: replay.append(('A6b',key,int(new_by[key]['shots']),item))

        singles=_single_groups_from_paired(paired)
        single_syns=sorted({int(g.syndrome) for g in singles})
        targets_pure=set(); targets_mixed=set()
        for _run,key,_n,(_idx,r,da,db) in replay:
            full=int(r['full_syndrome_A_hex'],16) if da.predicted is None else int(r['full_syndrome_B_hex'],16)
            if key in ('ESC_A3','ESC_B3'): targets_pure.add(full)
            elif key in ('CORE_POST1+ESC_A2','CORE_POST1+ESC_B2'): targets_mixed.add(full)
        wanted=set()
        for t in targets_pure:
            for s in single_syns: wanted.add(int(t)^int(s))
        for t in targets_mixed:
            for s in sectors: wanted.add(int(t)^int(s.full))
        pair_flags,pair_common,pmeta=a4._build_filtered_pair_catalogue(paired,wanted,progress=int(args.pair_progress))
        closure['pair_catalogue_meta']=pmeta

        resolved_by_key=collections.defaultdict(list)
        actual_not_in=0; remain=0
        for run,key,nrun,(sample_i,r,da,db) in replay:
            amb_blocks=[]
            if da.predicted is None: amb_blocks.append('A')
            if db.predicted is None: amb_blocks.append('B')
            if len(amb_blocks) != 1:
                raise AssertionError(('A6b closure expects exactly one ambiguous block in the currently supported families', key, sample_i, amb_blocks))
            ambA=(amb_blocks[0]=='A')
            # In the currently supported strata, the ambiguous block must be the
            # block carrying the escape faults.  Abort rather than silently
            # applying a one-block over-approximation to the wrong side.
            expected_amb = 'A' if key in ('ESC_A3','CORE_POST1+ESC_A2') else 'B' if key in ('ESC_B3','CORE_POST1+ESC_B2') else None
            if expected_amb is None or amb_blocks[0] != expected_amb:
                raise AssertionError(('unsupported ambiguous-block pattern', key, sample_i, amb_blocks[0], expected_amb))
            full=int(r['full_syndrome_A_hex'],16) if ambA else int(r['full_syndrome_B_hex'],16)
            actual=int(r['actual_frame_A']) if ambA else int(r['actual_frame_B'])
            if key in ('ESC_A3','ESC_B3'):
                flags=a4._exact_escape3_flags(full,singles,pair_flags,pair_common); method='exact_distinct_location_E3'
            elif key in ('CORE_POST1+ESC_A2','CORE_POST1+ESC_B2'):
                flags=_all64_e2_flags(full,pair_flags,sectors); method='conservative_ALL64_plus_exact_E2'
            else:
                flags=0; method='unsupported_retained_family'
            if not (flags & (1<<actual)):
                actual_not_in+=1
            pred=a4._singleton(flags)
            entry={'run':run,'stratum':key,'sample':sample_i,'ambiguous_block':amb_blocks[0],
                   'full':hex(full),'method':method,'candidate_frames':[FRAME_LABELS[x] for x in a4._frames(flags)],
                   'actual_frame':FRAME_LABELS[actual]}
            if pred is None:
                remain+=1; entry['resolved']=False
            else:
                pa=int(pred) if ambA else int(da.predicted)
                pb=int(db.predicted) if ambA else int(pred)
                aa=int(r['actual_frame_A']); ab=int(r['actual_frame_B'])
                lv=np.asarray(r['_logical_vector'],dtype=complex); target=a1._target(r['record'])
                corr=a1.s10mc._apply(lv,pa^aa,pb^ab); inf=float(a1._phase_inf(corr,target))
                entry.update({'resolved':True,'predicted_frame':FRAME_LABELS[int(pred)],'resolved_infidelity':inf})
                resolved_by_key[key].append(inf)
            closure['items'].append(entry)
        closure['actual_frame_not_in_candidate_union']=actual_not_in
        if actual_not_in:
            raise AssertionError(('closure candidate union omitted sampled actual frame',actual_not_in))
        closure['resolved_count']=sum(len(v) for v in resolved_by_key.values())
        closure['remaining_count']=remain

        post=[]
        for r in pooled:
            post.append(_apply_closure_to_row(r,resolved_by_key.get(r['key'],[])))
        _dump(args.out_dir/'s11a6b_unresolved_closure.json',closure)
    _dump(args.out_dir/'s11a6b_pooled_strata_postclosure.json',post)
    postagg=_aggregate(post)

    summary={
        'stage':'S11-A6b','status':'TARGETED_PAPER_FINAL_A3_CLOSURE',
        'scope':'A6-independent global audit pooled with predeclared independent augmentation on variance-dominant strata; exact pure-ESC3 retained-ambiguity closure plus conservative mixed CORE_POST1+ESC2 closure',
        'plan':plan,'exact_policy_certificate':cert,'preclosure_aggregate':preagg,'postclosure_aggregate':postagg,
        'closure':{k:v for k,v in closure.items() if k!='items'},
        'interpretation_gate':{
            'target_pooled_A3_lower_se':float(args.target_pooled_a3_se),
            'A3_lower_SE_target_met':bool(postagg['A3_lower_naive_se']<=float(args.target_pooled_a3_se)),
            'relative_A3_lower_SE':float(postagg['A3_lower_naive_se']/postagg['A3_lower_estimate']) if postagg['A3_lower_estimate'] else None,
            'relative_unresolved_width':float(postagg['A3_unresolved_interval_width']/postagg['A3_lower_estimate']) if postagg['A3_lower_estimate'] else None,
            'paper_final_numeric_freeze_recommended':bool(postagg['A3_lower_naive_se']<=float(args.target_pooled_a3_se) and postagg['A3_unresolved_interval_width']<=0.05*postagg['A3_lower_estimate']),
        },
        'important_limits':[
            'The <=2 / no-new-rejection statement is exact; A3/B3 remain stratified Monte-Carlo estimates.',
            'Pure ESC3 retained-syndrome closure is exact with respect to distinct escape locations.',
            'CORE_POST1+ESC2 retained-syndrome closure uses an ALL-64-sector over-approximation plus exact two-escape reachability. Singleton resolution is safe; remaining ambiguity is conservative.',
            'The 195-bit tail remains deferred/ideal decoder context, not an immediate 171-bit live frame decoder.',
        ],
        'elapsed_seconds':time.time()-t0,
    }
    _dump(args.out_dir/'s11a6b_summary.json',summary)
    print(json.dumps(summary['interpretation_gate'],indent=2),flush=True)
    print(f"S11-A6b COMPLETE: A3=[{postagg['A3_lower_estimate']:.8g},{postagg['A3_upper_estimate']:.8g}] lower_SE={postagg['A3_lower_naive_se']:.6g} unresolved_width={postagg['A3_unresolved_interval_width']:.6g}",flush=True)


if __name__=='__main__':
    main()
