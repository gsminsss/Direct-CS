#!/usr/bin/env python3
"""S11-A4: exact escape-overlap closure + selective-rejection diagnostics.

Scientific purpose
------------------
S11-A3 reduced the order-3 decoder problem to a finite unresolved interval, but
its order-3 extension deliberately ignored distinct-location overlap and used
all 64 canonical Steane sectors in the upstream-one-fault + escape<=2 family.
Those choices are safe for uniqueness but create false ambiguity.

S11-A4 tightens exactly the samples that were unresolved in S11-A3, without
rerunning or replacing already-resolved contributions:

  * exact escape-pair reachability is built from the physical ZXZ single-fault
    catalogue with the two faults required to occupy distinct locations;
  * for each exact pair syndrome/frame, we retain the intersection of physical
    locations appearing in every legal pair witness.  This is sufficient to
    decide whether a third single fault can be placed at a distinct location,
    so pure ESC3 reachability is exact without enumerating all class triples;
  * upstream-one-fault + escape2 uses only canonical weight-1 Steane sectors
    (0 is excluded because that is pure escape and treated separately);
  * upstream-two-fault + escape1 uses canonical weight-2 sectors.  Lower-order
    states remain handled by the exact S8-C <=2 table.

The A4 structured order-3 set is used only when the exact <=2 table has no
explanation, preserving the S11-A3 minimum-order decoder.  Each S11-A3
unresolved sample is replayed with the same random seed and rescored.  The
resulting rows replace only the unresolved part of the same S11-A3 sample set,
so the global A3 interval is directly comparable sample-for-sample.

Selective rejection
-------------------
If a full 195-bit syndrome has no <=2 explanation and remains multi-frame after
A4 exactification, rejecting that full syndrome cannot reject an exact <=2
*full-syndrome state*.  The current live handoff exposes only 171 bits, however.
A4 therefore also projects each remaining ambiguous full syndrome to its 171-bit
prefix and checks whether that prefix is used by any accepted <=2 boundary
state.  Prefixes with no such collision are candidates for an operational
live-handoff rejection rule with no <=2 acceptance penalty in the certified
boundary table.

This stage does NOT claim an exhaustive order-3 rejection table: it exactifies
and classifies the S11-A3 sampled unresolved trajectories.  If the residual
interval becomes small, S11-A5 can perform high-statistics independent-seed
estimation; if not, the remaining prefix-collision mechanisms identify the
next exactification target.
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
TOL = 1e-10
FRAME_LABELS = ('I', 'X', 'Z', 'Y')


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError((name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a1 = _load('s11a4_a1', HERE / 'direct_cs_s11a1_exact3_stratified_pilot.py')
a2 = _load('s11a4_a2', HERE / 'direct_cs_s11a2_decoder_aware_exact3.py')
a3 = _load('s11a4_a3', HERE / 'direct_cs_s11a3_priority_decoder_a3.py')
s8c = _load('s11a4_s8c', HERE / 'zxz_core_sector_escape_composability.py')
s7b2 = s8c.s7b2


@dataclass(frozen=True)
class SingleGroup:
    syndrome: int
    frame: int
    locations_mask: int


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


def _frames(flags: int) -> list[int]:
    return [i for i in range(4) if int(flags) & (1 << i)]


def _singleton(flags: int) -> int | None:
    f = int(flags)
    return f.bit_length() - 1 if f and f.bit_count() == 1 else None


def _combine_flags(flags: int, frame: int) -> int:
    out = 0
    for f in _frames(flags):
        out |= 1 << (int(f) ^ int(frame))
    return int(out)


def _mask_from_locations(ids) -> int:
    out = 0
    for lid in ids:
        out |= 1 << int(lid)
    return int(out)


def _single_groups(paired) -> list[SingleGroup]:
    groups = s7b2._build_groups(paired)
    out = []
    for g in groups:
        out.append(SingleGroup(
            syndrome=int(g.syndrome),
            frame=int(g.frame),
            locations_mask=_mask_from_locations(g.examples_by_location.keys()),
        ))
    return out


def _only_after_excluding(vals: set[int], x: int) -> int | None:
    for v in vals:
        if v != x:
            return int(v)
    return None


def _family_common_locations_mask(a, b, same: bool) -> int:
    """Intersection of location supports across ALL legal physical pairs.

    For a==b, legal pairs are all distinct two-subsets of the group's location
    set.  For a!=b, legal pairs are all (la,lb) with la!=lb.

    A location x is common to every a!=b pair iff there is no legal pair after
    deleting x from both location sets.  Only members of a location set of size
    <=2 can satisfy this, so the test is O(1) per group pair.
    """
    A = set(int(x) for x in a.examples_by_location)
    B = set(int(x) for x in b.examples_by_location)
    if same:
        if len(A) < 2:
            raise AssertionError('illegal same-group pair')
        return _mask_from_locations(A) if len(A) == 2 else 0

    # Must agree with the exact pair-legality predicate used in S7-B2.
    if not s7b2._pair_legal(a, b, False):
        raise AssertionError('called on illegal group pair')

    candidates = set()
    if len(A) <= 2:
        candidates |= A
    if len(B) <= 2:
        candidates |= B
    common = 0
    for x in candidates:
        na = len(A) - int(x in A)
        nb = len(B) - int(x in B)
        no_pair_avoiding_x = False
        if na == 0 or nb == 0:
            no_pair_avoiding_x = True
        elif na == 1 and nb == 1:
            aa = _only_after_excluding(A, x)
            bb = _only_after_excluding(B, x)
            if aa is not None and bb is not None and aa == bb:
                no_pair_avoiding_x = True
        if no_pair_avoiding_x:
            common |= 1 << int(x)
    return int(common)


def _desired_pair_syndromes(targets: set[int], singles: list[SingleGroup], w1_sectors) -> set[int]:
    # Pure escape3 requires target XOR escape-single syndrome.
    single_syns = sorted(set(int(g.syndrome) for g in singles))
    out: set[int] = set()
    for t in targets:
        ti = int(t)
        for s in single_syns:
            out.add(ti ^ s)
        # one upstream physical fault + exactly two escape faults
        for sec in w1_sectors:
            out.add(ti ^ int(sec.full))
    return out


def _build_filtered_pair_catalogue(paired, wanted_syndromes: set[int], progress: int = 250):
    """Exact two-distinct-location escape pair table, filtered by syndrome.

    Returns:
      pair_flags[syn] -> reachable frame bitmask at exactly 2 escape faults.
      pair_common[(syn,frame)] -> intersection of locations appearing in every
        exact pair witness for that syndrome/frame.
    """
    groups = s7b2._build_groups(paired)
    pair_flags: dict[int, int] = {}
    pair_common: dict[tuple[int, int], int] = {}
    legal_families = kept_families = 0
    for i, ga in enumerate(groups):
        if progress and (i % int(progress) == 0):
            print(f'S11-A4 pair catalogue: group {i}/{len(groups)}, kept_effects={len(pair_common)}', flush=True)
        for j in range(i, len(groups)):
            gb = groups[j]
            same = i == j
            if not s7b2._pair_legal(ga, gb, same):
                continue
            legal_families += 1
            syn = int(ga.syndrome) ^ int(gb.syndrome)
            if syn not in wanted_syndromes:
                continue
            kept_families += 1
            fr = int(ga.frame) ^ int(gb.frame)
            k = (syn, fr)
            fam_common = _family_common_locations_mask(ga, gb, same)
            pair_flags[syn] = int(pair_flags.get(syn, 0)) | (1 << fr)
            if k in pair_common:
                pair_common[k] &= int(fam_common)
            else:
                pair_common[k] = int(fam_common)
    return pair_flags, pair_common, {
        'signature_groups': len(groups),
        'wanted_pair_syndromes': len(wanted_syndromes),
        'legal_group_pair_families': int(legal_families),
        'kept_group_pair_families': int(kept_families),
        'kept_pair_syndromes': len(pair_flags),
        'kept_pair_syndrome_frame_effects': len(pair_common),
    }


def _single_frame_map(singles: list[SingleGroup]) -> dict[int, int]:
    out: dict[int, int] = {}
    for g in singles:
        out[int(g.syndrome)] = int(out.get(int(g.syndrome), 0)) | (1 << int(g.frame))
    return out


def _exact_escape3_flags(target: int, singles: list[SingleGroup], pair_flags, pair_common) -> int:
    target = int(target)
    out = 0
    for g in singles:
        psyn = target ^ int(g.syndrome)
        pflags = int(pair_flags.get(psyn, 0))
        if not pflags:
            continue
        for pf in _frames(pflags):
            common = int(pair_common[(psyn, int(pf))])
            # There exists a single-fault location in this syndrome/frame group
            # that is absent from at least one exact pair witness.
            if int(g.locations_mask) & ~common:
                out |= 1 << (int(g.frame) ^ int(pf))
    return int(out)


def _structured_order3_flags(target: int, singles, single_frames, pair_flags, pair_common, w1, w2) -> tuple[int, dict[str, int]]:
    """Tight physical-order-3 frame candidates for a base-absent syndrome.

    Families:
      E3: exact three escape faults at distinct locations.
      W1+E2: one upstream local fault -> canonical weight-1 sector + exact
             two-distinct-location escape faults.
      W2+E1: two upstream local faults -> canonical weight-2 sector + one escape
             fault.  This is a syndrome/frame over-approximation of the physical
             two-upstream-fault interface and is retained conservatively.

    Exact <=2 states are intentionally excluded; if one existed, the priority
    decoder would already have used the base table.
    """
    t = int(target)
    f_e3 = _exact_escape3_flags(t, singles, pair_flags, pair_common)

    f_w1e2 = 0
    for s in w1:
        p = int(pair_flags.get(t ^ int(s.full), 0))
        if p:
            f_w1e2 |= _combine_flags(p, int(s.frame))

    f_w2e1 = 0
    for s in w2:
        sf = int(single_frames.get(t ^ int(s.full), 0))
        if sf:
            f_w2e1 |= _combine_flags(sf, int(s.frame))

    flags = int(f_e3 | f_w1e2 | f_w2e1)
    return flags, {'E3':int(f_e3), 'W1E2':int(f_w1e2), 'W2E1':int(f_w2e1)}


def _plan_rows(rows: list[dict[str, Any]], cumulative_fraction: float = 1.0):
    pos = [r for r in rows if float(r.get('A3_priority_interval_width_estimate', 0.0)) > 0]
    pos.sort(key=lambda r: float(r['A3_priority_interval_width_estimate']), reverse=True)
    total = sum(float(r['A3_priority_interval_width_estimate']) for r in pos)
    out = []
    c = 0.0
    for r in pos:
        w = float(r['A3_priority_interval_width_estimate'])
        c += w
        out.append({
            'key': r['key'], 'shots': int(r['shots']), 'unresolved': int(r['priority_unresolved_accepted']),
            'width': w, 'cumulative_fraction': (c/total if total else 1.0),
        })
        if total and c/total >= float(cumulative_fraction):
            break
    return out, total


def _replay_unresolved(meta, row_a3, seed, groups, bridge, clean_table, singles_conservative, sectors_conservative):
    """Replay the exact same S11-A3 sample set and retain only unresolved rows."""
    rng = np.random.default_rng(int(seed))
    unresolved_rows = []
    accepted = 0
    unresolved = 0
    for i in range(int(row_a3['shots'])):
        locs = a1._sample_locations(meta, groups, rng)
        r = a1._simulate_exact3(locs, bridge, rng, want_trace=False, want_state=True)
        if not r['accepted']:
            continue
        accepted += 1
        fa = int(r['full_syndrome_A_hex'], 16); fb = int(r['full_syndrome_B_hex'], 16)
        da = a3._decode_one(fa, bridge['table'], clean_table, singles_conservative, sectors_conservative)
        db = a3._decode_one(fb, bridge['table'], clean_table, singles_conservative, sectors_conservative)
        if da.predicted is None or db.predicted is None:
            unresolved += 1
            unresolved_rows.append((i, r, da, db))
    # Same-seed replay is a hard reproducibility gate.
    if accepted != int(row_a3['accepted']) or unresolved != int(row_a3['priority_unresolved_accepted']):
        raise AssertionError(('S11-A3 replay fingerprint mismatch', meta['key'], accepted, row_a3['accepted'], unresolved, row_a3['priority_unresolved_accepted']))
    return unresolved_rows


def _safe_le2_prefixes(bridge) -> set[int]:
    mask = (1 << a1.ACTUAL_BITS) - 1
    out = set()
    for full, flags in bridge['table'].items():
        if _singleton(int(flags)) is None:
            continue
        pref = int(full) & mask
        if pref not in bridge['unsafe']:
            out.add(pref)
    return out


def _rescore_unresolved(meta, row_a3, items, bridge, exact_flags_by_target, exact_families_by_target, safe_prefixes, max_examples=20):
    ways = int(meta['location_triples']); shots = int(row_a3['shots'])
    add_lower_sum = 0.0
    remain_unresolved = 0
    exact_resolved = 0
    exact_resolved_harmful = 0
    actual_not_in = 0
    rejectable_171 = 0
    colliding_171 = 0
    source = collections.Counter()
    examples = []
    mask = (1 << a1.ACTUAL_BITS) - 1

    for sample_i, r, old_da, old_db in items:
        dec = []
        fams = []
        for block, (full_hex, actual) in enumerate(((r['full_syndrome_A_hex'], r['actual_frame_A']), (r['full_syndrome_B_hex'], r['actual_frame_B']))):
            full = int(full_hex, 16)
            base = int(bridge['table'].get(full, 0))
            if base:
                pred = _singleton(base)
                dflags = base
                fam = {'base':base}
                src = 'le2_unique' if pred is not None else 'le2_ambiguous'
            else:
                dflags = int(exact_flags_by_target.get(full, 0))
                fam = dict(exact_families_by_target.get(full, {}))
                pred = _singleton(dflags)
                src = 'a4_exact3_unique' if pred is not None else ('a4_exact3_ambiguous' if dflags else 'a4_exact3_empty')
                if not (int(dflags) & (1 << int(actual))):
                    actual_not_in += 1
                    if len(examples) < max_examples:
                        examples.append({'kind':'actual_not_in_a4_union','sample':sample_i,'block':block,'stratum':meta['key'],'full':hex(full),'actual':FRAME_LABELS[int(actual)],'flags':[FRAME_LABELS[x] for x in _frames(dflags)],'families':{k:[FRAME_LABELS[x] for x in _frames(v)] for k,v in fam.items()}})
            dec.append((pred, src, dflags, full)); fams.append(fam); source[src] += 1

        pa, pb = dec[0][0], dec[1][0]
        if pa is not None and pb is not None:
            exact_resolved += 1
            da = a3.DecodeOne(int(pa), dec[0][1], int(bridge['table'].get(dec[0][3],0)), int(dec[0][2]), int(dec[0][2]))
            db = a3.DecodeOne(int(pb), dec[1][1], int(bridge['table'].get(dec[1][3],0)), int(dec[1][2]), int(dec[1][2]))
            inf = float(a3._score_resolved(r, da, db))
            add_lower_sum += inf
            if inf > TOL:
                exact_resolved_harmful += 1
                if len(examples) < max_examples:
                    examples.append({'kind':'newly_resolved_harmful','sample':sample_i,'stratum':meta['key'],'infidelity':inf,'full_A':hex(dec[0][3]),'full_B':hex(dec[1][3]),'pred_A':FRAME_LABELS[int(pa)],'pred_B':FRAME_LABELS[int(pb)],'actual_A':FRAME_LABELS[int(r['actual_frame_A'])],'actual_B':FRAME_LABELS[int(r['actual_frame_B'])]})
        else:
            remain_unresolved += 1
            # A full syndrome is base-absent by construction on every unresolved
            # block.  For live 171-bit rejection, rejectability additionally
            # requires the prefix to be absent from all accepted <=2 states.
            live_ok = True
            collided = False
            for bi, dd in enumerate(dec):
                if dd[0] is not None:
                    continue
                pref = int(dd[3]) & mask
                if pref in safe_prefixes:
                    live_ok = False; collided = True
            if live_ok:
                rejectable_171 += 1
            if collided:
                colliding_171 += 1
            if len(examples) < max_examples:
                examples.append({'kind':'still_unresolved','sample':sample_i,'stratum':meta['key'],'full_A':hex(dec[0][3]),'full_B':hex(dec[1][3]),'source_A':dec[0][1],'source_B':dec[1][1],'candidates_A':[FRAME_LABELS[x] for x in _frames(dec[0][2])],'candidates_B':[FRAME_LABELS[x] for x in _frames(dec[1][2])],'live171_rejectable_without_le2_collision':bool(live_ok)})

    # Replace only the old unresolved portion on the exact same samples.
    old_lower = float(row_a3['A3_priority_resolved_lower_estimate'])
    new_lower = old_lower + ways * (add_lower_sum / shots)
    new_upper = new_lower + ways * (remain_unresolved / shots)
    live171_upper = new_lower + ways * (colliding_171 / shots)
    return {
        'key': meta['key'],
        'shots': shots,
        'old_unresolved': int(row_a3['priority_unresolved_accepted']),
        'a4_newly_resolved': int(exact_resolved),
        'a4_newly_resolved_harmful': int(exact_resolved_harmful),
        'a4_still_unresolved': int(remain_unresolved),
        'a4_actual_frame_not_in_structured_union': int(actual_not_in),
        'a4_decoder_source_histogram': dict(sorted(source.items())),
        'a4_live171_rejectable_unresolved_samples': int(rejectable_171),
        'a4_live171_prefix_collision_unresolved_samples': int(colliding_171),
        'A3_old_lower': old_lower,
        'A3_old_upper': float(row_a3['A3_priority_unresolved_as_failure_upper_estimate']),
        'A3_a4_exactified_lower': float(new_lower),
        'A3_a4_exactified_upper': float(new_upper),
        'A3_a4_interval_width': float(new_upper-new_lower),
        'A3_a4_sampled_live171_rejection_upper': float(live171_upper),
        'A3_a4_sampled_live171_rejection_width': float(live171_upper-new_lower),
        'A3_a4_candidate_rejection_coefficient': float(ways * (rejectable_171 / shots)),
        'examples': examples,
    }


def _merge_global(rows_a3, replacements: dict[str, dict[str, Any]]):
    low = up = liveup = 0.0
    for r in rows_a3:
        k = r['key']
        if k in replacements:
            x = replacements[k]
            low += float(x['A3_a4_exactified_lower'])
            up += float(x['A3_a4_exactified_upper'])
            liveup += float(x['A3_a4_sampled_live171_rejection_upper'])
        else:
            low += float(r['A3_priority_resolved_lower_estimate'])
            up += float(r['A3_priority_unresolved_as_failure_upper_estimate'])
            liveup += float(r['A3_priority_unresolved_as_failure_upper_estimate'])
    return {
        'A3_a4_exactified_lower_same_A3_samples': low,
        'A3_a4_exactified_upper_same_A3_samples': up,
        'A3_a4_exactified_interval_width': up-low,
        'A3_a4_sampled_live171_rejection_upper': liveup,
        'A3_a4_sampled_live171_rejection_interval_width': liveup-low,
        'p3_predictions_exactified_lower_upper': {str(p):[low*p**3,up*p**3] for p in (2e-4,5e-4,1e-3)},
        'p3_predictions_with_sampled_live171_rejection': {str(p):[low*p**3,liveup*p**3] for p in (2e-4,5e-4,1e-3)},
    }


def _synthetic_self_test():
    # Build a small fake RecoveryFault catalogue with repeated syndrome/frame
    # effects across physical locations, then compare the compressed exact E3
    # reachability against brute force over all triples of distinct locations.
    RF = s7b2.RecoveryFault
    rows = [
        RF(0,0,0,'X',(0,),'a',1,0),
        RF(1,1,0,'X',(1,),'b',2,1),
        RF(2,2,0,'X',(2,),'c',4,2),
        RF(3,3,0,'X',(3,),'d',8,3),
        RF(4,0,0,'X',(0,),'e',3,1),
        RF(5,2,0,'X',(2,),'f',3,1),
    ]
    singles = _single_groups(rows)
    # Want every syndrome in this tiny universe, so no filtering artifact.
    wanted = set(range(16))
    pf, pc, meta = _build_filtered_pair_catalogue(rows, wanted, progress=0)

    brute: dict[int,int] = collections.defaultdict(int)
    # one fault class per selected physical location, with potentially multiple
    # classes at a location; enumerate class triples from three distinct locs.
    byloc = collections.defaultdict(list)
    for r in rows: byloc[int(r.location_id)].append(r)
    lids = sorted(byloc)
    import itertools
    for la,lb,lc in itertools.combinations(lids,3):
        for a in byloc[la]:
            for b in byloc[lb]:
                for c in byloc[lc]:
                    syn=int(a.recovery_syndrome)^int(b.recovery_syndrome)^int(c.recovery_syndrome)
                    fr=int(a.frame)^int(b.frame)^int(c.frame)
                    brute[syn] |= 1<<fr
    got = {s:_exact_escape3_flags(s,singles,pf,pc) for s in range(16)}
    if any(int(got[s]) != int(brute.get(s,0)) for s in range(16)):
        raise AssertionError(('compressed E3 != brute', {s:(got[s],brute.get(s,0)) for s in range(16) if got[s]!=brute.get(s,0)}))

    # Pair support intersection corner case A={0}, B={0,1}: the only legal pair
    # is (0,1), so both locations are common to every witness.
    gs=s7b2._build_groups(rows)
    # Find group (syn=1,frame=0) at loc0 and (syn=3,frame=1) at loc0/2, etc.
    # The general brute-force E3 equality above already exercises overlap; just
    # report the exact fingerprint here.
    return {
        'passed': True,
        'synthetic_single_groups': len(singles),
        'synthetic_kept_pair_effects': int(meta['kept_pair_syndrome_frame_effects']),
        'synthetic_exact3_states': sum(1 for v in got.values() if v),
        'synthetic_exact3_matches_bruteforce': True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hirano-src', type=Path)
    ap.add_argument('--signature-p', type=float, default=1e-3)
    ap.add_argument('--s11a3-strata', type=Path, default=Path('stim_pipeline/interface/s11a3_results/s11a3_strata.json'))
    ap.add_argument('--a3-seed', type=int, default=20260823)
    ap.add_argument('--target-cumulative-width', type=float, default=1.0)
    ap.add_argument('--pair-progress', type=int, default=500)
    ap.add_argument('--target-batch-size', type=int, default=256)
    ap.add_argument('--max-examples-per-stratum', type=int, default=20)
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--out-dir', type=Path, default=HERE/'s11a4_results')
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    st = _synthetic_self_test(); _dump(args.out_dir/'s11a4_self_test.json', st)
    if args.self_test:
        print(json.dumps(st, indent=2)); print('S11-A4 SELF-TEST PASS'); return

    if not args.s11a3_strata.exists():
        raise SystemExit(f'--s11a3-strata not found: {args.s11a3_strata}')
    rows_a3 = json.loads(args.s11a3_strata.read_text(encoding='utf-8'))
    plan, total_width = _plan_rows(rows_a3, float(args.target_cumulative_width))
    plan_obj = {
        'stage':'S11-A4-plan',
        'source_s11a3_strata':str(args.s11a3_strata),
        's11a3_total_interval_width':float(total_width),
        'target_cumulative_width':float(args.target_cumulative_width),
        'targets':plan,
        'target_width_fraction':(plan[-1]['cumulative_fraction'] if plan else 0.0),
    }
    _dump(args.out_dir/'s11a4_plan.json', plan_obj)
    print(json.dumps(plan_obj, indent=2), flush=True)
    if args.plan_only:
        print('S11-A4 PLAN COMPLETE'); return
    if args.hirano_src is None:
        raise SystemExit('--hirano-src required unless --self-test/--plan-only')

    t0=time.time()
    print('S11-A4: building complete S10 boundary bridge...', flush=True)
    bridge=a1.s10mc._bridge(args.hirano_src.resolve(),float(args.signature_p))
    print('S11-A4: rebuilding escape catalogue / exact <=2 table...', flush=True)
    paired,clean_analysis,_pol,_meta=s8c._escape_catalogue(args.hirano_src.resolve(),float(args.signature_p))
    clean_table={int(k):int(v) for k,v in clean_analysis['state_table'].items()}
    singles_cons=a2._single_effects_from_bridge(bridge)
    sectors_cons=list(bridge['sector'].values())
    singles=_single_groups(paired)
    single_frames=_single_frame_map(singles)
    w1=[s for s in sectors_cons if int(s.weight)==1]
    w2=[s for s in sectors_cons if int(s.weight)==2]
    if len(w1)!=21 or len(w2)!=42:
        raise AssertionError((len(w1),len(w2)))
    groups=a1._build_groups(int(bridge['meta']['physical_locations']))
    strata=a1._all_strata(groups); bymeta={s['key']:s for s in strata}; byrow={r['key']:r for r in rows_a3}

    # Pass 1: reproduce exactly the S11-A3 unresolved samples in the target strata.
    unresolved_by_key={}; target_full=set()
    for pi,p in enumerate(plan,1):
        key=p['key']; meta=bymeta[key]; row=byrow[key]
        seed=int(args.a3_seed)+1009*(strata.index(meta)+1)
        print(f'S11-A4 replay {pi}/{len(plan)} {key}: shots={row["shots"]}', flush=True)
        items=_replay_unresolved(meta,row,seed,groups,bridge,clean_table,singles_cons,sectors_cons)
        unresolved_by_key[key]=items
        for _si,r,da,db in items:
            if da.predicted is None: target_full.add(int(r['full_syndrome_A_hex'],16))
            if db.predicted is None: target_full.add(int(r['full_syndrome_B_hex'],16))
    print(f'S11-A4 unresolved full-syndrome targets: {len(target_full)}', flush=True)

    # Exactify target syndromes in bounded-memory batches.  We keep only the
    # final order-3 frame masks; the large temporary pair-support maps are
    # discarded after each batch.
    target_list=sorted(int(x) for x in target_full)
    bsz=max(1,int(args.target_batch_size))
    exact_flags_by_target={}; exact_families_by_target={}
    batch_meta=[]
    for b0 in range(0,len(target_list),bsz):
        batch=target_list[b0:b0+bsz]
        wanted=_desired_pair_syndromes(set(batch),singles,w1)
        print(f'S11-A4 pair batch {b0//bsz+1}/{math.ceil(len(target_list)/bsz)}: targets={len(batch)}, desired_pair_syndromes={len(wanted)}',flush=True)
        pf,pc,pm=_build_filtered_pair_catalogue(paired,wanted,int(args.pair_progress))
        for t in batch:
            flags,fam=_structured_order3_flags(t,singles,single_frames,pf,pc,w1,w2)
            exact_flags_by_target[int(t)]=int(flags)
            exact_families_by_target[int(t)]={k:int(v) for k,v in fam.items()}
        pm=dict(pm);pm['batch_index']=b0//bsz;pm['target_count']=len(batch);pm['target_start']=b0;pm['target_stop']=b0+len(batch)
        batch_meta.append(pm)
        del pf,pc,wanted
    pair_meta={
        'target_batch_size':bsz,
        'batches':len(batch_meta),
        'target_count':len(target_list),
        'batch_details':batch_meta,
    }
    _dump(args.out_dir/'s11a4_pair_catalogue_meta.json',pair_meta)
    _dump(args.out_dir/'s11a4_exact_flags.json',{hex(k):{'flags':v,'frames':[FRAME_LABELS[x] for x in _frames(v)],'families':{fk:[FRAME_LABELS[x] for x in _frames(fv)] for fk,fv in exact_families_by_target[k].items()}} for k,v in exact_flags_by_target.items()})

    safe_prefixes=_safe_le2_prefixes(bridge)
    replacements={}; all_examples=[]
    for pi,p in enumerate(plan,1):
        key=p['key']; meta=bymeta[key]; row=byrow[key]
        rr=_rescore_unresolved(meta,row,unresolved_by_key[key],bridge,exact_flags_by_target,exact_families_by_target,safe_prefixes,int(args.max_examples_per_stratum))
        if rr['a4_actual_frame_not_in_structured_union']:
            raise AssertionError(('A4 structured order3 union lost actual frame',key,rr['a4_actual_frame_not_in_structured_union']))
        replacements[key]=rr; all_examples.extend(rr['examples'])
        _dump(args.out_dir/'checkpoints'/f'{key}.json',rr)
        print(f"S11-A4 exactify {pi}/{len(plan)} {key}: unresolved {rr['old_unresolved']} -> {rr['a4_still_unresolved']}, newly_resolved={rr['a4_newly_resolved']}, A3=[{rr['A3_a4_exactified_lower']:.6g},{rr['A3_a4_exactified_upper']:.6g}], live171_upper={rr['A3_a4_sampled_live171_rejection_upper']:.6g}",flush=True)

    agg=_merge_global(rows_a3,replacements)
    summary={
        'stage':'S11-A4',
        'status':'SAMPLED_EXACT_OVERLAP_CLOSURE_NOT_PAPER_FINAL_A3',
        'scope':'same-sample exactification of S11-A3 unresolved exact-three trajectories in the dominant interval-width strata',
        'scientific_semantics':[
            'Exact <=2 S8-C full-syndrome singleton remains priority decoder.',
            'Pure escape3 candidate reachability enforces three distinct escape physical locations exactly.',
            'One-upstream-fault + escape2 uses only the 21 canonical weight-1 Steane sectors and exact distinct-location escape pairs.',
            'Two-upstream-fault + escape1 uses the 42 canonical weight-2 sectors plus exact escape singles as a conservative local frame reachability family.',
            'Only S11-A3 sampled unresolved trajectories are exactified; this is not yet an exhaustive order-3 syndrome table.',
            '195-bit selective rejection is a decoder/scoring diagnostic because the final 24 bits come from a future noiseless audit round outside the current factory fault budget.',
            '171-bit rejectability is reported only when the ambiguous sample prefix collides with no accepted <=2 full-syndrome state in the certified boundary table.',
        ],
        'source_s11a3_strata':str(args.s11a3_strata),
        'a3_seed_replayed':int(args.a3_seed),
        'target_plan':plan,
        'target_width_fraction':(plan[-1]['cumulative_fraction'] if plan else 0.0),
        'unresolved_full_syndrome_targets':len(target_full),
        'pair_catalogue':pair_meta,
        'replacement_rows':[replacements[k] for k in [x['key'] for x in plan]],
        'aggregate':agg,
        'elapsed_seconds':time.time()-t0,
        'next_gate':'If the interval is now narrow, run independent-seed high-statistics A5. If it remains wide, target only the remaining exact3-ambiguous prefixes that collide with accepted <=2 171-bit prefixes; these cannot be removed by a no-cost live rejection rule.',
    }
    _dump(args.out_dir/'s11a4_summary.json',summary)
    _dump(args.out_dir/'s11a4_replacements.json',replacements)
    _dump(args.out_dir/'s11a4_examples.json',all_examples)
    print(json.dumps(_json_safe(summary),indent=2),flush=True)
    print('S11-A4 COMPLETE: upload s11a4_summary.json and s11a4_replacements.json.',flush=True)


if __name__=='__main__':
    main()
