#!/usr/bin/env python3
"""S18-B3: fresh independent validation of the B2R-firewalled 638-bit D2 decoder.

This stage follows a full S18-B2R PASS. The B2R exact <=2 firewall and gap=10
are frozen before B3. No threshold/firewall tuning is performed on B3 data.

Frozen policy used by this paper workflow
-----------------------------------------
    gap >= 10

Freeze rationale (B1 calibration only): among calibrated candidates that kept
stress-point (p=1e-3) operational acceptance >= 0.20 and preserved the sampled
<=2 regression gate, gap=10 had the lowest observed closed-boundary harmful
fraction while retaining useful acceptance.  The B1 program's automatic gap=16
recommendation was explicitly a fallback after its over-conservative zero-
harmful criterion failed; gap=16 collapsed stress-point acceptance to ~0.03.

Decoder input is exactly the 638 operational detector bits.  The 168 terminal
truth bits are scoring-only and are never visible to the decoder.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_GAP = 10.0
OP_DETECTORS = 638
TRUTH_BITS = 168
PAULI_NAMES = ["I", "X", "Z", "Y"]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find(root: Path, names: list[str], prefer: tuple[str, ...] = ()) -> Path:
    # IMPORTANT: preserve B1 path semantics.  A name containing a path
    # separator (e.g. ``msc-ls/src``) is an exact root-relative path, not
    # a request to recursively search for every directory called ``src``.
    hits: list[Path] = []
    for n in names:
        p = Path(n)
        if p.is_absolute() and p.exists():
            hits.append(p)
        elif "/" in n or "\\" in n:
            q = root / p
            if q.exists():
                hits.append(q)
        else:
            hits.extend(root.rglob(n))
    if not hits:
        raise FileNotFoundError((str(root), names))
    hits = list(dict.fromkeys(x.resolve() for x in hits))
    hits.sort(key=lambda p: (
        tuple(0 if token in str(p) else 1 for token in prefer),
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)), str(p),
    ))
    return hits[0]


def _tag(p: float) -> str:
    return f"{p:.12g}".replace('.', 'p').replace('-', 'm')


def _cache_paths(out: Path, p: float):
    t = _tag(p)
    return out / f"s18b3_cache_{t}.jsonl", out / f"s18b3_cache_{t}_meta.json"


def _deserialize_d2(d2, obj):
    # d2._serialize() stores Pauli labels as strings (I/X/Z/Y) and names the
    # confidence field ``finite_coset_gap``.  Accept both that canonical schema
    # and the older integer/``gap`` schema so interrupted B2 runs can resume.
    inv = {name: i for i, name in enumerate(PAULI_NAMES)}

    def pval(x):
        if x is None:
            return None
        if isinstance(x, str):
            if x in inv:
                return int(inv[x])
            return int(x)
        return int(x)

    def cand(c):
        if c is None:
            return None
        return d2.CosetCandidate(
            sector=int(c["sector"]),
            final_pauli=pval(c["final_pauli"]),
            downstream_pauli=pval(c["downstream_pauli"]),
            score=float(c["score"]),
            correction_effect_count=int(c["correction_effect_count"]),
            assigned_locations=tuple(map(int, c["assigned_locations"])),
            selected_columns=tuple(map(int, c["selected_columns"])),
            prefix_distance=int(c["prefix_distance"]),
            source=str(c.get("source", "cache")),
        )

    gap_key = "finite_coset_gap" if "finite_coset_gap" in obj else "gap"
    return d2.CosetDecodeResult(
        success=bool(obj["success"]),
        pauli=pval(obj.get("pauli")),
        gap=float(obj[gap_key]),
        coverage_paulis=tuple(pval(x) for x in obj.get("coverage_paulis", [])),
        candidate_count=int(obj["candidate_count"]),
        invalid_candidate_count=int(obj["invalid_candidate_count"]),
        best=cand(obj.get("best")),
        best_different=cand(obj.get("best_different")),
        pauli_logsumexp={pval(k): float(v) for k, v in obj.get("pauli_logsumexp", {}).items()},
        pauli_best_score={pval(k): float(v) for k, v in obj.get("pauli_best_score", {}).items()},
        pauli_candidate_counts={pval(k): int(v) for k, v in obj.get("pauli_candidate_counts", {}).items()},
        sectors_tested=tuple(map(int, obj.get("sectors_tested", []))),
    )


def _append_cache(path: Path, q: int, d2, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"record":hex(int(q)),"decode":d2._serialize(obj)},sort_keys=True)+"\n")


def _load_cache(d2, path: Path, meta: Path, wanted: dict[str,Any], resume: bool):
    if not resume or not path.exists(): return {}
    if not meta.exists() or _load(meta) != wanted:
        raise RuntimeError("Stale S18-B3 cache. Use a fresh --out-dir or --no-resume-cache.")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o = json.loads(line)
        out[int(o["record"],16)] = _deserialize_d2(d2,o["decode"])
    return out


def _hash_obj(x: Any) -> str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def _decode_b3(q: int, fw_exact: dict[int,int], fw_amb: set[int], cheap_exact: dict[int,int], cheap_amb: set[int], high: dict[int,Any], gap: float, max_effects: int):
    """Frozen decision order: zero -> B2R firewall -> cheap exact -> D2."""
    q = int(q)
    if q == 0:
        return True, 0, "zero_structural"
    if q in fw_amb:
        return False, None, "firewall_ambiguous"
    if q in fw_exact:
        return True, int(fw_exact[q]), "firewall_exact"
    if q in cheap_amb:
        return False, None, "cheap_ambiguous"
    if q in cheap_exact:
        return True, int(cheap_exact[q]), "cheap_exact"
    d = high.get(q)
    if d is None or not bool(d.success) or d.pauli is None:
        return False, None, "d2_failure"
    if not math.isfinite(float(d.gap)) or float(d.gap) < float(gap):
        return False, None, "d2_gap_abstain"
    k = int(d.best.correction_effect_count) if d.best is not None else 10**9
    if k > int(max_effects):
        return False, None, "d2_effect_count_abstain"
    return True, int(d.pauli), "d2_high"


def _score_b3(sampled, fw_exact, fw_amb, cheap_exact, cheap_amb, high, b1mod, s10mc, gap: float, max_effects: int, max_examples: int):
    shots = int(sampled["shots"]); accepted = closed_acc = residual_acc = 0
    closed_harm = closed_harm_le2 = 0; raw_harm = 0; inf_closed = 0.0; intrinsic_closed = 0.0
    tiers = collections.Counter(); residual_fault_hist = collections.Counter(); accepted_fault_hist = collections.Counter()
    harmful_ex = []; residual_ex = []
    for tr in sampled["trajectories"]:
        oka, pa, ta = _decode_b3(tr.record_a, fw_exact, fw_amb, cheap_exact, cheap_amb, high, gap, max_effects)
        okb, pb, tb = _decode_b3(tr.record_b, fw_exact, fw_amb, cheap_exact, cheap_amb, high, gap, max_effects)
        if not oka or not okb:
            tiers[f"reject_A:{ta}" if not oka else f"reject_B:{tb}"] += 1
            continue
        tiers[f"A:{ta}"] += 1; tiers[f"B:{tb}"] += 1
        accepted += 1; accepted_fault_hist[tr.total_fault_locations] += 1
        inf = float(b1mod._raw_corrected_infidelity(s10mc, tr, int(pa), int(pb)))
        raw_harm += int(inf > b1mod.TOL)
        closed = (int(tr.truth_a) == 0 and int(tr.truth_b) == 0)
        if closed:
            closed_acc += 1; inf_closed += inf; intrinsic_closed += float(tr.intrinsic_oracle_infidelity)
            if inf > b1mod.TOL:
                closed_harm += 1; closed_harm_le2 += int(tr.total_fault_locations <= 2)
                if len(harmful_ex) < max_examples:
                    harmful_ex.append({
                        "record_A": hex(tr.record_a), "record_B": hex(tr.record_b),
                        "pred_A": PAULI_NAMES[int(pa)], "pred_B": PAULI_NAMES[int(pb)],
                        "actual_A": PAULI_NAMES[tr.actual_pauli_a], "actual_B": PAULI_NAMES[tr.actual_pauli_b],
                        "truth_A": hex(tr.truth_a), "truth_B": hex(tr.truth_b), "infidelity": inf,
                        "total_fault_locations": tr.total_fault_locations, "tier_A": ta, "tier_B": tb,
                    })
        else:
            residual_acc += 1; residual_fault_hist[tr.total_fault_locations] += 1
            if len(residual_ex) < max_examples:
                residual_ex.append({
                    "record_A": hex(tr.record_a), "record_B": hex(tr.record_b),
                    "truth_A": hex(tr.truth_a), "truth_B": hex(tr.truth_b),
                    "raw_proxy_infidelity": inf, "total_fault_locations": tr.total_fault_locations,
                    "tier_A": ta, "tier_B": tb,
                })
    front = int(sampled["predecode"]["frontend_accepted"]); core = int(sampled["predecode"]["core_accepted"])
    executed = 194.0 + (front/shots)*696.0 + (core/shots)*(2*b1mod.TOTAL_LOCATIONS)
    return {
        "p": float(sampled["p"]), "shots": shots, "evaluation_seed": int(sampled["seed"]), "gap": float(gap),
        "operational_accepted": accepted, "operational_acceptance": accepted/shots,
        "operational_acceptance_wilson95": b1mod._wilson(accepted,shots),
        "attempts_per_operational_accept": shots/accepted if accepted else None,
        "truth_closed_accepted": closed_acc, "truth_closed_fraction_among_accepted": closed_acc/accepted if accepted else None,
        "detectable_residual_accepted": residual_acc, "detectable_residual_fraction_among_accepted": residual_acc/accepted if accepted else None,
        "closed_boundary_harmful": closed_harm,
        "closed_boundary_harmful_fraction_among_truth_closed": closed_harm/closed_acc if closed_acc else None,
        "closed_boundary_harmful_wilson95": b1mod._wilson(closed_harm,closed_acc) if closed_acc else None,
        "closed_boundary_zero_harmful_upper95_if_applicable": b1mod._zero_upper95(closed_acc) if closed_acc and closed_harm == 0 else None,
        "sampled_closed_boundary_harmful_with_total_fault_locations_le2": closed_harm_le2,
        "sampled_le2_regression_gate_pass": closed_harm_le2 == 0,
        "closed_boundary_conditional_raw_frame_infidelity": inf_closed/closed_acc if closed_acc else None,
        "closed_boundary_intrinsic_oracle_infidelity": intrinsic_closed/closed_acc if closed_acc else None,
        "raw_terminal_proxy_harmful_all_accepted_DO_NOT_QUOTE_AS_LOGICAL": raw_harm,
        "tier_counts": dict(sorted(tiers.items())),
        "accepted_fault_histogram": {str(k): int(v) for k,v in sorted(accepted_fault_hist.items())},
        "residual_fault_histogram": {str(k): int(v) for k,v in sorted(residual_fault_hist.items())},
        "stage_level_expected_active_locations_per_attempt": executed,
        "stage_level_active_location_proxy_per_operational_accept": executed/(accepted/shots) if accepted else None,
        "examples": {"closed_boundary_harmful": harmful_ex, "detectable_residual": residual_ex},
    }


def _collect_b2_seeds(b2_summary: dict[str,Any]) -> set[int]:
    out: set[int] = set()
    for r in b2_summary.get("evaluation_rows", []):
        for k in ("evaluation_seed", "prior_seed", "calibration_seed"):
            if r.get(k) is not None: out.add(int(r[k]))
    for r in b2_summary.get("fresh_sector_priors", []):
        if r.get("seed") is not None: out.add(int(r["seed"]))
    for r in b2_summary.get("decoder_audit", []):
        for k in ("evaluation_seed", "prior_seed"):
            if r.get(k) is not None: out.add(int(r[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s18m-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m_results"))
    ap.add_argument("--s18m2-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--b1-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b1_calibration_results"))
    ap.add_argument("--b2-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b2_validation_results"))
    ap.add_argument("--b2r-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b2r_firewall_results"))
    ap.add_argument("--p-values", type=float, nargs="+", default=[2e-4,5e-4,1e-3])
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=[5000,5000,3000])
    ap.add_argument("--frozen-gap", type=float, default=FROZEN_GAP)
    ap.add_argument("--prior-calibration-shots", type=int, default=30000)
    ap.add_argument("--prior-seed", type=int, default=20261201)
    ap.add_argument("--evaluation-seed", type=int, default=20261221)
    ap.add_argument("--signature-p", type=float, default=0.001)
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd_cs")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--no-resume-cache", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s18b3_validation_results"))
    args = ap.parse_args()

    if args.self_test_only:
        t1=_decode_b3(0x11,{0x11:1},set(),{0x11:0},set(),{},FROZEN_GAP,8)
        t2=_decode_b3(0x22,{}, {0x22},{0x22:2},set(),{},FROZEN_GAP,8)
        t3=_decode_b3(0x33,{},set(),{0x33:2},set(),{},FROZEN_GAP,8)
        passed=(t1==(True,1,"firewall_exact") and t2==(False,None,"firewall_ambiguous") and t3==(True,2,"cheap_exact"))
        print(json.dumps({"passed": passed, "stage": "S18-B3 firewalled fresh independent validation", "frozen_gap": FROZEN_GAP, "operational_detector_bits": OP_DETECTORS, "truth_bits_decoder_visible": False, "threshold_sweep_on_evaluation": False, "firewall_order": "zero -> B2R exact<=2 -> cheap exact -> D2", "default_prior_seed": 20261201, "default_evaluation_seed": 20261221, "priority_tests":{"firewall_exact_over_cheap":t1[2],"firewall_ambiguous_over_cheap":t2[2],"cheap_fallback":t3[2]}}, indent=2))
        if not passed: raise RuntimeError("S18-B3 self-test failed")
        return

    if abs(float(args.frozen_gap)-FROZEN_GAP) > 1e-12:
        raise SystemExit(f"B3 threshold is frozen at {FROZEN_GAP:g}. Do not retune on evaluation data.")
    if len(args.shots_per_p) != len(args.p_values):
        raise SystemExit("--shots-per-p length must match --p-values")
    if args.prior_seed == args.evaluation_seed:
        raise SystemExit("prior and evaluation seeds must differ")

    root = args.search_root.resolve()
    out = (root/args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    b1dir = (root/args.b1_result_dir).resolve() if not args.b1_result_dir.is_absolute() else args.b1_result_dir.resolve()
    b2dir = (root/args.b2_result_dir).resolve() if not args.b2_result_dir.is_absolute() else args.b2_result_dir.resolve()
    b2rdir = (root/args.b2r_result_dir).resolve() if not args.b2r_result_dir.is_absolute() else args.b2r_result_dir.resolve()
    b1summary = b1dir/"s18b1_summary.json"; b2summary = b2dir/"s18b2_summary.json"; b2rsummary = b2rdir/"s18b2r_summary.json"
    for pth in (b1summary,b2summary,b2rsummary):
        if not pth.exists(): raise FileNotFoundError(pth)
    b1 = _load(b1summary); b2prev = _load(b2summary); b2rprev = _load(b2rsummary)
    if not bool(b1.get("calibration_only",False)):
        raise RuntimeError("Expected calibration-only S18-B1 summary")
    if bool(b2prev.get("validation_gate",{}).get("passed",True)):
        raise RuntimeError("Expected preserved S18-B2 v1 FAIL provenance before B3")
    if int(b2prev.get("validation_gate",{}).get("sampled_closed_boundary_harmful_le2_total",0)) <= 0:
        raise RuntimeError("S18-B2 v1 summary does not preserve the observed <=2 decoder regression")
    if not bool(b2rprev.get("validation_gate",{}).get("passed",False)):
        raise RuntimeError("S18-B2R full firewall audit is not PASS")
    if bool(b2rprev.get("validation_gate",{}).get("truth_bits_decoder_visible",True)):
        raise RuntimeError("B2R reports truth bits decoder-visible")
    # Guard against accidentally reusing B1/B2 seeds.
    b1_prior = int(b1.get("independence_plan",{}).get("B1_prior_seed_base",-1))
    b1_eval = int(b1.get("independence_plan",{}).get("B1_calibration_seed_base",-1))
    new_seeds = {args.prior_seed+i for i in range(len(args.p_values))} | {args.evaluation_seed+i for i in range(len(args.p_values))}
    old_seeds = {x for x in (b1_prior,b1_eval) if x >= 0} | _collect_b2_seeds(b2prev)
    overlap = sorted(new_seeds & old_seeds)
    if overlap:
        raise RuntimeError(("B3 seeds overlap prior calibration/validation seeds",overlap))

    b1mod = _import("s18b2_b1", _find(root,["direct_cs_s18b1_638_d2_calibration.py"],("stim_pipeline/interface",)))
    d2 = _import("s18b2_d2", _find(root,["direct_cs_s17c_d2_logical_coset_list_decoder.py"],("stim_pipeline/interface",)))
    d1 = _import("s18b2_d1", _find(root,["direct_cs_s17c_d_hybrid_bposd_abstention.py"],("stim_pipeline/interface",)))
    s10mc = _import("s18b2_s10", _find(root,["direct_cs_s10_complete_physical_factory_mc.py"],("stim_pipeline/interface",)))
    cert = _import("s18b2_v13d", _find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"],("stim_pipeline/interface",)))
    s18m = _import("s18b2_s18m", _find(root,["direct_cs_s18m_modular_noisy_d13_certificate.py"],("stim_pipeline/interface",)))
    m2 = _import("s18b2_m2", _find(root,["direct_cs_s18m2_terminal_truth_closure.py"],("stim_pipeline/interface",)))
    ca = _import("s18b2_ca", _find(root,["direct_cs_s17c_a_sector_to_470_map.py"],("stim_pipeline/interface",)))
    s8b = _import("s18b2_s8b", _find(root,["zxz_core_boundary_signature_map.py"],("stim_pipeline/interface",)))
    v7 = _import("s18b2_v7", _find(root,["direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"],("stim_pipeline/interface",)))
    v11 = _import("s18b2_v11", _find(root,["direct_cs_s17b_v11_complete_missing_basis.py"],("stim_pipeline/interface",)))
    s18a = _import("s18b3_s18a", _find(root,["direct_cs_s18a_noisy_d13_boundary_certificate.py"],("stim_pipeline/interface",)))
    b2rmod = _import("s18b3_b2r", _find(root,["direct_cs_s18b2r_low_order_firewall.py"],("stim_pipeline/interface",)))
    d1.EXPECTED_DETECTORS=OP_DETECTORS; d2.EXPECTED_DETECTORS=OP_DETECTORS
    BpOsdDecoder=d1._require_ldpc()

    s18mr=(root/args.s18m_result_dir).resolve() if not args.s18m_result_dir.is_absolute() else args.s18m_result_dir.resolve()
    s18m2r=(root/args.s18m2_result_dir).resolve() if not args.s18m2_result_dir.is_absolute() else args.s18m2_result_dir.resolve()
    xtruth=s18m2r/"s18m2_X_truth_closed.stim"; ztruth=s18m2r/"s18m2_Z_truth_closed.stim"
    sector_op=s18mr/"s18m_sector_map_638.json"; m2sum=s18m2r/"s18m2_summary.json"
    for pth in (xtruth,ztruth,sector_op,m2sum):
        if not pth.exists(): raise FileNotFoundError(pth)
    ms=_load(m2sum)
    if not bool(ms.get("main_gate",{}).get("combined_modular_boundary_output_contract_pass",False)):
        raise RuntimeError("S18-M2 structural gate is not PASS")
    if bool(ms.get("policy",{}).get("truth_bits_decoder_visible",True)):
        raise RuntimeError("truth bits unexpectedly decoder-visible")

    hirano=_find(root,["msc-ls/src"]); sahay=_find(root,["reference/fd3_baseline_ckt.stim"])
    model=b1mod._build_truth_operational_model(cert,xtruth,ztruth)
    sectors,smeta=b1mod._build_sector_truth_map(root,s18m,m2,ca,s8b,v7,v11,s18a,hirano,sahay,sector_op,float(args.signature_p))
    sector_by={int(r["steane_syndrome"]):r for r in sectors}
    _dump(out/"s18b3_sector_map_638_truth.json",sectors)

    print("S18-B3: building frozen 638-bit D2 matrix...",flush=True)
    cols=d1._effect_columns(model["effects"])
    h=d1._build_sparse_check_matrix(cols); h_aug=d2._build_augmented_matrix(h,cols)
    rank_h=d2._gf2_row_rank_sparse(h); rank_aug=d2._gf2_row_rank_sparse(h_aug)
    exact,amb,cmeta=b1mod._build_fast_catalogue(sectors,model["effects"])
    matrix_meta={"H_shape":list(map(int,h.shape)),"H_aug_shape":list(map(int,h_aug.shape)),"H_rank":int(rank_h),"H_aug_rank":int(rank_aug),
                 "logical_rows_independent":bool(rank_aug>=rank_h+2),"effect_columns":len(cols),"raw_effect_classes":len(model["effects"]),
                 "fast_catalogue":cmeta,"downstream":model["meta"],"sector":smeta}
    if not (rank_h==638 and rank_aug>=640 and smeta["existing_S18M_operational_and_raw_labels_reproduced"]):
        raise RuntimeError(("S18-B3 matrix/sector selftest failed",matrix_meta))

    print("S18-B3: building frozen B2R exact <=2 firewall index...",flush=True)
    fw_idx = b2rmod.build_firewall_index(sectors,model)
    zero_fw = b2rmod.resolve_query(0,fw_idx,max_examples=2)
    if not (zero_fw["status"] == "exact" and zero_fw["pauli"] == "I"):
        raise RuntimeError(("B2R zero-record contract failed inside B3",zero_fw))

    rows=[]; audits=[]; priors=[]; firewall_audits=[]; fw_cache={0:zero_fw}; t0=time.time()
    for ip,(p,shots) in enumerate(zip(map(float,args.p_values),map(int,args.shots_per_p))):
        ps=int(args.prior_seed+ip); es=int(args.evaluation_seed+ip)
        print(f"S18-B3: fresh prior p={p:g}, seed={ps} ...",flush=True)
        prior,pmeta=d1._calibrate_sector_prior(p,int(args.prior_calibration_shots),ps,s10mc); pmeta["seed"]=ps; priors.append(pmeta)
        print(f"S18-B3: fresh independent evaluation p={p:g}, shots={shots}, seed={es} ...",flush=True)
        sampled=b1mod._sample_one_p(p,shots,es,s10mc,sector_by,model,int(args.max_examples))
        queries=set(map(int,sampled["queries"]))

        # Resolve every newly observed operational record through the frozen B2R firewall first.
        new_fw=sorted(q for q in queries if q not in fw_cache)
        print(f"S18-B3: firewall queries={len(queries)}, reused_in_B3={len(queries)-len(new_fw)}, new_exact_joins={len(new_fw)}",flush=True)
        tfw=time.time()
        for ii,q in enumerate(new_fw):
            fw_cache[q]=b2rmod.resolve_query(q,fw_idx,max_examples=2)
            if (ii+1)%25==0 or ii+1==len(new_fw):
                print(f"  firewall resolved {ii+1}/{len(new_fw)} new records",flush=True)
        fw_exact={q:int(fw_cache[q]["pauli_int"]) for q in queries if fw_cache[q]["status"]=="exact"}
        fw_amb={q for q in queries if fw_cache[q]["status"]=="ambiguous_reject"}
        fw_high={q for q in queries if fw_cache[q]["status"]=="higher_order_only"}
        # Firewall has priority over the older cheap catalogue. Conflicts are reported, not tuned away.
        fw_conflicts=[]
        for q in sorted(queries):
            if q in fw_exact and q in exact and int(fw_exact[q]) != int(exact[q]):
                fw_conflicts.append({"record":hex(q),"kind":"exact_vs_cheap_exact","firewall":PAULI_NAMES[fw_exact[q]],"cheap":PAULI_NAMES[int(exact[q])]})
            elif q in fw_exact and q in amb:
                fw_conflicts.append({"record":hex(q),"kind":"firewall_exact_vs_cheap_ambiguous","firewall":PAULI_NAMES[fw_exact[q]]})
            elif q in fw_amb and q in exact:
                fw_conflicts.append({"record":hex(q),"kind":"firewall_ambiguous_vs_cheap_exact","cheap":PAULI_NAMES[int(exact[q])]})

        unknown=sorted(q for q in queries if q!=0 and q not in fw_exact and q not in fw_amb and q not in exact and q not in amb)
        # By construction every D2 query must have no <=2 truth-closed realization.
        leakage_to_d2=[q for q in unknown if q not in fw_high]
        if leakage_to_d2:
            raise RuntimeError(("B2R firewall routing violation: low-order-resolvable query reached D2",[hex(q) for q in leakage_to_d2[:10]]))

        pri=d1._effect_parity_priors(cols,p)
        decoder=d1._make_bposd(BpOsdDecoder,h_aug,pri,args.bp_max_iter,args.bp_method,args.ms_scaling_factor,args.osd_method,args.osd_order)

        cache,meta=_cache_paths(out,p)
        wanted={"stage":"S18-B3","frozen_gap":FROZEN_GAP,"p":p,
                "firewall_policy":"S18-B2R exact<=2 before cheap catalogue/D2",
                "firewall_meta_hash":_hash_obj(fw_idx.get("meta",{})),
                "matrix_hash":_hash_obj({k:matrix_meta[k] for k in ("H_shape","H_aug_shape","H_rank","H_aug_rank","effect_columns")}),
                "prior_hash":hashlib.sha256(np.asarray(prior,dtype=np.float64).tobytes()).hexdigest(),
                "sector_candidates":args.sector_candidates,"bp_max_iter":args.bp_max_iter,"bp_method":args.bp_method,
                "ms_scaling_factor":args.ms_scaling_factor,"osd_method":args.osd_method,"osd_order":args.osd_order}
        if args.no_resume_cache:
            if cache.exists(): cache.unlink()
            if meta.exists(): meta.unlink()
            _dump(meta,wanted); high={}
        else:
            if not meta.exists(): _dump(meta,wanted)
            high=_load_cache(d2,cache,meta,wanted,True)
        reused=sum(q in high for q in unknown); todo=[q for q in unknown if q not in high]
        print(f"S18-B3: records={len(queries)}, firewall_exact={len(fw_exact)}, firewall_ambiguous={len(fw_amb)}, firewall_higher={len(fw_high)}, cheap_exact_after_fw={sum(q in exact and q not in fw_exact and q not in fw_amb for q in queries)}, cheap_ambiguous_after_fw={sum(q in amb and q not in fw_exact and q not in fw_amb for q in queries)}, D2 unknown={len(unknown)}, reused={reused}, to_decode={len(todo)}",flush=True)
        td=time.time()
        for ii,q in enumerate(todo):
            d=d2._decode_unknown_coset(d1,q,sectors,prior,p,decoder,h_aug,cols,int(args.sector_candidates),require_all_paulis=True)
            high[q]=d; _append_cache(cache,q,d2,d)
            if (ii+1)%25==0 or ii+1==len(todo): print(f"  decoded {ii+1}/{len(todo)} new records",flush=True)
        gaps=[float(high[q].gap) for q in unknown if high[q].success and math.isfinite(float(high[q].gap))]
        cover=collections.Counter(len(high[q].coverage_paulis) for q in unknown)
        audit={"p":p,"shots":shots,"evaluation_seed":es,"prior_seed":ps,"observed_records":len(queries),
               "firewall_exact_observed":len(fw_exact),"firewall_ambiguous_observed":len(fw_amb),"firewall_higher_order_only_observed":len(fw_high),
               "firewall_conflict_count":len(fw_conflicts),"firewall_new_resolution_seconds":time.time()-tfw,
               "cheap_exact_observed_after_firewall":sum(q in exact and q not in fw_exact and q not in fw_amb for q in queries),
               "cheap_ambiguous_observed_after_firewall":sum(q in amb and q not in fw_exact and q not in fw_amb for q in queries),
               "d2_unknown":len(unknown),"cache_reused":reused,"new_decodes":len(todo),
               "d2_success":sum(bool(high[q].success) for q in unknown),"d2_failure":sum(not bool(high[q].success) for q in unknown),
               "coverage_histogram":{str(k):int(v) for k,v in sorted(cover.items())},"finite_gap_count":len(gaps),
               "gap_median":float(np.median(gaps)) if gaps else None,"gap_p10":float(np.quantile(gaps,.1)) if gaps else None,
               "gap_p90":float(np.quantile(gaps,.9)) if gaps else None,"new_decode_seconds":time.time()-td}
        audits.append(audit)
        firewall_audits.append({
            "p":p,"evaluation_seed":es,"status_counts":{"exact":len(fw_exact),"ambiguous_reject":len(fw_amb),"higher_order_only":len(fw_high)},
            "conflicts_with_cheap_catalogue":fw_conflicts[:50],
            "non_higher_records":[fw_cache[q] for q in sorted(queries) if fw_cache[q]["status"]!="higher_order_only"][:200],
        })
        r=_score_b3(sampled,fw_exact,fw_amb,exact,amb,high,b1mod,s10mc,FROZEN_GAP,int(args.max_correction_effects),int(args.max_examples))
        r["evaluation_seed"]=es; r["prior_seed"]=ps; r["frozen_gap"]=FROZEN_GAP
        rows.append(r)
        print(json.dumps({"p":p,"frozen_gap":FROZEN_GAP,"acc":r["operational_acceptance"],"truth_closed":r["truth_closed_accepted"],
                          "residual_frac":r["detectable_residual_fraction_among_accepted"],"closed_harmful":r["closed_boundary_harmful"],
                          "le2_harmful":r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"],
                          "firewall_exact":len(fw_exact),"firewall_ambiguous":len(fw_amb),"D2_unknown":len(unknown)},indent=2),flush=True)

    validation_gate={
        "b2r_full_audit_pass":bool(b2rprev.get("validation_gate",{}).get("passed",False)),
        "b2r_b2_v1_witnesses_intercepted":bool(b2rprev.get("validation_gate",{}).get("b2_v1_witnesses_intercepted_before_D2",False)),
        "frozen_gap_not_swept":True,
        "firewall_policy_frozen_before_B3":True,
        "fresh_seed_nonoverlap_with_B1_B2":True,
        "truth_bits_decoder_visible":False,
        "all_D2_queries_have_no_le2_truth_closed_realization":True,
        "all_d2_decodes_successful":all(a["d2_failure"]==0 for a in audits),
        "all_unknown_records_have_four_cosets":all(set(a["coverage_histogram"].keys()) <= {"4"} and a["d2_success"]==a["d2_unknown"] for a in audits),
        "sampled_le2_regression_pass_all_p":all(r["sampled_le2_regression_gate_pass"] for r in rows),
        "sampled_closed_boundary_harmful_le2_total":sum(int(r["sampled_closed_boundary_harmful_with_total_fault_locations_le2"]) for r in rows),
        "closed_boundary_harmful_total":sum(int(r["closed_boundary_harmful"]) for r in rows),
        "firewall_conflict_total_informational":sum(int(a["firewall_conflict_count"]) for a in audits),
    }
    validation_gate["passed"]=bool(validation_gate["b2r_full_audit_pass"] and validation_gate["b2r_b2_v1_witnesses_intercepted"] and
        validation_gate["frozen_gap_not_swept"] and validation_gate["firewall_policy_frozen_before_B3"] and
        validation_gate["fresh_seed_nonoverlap_with_B1_B2"] and not validation_gate["truth_bits_decoder_visible"] and
        validation_gate["all_D2_queries_have_no_le2_truth_closed_realization"] and validation_gate["all_d2_decodes_successful"] and
        validation_gate["all_unknown_records_have_four_cosets"] and validation_gate["sampled_le2_regression_pass_all_p"] and
        validation_gate["sampled_closed_boundary_harmful_le2_total"]==0)

    summary={
        "stage":"S18-B3 fresh independent 638-bit D2 validation with frozen S18-B2R firewall",
        "independent_evaluation":True,"threshold_sweep_on_evaluation":False,"frozen_gap":FROZEN_GAP,
        "freeze_source":"S18-B1 calibration Pareto decision; gap=10 selected before B2 evaluation",
        "freeze_rule":"among B1 candidates with stress p=1e-3 operational acceptance >=0.20 and sampled <=2 regression PASS, choose the lowest observed stress-point closed-boundary harmful fraction; ties favor higher acceptance then smaller gap",
        "b1_automatic_fallback_gap_16_used":False,
        "structural_theorem_source":"S18-M2 plus S18-B2R low-order firewall audit; B3 is fresh empirical finite-p validation and does not replace either structural certificate",
        "decoder_input":{"operational_detector_bits":638,"terminal_truth_bits_decoder_visible":False,"truth_bits_scoring_only":168},
        "decoder":{"algorithm":"D2 four-logical-coset BP+OSD list decoder","sector_candidates":int(args.sector_candidates),"osd_method":args.osd_method,"osd_order":int(args.osd_order),"max_correction_effects":int(args.max_correction_effects)},
        "matrix":matrix_meta,"firewall":{"policy":"zero -> B2R exact<=2 -> cheap exact -> D2","meta":fw_idx.get("meta",{}),"prior_B2R_gate":b2rprev.get("validation_gate",{})},"fresh_sector_priors":priors,"decoder_audit":audits,"firewall_audit":firewall_audits,"evaluation_rows":rows,
        "validation_gate":validation_gate,"timing_seconds":time.time()-t0,
        "paper_use":"B3 is the fresh post-repair practical-decoder validation. Structural <=2 claims remain sourced from S18-M2/B2R; delivered-resource claims still wait for S19/S20."
    }
    _dump(out/"s18b3_summary.json",summary); _dump(out/"s18b3_evaluation_rows.json",rows); _dump(out/"s18b3_decoder_audit.json",audits); _dump(out/"s18b3_firewall_audit.json",firewall_audits); _dump(out/"s18b3_sector_priors.json",priors)
    print("S18-B3 FIREWALLED FRESH VALIDATION COMPLETE",flush=True)
    print(json.dumps(validation_gate,indent=2),flush=True)
    if not validation_gate["passed"]:
        raise RuntimeError(("S18-B3 validation gate failed",validation_gate))

if __name__ == "__main__":
    main()
