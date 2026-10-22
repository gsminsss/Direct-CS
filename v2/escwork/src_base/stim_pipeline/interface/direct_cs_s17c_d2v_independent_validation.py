#!/usr/bin/env python3
"""S17-C/D2V: frozen-threshold independent validation of the D2 logical-coset decoder.

Purpose
=======
D2 pilot data were used to choose a practical abstention threshold. D2V freezes that
choice *before* evaluation and uses independent sector-prior and evaluation seeds.
It does not sweep thresholds on the evaluation sample.

Default frozen policy:
  - logical-coset D2 decoder unchanged;
  - require explicit I/X/Z/Y coset coverage;
  - top-8 latent Steane sectors;
  - BP+OSD-CS, order 2;
  - max 8 correction effects;
  - finite logical-coset score gap >= 12;
  - Tier 0 and Tier 1 exact low-order policies unchanged.

A persistent per-p syndrome cache is used so interrupted long decodes can resume
without recomputing completed D2 records. Cache metadata includes the calibrated
sector prior and decoder hyperparameters; mismatches are rejected instead of
silently reusing stale entries.

This is still an empirical higher-order decoder validation. The exact structural
claim remains the frozen <=2 correct-or-reject certificate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_DETECTORS = 470
EXPECTED_LOCATIONS = 2407
EXPECTED_EFFECTS = 6918
PAULI_NAMES = {0: "I", 1: "X", 2: "Z", 3: "Y"}


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _prior_hash(prior: np.ndarray) -> str:
    a = np.asarray(prior, dtype=np.float64)
    return _sha256_bytes(a.tobytes(order="C"))


def _cache_paths(out_dir: Path, p: float) -> tuple[Path, Path]:
    tag = f"{p:g}".replace("-", "m").replace(".", "p")
    return out_dir / f"s17c_d2v_cache_{tag}.jsonl", out_dir / f"s17c_d2v_cache_{tag}_meta.json"


def _cache_meta(*, p: float, prior: np.ndarray, args, matrix_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "p": float(p),
        "prior_sha256": _prior_hash(prior),
        "matrix": {
            "augmented_matrix_shape": matrix_meta["augmented_matrix_shape"],
            "augmented_matrix_gf2_row_rank": matrix_meta["augmented_matrix_gf2_row_rank"],
            "raw_joint_unique_effects": matrix_meta["raw_joint_unique_effects"],
            "physical_locations_per_patch": matrix_meta["physical_locations_per_patch"],
        },
        "decoder": {
            "sector_candidates": int(args.sector_candidates),
            "bp_max_iter": int(args.bp_max_iter),
            "bp_method": str(args.bp_method),
            "ms_scaling_factor": float(args.ms_scaling_factor),
            "osd_method": str(args.osd_method),
            "osd_order": int(args.osd_order),
            "require_all_four_logical_cosets": True,
        },
    }


def _deserialize_d2(d2, obj: dict[str, Any]):
    def dec_c(c):
        if c is None:
            return None
        inv = {v: k for k, v in PAULI_NAMES.items()}
        return d2.CosetCandidate(
            sector=int(c["sector"]),
            final_pauli=int(inv[c["final_pauli"]]),
            downstream_pauli=int(inv[c["downstream_pauli"]]),
            score=float(c["score"]),
            correction_effect_count=int(c["correction_effect_count"]),
            assigned_locations=tuple(map(int, c["assigned_locations"])),
            selected_columns=tuple(map(int, c["selected_columns"])),
            prefix_distance=int(c["prefix_distance"]),
            source=str(c.get("source", "cache")),
        )

    inv = {v: k for k, v in PAULI_NAMES.items()}
    return d2.CosetDecodeResult(
        success=bool(obj["success"]),
        pauli=None if obj["pauli"] is None else int(inv[obj["pauli"]]),
        gap=float(obj["finite_coset_gap"]),
        coverage_paulis=tuple(int(inv[x]) for x in obj["coverage_paulis"]),
        candidate_count=int(obj["candidate_count"]),
        invalid_candidate_count=int(obj["invalid_candidate_count"]),
        best=dec_c(obj.get("best")),
        best_different=dec_c(obj.get("best_different")),
        pauli_logsumexp={int(inv[k]): float(v) for k, v in obj.get("pauli_logsumexp", {}).items()},
        pauli_best_score={int(inv[k]): float(v) for k, v in obj.get("pauli_best_score", {}).items()},
        pauli_candidate_counts={int(inv[k]): int(v) for k, v in obj.get("pauli_candidate_counts", {}).items()},
        sectors_tested=tuple(map(int, obj.get("sectors_tested", []))),
    )


def _load_cache(d2, cache_file: Path, meta_file: Path, wanted_meta: dict[str, Any], resume: bool):
    if not resume or not cache_file.exists():
        return {}
    if not meta_file.exists():
        raise RuntimeError(f"Cache exists without metadata: {cache_file}")
    got = _load_json(meta_file)
    if got != wanted_meta:
        raise RuntimeError(
            "Existing D2V cache metadata does not match this run. "
            "Use a new --out-dir or remove the stale cache files."
        )
    out = {}
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = int(r["record"], 16)
        out[q] = _deserialize_d2(d2, r["result"])
    return out


def _append_cache(cache_file: Path, q: int, result_obj: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"record": hex(int(q)), "result": result_obj}, sort_keys=True) + "\n")
        f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--d2-script", type=Path, default=None)
    ap.add_argument("--d1-script", type=Path, default=None)
    ap.add_argument("--s17c-c-script", type=Path, default=None)
    ap.add_argument("--s10-script", type=Path, default=None)
    ap.add_argument("--v13d-script", type=Path, default=None)
    ap.add_argument("--sector-map", type=Path, default=None)
    ap.add_argument("--sector-aware-reject-policy", type=Path, default=None)
    ap.add_argument("--x-circuit", type=Path, default=None)
    ap.add_argument("--z-circuit", type=Path, default=None)

    ap.add_argument("--p-values", type=float, nargs="+", default=[2e-4, 5e-4, 1e-3])
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--shots-per-p", type=int, nargs="+", default=None)
    ap.add_argument("--prior-calibration-shots", type=int, default=20000)
    ap.add_argument("--prior-seed", type=int, default=20260911)
    ap.add_argument("--evaluation-seed", type=int, default=20260921)

    ap.add_argument("--frozen-gap-threshold", type=float, default=12.0)
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd_cs")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--cache-flush-every", type=int, default=1,
                    help="Reserved for compatibility; cache is appended atomically after every decoded record.")
    ap.add_argument("--no-resume-cache", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.prior_seed == args.evaluation_seed:
        raise SystemExit("--prior-seed and --evaluation-seed must differ")
    if args.frozen_gap_threshold < 0:
        raise SystemExit("--frozen-gap-threshold must be >= 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.search_root.resolve()

    # Import D2 first; use its tested discovery/import helpers and exact decoder routines.
    # Prefer the interface copy, never a result-directory copy.
    d2_path = (args.d2_script or None)
    if d2_path is None:
        xs = list(root.rglob("direct_cs_s17c_d2_logical_coset_list_decoder.py"))
        if not xs:
            raise FileNotFoundError("Could not find direct_cs_s17c_d2_logical_coset_list_decoder.py")
        xs.sort(key=lambda p: (0 if "stim_pipeline\\interface" in str(p) or "stim_pipeline/interface" in str(p) else 1,
                               len(str(p)), str(p)))
        d2_path = xs[0]
    d2_path = Path(d2_path).resolve()

    # Local dynamic import (same semantics as D2).
    import importlib.util
    spec = importlib.util.spec_from_file_location("s17cd2v_d2", d2_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Can't import D2 from {d2_path}")
    d2 = importlib.util.module_from_spec(spec)
    sys.modules["s17cd2v_d2"] = d2
    spec.loader.exec_module(d2)

    d1_path = (args.d1_script or d2._find(root, "direct_cs_s17c_d_hybrid_bposd_abstention.py", ("stim_pipeline/interface",))).resolve()
    c_path = (args.s17c_c_script or d2._find(root, "direct_cs_s17c_c_finite_p_e2e_certified_lookup.py", ("stim_pipeline/interface",))).resolve()
    s10_path = (args.s10_script or d2._find(root, "direct_cs_s10_complete_physical_factory_mc.py", ("stim_pipeline/interface",))).resolve()
    v13d_path = (args.v13d_script or d2._find(root, "direct_cs_s17b_v13d_joint_postselection_certificate.py", ("stim_pipeline/interface",))).resolve()
    sector_path = (args.sector_map or d2._find(root, "s17c_a_sector_map.json", ("s17c_a_results",))).resolve()
    reject_path = (args.sector_aware_reject_policy or d2._find(root, "s17c_b_sector_aware_reject_policy.json", ("s17c_b_results",))).resolve()
    x_path = (args.x_circuit or d2._find(root, "s17b_v11_X_transition_circuit.stim", ("s17b_v11_order2_results",))).resolve()
    z_path = (args.z_circuit or d2._find(root, "s17b_v11_Z_transition_circuit.stim", ("s17b_v11_order2_results",))).resolve()

    d1 = d2._import_module("s17cd2v_d1", d1_path)
    s17cc = d2._import_module("s17cd2v_s17cc", c_path)
    s10mc = d2._import_module("s17cd2v_s10mc", s10_path)
    v13d = d2._import_module("s17cd2v_v13d", v13d_path)
    BpOsdDecoder = d1._require_ldpc()

    sectors = d1._sector_rows(sector_path)
    sector_dict = {int(r["steane_syndrome"]): r for r in sectors}
    reject_obj = _load_json(reject_path)
    reject_set = {int(x, 16) for x in reject_obj["total_sector_aware_rejected_syndromes_hex"]}
    if len(reject_set) != 14197 or 0 in reject_set:
        raise RuntimeError(f"Expected frozen 14,197 nonzero reject records, got {len(reject_set)}")

    print("S17-C/D2V: building downstream model ...", flush=True)
    downstream = s17cc._build_downstream_model(v13d, x_path, z_path)
    if len(downstream["effects"]) != EXPECTED_EFFECTS:
        raise RuntimeError((len(downstream["effects"]), EXPECTED_EFFECTS))
    cols = d1._effect_columns(downstream["effects"])
    h = d1._build_sparse_check_matrix(cols)
    h_aug = d2._build_augmented_matrix(h, cols)
    rank_h = d2._gf2_row_rank_sparse(h)
    rank_aug = d2._gf2_row_rank_sparse(h_aug)
    matrix_meta = {
        "detector_matrix_shape": list(map(int, h.shape)),
        "augmented_matrix_shape": list(map(int, h_aug.shape)),
        "detector_matrix_gf2_row_rank": int(rank_h),
        "augmented_matrix_gf2_row_rank": int(rank_aug),
        "logical_rows_independent_of_detector_rows": bool(rank_aug >= rank_h + 2),
        "raw_joint_unique_effects": len(downstream["effects"]),
        "nonzero_syndrome_effect_columns": len(cols),
        "frozen_reject_records": len(reject_set),
        "physical_locations_per_patch": downstream["meta"]["physical_locations_per_patch"],
        "sector_count": len(sectors),
    }

    # Structural/self-test: same core D2 conditions, and explicitly enforce frozen policy.
    p_smoke = max(1e-6, min(map(float, args.p_values)))
    pri_smoke = d1._effect_parity_priors(cols, p_smoke)
    dec_smoke = d1._make_bposd(BpOsdDecoder, h_aug, pri_smoke, args.bp_max_iter, args.bp_method,
                                args.ms_scaling_factor, args.osd_method, args.osd_order)
    smoke = {}
    for q in range(4):
        target = np.zeros(EXPECTED_DETECTORS + 2, dtype=np.uint8)
        target[EXPECTED_DETECTORS] = q & 1
        target[EXPECTED_DETECTORS + 1] = (q >> 1) & 1
        ok = False
        try:
            for _, v in d2._collect_decoder_vectors(dec_smoke, target, len(cols)):
                chk = np.asarray(h_aug @ v, dtype=np.uint8).reshape(-1) & 1
                if np.array_equal(chk, target):
                    ok = True
                    break
        except Exception:
            ok = False
        smoke[PAULI_NAMES[q]] = bool(ok)

    st = {
        "passed": bool(
            len(reject_set) == 14197 and 0 not in reject_set and rank_aug >= rank_h + 2
            and all(smoke.values()) and args.prior_seed != args.evaluation_seed
        ),
        "matrix": matrix_meta,
        "zero_detector_four_logical_coset_solver_smoke": smoke,
        "frozen_policy": {
            "gap_threshold": float(args.frozen_gap_threshold),
            "sector_candidates": int(args.sector_candidates),
            "osd_method": args.osd_method,
            "osd_order": int(args.osd_order),
            "max_correction_effects": int(args.max_correction_effects),
            "require_all_four_logical_cosets": True,
            "threshold_sweep_on_evaluation_set": False,
        },
        "independence": {
            "prior_seed": int(args.prior_seed),
            "evaluation_seed": int(args.evaluation_seed),
            "seeds_distinct": bool(args.prior_seed != args.evaluation_seed),
            "policy_frozen_before_evaluation": True,
        },
    }
    _dump(args.out_dir / "s17c_d2v_self_test.json", st)
    if not st["passed"]:
        raise RuntimeError(st)
    if args.self_test_only:
        print(json.dumps(st, indent=2), flush=True)
        print("S17-C/D2V SELF-TEST PASS", flush=True)
        return

    if args.shots_per_p is not None:
        if len(args.shots_per_p) != len(args.p_values):
            raise SystemExit("--shots-per-p length must match --p-values")
        shot_list = [int(x) for x in args.shots_per_p]
    else:
        shot_list = [int(args.shots)] * len(args.p_values)

    all_rows = []
    prior_rows = []
    audits = []
    t_all = time.time()

    for ip, (p, shots) in enumerate(zip(args.p_values, shot_list)):
        p = float(p)
        print(f"S17-C/D2V: independent sector-prior calibration p={p:g} ...", flush=True)
        prior, prior_meta = d1._calibrate_sector_prior(
            p, int(args.prior_calibration_shots), int(args.prior_seed + ip), s10mc
        )
        prior_meta = dict(prior_meta)
        prior_meta["validation_prior_seed"] = int(args.prior_seed + ip)
        prior_rows.append(prior_meta)

        print(f"S17-C/D2V: evaluation sampling p={p:g}, shots={shots}, seed={args.evaluation_seed + ip} ...", flush=True)
        sampled = s17cc._sample_one_p(
            p, int(shots), int(args.evaluation_seed + ip), s10mc, sector_dict, downstream, args.max_examples
        )
        queries = set(sampled["queries"])
        print(f"S17-C/D2V: exact <=2 resolving {len(queries)} observed patch records ...", flush=True)
        exact_decoder, exact_meta = s17cc._resolve_observed_records(
            queries, sectors, downstream["effects"], v13d, reject_set
        )
        unknown = sorted(q for q in queries if q not in reject_set and q not in exact_decoder)

        pri = d1._effect_parity_priors(cols, p)
        decoder = d1._make_bposd(
            BpOsdDecoder, h_aug, pri, args.bp_max_iter, args.bp_method,
            args.ms_scaling_factor, args.osd_method, args.osd_order
        )

        cache_file, meta_file = _cache_paths(args.out_dir, p)
        wanted_meta = _cache_meta(p=p, prior=prior, args=args, matrix_meta=matrix_meta)
        if meta_file.exists():
            got = _load_json(meta_file)
            if got != wanted_meta and not args.no_resume_cache:
                raise RuntimeError(
                    f"Stale cache metadata for p={p:g}. Use a fresh --out-dir or --no-resume-cache."
                )
        if args.no_resume_cache:
            if cache_file.exists():
                cache_file.unlink()
            _dump(meta_file, wanted_meta)
            high_raw = {}
        else:
            if not meta_file.exists():
                _dump(meta_file, wanted_meta)
            high_raw = _load_cache(d2, cache_file, meta_file, wanted_meta, resume=True)

        reused = sum(q in high_raw for q in unknown)
        todo = [q for q in unknown if q not in high_raw]
        print(
            f"S17-C/D2V: D2 unknown={len(unknown)}, cache_reused={reused}, to_decode={len(todo)}, "
            f"top-{args.sector_candidates} sectors x 4 cosets ...",
            flush=True,
        )
        tdec = time.time()
        for qi, q in enumerate(todo):
            d = d2._decode_unknown_coset(
                d1, q, sectors, prior, p, decoder, h_aug, cols, args.sector_candidates,
                require_all_paulis=True,
            )
            high_raw[q] = d
            _append_cache(cache_file, q, d2._serialize(d))
            if (qi + 1) % 25 == 0 or qi + 1 == len(todo):
                print(f"  decoded {qi+1}/{len(todo)} new records (cache total {reused+qi+1}/{len(unknown)})", flush=True)
        dec_s = time.time() - tdec

        # Keep only records in this evaluation query set when reporting audit metrics.
        eval_high = {q: high_raw[q] for q in unknown}
        high_adapter = {q: d2._to_d1_result(d1, d) for q, d in eval_high.items()}
        cover_hist = collections.Counter(len(d.coverage_paulis) for d in eval_high.values())
        success = sum(bool(d.success) for d in eval_high.values())
        gaps = [float(d.gap) for d in eval_high.values() if d.success]
        cand_counts = [int(d.candidate_count) for d in eval_high.values() if d.success]
        finite_all = all(math.isfinite(x) for x in gaps)
        all_four = all(len(d.coverage_paulis) == 4 for d in eval_high.values())

        audit = {
            "p": p,
            "evaluation_shots": int(shots),
            "evaluation_seed": int(args.evaluation_seed + ip),
            "prior_seed": int(args.prior_seed + ip),
            "observed_patch_records": len(queries),
            "exact_unique_records": len(exact_decoder),
            "frozen_reject_records_observed": sum(q in reject_set for q in queries),
            "out_of_certificate_unique_records": len(unknown),
            "cache_reused_records": int(reused),
            "newly_decoded_records": int(len(todo)),
            "coset_decode_success_records": int(success),
            "coset_decode_failure_records": int(len(unknown) - success),
            "logical_coset_coverage_histogram": {str(k): int(v) for k, v in sorted(cover_hist.items())},
            "all_unknown_records_have_four_cosets": bool(all_four),
            "all_success_gaps_finite": bool(finite_all),
            "finite_gap_summary": None if not gaps else {
                "count": len(gaps),
                "median": float(np.median(gaps)),
                "p10": float(np.quantile(gaps, 0.1)),
                "p90": float(np.quantile(gaps, 0.9)),
            },
            "candidate_count_summary": None if not cand_counts else {
                "median": float(np.median(cand_counts)),
                "p10": float(np.quantile(cand_counts, 0.1)),
                "p90": float(np.quantile(cand_counts, 0.9)),
                "max": int(max(cand_counts)),
            },
            "new_decode_seconds": float(dec_s),
            "exact_catalogue_meta": exact_meta,
            "effect_prior_min": float(np.min(pri)),
            "effect_prior_max": float(np.max(pri)),
        }
        audits.append(audit)

        # ONE pre-registered threshold only. No evaluation-set sweep.
        row = d1._finish_threshold(
            sampled, exact_decoder, reject_set, high_adapter, s10mc,
            float(args.frozen_gap_threshold), int(args.max_correction_effects), int(args.max_examples)
        )
        row["validation_policy"] = {
            "frozen_gap_threshold": float(args.frozen_gap_threshold),
            "threshold_sweep_on_this_evaluation_sample": False,
            "evaluation_seed": int(args.evaluation_seed + ip),
            "prior_seed": int(args.prior_seed + ip),
        }

        front = {k: v for k, v in sampled["predecode"].items() if k != "examples"}
        all_rows.append({
            "p": p,
            "shots": int(shots),
            "evaluation_seed": int(args.evaluation_seed + ip),
            "frontend": front,
            "conservative_reference": {
                "core_accepted": int(front["core_accepted"]),
                "exact_unique_query_records": len(exact_decoder),
                "out_of_certificate_unique_query_records": len(unknown),
            },
            "frozen_threshold_result": row,
        })
        print(json.dumps({
            "p": p,
            "frozen_gap": float(args.frozen_gap_threshold),
            "acceptance": row["end_to_end_acceptance"],
            "conditional_infidelity": row["conditional_logical_infidelity"],
            "intrinsic_oracle": row["conditional_intrinsic_oracle_infidelity"],
            "decoder_excess": row["decoder_excess_infidelity_conditional"],
            "harmful": row["harmful_accepted_samples"],
            "accepted_harmful_le2": row["accepted_harmful_samples_with_total_fault_locations_le2"],
            "le2_gate": row["global_le2_regression_gate_pass"],
            "higher_order_accept": row["accepted_with_any_higher_order_decode"],
            "higher_order_abstain": row["higher_order_abstain"],
        }, indent=2), flush=True)

        sample_keys = unknown[: min(100, len(unknown))]
        _dump(
            args.out_dir / f"s17c_d2v_coset_examples_p{p:g}.json",
            [{"record": hex(q), **d2._serialize(eval_high[q])} for q in sample_keys],
        )

    global_le2 = all(r["frozen_threshold_result"]["global_le2_regression_gate_pass"] for r in all_rows)
    all_four = all(a["all_unknown_records_have_four_cosets"] for a in audits)
    finite = all(a["all_success_gaps_finite"] for a in audits)
    no_fail = all(a["coset_decode_failure_records"] == 0 for a in audits)

    summary = {
        "stage": "S17-C/D2V frozen-threshold independent validation",
        "purpose": "Validate the D2 pilot decision without tuning on the evaluation samples.",
        "pre_registered_policy": {
            "gap_threshold": float(args.frozen_gap_threshold),
            "threshold_source": "frozen after S17-C/D2 pilot; not optimized on D2V evaluation samples",
            "tier0": "frozen 14,197 certified ambiguous records -> reject",
            "tier1": "exact total-budget<=2 unique catalogue -> exact frame",
            "tier2": "D2 augmented logical-coset BP+OSD on unknown records only",
            "sector_candidates": int(args.sector_candidates),
            "bp_max_iter": int(args.bp_max_iter),
            "bp_method": args.bp_method,
            "ms_scaling_factor": float(args.ms_scaling_factor),
            "osd_method": args.osd_method,
            "osd_order": int(args.osd_order),
            "max_correction_effects": int(args.max_correction_effects),
            "require_all_four_logical_cosets": True,
        },
        "independence": {
            "prior_seed_base": int(args.prior_seed),
            "evaluation_seed_base": int(args.evaluation_seed),
            "seeds_distinct": bool(args.prior_seed != args.evaluation_seed),
            "threshold_sweep_on_evaluation_samples": False,
        },
        "matrix": matrix_meta,
        "self_test": st,
        "sector_prior_calibration": prior_rows,
        "decoder_audit": audits,
        "finite_p_rows": all_rows,
        "main_gate": {
            "frozen_low_order_reject_policy_unchanged": True,
            "global_no_sampled_accepted_harmful_event_with_total_physical_fault_locations_le2": bool(global_le2),
            "all_unknown_records_explicitly_cover_I_X_Z_Y": bool(all_four),
            "all_D2_success_gaps_are_finite": bool(finite),
            "zero_coset_decode_failures": bool(no_fail),
            "policy_frozen_before_evaluation": True,
            "evaluation_threshold_sweep_prohibited": True,
            "passed": bool(global_le2 and all_four and finite and no_fail),
        },
        "interpretation_rule": (
            "If the frozen D2 policy reproduces its large D1 error reduction while retaining useful acceptance, "
            "stop decoder development and proceed to S18 noisy d13 delivery boundary. If it does not, do not "
            "retune the gap on these evaluation samples; move to a categorical physical-location likelihood model."
        ),
        "ideal_d13_close_warning": (
            "The 470-bit boundary still ends with an ideal downstream d13 close. D2V validates only the practical "
            "higher-order decoder policy, not final noisy delivered-d13 performance."
        ),
        "timing_seconds": float(time.time() - t_all),
    }

    _dump(args.out_dir / "s17c_d2v_summary.json", summary)
    _dump(args.out_dir / "s17c_d2v_finite_p_rows.json", all_rows)
    _dump(args.out_dir / "s17c_d2v_decoder_audit.json", audits)
    _dump(args.out_dir / "s17c_d2v_sector_priors.json", prior_rows)
    print(json.dumps({
        "main_gate_pass": summary["main_gate"]["passed"],
        "global_le2_regression_gate_pass": bool(global_le2),
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
