#!/usr/bin/env python3
"""S21-OPT4C: targeted high-stat exact-three follow-up for optimized direct d5->d13.

This stage is explicitly PILOT-INFORMED.  The target set and shot allocation
below are frozen before looking at OPT4C outcomes.

Why targeted
------------
OPT4B found:
  * 2 harmful strata, both decoder-prior invariant and intrinsic;
  * the dominant unobserved decoder-risk mass sits in DOWN_DIRECT_D5_D13-heavy
    strata, especially pure DOWN^3;
  * the omitted pilot-zero strata outside the fixed target set have only a
    small aggregate one-sided pilot risk bound compared with the dominant
    selected strata.

The fixed target plan contains:
  * the 2 observed harmful strata;
  * all pilot strata with large location-triple mass and nonzero truth-closed
    acceptance whose crude one-sided pilot risk exceeded the fixed screening
    threshold;
  * one CORE_PRE+DOWN^2 control stratum that had zero pilot acceptance.

No decoder hyperparameter is retuned.

Outputs
-------
  s21opt4c_plan.json
  s21opt4c_summary.json
  s21opt4c_target_rows_by_decoder_p.json
  s21opt4c_harmful_witnesses.json
  s21opt4c_d2_cache_<p>.jsonl

Interpretation
--------------
OPT4C is a targeted precision/prior-stability follow-up.  Its reconstructed
global A3 combines fresh target-stratum estimates with the untouched OPT4B
pilot rows outside the target set.  Because target selection was pilot-informed,
the reconstructed global value is still labelled "targeted/reconstructed" and
not automatically paper-final.  The script separately reports the fresh target
contribution and zero-event upper bounds so that a final closure decision can
be made without hiding selection effects.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_GAP = 10.0
OP_DETECTORS = 263
TRUTH_PER_PROBE = 168
COMBINED_TRUTH_BITS = 336
DOWNSTREAM_LOCATIONS_PER_PATCH = 2132
TOTAL_POTENTIAL_LOCATIONS = 5154
DECODER_PS = (5e-4, 1e-3)

# Fixed-before-OPT4C-outcomes, pilot-informed target plan.
TARGET_SHOTS = {
    # Observed harmful strata: precision targets.
    "MIDED1+CORE_POST1+DOWN_DIRECT_D5_D131": 10000,
    "CORE_POST2+DOWN_DIRECT_D5_D131": 10000,

    # Dominant pure-downstream decoder-risk stratum.
    "DOWN_DIRECT_D5_D133": 20000,

    # Large one-pre + two-down strata with pilot truth-closed acceptance.
    "FRONT1+DOWN_DIRECT_D5_D132": 5000,
    "CAT_PRE1+DOWN_DIRECT_D5_D132": 5000,
    "MIDED1+DOWN_DIRECT_D5_D132": 5000,
    "CAT_POST1+DOWN_DIRECT_D5_D132": 5000,
    "CORE_POST1+DOWN_DIRECT_D5_D132": 5000,

    # Negative-control screen: pilot had zero operational acceptance.
    "CORE_PRE1+DOWN_DIRECT_D5_D132": 1000,

    # Remaining large one-down pilot-zero risk strata.
    "FRONT1+MIDED1+DOWN_DIRECT_D5_D131": 2000,
    "FRONT1+CORE_POST1+DOWN_DIRECT_D5_D131": 2000,
    "FRONT1+CAT_PRE1+DOWN_DIRECT_D5_D131": 2000,
    "FRONT2+DOWN_DIRECT_D5_D131": 2000,
    "MIDED1+CAT_POST1+DOWN_DIRECT_D5_D131": 2000,
    "CAT_POST1+CORE_POST1+DOWN_DIRECT_D5_D131": 2000,
    "MIDED2+DOWN_DIRECT_D5_D131": 2000,

    # Borderline pilot-risk strata.
    "CAT_PRE2+DOWN_DIRECT_D5_D131": 1000,
    "CAT_POST2+DOWN_DIRECT_D5_D131": 1000,
}

PAULI_NAMES = ("I", "X", "Z", "Y")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (complex, np.complexfloating)):
        z = complex(x)
        return {"__complex__": [float(z.real), float(z.imag)]}
    raise TypeError(type(x).__name__)


def imp(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def find(root: Path, names: list[str], prefer: tuple[str, ...] = ()):
    hits = []
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
        len(str(p)),
        str(p),
    ))
    return hits[0]


def scan_firewall(path: Path, queries: set[int]):
    wanted = set(map(int, queries))
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for ii, line in enumerate(f, 1):
            r = json.loads(line)
            q = int(r["record"], 16)
            if q in wanted:
                out[q] = r
            if ii % 1000000 == 0:
                print(f"  firewall {ii:,}: matched {len(out)}/{len(wanted)}", flush=True)
    return out


def sector_rows(path: Path):
    raw = load(path)
    out = []
    for q0 in raw:
        q = dict(q0)
        q["detector_638_hex"] = q["detector_263_hex"]
        q["detector_638_weight"] = q["detector_263_weight"]
        q["detector_470_hex"] = q["detector_263_hex"]
        q["terminal_truth_168_hex"] = q["future_truth_168_hex"]
        q["terminal_truth_168_weight"] = q["future_truth_168_weight"]
        out.append(q)
    if len(out) != 64 or len({q["detector_638_hex"] for q in out}) != 64:
        raise RuntimeError("bad 263-bit sector map")
    if any(int(q["future_truth_168_hex"], 16) != 0 for q in out):
        raise RuntimeError("canonical sector future truth is nonzero")
    z = [q for q in out if int(q["detector_638_hex"], 16) == 0]
    if len(z) != 1 or z[0]["raw_terminal_pauli"] != "I":
        raise RuntimeError(("bad zero sector", z))
    return out


def closure_condition_decoder_effects(d1, effects):
    kept, latent = [], []
    for e in effects:
        try:
            d1._effect_columns([e])
            kept.append(e)
        except RuntimeError as ex:
            if "zero-syndrome nonidentity downstream effect found" in str(ex):
                latent.append(e)
            else:
                raise
    if len(latent) != 3:
        raise RuntimeError(("optimized latent residual count drift", len(latent)))
    return kept, latent


def prior_from_c1(summary, p):
    for r in summary["prior_calibrations"]:
        if abs(float(r["p"]) - float(p)) < 1e-15:
            sp = r["sector_probability"]
            arr = np.asarray([float(sp[str(i)]) for i in range(64)], dtype=float)
            if abs(float(arr.sum()) - 1) > 1e-10:
                raise RuntimeError("prior norm")
            return arr, r
    raise RuntimeError(("prior missing", p))


def load_d2_cache(b3, d2, paths: list[Path]):
    out = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[int(r["record"], 16)] = b3._deserialize_d2(d2, r["decode"])
    return out


def append_d2(path: Path, q: int, d2, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"record": hex(int(q)), "decode": d2._serialize(obj)},
            sort_keys=True
        ) + "\n")


def one_sided_zero_event_upper(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return 1.0
    # Exact Clopper-Pearson upper bound for k=0.
    return 1.0 - alpha ** (1.0 / n)


def aggregate_rows(a2, rows):
    return a2._aggregate(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--opt3b-result-dir", type=Path, default=Path("stim_pipeline/interface/s21opt3b_results"))
    ap.add_argument("--opt3c0-result-dir", type=Path, default=Path("stim_pipeline/interface/s21opt3c0_results"))
    ap.add_argument("--opt3c1-result-dir", type=Path, default=Path("stim_pipeline/interface/s21opt3c1_results"))
    ap.add_argument("--opt4b-result-dir", type=Path, default=Path("stim_pipeline/interface/s21opt4b_results"))
    ap.add_argument("--seed", type=int, default=20271031)
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd_cs")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--max-examples-per-stratum", type=int, default=8)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s21opt4c_results"))
    args = ap.parse_args()

    root = args.search_root.resolve()

    def R(p: Path):
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    bdir = R(args.opt3b_result_dir)
    c0dir = R(args.opt3c0_result_dir)
    c1dir = R(args.opt3c1_result_dir)
    b4dir = R(args.opt4b_result_dir)
    out = R(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    opt4b = imp(
        "s21opt4c_opt4b",
        find(root, ["direct_cs_s21opt4b_optimized_exact3_pilot.py"], ("stim_pipeline/interface",))
    )
    a2 = imp(
        "s21opt4c_a2",
        find(root, ["direct_cs_s19a2a_exact3_independent_highstat.py"], ("stim_pipeline/interface", "s19work"))
    )
    a1 = imp(
        "s21opt4c_a1",
        find(root, ["direct_cs_s11a1_exact3_stratified_pilot.py"], ("stim_pipeline/interface",))
    )
    b1 = imp(
        "s21opt4c_b1",
        find(root, ["direct_cs_s18b1_638_d2_calibration.py"], ("stim_pipeline/interface",))
    )
    b3 = imp(
        "s21opt4c_b3",
        find(root, ["direct_cs_s18b3_firewalled_fresh_validation.py"], ("stim_pipeline/interface",))
    )
    d1 = imp(
        "s21opt4c_d1",
        find(root, ["direct_cs_s17c_d_hybrid_bposd_abstention.py"], ("stim_pipeline/interface",))
    )
    d2 = imp(
        "s21opt4c_d2",
        find(root, ["direct_cs_s17c_d2_logical_coset_list_decoder.py"], ("stim_pipeline/interface",))
    )
    s10 = imp(
        "s21opt4c_s10",
        find(root, ["direct_cs_s10_complete_physical_factory_mc.py"], ("stim_pipeline/interface",))
    )
    cert = imp(
        "s21opt4c_cert",
        find(root, ["direct_cs_s17b_v13d_joint_postselection_certificate.py"], ("stim_pipeline/interface",))
    )

    opt4b.patch_globals(b1, d1, d2, b3, a2)
    b1._stage_from_tick = opt4b.direct_stage
    cert.EXPECTED_LOCATIONS = DOWNSTREAM_LOCATIONS_PER_PATCH
    cert.EXPECTED_DETECTORS = 431

    # Required prior-stage artifacts.
    c1sum = load(c1dir / "s21opt3c1_summary.json")
    b4sum = load(b4dir / "s21opt4b_summary.json")
    b4rows = load(b4dir / "s21opt4b_rows_by_decoder_p.json")
    b4plan = load(b4dir / "s21opt4b_plan.json")

    if c1sum.get("status") != "PASS" or not c1sum.get("main_gate", {}).get("passed", False):
        raise RuntimeError("OPT3C1 not PASS")
    if b4sum.get("status") != "PILOT_NOT_PAPER_FINAL":
        raise RuntimeError(("unexpected OPT4B status", b4sum.get("status")))
    if int(b4plan["stratum_count"]) != 84 or int(b4plan["total_location_triples"]) != math.comb(TOTAL_POTENTIAL_LOCATIONS, 3):
        raise RuntimeError("OPT4B plan drift")

    fwpath = c0dir / "s21opt3c0_firewall.jsonl.gz"
    sectors = sector_rows(c0dir / "s21opt3c0_sector_map_263_truth.json")
    sector_by = {int(r["steane_syndrome"]): r for r in sectors}
    xtruth = bdir / "s21opt3b_X_direct_d5_d13_noisy.stim"
    ztruth = bdir / "s21opt3b_Z_direct_d5_d13_noisy.stim"
    for p in (fwpath, xtruth, ztruth):
        if not p.exists():
            raise FileNotFoundError(p)

    print("S21-OPT4C: rebuilding optimized physical model...", flush=True)
    model = b1._build_truth_operational_model(cert, xtruth, ztruth)
    decoder_effects, latent = closure_condition_decoder_effects(d1, model["effects"])
    groups = opt4b.build_groups(a1, a2, model)
    strata = opt4b.all_strata(groups, a2)
    meta_by_key = {r["key"]: r for r in strata}

    missing = [k for k in TARGET_SHOTS if k not in meta_by_key]
    if missing:
        raise RuntimeError(("target strata missing", missing))

    # Pilot-informed fixed target-selection diagnostics.
    pilot_low = {r["key"]: r for r in b4rows["0.0005"]}
    non_target_zero_upper = 0.0
    omitted_rows = []
    pilot_zero_u = one_sided_zero_event_upper(200)
    for k, r in pilot_low.items():
        if k in TARGET_SHOTS:
            continue
        if int(r["closed_boundary_harmful_samples"]) == 0 and int(r["truth_closed_operational_accept"]) > 0:
            u = int(r["location_triples"]) * pilot_zero_u
            non_target_zero_upper += u
            omitted_rows.append({"key": k, "pilot_binary_A3_upper95": u})

    plan = {
        "stage": "S21-OPT4C targeted high-stat fixed-before-OPT4C-outcomes plan",
        "selection_basis": "pilot-informed; target set fixed before OPT4C outcomes",
        "seed": int(args.seed),
        "decoder_ps": list(DECODER_PS),
        "threshold_sweep": False,
        "target_strata_count": len(TARGET_SHOTS),
        "total_fresh_target_shots": int(sum(TARGET_SHOTS.values())),
        "targets": [
            {
                "key": k,
                "shots": int(TARGET_SHOTS[k]),
                "location_triples": int(meta_by_key[k]["location_triples"]),
                "pilot_harmful_low_p": int(pilot_low[k]["closed_boundary_harmful_samples"]),
                "pilot_truth_closed_accept_low_p": int(pilot_low[k]["truth_closed_operational_accept"]),
            }
            for k in TARGET_SHOTS
        ],
        "pilot_non_target_zero_event_binary_A3_upper95_sum": float(non_target_zero_upper),
        "note": (
            "The non-target upper bound is a crude sum of per-stratum pilot k=0 "
            "one-sided bounds and is reported only as a screening diagnostic."
        ),
    }
    dump(out / "s21opt4c_plan.json", plan)

    print(json.dumps({
        "target_strata": plan["target_strata_count"],
        "total_fresh_target_shots": plan["total_fresh_target_shots"],
        "pilot_non_target_zero_upper95_sum": plan["pilot_non_target_zero_event_binary_A3_upper95_sum"],
    }, indent=2), flush=True)

    if args.plan_only:
        print("S21-OPT4C plan-only PASS")
        return

    # Frozen practical decoder.
    cols = d1._effect_columns(decoder_effects)
    h = d1._build_sparse_check_matrix(cols)
    h_aug = d2._build_augmented_matrix(h, cols)
    rh = int(d2._gf2_row_rank_sparse(h))
    ra = int(d2._gf2_row_rank_sparse(h_aug))
    if rh != 263 or ra < 265:
        raise RuntimeError(("matrix drift", rh, ra))
    cheap_exact, cheap_amb, cheap_meta = b1._build_fast_catalogue(sectors, decoder_effects)

    # Generate fresh physical samples for the fixed target set.
    sampled = {}
    all_queries = set()
    t0 = time.time()
    for ti, (key, shots) in enumerate(TARGET_SHOTS.items()):
        meta = meta_by_key[key]
        arr = []
        base_seed = int(args.seed) + ti * 10000019
        for j in range(int(shots)):
            shot_seed = (base_seed + j * 1000000007) & ((1 << 63) - 1)
            rng = np.random.default_rng(shot_seed)
            locs = a2._sample_locations(meta, groups, rng)
            rr = a2._simulate_exact3(a1, b1, sector_by, model, locs, rng, want_trace=False)
            arr.append(rr)
            if rr.get("accepted_predecode", False):
                all_queries.add(int(rr["record_a"]))
                all_queries.add(int(rr["record_b"]))
        sampled[key] = arr
        print(
            f"S21-OPT4C sampled {ti+1}/{len(TARGET_SHOTS)} "
            f"{key}: {shots} shots; queries={len(all_queries)}",
            flush=True,
        )

    print(f"S21-OPT4C: scanning firewall for {len(all_queries)} observed records...", flush=True)
    fw = scan_firewall(fwpath, all_queries)
    fw_exact = {q: int(r["pauli_int"]) for q, r in fw.items() if r["status"] == "exact"}
    fw_amb = {q for q, r in fw.items() if r["status"] == "ambiguous_reject"}
    if fw_exact.get(0, None) != 0:
        raise RuntimeError("zero firewall drift")

    unknown = sorted(
        q for q in all_queries
        if q != 0
        and q not in fw_exact
        and q not in fw_amb
        and q not in cheap_exact
        and q not in cheap_amb
    )

    BpOsdDecoder = d1._require_ldpc()
    priors, decoders, caches, cache_paths = {}, {}, {}, {}

    for p in DECODER_PS:
        prior, pmeta = prior_from_c1(c1sum, p)
        priors[p] = prior
        ep = d1._effect_parity_priors(cols, p)
        decoders[p] = d1._make_bposd(
            BpOsdDecoder, h_aug, ep,
            int(args.bp_max_iter), args.bp_method, float(args.ms_scaling_factor),
            args.osd_method, int(args.osd_order)
        )
        tag = f"{p:.12g}".replace(".", "p").replace("-", "m")
        inherited = [
            c1dir / f"s21opt3c1_d2_cache_{tag}.jsonl",
            b4dir / f"s21opt4b_d2_cache_{tag}.jsonl",
        ]
        caches[p] = load_d2_cache(b3, d2, inherited)
        cp = out / f"s21opt4c_d2_cache_{tag}.jsonl"
        cache_paths[p] = cp
        if cp.exists():
            cp.unlink()

        todo = [q for q in unknown if q not in caches[p]]
        print(f"S21-OPT4C pdec={p:g}: inherited={len(caches[p])}, new D2={len(todo)}", flush=True)
        for ii, q in enumerate(todo):
            dd = d2._decode_unknown_coset(
                d1, q, sectors, prior, p, decoders[p], h_aug, cols,
                int(args.sector_candidates), require_all_paulis=True
            )
            caches[p][q] = dd
            append_d2(cp, q, d2, dd)
            if (ii + 1) % 25 == 0 or ii + 1 == len(todo):
                print(f"  pdec={p:g} decoded {ii+1}/{len(todo)}", flush=True)

        bad = [
            q for q in unknown
            if (not bool(caches[p][q].success))
            or len(caches[p][q].coverage_paulis) != 4
        ]
        if bad:
            raise RuntimeError(("D2 completeness failure", p, [hex(x) for x in bad[:10]]))

    target_rows_by_p = {}
    reconstructed = {}
    witnesses = []

    for p in DECODER_PS:
        pkey = str(p)
        fresh_rows = []
        fresh_map = {}

        for ti, (key, shots) in enumerate(TARGET_SHOTS.items()):
            meta = meta_by_key[key]
            row = a2._score_samples(
                sampled[key], b3, b1, s10,
                fw_exact, fw_amb, cheap_exact, cheap_amb, caches[p],
                int(args.max_correction_effects),
                int(args.max_examples_per_stratum),
                meta,
            )
            row["decoder_p"] = p
            fresh_rows.append(row)
            fresh_map[key] = row

            # Recreate exact harmful witnesses using deterministic replay.
            for ex in row.get("harmful_examples", []):
                j = int(ex["sample_index"])
                base_seed = int(args.seed) + ti * 10000019
                shot_seed = (base_seed + j * 1000000007) & ((1 << 63) - 1)
                rng = np.random.default_rng(shot_seed)
                locs = a2._sample_locations(meta, groups, rng)
                rep = a2._simulate_exact3(a1, b1, sector_by, model, locs, rng, want_trace=True)
                if int(rep["record_a"]) != int(ex["record_A"], 16) or int(rep["record_b"]) != int(ex["record_B"], 16):
                    raise RuntimeError(("witness replay drift", p, key, j))
                ex["trace"] = rep.get("trace")
                ex["shot_seed"] = int(shot_seed)
                ex["decoder_p"] = p
                witnesses.append(ex)

        target_rows_by_p[pkey] = fresh_rows
        fresh_target_agg = aggregate_rows(a2, fresh_rows)

        # Pilot-informed reconstruction: replace pilot target rows by fresh target rows.
        base_rows = []
        for oldrow in b4rows[pkey]:
            k = oldrow["key"]
            base_rows.append(fresh_map[k] if k in fresh_map else oldrow)
        recon = aggregate_rows(a2, base_rows)

        # Exact k=0 upper-bound diagnostics for fresh zero-event target strata.
        zero_bounds = []
        zero_upper_sum = 0.0
        for row in fresh_rows:
            if int(row["closed_boundary_harmful_samples"]) != 0:
                continue
            n = int(row["shots"])
            up_prob = one_sided_zero_event_upper(n)
            up_a3 = int(row["location_triples"]) * up_prob
            zero_upper_sum += up_a3
            zero_bounds.append({
                "key": row["key"],
                "shots": n,
                "location_triples": int(row["location_triples"]),
                "binary_harmful_probability_upper95": up_prob,
                "binary_A3_upper95": up_a3,
                "truth_closed_operational_accept": int(row["truth_closed_operational_accept"]),
            })
        zero_bounds.sort(key=lambda r: r["binary_A3_upper95"], reverse=True)

        reconstructed[pkey] = {
            "decoder_p": p,
            "fresh_target_aggregate": fresh_target_agg,
            "reconstructed_global_aggregate": recon,
            "fresh_zero_event_target_binary_A3_upper95_sum": zero_upper_sum,
            "fresh_zero_event_target_bounds": zero_bounds,
            "pilot_non_target_zero_event_binary_A3_upper95_sum":
                float(non_target_zero_upper),
        }

    low = reconstructed[str(DECODER_PS[0])]["reconstructed_global_aggregate"]
    high = reconstructed[str(DECODER_PS[1])]["reconstructed_global_aggregate"]

    main = {
        "stage": "S21-OPT4C targeted high-stat optimized exact-three follow-up",
        "status": "TARGETED_COMPLETE_NOT_AUTOMATICALLY_PAPER_FINAL",
        "pilot_informed_targeting": True,
        "target_selection_frozen_before_OPT4C_outcomes": True,
        "truth_bits_decoder_visible": False,
        "threshold_sweep": False,
        "decoder_ps": list(DECODER_PS),
        "seed": int(args.seed),
        "target_strata_count": len(TARGET_SHOTS),
        "total_fresh_target_shots": int(sum(TARGET_SHOTS.values())),
        "matrix": {
            "H_rank": rh,
            "H_aug_rank": ra,
            "raw_effect_classes": len(model["effects"]),
            "closure_conditioned_effect_classes": len(decoder_effects),
            "latent_zero_op_residuals": len(latent),
        },
        "observed_queries": len(all_queries),
        "D2_unknown_records": len(unknown),
        "results": reconstructed,
        "paired_decoder_prior_difference": {
            "A3_low": float(low["A3_closed_boundary_infidelity_estimate"]),
            "A3_high": float(high["A3_closed_boundary_infidelity_estimate"]),
            "delta_high_minus_low":
                float(high["A3_closed_boundary_infidelity_estimate"])
                - float(low["A3_closed_boundary_infidelity_estimate"]),
            "intrinsic_low": float(low["A3_intrinsic_oracle_estimate"]),
            "intrinsic_high": float(high["A3_intrinsic_oracle_estimate"]),
            "decoder_excess_low": float(low["A3_decoder_excess_estimate"]),
            "decoder_excess_high": float(high["A3_decoder_excess_estimate"]),
        },
        "harmful_witness_count": len(witnesses),
        "elapsed_seconds": time.time() - t0,
        "decision_rule": {
            "if_pure_DOWN3_or_large_DOWN2_harmful_found":
                "Run a final fixed-allocation precision stage on those decoder-excess strata.",
            "if_no_decoder_excess_harmful_found_and_intrinsic_two_strata_stabilize":
                "Use OPT4C to design a smaller final closure run; do not silently claim unobserved decoder-excess is zero.",
            "paper_final_global_A3":
                "Not automatic because target selection was pilot-informed; report reconstructed estimate separately until final closure."
        },
    }

    dump(out / "s21opt4c_summary.json", main)
    dump(out / "s21opt4c_target_rows_by_decoder_p.json", target_rows_by_p)
    dump(out / "s21opt4c_harmful_witnesses.json", witnesses)

    print("S21-OPT4C TARGETED COMPLETE")
    print(json.dumps({
        "status": main["status"],
        "target_shots": main["total_fresh_target_shots"],
        "queries": main["observed_queries"],
        "D2_unknown": main["D2_unknown_records"],
        "paired": main["paired_decoder_prior_difference"],
        "low_p_reconstructed": reconstructed[str(DECODER_PS[0])]["reconstructed_global_aggregate"],
        "low_p_zero_event_target_upper95_sum":
            reconstructed[str(DECODER_PS[0])]["fresh_zero_event_target_binary_A3_upper95_sum"],
        "pilot_non_target_zero_upper95_sum":
            reconstructed[str(DECODER_PS[0])]["pilot_non_target_zero_event_binary_A3_upper95_sum"],
    }, indent=2))


if __name__ == "__main__":
    main()
