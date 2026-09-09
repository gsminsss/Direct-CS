#!/usr/bin/env python3
"""S21-OPT5C: crossover-bracket A3 validation for optimized direct-CS.

Scientific purpose
------------------
OPT5A measured the direct-CS acceptance/resource curve on
p = 1e-4, 2e-4, ..., 1e-3 with a frozen practical decoder architecture and
p-specific sector priors.

OPT4D provided the final targeted-precision hierarchical exact-three estimate.
Its p-dependence was concentrated in the three dominant decoder-excess strata.
This stage therefore:

  1. deterministically regenerates the exact same 140,000 OPT4D physical
     exact-three samples in those three dominant strata;
  2. scores the SAME physical samples under all ten OPT5A p-specific priors;
  3. keeps gap/BP/OSD/sector-candidate hyperparameters frozen;
  4. adds the fixed hierarchical background from OPT4D;
  5. outputs the leading-order direct-CS LER curve
         LER(p) = A3(p) * p^3
     together with OPT5A acceptance/resource values.

Important reporting rule
------------------------
This is NOT a fresh global exact-three resample.  It is a dense p-specific
rescoring of the OPT4D dominant strata plus the fixed hierarchical
OPT4B/OPT4C background.  Report it as a hierarchical reconstructed A3(p).

Outputs
-------
  s21opt5c_plan.json
  s21opt5c_summary.json
  s21opt5c_curve.csv
  s21opt5c_target_rows_by_p.json
  s21opt5c_harmful_witnesses.json
  s21opt5c_d2_cache_<p>.jsonl
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

P_VALUES = (8e-4, 9e-4)
OPT4D_SEED = 20271031
FROZEN_GAP = 10.0
DOWNSTREAM_LOCATIONS_PER_PATCH = 2132

# Exact OPT4D insertion order matters because it determines deterministic seeds.
TARGET_SHOTS = {
    "DOWN_DIRECT_D5_D133": 100000,
    "CORE_POST1+DOWN_DIRECT_D5_D132": 20000,
    "CAT_POST1+DOWN_DIRECT_D5_D132": 20000,
}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


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
    hits.sort(
        key=lambda p: (
            tuple(0 if token in str(p) else 1 for token in prefer),
            0 if "stim_pipeline" in str(p) else 1,
            len(str(p)),
            str(p),
        )
    )
    return hits[0]


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


def prior_from_summary(summary, p):
    for r in summary["prior_calibrations"]:
        if abs(float(r["p"]) - float(p)) < 1e-15:
            sp = r["sector_probability"]
            arr = np.asarray([float(sp[str(i)]) for i in range(64)], dtype=float)
            if abs(float(arr.sum()) - 1.0) > 1e-10:
                raise RuntimeError(("prior norm", p, float(arr.sum())))
            return arr, r
    raise RuntimeError(("prior missing", p))


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
                print(
                    f"  firewall {ii:,}: matched {len(out)}/{len(wanted)}",
                    flush=True,
                )
    return out


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
        f.write(
            json.dumps(
                {"record": hex(int(q)), "decode": d2._serialize(obj)},
                sort_keys=True,
            )
            + "\n"
        )


def tag_p(p: float) -> str:
    return f"{p:.12g}".replace(".", "p").replace("-", "m")


def derive_background(opt4d_summary):
    """Extract the p-independent non-target hierarchical background.

    In OPT4D, the same untouched OPT4B/OPT4C rows were used at pdec=5e-4 and
    pdec=1e-3.  Therefore global - fresh_target must match at both anchors.
    """
    vals = []
    for pk in ("0.0005", "0.001"):
        rr = opt4d_summary["results"][pk]
        g = rr["reconstructed_global_aggregate"]
        t = rr["fresh_target_aggregate"]

        total = (
            float(g["A3_closed_boundary_infidelity_estimate"])
            - float(t["A3_closed_boundary_infidelity_estimate"])
        )
        intrinsic = (
            float(g["A3_intrinsic_oracle_estimate"])
            - float(t["A3_intrinsic_oracle_estimate"])
        )
        decoder_excess = (
            float(g["A3_decoder_excess_estimate"])
            - float(t["A3_decoder_excess_estimate"])
        )

        gs = float(g["A3_closed_boundary_infidelity_naive_se"])
        ts = float(t["A3_closed_boundary_infidelity_naive_se"])
        total_se = math.sqrt(max(0.0, gs * gs - ts * ts))

        vals.append(
            {
                "anchor": pk,
                "total_A3": total,
                "total_A3_se": total_se,
                "intrinsic_A3": intrinsic,
                "decoder_excess_A3": decoder_excess,
            }
        )

    for field in ("total_A3", "total_A3_se", "intrinsic_A3", "decoder_excess_A3"):
        if abs(vals[0][field] - vals[1][field]) > max(
            1e-6, 1e-10 * max(abs(vals[0][field]), abs(vals[1][field]), 1.0)
        ):
            raise RuntimeError(("OPT4D background anchor mismatch", field, vals))

    return {
        "total_A3": vals[0]["total_A3"],
        "total_A3_se": vals[0]["total_A3_se"],
        "intrinsic_A3": vals[0]["intrinsic_A3"],
        "decoder_excess_A3": vals[0]["decoder_excess_A3"],
        "anchor_diagnostics": vals,
        "policy": (
            "Fixed hierarchical background inherited from OPT4D; only the "
            "three dominant OPT4D strata are rescored versus p-specific prior."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument(
        "--opt3b-result-dir",
        type=Path,
        default=Path("stim_pipeline/interface/s21opt3b_results"),
    )
    ap.add_argument(
        "--opt3c0-result-dir",
        type=Path,
        default=Path("stim_pipeline/interface/s21opt3c0_results"),
    )
    ap.add_argument(
        "--opt4d-result-dir",
        type=Path,
        default=Path("stim_pipeline/interface/s21opt4d_results"),
    )
    ap.add_argument(
        "--opt5a-result-dir",
        type=Path,
        default=Path("stim_pipeline/interface/s21opt5a_results"),
    )
    ap.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=list(P_VALUES),
        help="Crossover bracket only; defaults to p=8e-4 and 9e-4.",
    )
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--bp-max-iter", type=int, default=30)
    ap.add_argument("--bp-method", default="minimum_sum")
    ap.add_argument("--ms-scaling-factor", type=float, default=0.625)
    ap.add_argument("--osd-method", default="osd_cs")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--max-examples-per-stratum", type=int, default=8)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stim_pipeline/interface/s21opt5c_results"),
    )
    args = ap.parse_args()

    if abs(float(args.ms_scaling_factor) - 0.625) > 1e-15:
        raise SystemExit("OPT5C min-sum scaling is frozen at 0.625")
    if int(args.bp_max_iter) != 30:
        raise SystemExit("OPT5C BP max iter is frozen at 30")
    if float(FROZEN_GAP) != 10.0:
        raise RuntimeError("OPT5C gap drift")
    if int(args.sector_candidates) != 8:
        raise SystemExit("OPT5C sector candidates are frozen at 8")
    if args.osd_method != "osd_cs" or int(args.osd_order) != 2:
        raise SystemExit("OPT5C OSD is frozen at osd_cs order 2")

    p_values = [float(p) for p in args.p_values]
    allowed = {round(p, 12) for p in P_VALUES}
    if any(round(p, 12) not in allowed for p in p_values):
        raise SystemExit(("p-values must be OPT5A grid points", p_values))

    root = args.search_root.resolve()

    def R(p: Path):
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    bdir = R(args.opt3b_result_dir)
    c0dir = R(args.opt3c0_result_dir)
    d4dir = R(args.opt4d_result_dir)
    a5dir = R(args.opt5a_result_dir)
    out = R(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    opt4b = imp(
        "s21opt5c_opt4b",
        find(
            root,
            ["direct_cs_s21opt4b_optimized_exact3_pilot.py"],
            ("stim_pipeline/interface",),
        ),
    )
    a2 = imp(
        "s21opt5c_a2",
        find(
            root,
            ["direct_cs_s19a2a_exact3_independent_highstat.py"],
            ("stim_pipeline/interface", "s19work"),
        ),
    )
    a1 = imp(
        "s21opt5c_a1",
        find(
            root,
            ["direct_cs_s11a1_exact3_stratified_pilot.py"],
            ("stim_pipeline/interface",),
        ),
    )
    b1 = imp(
        "s21opt5c_b1",
        find(
            root,
            ["direct_cs_s18b1_638_d2_calibration.py"],
            ("stim_pipeline/interface",),
        ),
    )
    b3 = imp(
        "s21opt5c_b3",
        find(
            root,
            ["direct_cs_s18b3_firewalled_fresh_validation.py"],
            ("stim_pipeline/interface",),
        ),
    )
    d1 = imp(
        "s21opt5c_d1",
        find(
            root,
            ["direct_cs_s17c_d_hybrid_bposd_abstention.py"],
            ("stim_pipeline/interface",),
        ),
    )
    d2 = imp(
        "s21opt5c_d2",
        find(
            root,
            ["direct_cs_s17c_d2_logical_coset_list_decoder.py"],
            ("stim_pipeline/interface",),
        ),
    )
    s10 = imp(
        "s21opt5c_s10",
        find(
            root,
            ["direct_cs_s10_complete_physical_factory_mc.py"],
            ("stim_pipeline/interface",),
        ),
    )
    cert = imp(
        "s21opt5c_cert",
        find(
            root,
            ["direct_cs_s17b_v13d_joint_postselection_certificate.py"],
            ("stim_pipeline/interface",),
        ),
    )

    opt4b.patch_globals(b1, d1, d2, b3, a2)
    b1._stage_from_tick = opt4b.direct_stage
    cert.EXPECTED_LOCATIONS = DOWNSTREAM_LOCATIONS_PER_PATCH
    cert.EXPECTED_DETECTORS = 431

    opt4d_summary = load(d4dir / "s21opt4d_summary.json")
    opt5a_summary = load(a5dir / "s21opt5a_summary.json")

    if opt4d_summary.get("status") != "FINAL_TARGETED_PRECISION_COMPLETE":
        raise RuntimeError(("OPT4D status", opt4d_summary.get("status")))
    if not opt5a_summary.get("main_gate", {}).get("passed", False):
        raise RuntimeError("OPT5A main gate not PASS")
    frozen = opt5a_summary["frozen_decoder"]
    expected_frozen = {
        "gap": 10.0,
        "sector_candidates": 8,
        "bp_max_iter": 30,
        "bp_method": "minimum_sum",
        "ms_scaling_factor": 0.625,
        "osd_method": "osd_cs",
        "osd_order": 2,
        "max_correction_effects": 8,
        "threshold_sweep": False,
    }
    for k, v in expected_frozen.items():
        if frozen.get(k) != v:
            raise RuntimeError(("OPT5A frozen decoder drift", k, frozen.get(k), v))

    background = derive_background(opt4d_summary)

    fwpath = c0dir / "s21opt3c0_firewall.jsonl.gz"
    sectors = sector_rows(c0dir / "s21opt3c0_sector_map_263_truth.json")
    sector_by = {int(r["steane_syndrome"]): r for r in sectors}
    xtruth = bdir / "s21opt3b_X_direct_d5_d13_noisy.stim"
    ztruth = bdir / "s21opt3b_Z_direct_d5_d13_noisy.stim"
    for p in (fwpath, xtruth, ztruth):
        if not p.exists():
            raise FileNotFoundError(p)

    print("S21-OPT5C: rebuilding optimized physical model...", flush=True)
    model = b1._build_truth_operational_model(cert, xtruth, ztruth)
    decoder_effects, latent = closure_condition_decoder_effects(d1, model["effects"])
    groups = opt4b.build_groups(a1, a2, model)
    strata = opt4b.all_strata(groups, a2)
    meta_by_key = {r["key"]: r for r in strata}

    missing = [k for k in TARGET_SHOTS if k not in meta_by_key]
    if missing:
        raise RuntimeError(("target strata missing", missing))

    plan = {
        "stage": "S21-OPT5C dense p-specific A3 rescoring",
        "p_values": p_values,
        "target_strata": [
            {"key": k, "shots": int(v), "location_triples": int(meta_by_key[k]["location_triples"])}
            for k, v in TARGET_SHOTS.items()
        ],
        "total_exact3_samples": int(sum(TARGET_SHOTS.values())),
        "physical_samples_shared_across_all_p": True,
        "opt4d_seed_reused": OPT4D_SEED,
        "decoder_hyperparameters_retuned": False,
        "p_specific_sector_prior": True,
        "fixed_hierarchical_background": background,
        "reporting_label": "hierarchical reconstructed A3(p), not fresh global resample",
    }
    dump(out / "s21opt5c_plan.json", plan)

    print(
        json.dumps(
            {
                "p_count": len(p_values),
                "target_strata": len(TARGET_SHOTS),
                "shared_exact3_samples": sum(TARGET_SHOTS.values()),
                "background_A3": background["total_A3"],
                "background_A3_se": background["total_A3_se"],
            },
            indent=2,
        ),
        flush=True,
    )

    if args.plan_only:
        print("S21-OPT5C plan-only PASS")
        return

    cols = d1._effect_columns(decoder_effects)
    h = d1._build_sparse_check_matrix(cols)
    h_aug = d2._build_augmented_matrix(h, cols)
    rh = int(d2._gf2_row_rank_sparse(h))
    ra = int(d2._gf2_row_rank_sparse(h_aug))
    if rh != 263 or ra < 265:
        raise RuntimeError(("matrix drift", rh, ra))
    cheap_exact, cheap_amb, _cheap_meta = b1._build_fast_catalogue(
        sectors, decoder_effects
    )

    # Deterministically regenerate the exact same OPT4D target samples.
    sampled = {}
    all_queries = set()
    t0 = time.time()

    for ti, (key, shots) in enumerate(TARGET_SHOTS.items()):
        meta = meta_by_key[key]
        arr = []
        base_seed = OPT4D_SEED + ti * 10000019
        for j in range(int(shots)):
            shot_seed = (base_seed + j * 1000000007) & ((1 << 63) - 1)
            rng = np.random.default_rng(shot_seed)
            locs = a2._sample_locations(meta, groups, rng)
            rr = a2._simulate_exact3(
                a1, b1, sector_by, model, locs, rng, want_trace=False
            )
            arr.append(rr)
            if rr.get("accepted_predecode", False):
                all_queries.add(int(rr["record_a"]))
                all_queries.add(int(rr["record_b"]))
        sampled[key] = arr
        print(
            f"S21-OPT5C sampled {ti+1}/{len(TARGET_SHOTS)} "
            f"{key}: {shots} shots; queries={len(all_queries)}",
            flush=True,
        )

    print(
        f"S21-OPT5C: scanning firewall for {len(all_queries)} observed records...",
        flush=True,
    )
    fw = scan_firewall(fwpath, all_queries)
    fw_exact = {
        q: int(r["pauli_int"]) for q, r in fw.items() if r["status"] == "exact"
    }
    fw_amb = {
        q for q, r in fw.items() if r["status"] == "ambiguous_reject"
    }
    if fw_exact.get(0, None) != 0:
        raise RuntimeError("zero firewall drift")

    unknown = sorted(
        q
        for q in all_queries
        if q != 0
        and q not in fw_exact
        and q not in fw_amb
        and q not in cheap_exact
        and q not in cheap_amb
    )

    BpOsdDecoder = d1._require_ldpc()

    opt5a_eval = {round(float(r["p"]), 12): r for r in opt5a_summary["evaluation_rows"]}
    curve = []
    target_rows_by_p = {}
    witnesses = []
    d2_audit = []

    for p in p_values:
        prior, pmeta = prior_from_summary(opt5a_summary, p)
        ep = d1._effect_parity_priors(cols, p)
        decoder = d1._make_bposd(
            BpOsdDecoder,
            h_aug,
            ep,
            int(args.bp_max_iter),
            args.bp_method,
            float(args.ms_scaling_factor),
            args.osd_method,
            int(args.osd_order),
        )

        tag = tag_p(p)
        inherited = [
            a5dir / f"s21opt5a_d2_cache_{tag}.jsonl",
        ]
        if abs(p - 5e-4) < 1e-15 or abs(p - 1e-3) < 1e-15:
            inherited.append(d4dir / f"s21opt4d_d2_cache_{tag}.jsonl")

        cp = out / f"s21opt5c_d2_cache_{tag}.jsonl"
        inherited.append(cp)  # resume support
        cache = load_d2_cache(b3, d2, inherited)

        todo = [q for q in unknown if q not in cache]
        print(
            f"S21-OPT5C p={p:g}: inherited={len(cache)}, new D2={len(todo)}",
            flush=True,
        )

        for ii, q in enumerate(todo):
            dd = d2._decode_unknown_coset(
                d1,
                q,
                sectors,
                prior,
                p,
                decoder,
                h_aug,
                cols,
                int(args.sector_candidates),
                require_all_paulis=True,
            )
            cache[q] = dd
            append_d2(cp, q, d2, dd)
            if (ii + 1) % 25 == 0 or ii + 1 == len(todo):
                print(
                    f"  p={p:g} decoded {ii+1}/{len(todo)}",
                    flush=True,
                )

        bad = [
            q
            for q in unknown
            if (not bool(cache[q].success))
            or len(cache[q].coverage_paulis) != 4
        ]
        if bad:
            raise RuntimeError(
                ("D2 completeness failure", p, [hex(x) for x in bad[:10]])
            )

        fresh_rows = []
        for ti, (key, shots) in enumerate(TARGET_SHOTS.items()):
            meta = meta_by_key[key]
            row = a2._score_samples(
                sampled[key],
                b3,
                b1,
                s10,
                fw_exact,
                fw_amb,
                cheap_exact,
                cheap_amb,
                cache,
                int(args.max_correction_effects),
                int(args.max_examples_per_stratum),
                meta,
            )
            row["decoder_p"] = p
            fresh_rows.append(row)

            for ex in row.get("harmful_examples", []):
                j = int(ex["sample_index"])
                base_seed = OPT4D_SEED + ti * 10000019
                shot_seed = (base_seed + j * 1000000007) & ((1 << 63) - 1)
                ex2 = dict(ex)
                ex2["shot_seed"] = int(shot_seed)
                ex2["decoder_p"] = p
                witnesses.append(ex2)

        target_rows_by_p[f"{p:.12g}"] = fresh_rows
        targ = a2._aggregate(fresh_rows)

        target_A3 = float(targ["A3_closed_boundary_infidelity_estimate"])
        target_se = float(targ["A3_closed_boundary_infidelity_naive_se"])
        target_intrinsic = float(targ["A3_intrinsic_oracle_estimate"])

        global_A3 = target_A3 + float(background["total_A3"])
        global_se = math.sqrt(
            target_se * target_se
            + float(background["total_A3_se"]) ** 2
        )
        global_intrinsic = (
            target_intrinsic + float(background["intrinsic_A3"])
        )
        global_decoder_excess = global_A3 - global_intrinsic

        ler = global_A3 * p**3
        ler_se = global_se * p**3
        ler95 = [
            max(0.0, ler - 1.96 * ler_se),
            ler + 1.96 * ler_se,
        ]

        erow = opt5a_eval[round(p, 12)]
        curve.append(
            {
                "p": p,
                "A3_LER_coefficient": global_A3,
                "A3_LER_coefficient_se": global_se,
                "A3_intrinsic": global_intrinsic,
                "A3_decoder_excess": global_decoder_excess,
                "LER_leading_order": ler,
                "LER_leading_order_se": ler_se,
                "LER_leading_order_95pct": ler95,
                "operational_acceptance": float(erow["operational_acceptance"]),
                "average_attempts_per_CS": float(
                    erow["attempts_per_operational_accept"]
                ),
                "active_locations_per_accepted_CS": float(
                    erow[
                        "stage_level_active_location_proxy_per_operational_accept"
                    ]
                ),
                "expected_active_locations_per_attempt": float(
                    erow["stage_level_expected_active_locations_per_attempt"]
                ),
                "target_closed_boundary_harmful_samples": int(
                    targ["closed_boundary_harmful_samples"]
                ),
                "D2_unknown_records": len(unknown),
                "D2_cache_records": len(cache),
                "D2_new_records": len(todo),
                "D2_four_coset_pass": True,
            }
        )
        d2_audit.append(
            {
                "p": p,
                "unknown_records": len(unknown),
                "cache_records": len(cache),
                "new_records": len(todo),
                "failure": 0,
                "four_coset_coverage": len(unknown),
            }
        )

    # CSV for the immediate crossover analysis.
    csv_path = out / "s21opt5c_curve.csv"
    fields = [
        "p",
        "A3_LER_coefficient",
        "A3_LER_coefficient_se",
        "A3_intrinsic",
        "A3_decoder_excess",
        "LER_leading_order",
        "LER_leading_order_se",
        "operational_acceptance",
        "average_attempts_per_CS",
        "active_locations_per_accepted_CS",
        "expected_active_locations_per_attempt",
        "target_closed_boundary_harmful_samples",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in curve:
            w.writerow({k: r[k] for k in fields})

    main = {
        "stage": "S21-OPT5C dense p-specific A3 rescoring",
        "status": "CROSSOVER_BRACKET_A3_COMPLETE",
        "reporting_label": (
            "hierarchical reconstructed A3(p): OPT4D dominant-strata "
            "p-specific rescoring + fixed OPT4B/OPT4C background"
        ),
        "fresh_global_resample": False,
        "physical_samples_shared_across_all_p": True,
        "total_shared_exact3_samples": int(sum(TARGET_SHOTS.values())),
        "p_values": p_values,
        "frozen_decoder": expected_frozen,
        "truth_bits_decoder_visible": False,
        "matrix": {
            "H_rank": rh,
            "H_aug_rank": ra,
            "raw_effect_classes": len(model["effects"]),
            "closure_conditioned_effect_classes": len(decoder_effects),
            "latent_zero_op_residuals": len(latent),
        },
        "background": background,
        "observed_queries": len(all_queries),
        "D2_unknown_records": len(unknown),
        "D2_audit": d2_audit,
        "curve": curve,
        "harmful_witness_count": len(witnesses),
        "main_gate": {
            "OPT4D_complete": True,
            "OPT5A_pass": True,
            "matrix_rank_pass": rh == 263 and ra >= 265,
            "decoder_hyperparameters_frozen": True,
            "p_specific_prior_only": True,
            "D2_all_success": True,
            "D2_four_coset_all": True,
            "truth_bits_decoder_visible": False,
            "passed": True,
        },
        "elapsed_seconds": time.time() - t0,
        "next_if_pass": (
            "Combine this direct-CS LER/resource curve with the Sahay 3T->CS "
            "baseline and solve for the crossover/advantage region."
        ),
    }

    dump(out / "s21opt5c_summary.json", main)
    dump(out / "s21opt5c_target_rows_by_p.json", target_rows_by_p)
    dump(out / "s21opt5c_harmful_witnesses.json", witnesses)

    print("S21-OPT5C COMPLETE")
    print(
        json.dumps(
            {
                "status": main["status"],
                "p_count": len(curve),
                "shared_exact3_samples": main["total_shared_exact3_samples"],
                "observed_queries": main["observed_queries"],
                "D2_unknown_records": main["D2_unknown_records"],
                "curve": [
                    {
                        "p": r["p"],
                        "A3": r["A3_LER_coefficient"],
                        "LER": r["LER_leading_order"],
                        "acceptance": r["operational_acceptance"],
                        "loc_per_CS": r["active_locations_per_accepted_CS"],
                    }
                    for r in curve
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
