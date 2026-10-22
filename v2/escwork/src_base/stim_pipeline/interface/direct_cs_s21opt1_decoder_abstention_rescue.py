#!/usr/bin/env python3
"""S21-OPT1: frozen-threshold second-stage BP+OSD rescue of S18-B3 abstentions.

Purpose
-------
Retrospectively rescore the *same* S18-B3 p=1e-3 physical evaluation sample.
No new quantum/physical Monte Carlo is used beyond deterministic replay of the frozen
B3 seed. The primary decoder/acceptance policy is not changed for already accepted
records. Only records rejected as ``d2_gap_abstain`` or
``d2_effect_count_abstain`` are offered to one pre-frozen stronger classical decoder.

Frozen rescue policy (chosen before looking at rescue outcomes)
--------------------------------------------------------------
- operational detector bits: 638 only
- B2R firewall remains before every D2 decoder
- primary policy remains gap >= 10 and correction effects <= 8
- fallback decoder: 16 highest-prior Steane sectors, BP max_iter=60, minimum-sum scaling=.625,
  OSD-CS order=3
- fallback acceptance uses the SAME gap >= 10 and effects <= 8 thresholds
- no threshold sweep and no rescue-setting sweep
- truth-closure bits (168) are scoring-only and never decoder-visible

This is a paired retrospective decoder-ablation/optimization screen. If it gives a
material improvement, a fresh-seed S21-OPT1B validation is the appropriate paper-final
follow-up; do not treat the reused B3 evaluation sample as an independent validation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

SCRIPT_VERSION = "S21-OPT1-2026-09-04"
TARGET_P = 1e-3
OP_DETECTORS = 638
TRUTH_BITS = 168
PRIMARY_GAP = 10.0
PRIMARY_MAX_EFFECTS = 8
RESCUE_SECTOR_CANDIDATES = 16
RESCUE_BP_MAX_ITER = 60
RESCUE_BP_METHOD = "minimum_sum"
RESCUE_MS_SCALING = 0.625
RESCUE_OSD_METHOD = "osd_cs"
RESCUE_OSD_ORDER = 3
PAULI_NAMES = ["I", "X", "Z", "Y"]
RESCUEABLE_PRIMARY_TIERS = {"d2_gap_abstain", "d2_effect_count_abstain"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_obj(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    # Required for Python 3.11 dataclasses and forward references.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def _find(root: Path, names: list[str], prefer: tuple[str, ...] = ("stim_pipeline/interface",)) -> Path:
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
        tuple(0 if token in str(p).replace("\\", "/") else 1 for token in prefer),
        0 if "stim_pipeline" in str(p) else 1,
        len(str(p)), str(p),
    ))
    return hits[0]


def _resolve(root: Path, p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _p_equal(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= max(1e-15, 1e-9 * max(abs(float(a)), abs(float(b))))


def _row_for_p(rows: list[dict[str, Any]], p: float) -> dict[str, Any]:
    for r in rows:
        if _p_equal(float(r.get("p", -1)), p):
            return r
    raise KeyError(f"No row for p={p:g}")


def _sector_prior_from_saved(saved: list[dict[str, Any]], p: float) -> np.ndarray:
    row = _row_for_p(saved, p)
    sp = row.get("sector_probability")
    if isinstance(sp, dict):
        arr = np.array([float(sp[str(i)]) for i in range(64)], dtype=np.float64)
    else:
        arr = np.asarray(sp, dtype=np.float64)
    if arr.shape != (64,):
        raise RuntimeError(("Expected 64 saved sector probabilities", arr.shape))
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise RuntimeError("Saved sector prior is non-positive/non-finite")
    arr = arr / float(np.sum(arr))
    return arr


def _primary_decision(
    q: int,
    fw_exact: dict[int, int], fw_amb: set[int],
    cheap_exact: dict[int, int], cheap_amb: set[int],
    high: dict[int, Any], gap: float = PRIMARY_GAP,
    max_effects: int = PRIMARY_MAX_EFFECTS,
):
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


def _rescue_decision(
    q: int,
    fw_exact: dict[int, int], fw_amb: set[int],
    cheap_exact: dict[int, int], cheap_amb: set[int],
    primary_high: dict[int, Any], rescue_high: dict[int, Any],
):
    base = _primary_decision(q, fw_exact, fw_amb, cheap_exact, cheap_amb, primary_high)
    if base[0] or base[2] not in RESCUEABLE_PRIMARY_TIERS:
        return (*base, False, None)
    d = rescue_high.get(int(q))
    if d is None or not bool(d.success) or d.pauli is None:
        return False, None, base[2], False, "rescue_failure"
    if not math.isfinite(float(d.gap)) or float(d.gap) < PRIMARY_GAP:
        return False, None, base[2], False, "rescue_gap_abstain"
    k = int(d.best.correction_effect_count) if d.best is not None else 10**9
    if k > PRIMARY_MAX_EFFECTS:
        return False, None, base[2], False, "rescue_effect_count_abstain"
    return True, int(d.pauli), "d2_rescue", True, "rescue_accepted"


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    ph = k / n
    den = 1 + z*z/n
    cen = (ph + z*z/(2*n)) / den
    half = z * math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    return [max(0.0, cen-half), min(1.0, cen+half)]


def _score(
    sampled: dict[str, Any],
    decode_fn: Callable[[int], tuple],
    b1mod: Any, s10mc: Any,
    primary_decode_fn: Callable[[int], tuple] | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    accepted = truth_closed = residual = 0
    closed_harm = closed_harm_le2 = 0
    inf_closed = intrinsic_closed = 0.0
    raw_harm_all_acc = 0
    tiers = collections.Counter()
    accepted_fault_hist = collections.Counter()
    residual_fault_hist = collections.Counter()
    examples: list[dict[str, Any]] = []
    rescued_samples = rescued_truth_closed = rescued_closed_harm = 0
    primary_rejected_rescued = 0

    for idx, tr in enumerate(sampled["trajectories"]):
        da = decode_fn(int(tr.record_a)); db = decode_fn(int(tr.record_b))
        oka, pa, ta = bool(da[0]), da[1], str(da[2])
        okb, pb, tb = bool(db[0]), db[1], str(db[2])
        if not oka or not okb:
            tiers[f"reject_A:{ta}" if not oka else f"reject_B:{tb}"] += 1
            continue

        rescued_this = bool(len(da) >= 4 and da[3]) or bool(len(db) >= 4 and db[3])
        if rescued_this:
            rescued_samples += 1
        if primary_decode_fn is not None:
            pda = primary_decode_fn(int(tr.record_a)); pdb = primary_decode_fn(int(tr.record_b))
            if not (bool(pda[0]) and bool(pdb[0])):
                primary_rejected_rescued += 1

        accepted += 1
        tiers[f"A:{ta}"] += 1; tiers[f"B:{tb}"] += 1
        accepted_fault_hist[int(tr.total_fault_locations)] += 1
        inf = float(b1mod._raw_corrected_infidelity(s10mc, tr, int(pa), int(pb)))
        raw_harm_all_acc += int(inf > b1mod.TOL)
        closed = (int(tr.truth_a) == 0 and int(tr.truth_b) == 0)
        if closed:
            truth_closed += 1
            inf_closed += inf
            intrinsic_closed += float(tr.intrinsic_oracle_infidelity)
            if rescued_this:
                rescued_truth_closed += 1
            if inf > b1mod.TOL:
                closed_harm += 1
                closed_harm_le2 += int(tr.total_fault_locations <= 2)
                if rescued_this:
                    rescued_closed_harm += 1
        else:
            residual += 1
            residual_fault_hist[int(tr.total_fault_locations)] += 1

        if rescued_this and len(examples) < max_examples:
            examples.append({
                "sample_index": idx,
                "record_A": hex(int(tr.record_a)), "record_B": hex(int(tr.record_b)),
                "pred_A": PAULI_NAMES[int(pa)], "pred_B": PAULI_NAMES[int(pb)],
                "actual_A": PAULI_NAMES[int(tr.actual_pauli_a)], "actual_B": PAULI_NAMES[int(tr.actual_pauli_b)],
                "truth_A": hex(int(tr.truth_a)), "truth_B": hex(int(tr.truth_b)),
                "infidelity": inf, "truth_closed": closed,
                "total_fault_locations": int(tr.total_fault_locations),
                "tier_A": ta, "tier_B": tb,
            })

    shots = int(sampled["shots"])
    return {
        "shots": shots,
        "operational_accepted": accepted,
        "operational_acceptance": accepted / shots,
        "attempts_per_operational_accept": (shots / accepted if accepted else None),
        "truth_closed_accepted": truth_closed,
        "truth_closed_fraction_among_accepted": (truth_closed/accepted if accepted else None),
        "detectable_residual_accepted": residual,
        "detectable_residual_fraction_among_accepted": (residual/accepted if accepted else None),
        "closed_boundary_harmful": closed_harm,
        "closed_boundary_harmful_fraction_among_truth_closed": (closed_harm/truth_closed if truth_closed else None),
        "closed_boundary_harmful_wilson95": _wilson(closed_harm, truth_closed),
        "closed_boundary_conditional_raw_frame_infidelity": (inf_closed/truth_closed if truth_closed else None),
        "closed_boundary_intrinsic_oracle_infidelity": (intrinsic_closed/truth_closed if truth_closed else None),
        "sampled_closed_boundary_harmful_with_total_fault_locations_le2": closed_harm_le2,
        "raw_terminal_proxy_harmful_all_accepted_DO_NOT_QUOTE_AS_LOGICAL": raw_harm_all_acc,
        "tier_counts": dict(sorted(tiers.items())),
        "accepted_fault_histogram": {str(k): int(v) for k,v in sorted(accepted_fault_hist.items())},
        "residual_fault_histogram": {str(k): int(v) for k,v in sorted(residual_fault_hist.items())},
        "rescued_accepted_samples": rescued_samples,
        "rescued_truth_closed_samples": rescued_truth_closed,
        "rescued_truth_closed_harmful": rescued_closed_harm,
        "primary_rejected_samples_that_become_accepted": primary_rejected_rescued,
        "rescued_examples": examples,
    }


def _load_primary_cache(b3mod: Any, d2: Any, path: Path) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        o = json.loads(ln)
        out[int(o["record"], 16)] = b3mod._deserialize_d2(d2, o["decode"])
    return out


def _load_rescue_cache(b3mod: Any, d2: Any, path: Path, meta_path: Path, wanted: dict[str, Any], fresh: bool) -> dict[int, Any]:
    if fresh:
        if path.exists(): path.unlink()
        if meta_path.exists(): meta_path.unlink()
    if meta_path.exists():
        got = _load(meta_path)
        if got != wanted:
            raise RuntimeError("Stale S21-OPT1 rescue cache. Use --fresh or a new --out-dir.")
    else:
        _dump(meta_path, wanted)
    out: dict[int, Any] = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.strip(): continue
            o = json.loads(ln)
            out[int(o["record"],16)] = b3mod._deserialize_d2(d2, o["decode"])
    return out


def _append_rescue_cache(path: Path, q: int, d2: Any, dec: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"record": hex(int(q)), "decode": d2._serialize(dec)}, sort_keys=True) + "\n")


def _dependency_map(root: Path) -> dict[str, Path]:
    req = {
        "b3": ["direct_cs_s18b3_firewalled_fresh_validation.py"],
        "b1": ["direct_cs_s18b1_638_d2_calibration.py"],
        "d2": ["direct_cs_s17c_d2_logical_coset_list_decoder.py"],
        "d1": ["direct_cs_s17c_d_hybrid_bposd_abstention.py"],
        "s10": ["direct_cs_s10_complete_physical_factory_mc.py"],
        "cert": ["direct_cs_s17b_v13d_joint_postselection_certificate.py"],
        "b2r": ["direct_cs_s18b2r_low_order_firewall.py"],
    }
    return {k: _find(root, v) for k,v in req.items()}


def _self_test() -> dict[str, Any]:
    good = SimpleNamespace(success=True, pauli=2, gap=12.0, best=SimpleNamespace(correction_effect_count=4))
    low = SimpleNamespace(success=True, pauli=1, gap=8.0, best=SimpleNamespace(correction_effect_count=3))
    highk = SimpleNamespace(success=True, pauli=3, gap=12.0, best=SimpleNamespace(correction_effect_count=9))
    primary = {0x1: low, 0x2: highk}
    rescue = {0x1: good, 0x2: good}
    a = _rescue_decision(0x1, {}, set(), {}, set(), primary, rescue)
    b = _rescue_decision(0x2, {}, set(), {}, set(), primary, rescue)
    c = _rescue_decision(0x3, {}, {0x3}, {}, set(), primary, rescue)
    passed = (a[:4] == (True,2,"d2_rescue",True) and b[:4] == (True,2,"d2_rescue",True) and c[:3] == (False,None,"firewall_ambiguous"))
    return {
        "passed": passed,
        "stage": "S21-OPT1 decoder abstention rescue",
        "script_version": SCRIPT_VERSION,
        "primary_gap": PRIMARY_GAP,
        "primary_max_effects": PRIMARY_MAX_EFFECTS,
        "rescue_sector_candidates": RESCUE_SECTOR_CANDIDATES,
        "rescue_bp_max_iter": RESCUE_BP_MAX_ITER,
        "rescue_osd_order": RESCUE_OSD_ORDER,
        "truth_bits_decoder_visible": False,
        "threshold_sweep": False,
        "rescue_setting_sweep": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s18m2-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--b3-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b3_validation_results"))
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s21opt1_results"))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    print(f"S21-OPT1 script version: {SCRIPT_VERSION}", flush=True)
    if args.self_test_only:
        r = _self_test(); print(json.dumps(r, indent=2));
        if not r["passed"]: raise RuntimeError("S21-OPT1 self-test failed")
        return

    root = args.search_root.resolve()
    b3dir = _resolve(root, args.b3_result_dir)
    m2dir = _resolve(root, args.s18m2_result_dir)
    out = _resolve(root, args.out_dir); out.mkdir(parents=True, exist_ok=True)

    deps = _dependency_map(root)
    inputs = {
        "b3_summary": b3dir / "s18b3_summary.json",
        "b3_sector_map": b3dir / "s18b3_sector_map_638_truth.json",
        "b3_sector_priors": b3dir / "s18b3_sector_priors.json",
        "primary_cache": b3dir / "s18b3_cache_0p001.jsonl",
        "primary_cache_meta": b3dir / "s18b3_cache_0p001_meta.json",
        "xtruth": m2dir / "s18m2_X_truth_closed.stim",
        "ztruth": m2dir / "s18m2_Z_truth_closed.stim",
    }
    missing = [str(p) for p in inputs.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing S21-OPT1 inputs:\n  " + "\n  ".join(missing))

    pf = {
        "stage": "S21-OPT1 preflight",
        "script_version": SCRIPT_VERSION,
        "dependencies": {k: str(v) for k,v in deps.items()},
        "inputs": {k: str(v) for k,v in inputs.items()},
        "frozen_policy": {
            "p": TARGET_P, "primary_gap": PRIMARY_GAP, "primary_max_effects": PRIMARY_MAX_EFFECTS,
            "rescue_sector_candidates": RESCUE_SECTOR_CANDIDATES,
            "rescue_bp_max_iter": RESCUE_BP_MAX_ITER,
            "rescue_bp_method": RESCUE_BP_METHOD,
            "rescue_ms_scaling": RESCUE_MS_SCALING,
            "rescue_osd_method": RESCUE_OSD_METHOD,
            "rescue_osd_order": RESCUE_OSD_ORDER,
            "threshold_sweep": False, "rescue_setting_sweep": False,
            "truth_bits_decoder_visible": False,
        },
        "passed": True,
    }
    _dump(out / "s21opt1_preflight.json", pf)
    if args.preflight_only:
        print("S21-OPT1 preflight PASS", flush=True)
        print(json.dumps(pf, indent=2), flush=True)
        return

    # Imports happen only after all file paths have passed preflight.
    b3mod = _import("s21opt1_b3", deps["b3"])
    b1mod = _import("s21opt1_b1", deps["b1"])
    d2 = _import("s21opt1_d2", deps["d2"])
    d1 = _import("s21opt1_d1", deps["d1"])
    s10mc = _import("s21opt1_s10", deps["s10"])
    cert = _import("s21opt1_cert", deps["cert"])
    b2rmod = _import("s21opt1_b2r", deps["b2r"])
    d1.EXPECTED_DETECTORS = OP_DETECTORS; d2.EXPECTED_DETECTORS = OP_DETECTORS
    BpOsdDecoder = d1._require_ldpc()

    b3sum = _load(inputs["b3_summary"])
    if not bool(b3sum.get("validation_gate",{}).get("passed",False)):
        raise RuntimeError("Primary S18-B3 validation gate is not PASS")
    if bool(b3sum.get("decoder_input",{}).get("terminal_truth_bits_decoder_visible",True)):
        raise RuntimeError("Primary B3 unexpectedly exposes truth bits")
    erow = _row_for_p(b3sum["evaluation_rows"], TARGET_P)
    arow = _row_for_p(b3sum["decoder_audit"], TARGET_P)
    shots = int(erow["shots"]); eval_seed = int(erow["evaluation_seed"]); prior_seed = int(erow["prior_seed"])
    expected = {
        "shots": shots,
        "operational_accepted": int(erow["operational_accepted"]),
        "truth_closed_accepted": int(erow["truth_closed_accepted"]),
        "closed_boundary_harmful": int(erow["closed_boundary_harmful"]),
        "detectable_residual_accepted": int(erow["detectable_residual_accepted"]),
        "d2_unknown": int(arow["d2_unknown"]),
    }

    sectors = _load(inputs["b3_sector_map"])
    if len(sectors) != 64:
        raise RuntimeError(("Expected 64 saved sector rows", len(sectors)))
    sector_by = {int(r["steane_syndrome"]): r for r in sectors}
    prior = _sector_prior_from_saved(_load(inputs["b3_sector_priors"]), TARGET_P)

    print("S21-OPT1: rebuilding frozen 638-bit operational model/matrix...", flush=True)
    model = b1mod._build_truth_operational_model(cert, inputs["xtruth"], inputs["ztruth"])
    cols = d1._effect_columns(model["effects"])
    h = d1._build_sparse_check_matrix(cols)
    h_aug = d2._build_augmented_matrix(h, cols)
    rank_h = d2._gf2_row_rank_sparse(h); rank_aug = d2._gf2_row_rank_sparse(h_aug)
    if int(h.shape[0]) != OP_DETECTORS or rank_h != OP_DETECTORS or rank_aug < OP_DETECTORS + 2:
        raise RuntimeError(("S21-OPT1 matrix contract failed", h.shape, rank_h, rank_aug))
    cheap_exact, cheap_amb, cheap_meta = b1mod._build_fast_catalogue(sectors, model["effects"])
    fw_idx = b2rmod.build_firewall_index(sectors, model)
    zero_fw = b2rmod.resolve_query(0, fw_idx, max_examples=2)
    if not (zero_fw["status"] == "exact" and zero_fw["pauli"] == "I"):
        raise RuntimeError(("B2R zero-record contract failed", zero_fw))

    print(f"S21-OPT1: deterministic B3 replay p={TARGET_P:g}, shots={shots}, seed={eval_seed} ...", flush=True)
    sampled = b1mod._sample_one_p(TARGET_P, shots, eval_seed, s10mc, sector_by, model, 20)
    queries = set(map(int, sampled["queries"]))
    fw_cache = {0: zero_fw}
    new_fw = sorted(q for q in queries if q != 0)
    for ii,q in enumerate(new_fw):
        fw_cache[q] = b2rmod.resolve_query(q, fw_idx, max_examples=2)
        if (ii+1) % 250 == 0 or ii+1 == len(new_fw):
            print(f"  firewall replay {ii+1}/{len(new_fw)}", flush=True)
    fw_exact = {q:int(fw_cache[q]["pauli_int"]) for q in queries if fw_cache[q]["status"] == "exact"}
    fw_amb = {q for q in queries if fw_cache[q]["status"] == "ambiguous_reject"}
    fw_high = {q for q in queries if fw_cache[q]["status"] == "higher_order_only"}
    unknown = sorted(q for q in queries if q != 0 and q not in fw_exact and q not in fw_amb and q not in cheap_exact and q not in cheap_amb)
    if any(q not in fw_high for q in unknown):
        raise RuntimeError("A non-higher-order-only record reached D2")

    primary_high = _load_primary_cache(b3mod, d2, inputs["primary_cache"])
    missing_primary = [q for q in unknown if q not in primary_high]
    if missing_primary:
        raise RuntimeError(("Primary B3 cache lacks replayed D2 records", len(missing_primary), [hex(q) for q in missing_primary[:10]]))

    primary_fn = lambda q: _primary_decision(q, fw_exact, fw_amb, cheap_exact, cheap_amb, primary_high)
    primary_score = _score(sampled, primary_fn, b1mod, s10mc, None, 0)
    replay_gate = {
        "expected": expected,
        "replayed": {k: primary_score[k] for k in ("shots","operational_accepted","truth_closed_accepted","closed_boundary_harmful","detectable_residual_accepted")},
        "replayed_d2_unknown": len(unknown),
    }
    replay_gate["passed"] = bool(
        all(primary_score[k] == expected[k] for k in ("shots","operational_accepted","truth_closed_accepted","closed_boundary_harmful","detectable_residual_accepted"))
        and len(unknown) == expected["d2_unknown"]
    )
    _dump(out / "s21opt1_primary_replay_gate.json", replay_gate)
    if not replay_gate["passed"]:
        raise RuntimeError(("S21-OPT1 primary replay gate failed; rescue result suppressed", replay_gate))
    print("S21-OPT1: primary B3 replay gate PASS", flush=True)

    target_records: list[int] = []
    primary_target_tiers = collections.Counter()
    gap_hist = collections.Counter()
    for q in unknown:
        ok, pa, tier = primary_fn(q)
        if tier in RESCUEABLE_PRIMARY_TIERS:
            target_records.append(q); primary_target_tiers[tier] += 1
            d = primary_high[q]
            g = float(d.gap)
            if not math.isfinite(g): bucket = "nonfinite"
            elif g < 2: bucket = "[0,2)"
            elif g < 5: bucket = "[2,5)"
            elif g < 8: bucket = "[5,8)"
            elif g < 9: bucket = "[8,9)"
            elif g < 10: bucket = "[9,10)"
            else: bucket = ">=10_effect_cap"
            gap_hist[bucket] += 1

    print(f"S21-OPT1: primary abstained unique D2 records={len(target_records)}; frozen rescue decode starts...", flush=True)
    effect_priors = d1._effect_parity_priors(cols, TARGET_P)
    rescue_decoder = d1._make_bposd(
        BpOsdDecoder, h_aug, effect_priors,
        RESCUE_BP_MAX_ITER, RESCUE_BP_METHOD, RESCUE_MS_SCALING,
        RESCUE_OSD_METHOD, RESCUE_OSD_ORDER,
    )
    cache = out / "s21opt1_rescue_d2_cache_0p001.jsonl"
    meta = out / "s21opt1_rescue_d2_cache_0p001_meta.json"
    wanted = {
        "stage": "S21-OPT1", "script_version": SCRIPT_VERSION, "p": TARGET_P,
        "primary_b3_summary_sha256": _sha256_file(inputs["b3_summary"]),
        "primary_cache_meta_sha256": _sha256_file(inputs["primary_cache_meta"]),
        "sector_map_sha256": _sha256_file(inputs["b3_sector_map"]),
        "matrix": {"shape": list(map(int,h_aug.shape)), "rank_h": int(rank_h), "rank_aug": int(rank_aug), "effect_columns": len(cols)},
        "rescue_policy": {
            "sector_candidates": RESCUE_SECTOR_CANDIDATES, "bp_max_iter": RESCUE_BP_MAX_ITER,
            "bp_method": RESCUE_BP_METHOD, "ms_scaling": RESCUE_MS_SCALING,
            "osd_method": RESCUE_OSD_METHOD, "osd_order": RESCUE_OSD_ORDER,
            "accept_gap": PRIMARY_GAP, "accept_max_effects": PRIMARY_MAX_EFFECTS,
            "rescueable_primary_tiers": sorted(RESCUEABLE_PRIMARY_TIERS),
            "threshold_sweep": False, "rescue_setting_sweep": False,
        },
    }
    rescue_high = _load_rescue_cache(b3mod, d2, cache, meta, wanted, args.fresh)
    todo = [q for q in target_records if q not in rescue_high]
    tdec = time.time()
    for ii,q in enumerate(todo):
        dec = d2._decode_unknown_coset(
            d1, q, sectors, prior, TARGET_P, rescue_decoder, h_aug, cols,
            RESCUE_SECTOR_CANDIDATES, require_all_paulis=True,
        )
        rescue_high[q] = dec; _append_rescue_cache(cache, q, d2, dec)
        if (ii+1) % 20 == 0 or ii+1 == len(todo):
            print(f"  rescue decoded {ii+1}/{len(todo)} new records", flush=True)

    rescue_record_status = collections.Counter()
    rescue_record_examples = []
    for q in target_records:
        d = rescue_high[q]
        if d is None or not bool(d.success) or d.pauli is None:
            st = "failure"
        elif not math.isfinite(float(d.gap)) or float(d.gap) < PRIMARY_GAP:
            st = "gap_abstain"
        elif d.best is None or int(d.best.correction_effect_count) > PRIMARY_MAX_EFFECTS:
            st = "effect_count_abstain"
        else:
            st = "accepted"
        rescue_record_status[st] += 1
        if len(rescue_record_examples) < 30 and st == "accepted":
            rescue_record_examples.append({
                "record": hex(q), "primary_tier": primary_fn(q)[2],
                "primary_gap": float(primary_high[q].gap),
                "primary_effects": int(primary_high[q].best.correction_effect_count) if primary_high[q].best is not None else None,
                "rescue_pauli": PAULI_NAMES[int(d.pauli)], "rescue_gap": float(d.gap),
                "rescue_effects": int(d.best.correction_effect_count) if d.best is not None else None,
            })

    cascade_fn = lambda q: _rescue_decision(q, fw_exact, fw_amb, cheap_exact, cheap_amb, primary_high, rescue_high)
    rescue_score = _score(sampled, cascade_fn, b1mod, s10mc, primary_fn, 30)

    old_acc = float(primary_score["operational_acceptance"]); new_acc = float(rescue_score["operational_acceptance"])
    expected_active_per_attempt = float(erow.get("stage_level_expected_active_locations_per_attempt", float("nan")))
    resource = {
        "expected_quantum_work_per_attempt_unchanged": True,
        "primary_acceptance": old_acc, "rescue_acceptance": new_acc,
        "absolute_acceptance_gain": new_acc-old_acc,
        "relative_accepted_output_gain": (new_acc/old_acc-1 if old_acc else None),
        "attempts_per_accept_primary": (1/old_acc if old_acc else None),
        "attempts_per_accept_rescue": (1/new_acc if new_acc else None),
        "active_locations_per_attempt_from_B3": expected_active_per_attempt,
        "active_locations_per_accept_primary": (expected_active_per_attempt/old_acc if old_acc and math.isfinite(expected_active_per_attempt) else None),
        "active_locations_per_accept_rescue": (expected_active_per_attempt/new_acc if new_acc and math.isfinite(expected_active_per_attempt) else None),
    }
    # S20-A2 p=1e-3 normalized qstep proxy. Decoder rescue changes only the denominator.
    qsteps_per_attempt_reference = 10109.784
    threeT_fresh_upper_reference = 29289.388734683263
    threeT_fresh_lower_reference = 23689.42181460972
    resource.update({
        "S20A2_qsteps_per_attempt_reference": qsteps_per_attempt_reference,
        "qsteps_per_accept_primary_reference": qsteps_per_attempt_reference/old_acc,
        "qsteps_per_accept_rescue_reference": qsteps_per_attempt_reference/new_acc if new_acc else None,
        "threeT_fresh_qstep_bracket_reference": [threeT_fresh_lower_reference, threeT_fresh_upper_reference],
        "acceptance_needed_to_beat_threeT_fresh_upper": qsteps_per_attempt_reference/threeT_fresh_upper_reference,
        "acceptance_needed_to_beat_threeT_fresh_lower": qsteps_per_attempt_reference/threeT_fresh_lower_reference,
        "beats_threeT_fresh_upper_after_rescue": bool(new_acc > qsteps_per_attempt_reference/threeT_fresh_upper_reference),
        "beats_threeT_fresh_lower_after_rescue": bool(new_acc > qsteps_per_attempt_reference/threeT_fresh_lower_reference),
        "reference_warning": "qstep and 3T bracket are copied from S20-A2 deterministic normalization; rerun S20-A2 after freezing an optimized decoder for paper-final resource numbers.",
    })

    if rescue_score["rescued_truth_closed_harmful"] > 0:
        recommendation = "REVIEW_NEW_RESCUED_HARMFUL_BEFORE_ANY_FRESH_VALIDATION"
    elif resource["beats_threeT_fresh_upper_after_rescue"]:
        recommendation = "HIGH_VALUE_PROMISING_RUN_FRESH_SEED_S21_OPT1B"
    elif (new_acc - old_acc) >= 0.03:
        recommendation = "PROMISING_BUT_NOT_YET_THREET_UPPER_BREAK_EVEN_CONSIDER_FRESH_VALIDATION"
    else:
        recommendation = "LOW_GAIN_STOP_DECODER_OPTIMIZATION_AND_MOVE_TO_CIRCUIT_OPTIMIZATION"

    summary = {
        "stage": "S21-OPT1 frozen-threshold second-stage BP+OSD abstention rescue",
        "status": "COMPLETE_PAIRED_RETROSPECTIVE_ABLATION",
        "script_version": SCRIPT_VERSION,
        "p": TARGET_P, "shots": shots, "evaluation_seed": eval_seed, "prior_seed": prior_seed,
        "comparison_contract": "same S18-B3 physical sample and primary accepted decisions; only primary D2 gap/effect abstentions are offered to one pre-frozen stronger classical decoder",
        "decoder_input": {"operational_detector_bits": OP_DETECTORS, "truth_bits_scoring_only": TRUTH_BITS, "truth_bits_decoder_visible": False},
        "primary_policy": {"gap": PRIMARY_GAP, "max_effects": PRIMARY_MAX_EFFECTS, "sector_candidates": 8, "bp_max_iter": 30, "osd_order": 2},
        "rescue_policy": wanted["rescue_policy"],
        "primary_replay_gate": replay_gate,
        "primary_abstained_unique_records": len(target_records),
        "primary_abstained_tiers": dict(primary_target_tiers),
        "primary_abstained_gap_histogram": dict(gap_hist),
        "rescue_unique_record_status": dict(rescue_record_status),
        "primary_score": primary_score,
        "rescue_score": rescue_score,
        "delta": {
            "operational_accepted": rescue_score["operational_accepted"] - primary_score["operational_accepted"],
            "operational_acceptance": new_acc-old_acc,
            "truth_closed_accepted": rescue_score["truth_closed_accepted"] - primary_score["truth_closed_accepted"],
            "closed_boundary_harmful": rescue_score["closed_boundary_harmful"] - primary_score["closed_boundary_harmful"],
            "detectable_residual_accepted": rescue_score["detectable_residual_accepted"] - primary_score["detectable_residual_accepted"],
            "sampled_closed_boundary_harmful_le2": rescue_score["sampled_closed_boundary_harmful_with_total_fault_locations_le2"] - primary_score["sampled_closed_boundary_harmful_with_total_fault_locations_le2"],
        },
        "resource_projection": resource,
        "next_step_recommendation": recommendation,
        "interpretation_guardrails": [
            "No physical/quantum circuit is changed; expected quantum work per attempt is identical to S18-B3.",
            "The B2R <=2 firewall, gap=10, and max-effects=8 acceptance thresholds are unchanged.",
            "Already accepted primary records are never re-decoded or revoked; this is a cascade/fallback decoder.",
            "The rescue settings are frozen once (16 sectors, BP60, OSD3); no parameter or threshold sweep is performed on the B3 evaluation sample.",
            "Because this reuses the B3 evaluation sample, a material improvement should be confirmed on fresh seeds before becoming a paper-final performance claim.",
        ],
        "timing": {"new_rescue_decode_seconds": time.time()-tdec, "new_records_decoded": len(todo), "cache_reused": len(target_records)-len(todo)},
    }
    _dump(out / "s21opt1_summary.json", summary)
    _dump(out / "s21opt1_rescue_record_examples.json", rescue_record_examples)
    _dump(out / "s21opt1_rescued_sample_examples.json", rescue_score["rescued_examples"])
    _dump(out / "s21opt1_self_test.json", _self_test())

    print("S21-OPT1 COMPLETE", flush=True)
    print(json.dumps({
        "primary_acceptance": old_acc,
        "rescue_acceptance": new_acc,
        "accepted_gain": summary["delta"]["operational_accepted"],
        "primary_closed_harmful": primary_score["closed_boundary_harmful"],
        "rescue_closed_harmful": rescue_score["closed_boundary_harmful"],
        "rescued_truth_closed_harmful": rescue_score["rescued_truth_closed_harmful"],
        "sampled_harmful_le2": rescue_score["sampled_closed_boundary_harmful_with_total_fault_locations_le2"],
        "qsteps_per_accept_reference": resource["qsteps_per_accept_rescue_reference"],
        "beats_threeT_fresh_upper": resource["beats_threeT_fresh_upper_after_rescue"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
