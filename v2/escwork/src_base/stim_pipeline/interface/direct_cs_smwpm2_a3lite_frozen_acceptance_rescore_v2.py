from __future__ import annotations

"""S-MWPM-2: paired MWPM reframe of the completed S19-A3-Lite sample.

Purpose
-------
Replay the exact 10,000 physical DOWN_HIRANO_D5^3 samples used by S19-A3-Lite,
first reproduce the frozen BP+OSD primary result exactly, then rescore ONLY the
Pauli frame of BP-accepted d2_high records using the graphlike-only MWPM baseline.

Crucial comparison contract
---------------------------
* Same physical samples (same S19-A3-Lite seed and per-shot seed rule).
* Same structural zero / B2R firewall / cheap exact decisions.
* Same BP+OSD accept/reject decision (frozen acceptance).
* MWPM changes only the predicted Pauli frame on records whose accepted BP tier
  is d2_high.
* 638 operational detector bits only; terminal truth bits are scoring-only.
* Plain PyMatching is a graphlike-only approximation; irreducible >2-detector
  mechanisms are audited and not hidden.

This is a decoder ablation, not a replacement of the S19 primary result.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

TOL = 1e-12
PAULI = {0: "I", 1: "X", 2: "Z", 3: "Y"}
TARGET_KEY = "DOWN_HIRANO_D53"
DEFAULT_DECODER_P = 2e-4
EXPECTED_SHOTS = 10000
EXPECTED_SEED = 20270311
EXPECTED_BP_A3 = 839623.8374999999
EXPECTED_BP_HARMFUL = 6
EXPECTED_BP_ACCEPT = 8371
EXPECTED_LOCATION_TRIPLES = 1599283500


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


SMWPM2_SCRIPT_VERSION = "SMWPM2-V2-2026-09-04"

def _json_default(x: Any):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    if hasattr(x, "item"):
        try: return x.item()
        except Exception: pass
    raise TypeError(type(x).__name__)


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses (and some typing helpers) resolve cls.__module__ through
    # sys.modules while the module body is executing.  Dynamic imports must
    # therefore register the module before exec_module, just like normal import.
    previous = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return mod


def _find(root: Path, names: list[str], prefer: tuple[str, ...] = ("stim_pipeline/interface",)) -> Path:
    hits: list[Path] = []
    for n in names:
        p = root / n
        if p.exists(): hits.append(p)
        hits.extend(root.rglob(n))
    hits = list(dict.fromkeys(p.resolve() for p in hits))
    if not hits:
        raise FileNotFoundError(names)
    def key(p: Path):
        s = str(p).replace("\\", "/")
        pref = min([i for i,x in enumerate(prefer) if x in s] or [len(prefer)])
        return (pref, len(p.parts), s)
    hits.sort(key=key)
    return hits[0]


def _resolve_dir(root: Path, p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0: return 0.0, 0.0
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0
    return m, se


def _self_test() -> dict[str, Any]:
    # Synthetic paired reframe: same acceptance, only d2_high frame changes.
    bp = np.asarray([0.0, 0.75, 0.0, 0.75, 0.0])
    mw = np.asarray([0.0, 0.0, 0.0, 0.75, 0.0])
    d = mw - bp
    m, se = _mean_se(d)
    passed = (
        int(np.count_nonzero(bp > TOL)) == 2
        and int(np.count_nonzero(mw > TOL)) == 1
        and math.isclose(m, -0.15, rel_tol=0, abs_tol=1e-15)
        and se > 0
    )
    return {
        "passed": bool(passed),
        "stage": "S-MWPM-2 A3-Lite frozen-acceptance paired MWPM reframe",
        "comparison_contract": "same physical samples + same BP acceptance; MWPM changes d2_high Pauli frame only",
        "operational_detector_bits": 638,
        "truth_bits_decoder_visible": False,
        "synthetic_bp_harmful": 2,
        "synthetic_mwpm_harmful": 1,
        "synthetic_mean_delta": m,
        "synthetic_delta_se": se,
    }



def _pack_sample(i: int, sample: dict[str, Any]) -> dict[str, Any]:
    v = np.asarray(sample["logical_vector"], dtype=complex).reshape(-1)
    return {
        "sample_index": int(i),
        "accepted_predecode": bool(sample["accepted_predecode"]),
        "reason": str(sample.get("reason", "")),
        "record_a": hex(int(sample["record_a"])), "record_b": hex(int(sample["record_b"])),
        "actual_pauli_a": int(sample["actual_pauli_a"]), "actual_pauli_b": int(sample["actual_pauli_b"]),
        "truth_a": hex(int(sample["truth_a"])), "truth_b": hex(int(sample["truth_b"])),
        "logical_vector": [[float(z.real), float(z.imag)] for z in v],
        "core_record": [int(x) for x in sample["core_record"]],
        "intrinsic_oracle_infidelity": float(sample["intrinsic_oracle_infidelity"]),
    }


def _unpack_sample(o: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_predecode": bool(o["accepted_predecode"]),
        "reason": str(o.get("reason", "")),
        "record_a": int(str(o["record_a"]),16), "record_b": int(str(o["record_b"]),16),
        "actual_pauli_a": int(o["actual_pauli_a"]), "actual_pauli_b": int(o["actual_pauli_b"]),
        "truth_a": int(str(o["truth_a"]),16), "truth_b": int(str(o["truth_b"]),16),
        "logical_vector": np.asarray([complex(float(x[0]),float(x[1])) for x in o["logical_vector"]],dtype=complex),
        "core_record": tuple(int(x) for x in o["core_record"]),
        "intrinsic_oracle_infidelity": float(o["intrinsic_oracle_infidelity"]),
    }


def _load_sample_cache(path: Path) -> list[dict[str, Any]]:
    out=[]
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o=json.loads(line); i=int(o["sample_index"])
        if i != len(out): raise RuntimeError(("non-contiguous physical replay cache",i,len(out)))
        out.append(_unpack_sample(o))
    return out


def _append_sample_cache(path: Path, i: int, sample: dict[str, Any]) -> None:
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(_pack_sample(i,sample),sort_keys=True)+"\n")


def _load_mwpm_cache(path: Path) -> dict[int,dict[str,Any]]:
    out={}
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o=json.loads(line); out[int(o["record"],16)] = o["decode"]
    return out


def _append_mwpm_cache(path: Path, record: int, dd: dict[str,Any]) -> None:
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps({"record":hex(int(record)),"decode":dd},sort_keys=True,default=_json_default)+"\n")


def _trajectory_from_sample(sample: dict[str, Any], b1mod):
    return b1mod.Trajectory(
        int(sample["record_a"]), int(sample["record_b"]),
        int(sample["actual_pauli_a"]), int(sample["actual_pauli_b"]),
        int(sample["truth_a"]), int(sample["truth_b"]),
        np.asarray(sample["logical_vector"], complex),
        tuple(int(x) for x in sample["core_record"]),
        3, 0, 0, 0, float(sample["intrinsic_oracle_infidelity"]),
    )


def _mwpm_reframe_record(record: int, bp_pauli: int, bp_tier: str,
                         xmatching, zmatching, sectors, prior, mwmod) -> tuple[int, dict[str, Any] | None]:
    if bp_tier != "d2_high":
        return int(bp_pauli), None
    dd = mwmod._mwpm_sector_decode(int(record), xmatching, zmatching, sectors, prior)
    if not bool(dd.get("decode_success", False)):
        raise RuntimeError(("MWPM failed on BP-accepted d2_high record", hex(int(record)), dd))
    if not bool(dd.get("four_final_paulis_covered", False)):
        raise RuntimeError(("MWPM did not cover all four final Paulis", hex(int(record)), dd))
    return int(dd["predicted_pauli_int"]), dd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("."))
    ap.add_argument("--s18m-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m_results"))
    ap.add_argument("--s18m2-result-dir", type=Path, default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--b2r-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b2r_firewall_results"))
    ap.add_argument("--b3-result-dir", type=Path, default=Path("stim_pipeline/interface/s18b3_validation_results"))
    ap.add_argument("--s19a2a-result-dir", type=Path, default=Path("stim_pipeline/interface/s19a2a_results"))
    ap.add_argument("--s19a3-result-dir", type=Path, default=Path("stim_pipeline/interface/s19a3_lite_results"))
    ap.add_argument("--decoder-p", type=float, default=DEFAULT_DECODER_P)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--sector-candidates", type=int, default=8)
    ap.add_argument("--max-correction-effects", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/smwpm2_a3lite_results"))
    ap.add_argument("--write-dem", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="Discard S-MWPM-2 replay/MWPM caches and start this ablation from scratch.")
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    st = _self_test()
    if args.self_test_only:
        print(json.dumps(st, indent=2, sort_keys=True)); return
    if not st["passed"]: raise RuntimeError(st)
    if abs(float(args.decoder_p) - DEFAULT_DECODER_P) > 1e-15:
        raise SystemExit("S-MWPM-2 primary paired rescore is frozen to --decoder-p 0.0002. Use a separate later ablation for other priors.")

    root = args.search_root.resolve()
    out = _resolve_dir(root, args.out_dir); out.mkdir(parents=True, exist_ok=True)
    _dump(out / "smwpm2_self_test.json", st)

    # Import the completed A3-Lite implementation and the validated MWPM V5 implementation.
    a3 = _import("smwpm2_a3", _find(root, ["direct_cs_s19a3_lite_hirano3_precision.py"]))
    a2 = _import("smwpm2_a2", _find(root, ["direct_cs_s19a2a_exact3_independent_highstat.py"]))
    mw = _import("smwpm2_mw", _find(root, ["direct_cs_smwpm1_hirano3_witness_ablation_v5.py"]))
    a1 = _import("smwpm2_a1", a2._find(root,["direct_cs_s11a1_exact3_stratified_pilot.py"],("stim_pipeline/interface",)))
    b1mod = _import("smwpm2_b1", a2._find(root,["direct_cs_s18b1_638_d2_calibration.py"],("stim_pipeline/interface",)))
    b3mod = _import("smwpm2_b3", a2._find(root,["direct_cs_s18b3_firewalled_fresh_validation.py"],("stim_pipeline/interface",)))
    b2rmod = _import("smwpm2_b2r", a2._find(root,["direct_cs_s18b2r_low_order_firewall.py"],("stim_pipeline/interface",)))
    d2 = _import("smwpm2_d2", a2._find(root,["direct_cs_s17c_d2_logical_coset_list_decoder.py"],("stim_pipeline/interface",)))
    d1 = _import("smwpm2_d1", a2._find(root,["direct_cs_s17c_d_hybrid_bposd_abstention.py"],("stim_pipeline/interface",)))
    s10mc = _import("smwpm2_s10", a2._find(root,["direct_cs_s10_complete_physical_factory_mc.py"],("stim_pipeline/interface",)))
    cert = _import("smwpm2_cert", a2._find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"],("stim_pipeline/interface",)))
    s18m = _import("smwpm2_s18m", a2._find(root,["direct_cs_s18m_modular_noisy_d13_certificate.py"],("stim_pipeline/interface",)))
    m2 = _import("smwpm2_m2", a2._find(root,["direct_cs_s18m2_terminal_truth_closure.py"],("stim_pipeline/interface",)))
    ca = _import("smwpm2_ca", a2._find(root,["direct_cs_s17c_a_sector_to_470_map.py"],("stim_pipeline/interface",)))
    s8b = _import("smwpm2_s8b", a2._find(root,["zxz_core_boundary_signature_map.py"],("stim_pipeline/interface",)))
    v7 = _import("smwpm2_v7", a2._find(root,["direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"],("stim_pipeline/interface",)))
    v11 = _import("smwpm2_v11", a2._find(root,["direct_cs_s17b_v11_complete_missing_basis.py"],("stim_pipeline/interface",)))
    s18a = _import("smwpm2_s18a", a2._find(root,["direct_cs_s18a_noisy_d13_boundary_certificate.py"],("stim_pipeline/interface",)))
    d1.EXPECTED_DETECTORS = 638; d2.EXPECTED_DETECTORS = 638

    # Authoritative completed A3-Lite inputs.
    a3dir = _resolve_dir(root, args.s19a3_result_dir)
    a3sum = _load(a3dir / "s19a3_lite_summary.json")
    a3plan = _load(a3dir / "s19a3_lite_plan.json")
    a3rows = _load(a3dir / "s19a3_lite_rows.json")
    if str(a3sum.get("status")) != "COMPLETE": raise RuntimeError("A3-Lite not COMPLETE")
    if int(a3sum.get("fresh_shots", -1)) != EXPECTED_SHOTS or int(a3sum.get("sampling_seed", -1)) != EXPECTED_SEED:
        raise RuntimeError(("A3-Lite sampling contract drift", a3sum.get("fresh_shots"), a3sum.get("sampling_seed")))
    if list(a3sum.get("selected_keys", [])) != [TARGET_KEY]: raise RuntimeError("A3-Lite key drift")
    src = a3sum["per_prior"][str(DEFAULT_DECODER_P)]
    if int(src["harmful_samples_selected"]) != EXPECTED_BP_HARMFUL or int(src["operational_accept_samples_selected"]) != EXPECTED_BP_ACCEPT:
        raise RuntimeError(("A3-Lite authoritative count drift", src))
    if not math.isclose(float(src["A3_selected"]), EXPECTED_BP_A3, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(("A3-Lite authoritative A3 drift", src["A3_selected"]))

    # Rebuild exactly the A3-Lite physical model.
    s18mr = _resolve_dir(root,args.s18m_result_dir); s18m2r = _resolve_dir(root,args.s18m2_result_dir)
    xtruth = s18m2r / "s18m2_X_truth_closed.stim"; ztruth = s18m2r / "s18m2_Z_truth_closed.stim"; sector_op = s18mr / "s18m_sector_map_638.json"
    hirano = a2._find(root,["msc-ls/src"]); sahay = a2._find(root,["reference/fd3_baseline_ckt.stim"])
    print("S-MWPM-2: rebuilding exact A3-Lite physical model for deterministic replay...", flush=True)
    model = b1mod._build_truth_operational_model(cert,xtruth,ztruth)
    sectors_truth,smeta = b1mod._build_sector_truth_map(root,s18m,m2,ca,s8b,v7,v11,s18a,hirano,sahay,sector_op,float(args.signature_p))
    sector_by = {int(r["steane_syndrome"]):r for r in sectors_truth}
    groups = a2._build_groups(a1,model); strata = a2._all_strata(groups); strata_by = {r["key"]:(i,r) for i,r in enumerate(strata)}
    if len(strata) != 220 or sum(int(r["location_triples"]) for r in strata) != math.comb(7960,3): raise RuntimeError("partition drift")
    si, meta = strata_by[TARGET_KEY]
    if int(meta["location_triples"]) != EXPECTED_LOCATION_TRIPLES: raise RuntimeError("Hirano3 location-triple drift")

    cheap_exact, cheap_amb, cheap_meta = b1mod._build_fast_catalogue(sectors_truth,model["effects"])
    fw_cache_path = a3dir / "s19a3_lite_firewall_cache.jsonl"
    fw_cache = a2._load_fw_cache(fw_cache_path)
    if 0 not in fw_cache or fw_cache[0]["status"] != "exact" or fw_cache[0]["pauli"] != "I": raise RuntimeError("A3 firewall cache invalid")

    # Load the existing completed BP D2 cache. No new BP decoding is done.
    bp_cache_path = a3dir / "s19a3_lite_d2_cache_0p0002.jsonl"
    bp_high = a2._load_d2_cache(b3mod,d2,bp_cache_path)

    # Deterministically replay the exact 10,000 physical samples.  A3-Lite did not
    # persist per-shot trajectories, so this is the only exact paired route.  We
    # persist the replay immediately so an interruption never loses completed work.
    sample_cache_path = out / "smwpm2_physical_replay_cache.jsonl"
    mwpm_cache_path = out / "smwpm2_mwpm_record_cache.jsonl"
    cache_meta_path = out / "smwpm2_cache_meta.json"
    cache_contract = {
        "stage":"S-MWPM-2", "target_stratum":TARGET_KEY, "shots":EXPECTED_SHOTS,
        "sampling_seed":EXPECTED_SEED, "decoder_p":float(args.decoder_p),
        "comparison_contract":"frozen BP acceptance; MWPM d2_high frame only",
    }
    if args.fresh:
        for cp in (sample_cache_path,mwpm_cache_path,cache_meta_path):
            if cp.exists(): cp.unlink()
    if cache_meta_path.exists():
        old_meta=_load(cache_meta_path)
        if old_meta != cache_contract:
            raise RuntimeError(("S-MWPM-2 cache contract mismatch; rerun with --fresh",old_meta,cache_contract))
    else:
        _dump(cache_meta_path,cache_contract)
    samples = _load_sample_cache(sample_cache_path)
    if len(samples) > EXPECTED_SHOTS: raise RuntimeError(("replay cache too long",len(samples)))
    if samples:
        print(f"S-MWPM-2: resuming physical replay cache {len(samples)}/{EXPECTED_SHOTS}",flush=True)
    for j in range(len(samples),EXPECTED_SHOTS):
        shot_seed = (EXPECTED_SEED + j*1000000007) & ((1<<63)-1)
        rng = np.random.default_rng(shot_seed)
        locs = a2._sample_locations(meta,groups,rng)
        smp = a2._simulate_exact3(a1,b1mod,sector_by,model,locs,rng,want_trace=False)
        samples.append(smp); _append_sample_cache(sample_cache_path,j,smp)
        if (j+1)%500==0 or j+1==EXPECTED_SHOTS:
            print(f"S-MWPM-2: physical replay {j+1}/{EXPECTED_SHOTS}",flush=True)
    queries = {int(r["record_a"]) for r in samples if r["accepted_predecode"]} | {int(r["record_b"]) for r in samples if r["accepted_predecode"]}
    missing_fw = [q for q in queries if q not in fw_cache]
    if missing_fw:
        raise RuntimeError(("Completed A3 firewall cache does not cover deterministic replay", len(missing_fw), [hex(q) for q in missing_fw[:8]]))
    fw_exact = {q:int(fw_cache[q]["pauli_int"]) for q in queries if fw_cache[q]["status"] == "exact"}
    fw_amb = {q for q in queries if fw_cache[q]["status"] == "ambiguous_reject"}
    fw_high = {q for q in queries if fw_cache[q]["status"] == "higher_order_only"}
    unknown = sorted(q for q in queries if q != 0 and q not in fw_exact and q not in fw_amb and q not in cheap_exact and q not in cheap_amb)
    bad = [q for q in unknown if q not in fw_high]
    if bad: raise RuntimeError(("low-order query leaked to D2", [hex(q) for q in bad[:8]]))
    missing_bp = [q for q in unknown if q not in bp_high]
    if missing_bp: raise RuntimeError(("Completed A3 BP cache missing replay queries", len(missing_bp), [hex(q) for q in missing_bp[:8]]))

    bp_rows = [a3._decode_one(x,DEFAULT_DECODER_P,bp_high,a2,b3mod,b1mod,s10mc,fw_exact,fw_amb,cheap_exact,cheap_amb,int(args.max_correction_effects)) for x in samples]
    ways = int(meta["location_triples"])
    bp_inf = np.asarray([float(x["inf"]) for x in bp_rows],float)
    bp_m,bp_se0 = _mean_se(bp_inf); replay_bp_a3 = ways*bp_m; replay_bp_se = ways*bp_se0
    replay_bp_harm = int(np.count_nonzero(bp_inf > TOL))
    replay_bp_accept = sum(x["status"] in ("closed_accept","residual_accept") for x in bp_rows)
    replay_gate = {
        "expected_A3": float(src["A3_selected"]), "replayed_A3": replay_bp_a3,
        "A3_difference": replay_bp_a3-float(src["A3_selected"]),
        "expected_harmful": int(src["harmful_samples_selected"]), "replayed_harmful": replay_bp_harm,
        "expected_operational_accept": int(src["operational_accept_samples_selected"]), "replayed_operational_accept": replay_bp_accept,
        "expected_d2_unknown_records": int(a3rows[0]["d2_unknown_records"]), "replayed_d2_unknown_records": len(unknown),
    }
    replay_gate["passed"] = bool(
        abs(replay_gate["A3_difference"]) <= max(1e-6,abs(float(src["A3_selected"]))*1e-12)
        and replay_bp_harm == int(src["harmful_samples_selected"])
        and replay_bp_accept == int(src["operational_accept_samples_selected"])
        and len(unknown) == int(a3rows[0]["d2_unknown_records"])
    )
    print("S-MWPM-2: BP exact-replay gate ->", json.dumps(replay_gate,indent=2), flush=True)
    if not replay_gate["passed"]: raise RuntimeError(("A3-Lite deterministic replay failed", replay_gate))

    # Build the graphlike-only MWPM at the SAME decoder prior p=2e-4.
    try:
        import stim, pymatching
    except Exception as e:
        raise RuntimeError("S-MWPM-2 requires the project's Python environment containing stim and pymatching") from e
    b3dir = _resolve_dir(root,args.b3_result_dir)
    sector_path = b3dir / "s18b3_sector_map_638_truth.json"; prior_path = b3dir / "s18b3_sector_priors.json"
    sectors = sorted(_load(sector_path),key=lambda r:int(r["steane_syndrome"]))
    if len(sectors) != 64 or [int(r["steane_syndrome"]) for r in sectors] != list(range(64)): raise RuntimeError("bad 64-sector map")
    if any(int(r.get("terminal_truth_168_hex","0x0"),16) != 0 for r in sectors): raise RuntimeError("truth bits leaked into sector map")
    prior, prior_meta = mw._load_b3_prior(prior_path,float(args.decoder_p))

    print(f"S-MWPM-2: building graphlike-only X/Z MWPM at decoder-p={args.decoder_p:g} ...", flush=True)
    # Saved S18-M circuits are p=1e-3, so for p=2e-4 we deliberately rebuild isolated channels.
    xc,_xclean,_xold,_xm = s18m._build_extended_channel(v7,v11,s18a,hirano,sahay,"X",float(args.decoder_p))
    zc,_zclean,_zold,_zm = s18m._build_extended_channel(v7,v11,s18a,hirano,sahay,"Z",float(args.decoder_p))
    xmatching,zmatching,dem_meta = mw._build_separate_matchings(xc,zc,out,bool(args.write_dem))
    dem_meta.update({"decoder_p":float(args.decoder_p),"circuit_source":"isolated_rebuild_at_decoder_p",
                     "pymatching_version":getattr(pymatching,"__version__",None),"stim_version":getattr(stim,"__version__",None)})
    _dump(out / "smwpm2_dem_audit.json", dem_meta)

    # Cache MWPM decisions by record: only BP-accepted d2_high records need a new frame.
    mw_decodes: dict[int,dict[str,Any]] = _load_mwpm_cache(mwpm_cache_path)
    new_mwpm_decodes = 0
    if mw_decodes: print(f"S-MWPM-2: resuming MWPM record cache {len(mw_decodes)} records",flush=True)
    mw_inf = np.zeros(EXPECTED_SHOTS,dtype=float)
    changed = fixed = introduced = both_harm = 0
    bp_d2_accept_samples = 0; bp_d2_accept_records = 0
    examples: list[dict[str,Any]] = []
    for j,(sample,bpr) in enumerate(zip(samples,bp_rows)):
        if bpr["status"] != "closed_accept":
            # Frozen acceptance: BP rejects remain rejected; detectable residual handling is unchanged.
            mw_inf[j] = float(bpr["inf"])
            continue
        pa,pb = (int(bpr["pred"][0]),int(bpr["pred"][1]))
        ta,tb = (str(bpr["tiers"][0]),str(bpr["tiers"][1]))
        if ta == "d2_high" or tb == "d2_high": bp_d2_accept_samples += 1
        new = []
        decs = []
        for rec,bpp,tier in ((int(sample["record_a"]),pa,ta),(int(sample["record_b"]),pb,tb)):
            if tier == "d2_high":
                bp_d2_accept_records += 1
                if rec not in mw_decodes:
                    dd = mw._mwpm_sector_decode(rec,xmatching,zmatching,sectors,prior)
                    if not bool(dd.get("decode_success",False)) or not bool(dd.get("four_final_paulis_covered",False)):
                        raise RuntimeError(("MWPM incomplete on BP-accepted D2 record",hex(rec),dd))
                    mw_decodes[rec] = dd; _append_mwpm_cache(mwpm_cache_path,rec,dd); new_mwpm_decodes += 1
                    if new_mwpm_decodes % 50 == 0:
                        print(f"S-MWPM-2: MWPM decoded {new_mwpm_decodes} new unique d2_high records ({len(mw_decodes)} cached total)",flush=True)
                dd = mw_decodes[rec]; newp = int(dd["predicted_pauli_int"]); decs.append(dd)
            else:
                newp = int(bpp); decs.append(None)
            new.append(newp)
        npa,npb = new
        tr = _trajectory_from_sample(sample,b1mod)
        inf = float(b1mod._raw_corrected_infidelity(s10mc,tr,npa,npb))
        mw_inf[j] = inf
        bp_h = float(bpr["inf"]) > TOL; mw_h = inf > TOL
        frame_changed = (npa != pa or npb != pb)
        changed += int(frame_changed)
        fixed += int(bp_h and not mw_h)
        introduced += int((not bp_h) and mw_h)
        both_harm += int(bp_h and mw_h)
        if (bp_h or mw_h or frame_changed) and len(examples) < 40:
            examples.append({
                "sample_index":j,"record_A":hex(int(sample["record_a"])),"record_B":hex(int(sample["record_b"])),
                "actual_A":PAULI[int(sample["actual_pauli_a"])],"actual_B":PAULI[int(sample["actual_pauli_b"])],
                "bp_pred_A":PAULI[pa],"bp_pred_B":PAULI[pb],"mwpm_pred_A":PAULI[npa],"mwpm_pred_B":PAULI[npb],
                "tier_A":ta,"tier_B":tb,"bp_infidelity":float(bpr["inf"]),"mwpm_infidelity":inf,
                "bp_harmful":bool(bp_h),"mwpm_harmful":bool(mw_h),"frame_changed":bool(frame_changed),
                "mwpm_gap_A":None if decs[0] is None else decs[0].get("gap_minpath_diagnostic"),
                "mwpm_gap_B":None if decs[1] is None else decs[1].get("gap_minpath_diagnostic"),
            })

    mw_m,mw_se0 = _mean_se(mw_inf); mw_a3 = ways*mw_m; mw_se = ways*mw_se0
    mw_harm = int(np.count_nonzero(mw_inf > TOL))
    diff = mw_inf - bp_inf; dm,dse0 = _mean_se(diff); delta_a3 = ways*dm; delta_se = ways*dse0

    # Diagnostic projected-global value: hold every non-Hirano3 A2A stratum fixed at the primary BP value.
    a2dir = _resolve_dir(root,args.s19a2a_result_dir); a2sum = _load(a2dir / "s19a2a_summary.json")
    global_a2 = float(a2sum["aggregate"]["A3_closed_boundary_infidelity_estimate"])
    old_h3 = None
    for r in a2sum.get("top_active_strata",[]):
        if str(r.get("key")) == TARGET_KEY:
            old_h3 = float(r["A3_closed_boundary_infidelity_estimate"]); break
    if old_h3 is None: old_h3 = 1199462.625
    stable_remainder = global_a2 - old_h3
    projected_bp_global = stable_remainder + replay_bp_a3
    projected_mw_global = stable_remainder + mw_a3

    summary = {
        "stage":"S-MWPM-2 A3-Lite frozen-acceptance paired graphlike-MWPM reframe",
        "status":"COMPLETE" if replay_gate["passed"] else "FAIL",
        "decoder_p":float(args.decoder_p),"b3_prior_seed":prior_meta.get("seed"),
        "comparison_contract":"same 10,000 physical shots and same BP+OSD accept/reject decisions; MWPM changes only accepted d2_high Pauli frames",
        "primary_decoder_unchanged":True,"truth_bits_decoder_visible":False,"operational_detector_bits":638,
        "target_stratum":TARGET_KEY,"shots":EXPECTED_SHOTS,"sampling_seed":EXPECTED_SEED,"location_triples":ways,
        "bp_exact_replay_gate":replay_gate,
        "bp_frozen_acceptance":{"operational_accept":replay_bp_accept,"harmful_samples":replay_bp_harm,"A3":replay_bp_a3,"A3_se":replay_bp_se},
        "mwpm_same_acceptance":{"operational_accept":replay_bp_accept,"harmful_samples":mw_harm,"A3":mw_a3,"A3_se":mw_se},
        "paired":{"delta_A3_mwpm_minus_bp":delta_a3,"paired_se":delta_se,"paired_z":(delta_a3/delta_se if delta_se>0 else None)},
        "frame_diagnostics":{"bp_accepted_samples_with_any_d2_high":bp_d2_accept_samples,
                             "bp_accepted_d2_high_record_occurrences":bp_d2_accept_records,
                             "unique_d2_high_records_mwpm_decoded":len(mw_decodes),
                             "new_unique_mwpm_decodes_this_run":new_mwpm_decodes,
                             "accepted_samples_with_frame_change":changed,
                             "bp_harmful_fixed_by_mwpm":fixed,
                             "new_harmful_introduced_by_mwpm":introduced,
                             "harmful_under_both":both_harm},
        "projected_global_A3_diagnostic":{
            "warning":"Not a full MWPM global A3. All non-Hirano3 A2A strata are held fixed at their BP+OSD primary values.",
            "A2A_primary_global":global_a2,"A2A_old_Hirano3":old_h3,"stable_remainder":stable_remainder,
            "with_fresh_BP_Hirano3":projected_bp_global,"with_MWPM_Hirano3_only":projected_mw_global,
        },
        "dem_audit":dem_meta,
        "interpretation":[
            "This is a paired decoder-frame ablation on the completed A3-Lite physical sample, not new physical Monte Carlo.",
            "BP+OSD acceptance is frozen exactly; MWPM is not allowed to gain performance by changing the accepted sample set.",
            "Only BP-accepted d2_high Pauli frames are replaced by MWPM; zero/firewall/cheap-exact frames are unchanged.",
            "Plain PyMatching omits irreducible >2-detector hyperedges, so the result is a graphlike-only MWPM baseline, not an exact full-hypergraph decoder.",
        ],
    }
    _dump(out / "smwpm2_a3_rescore_examples.json", examples)
    _dump(out / "smwpm2_a3_rescore_summary.json", summary)
    print("S-MWPM-2 COMPLETE", flush=True)
    print(json.dumps({
        "BP_A3":replay_bp_a3,"BP_harmful":replay_bp_harm,
        "MWPM_A3":mw_a3,"MWPM_harmful":mw_harm,
        "delta_A3":delta_a3,"paired_z":summary["paired"]["paired_z"],
        "fixed":fixed,"introduced":introduced,"frame_changed":changed,
        "projected_global_BP":projected_bp_global,"projected_global_MWPM_Hirano3_only":projected_mw_global,
    },indent=2),flush=True)
    print("->",out / "smwpm2_a3_rescore_summary.json",flush=True)


if __name__ == "__main__":
    main()
