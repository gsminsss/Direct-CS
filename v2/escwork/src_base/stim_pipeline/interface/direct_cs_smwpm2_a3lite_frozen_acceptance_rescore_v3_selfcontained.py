from __future__ import annotations

"""S-MWPM-2: paired MWPM reframe of the completed S19-A3-Lite sample.

Purpose
-------
Replay the exact 10,000 physical DOWN_HIRANO_D5^3 samples used by S19-A3-Lite,
first reproduce the frozen BP+OSD primary result exactly, then rescore ONLY the
Pauli frame of BP-accepted d2_high records using an embedded graphlike-only MWPM baseline.

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
The MWPM implementation is embedded in this file; no S-MWPM-1 source file is required.
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

TOL = 1e-12
OP_DETECTORS = 638
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


SMWPM2_SCRIPT_VERSION = "SMWPM2-V3-SELFCONTAINED-2026-09-04"

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



def _load_b3_prior(path: Path, p: float) -> tuple[np.ndarray, dict[str, Any]]:
    rows = _load(path)
    hit = None
    for r in rows:
        if abs(float(r.get("p", -1)) - float(p)) <= max(1e-15, abs(float(p))*1e-10):
            hit = r
            break
    if hit is None:
        raise RuntimeError(("requested decoder p missing from frozen B3 priors", p,
                            [r.get("p") for r in rows]))
    raw = hit["sector_probability"]
    # S18-B3 stores the frozen prior as a JSON object keyed by sector id
    # ("0", "1", ..., "63").  Do NOT rely on dict/value order because
    # lexicographic JSON key order would place "10" before "2".
    if isinstance(raw, dict):
        try:
            keys = {int(k) for k in raw.keys()}
        except Exception as e:
            raise RuntimeError(("bad B3 sector-prior keys", list(raw)[:8])) from e
        if keys != set(range(64)):
            raise RuntimeError(("B3 sector prior must contain exactly sectors 0..63", sorted(keys)))
        a = np.asarray([float(raw[str(i)]) if str(i) in raw else float(raw[i]) for i in range(64)], dtype=float)
    else:
        # Backward compatibility with an older/list representation.
        a = np.asarray(raw, dtype=float)
    if a.shape != (64,) or np.any(a < 0) or not np.isfinite(a).all() or a.sum() <= 0:
        raise RuntimeError("bad B3 sector prior")
    a = a / a.sum()
    return a, hit

def _parse_error_line(line: str) -> tuple[float, list[tuple[tuple[int, ...], int]]] | None:
    """Parse one flattened DEM error line.

    Returns probability and components [(detector_tuple, L0_parity), ...].
    This helper is probe-local; each probe DEM has exactly one logical observable L0.
    """
    s = line.strip()
    if not s.startswith("error("):
        return None
    m = re.match(r"error\(([^)]+)\)\s*(.*)$", s)
    if not m:
        raise RuntimeError(("cannot parse DEM error", line))
    p = float(m.group(1))
    rest = m.group(2).strip()
    raw_components = [x.strip() for x in rest.split("^")]
    comps: list[tuple[tuple[int, ...], int]] = []
    for rc in raw_components:
        ds: list[int] = []
        logical = 0
        if rc:
            for tok in rc.split():
                if tok.startswith("D"):
                    ds.append(int(tok[1:]))
                elif tok.startswith("L"):
                    # Probe circuits have one observable; use parity in case of duplicates.
                    if int(tok[1:]) != 0:
                        raise RuntimeError(("unexpected probe logical observable", tok))
                    logical ^= 1
                else:
                    raise RuntimeError(("unexpected DEM target", tok, line))
        comps.append((tuple(ds), int(logical)))
    return p, comps

def _error_rows_from_dem(dem) -> list[tuple[float, list[tuple[tuple[int, ...], int]]]]:
    # flattened() removes REPEAT and shift_detectors, making D ids absolute.
    text = str(dem.flattened())
    out = []
    for line in text.splitlines():
        q = _parse_error_line(line)
        if q is not None:
            out.append(q)
    return out

def _canonicalize_row(row):
    """Return a row with components sorted by detector support.

    Sorting by detector support removes harmless DEM component-order differences between
    the X and Z probe circuits. Duplicate component supports inside one row are rejected
    because their cross-probe logical pairing would be ambiguous for this diagnostic.
    """
    p, comps = row
    ds = [tuple(d) for d,_ in comps]
    if len(ds) != len(set(ds)):
        raise RuntimeError(("duplicate detector-support component inside one DEM row; refusing ambiguous X/Z merge", ds))
    comps2 = sorted(((tuple(d), int(l)&1) for d,l in comps), key=lambda q:q[0])
    return float(p), comps2

def _probe_dem_audit(rows, label: str) -> dict[str, Any]:
    """Audit one single-observable probe DEM without attempting cross-probe pairing.

    PyMatching can consume graphlike error instructions and graphlike components of a
    suggested decomposition.  Irreducible components with >2 detector symptoms are not
    represented by plain MWPM.  Detector-free logical-only components are also invisible
    to a syndrome-only matching graph.  We therefore report both explicitly.
    """
    component_hist=Counter()
    unresolved_hist=Counter()
    total_row_mass=0.0
    rows_with_hyper=0
    hyper_row_mass=0.0
    detector_free_logical_rows=0
    detector_free_logical_mass=0.0
    examples=[]
    max_det=-1
    for i,row in enumerate(rows):
        p, comps=_canonicalize_row(row)
        total_row_mass += float(p)
        has_hyper=False
        has_detector_free_logical=False
        for d,l in comps:
            w=len(d)
            component_hist[w]+=1
            if d:
                max_det=max(max_det,*d)
            if w>2:
                has_hyper=True
                unresolved_hist[w]+=1
            if w==0 and int(l):
                has_detector_free_logical=True
        if has_hyper:
            rows_with_hyper += 1
            hyper_row_mass += float(p)
            if len(examples)<16:
                examples.append({
                    "row_index":int(i),
                    "probability":float(p),
                    "component_detector_weights":[len(d) for d,_ in comps],
                    "components":[list(d) for d,_ in comps],
                    "logical_parities":[int(l) for _,l in comps],
                })
        if has_detector_free_logical:
            detector_free_logical_rows += 1
            detector_free_logical_mass += float(p)
    return {
        "probe":str(label),
        "error_instruction_count":len(rows),
        "decomposition_component_detector_weight_histogram":{str(k):int(v) for k,v in sorted(component_hist.items())},
        "irreducible_hyperedge_rows":int(rows_with_hyper),
        "irreducible_component_detector_weight_histogram":{str(k):int(v) for k,v in sorted(unresolved_hist.items())},
        "irreducible_hyperedge_examples":examples,
        "error_instruction_probability_mass_sum":float(total_row_mass),
        "rows_with_irreducible_hyperedge_probability_mass_sum":float(hyper_row_mass),
        "rows_with_irreducible_hyperedge_probability_mass_fraction_of_row_mass":float(hyper_row_mass/total_row_mass) if total_row_mass>0 else 0.0,
        "detector_free_logical_rows":int(detector_free_logical_rows),
        "detector_free_logical_probability_mass_sum":float(detector_free_logical_mass),
        "max_detector_id_seen":int(max_det),
    }

def _build_separate_matchings(xcircuit, zcircuit, out_dir: Path, write_dem: bool):
    """Build independent X- and Z-probe MWPM graphs; never pair DEM rows.

    This avoids the fundamentally ambiguous operation of reconstructing physical-mechanism
    identity from flattened DEM rows.  The two circuits are already authoritative probes
    of the SAME operational detector record; S18-M itself verifies probe-independent
    syndrome for physical mechanisms before constructing Pauli effects.
    """
    try:
        import stim
        import pymatching
    except Exception as e:
        raise RuntimeError("S-MWPM-1 requires stim and pymatching in this Python environment") from e

    if int(xcircuit.num_detectors) != OP_DETECTORS or int(zcircuit.num_detectors) != OP_DETECTORS:
        raise RuntimeError(("operational detector count drift", xcircuit.num_detectors, zcircuit.num_detectors))
    if int(xcircuit.num_observables) != 1 or int(zcircuit.num_observables) != 1:
        raise RuntimeError(("probe observable count drift", xcircuit.num_observables, zcircuit.num_observables))

    # Retain decomposition failures so the DEM can be audited instead of aborting.  Plain
    # PyMatching uses graphlike mechanisms/decomposition components and ignores irreducible
    # >2-detector components.  This is therefore explicitly an approximate MWPM baseline.
    xdem=xcircuit.detector_error_model(
        decompose_errors=True,
        allow_gauge_detectors=False,
        ignore_decomposition_failures=True,
    )
    zdem=zcircuit.detector_error_model(
        decompose_errors=True,
        allow_gauge_detectors=False,
        ignore_decomposition_failures=True,
    )
    xrows=_error_rows_from_dem(xdem)
    zrows=_error_rows_from_dem(zdem)
    xa=_probe_dem_audit(xrows,"X")
    za=_probe_dem_audit(zrows,"Z")

    # Crucially: no X/Z row matching or merging here.
    xmatching=pymatching.Matching.from_detector_error_model(xdem)
    zmatching=pymatching.Matching.from_detector_error_model(zdem)
    xmatching.ensure_num_fault_ids(1)
    zmatching.ensure_num_fault_ids(1)
    for label,m in (("X",xmatching),("Z",zmatching)):
        if int(m.num_detectors)!=OP_DETECTORS:
            raise RuntimeError((f"PyMatching {label} detector count",m.num_detectors))
        if int(m.num_fault_ids)<1:
            raise RuntimeError((f"PyMatching {label} logical fault-id count",m.num_fault_ids))

    if write_dem:
        (out_dir/"smwpm1_x_with_unresolved_hyperedges.dem").write_text(str(xdem),encoding="utf-8")
        (out_dir/"smwpm1_z_with_unresolved_hyperedges.dem").write_text(str(zdem),encoding="utf-8")

    meta={
        "construction":"two independent single-observable PyMatching graphs; no cross-probe DEM row pairing",
        "exact_full_model_mwpm":False,
        "baseline_scope":"graphlike-only approximation; irreducible >2-detector and detector-free logical-only mechanisms are audited explicitly",
        "truth_bits_decoder_visible":False,
        "x_probe":xa,
        "z_probe":za,
        "x_matching_detectors":int(xmatching.num_detectors),
        "z_matching_detectors":int(zmatching.num_detectors),
        "x_matching_edges":int(xmatching.num_edges),
        "z_matching_edges":int(zmatching.num_edges),
        "x_matching_fault_ids":int(xmatching.num_fault_ids),
        "z_matching_fault_ids":int(zmatching.num_fault_ids),
        "probe_error_count_difference":int(len(zrows)-len(xrows)),
        "reason_no_pairing":"flattened DEM rows do not preserve unique physical-mechanism identity; repeated support/probability rows are expected and need not pair uniquely",
    }
    return xmatching, zmatching, meta

def _int_to_syndrome(x: int) -> np.ndarray:
    a = np.zeros(OP_DETECTORS, dtype=np.uint8)
    y = int(x)
    while y:
        b = y & -y
        k = b.bit_length() - 1
        if k >= OP_DETECTORS:
            raise RuntimeError(("record has detector bit outside 638", k))
        a[k] = 1
        y ^= b
    return a

def _decode_one_probe(matching, syn: np.ndarray) -> tuple[int,float]:
    pred,w=matching.decode(syn,return_weight=True)
    pred=np.asarray(pred,dtype=np.uint8).reshape(-1)
    if pred.size < 1:
        raise RuntimeError(("MWPM returned no logical fault id",pred.size))
    return int(pred[0])&1,float(w)

def _mwpm_sector_decode(record: int, xmatching, zmatching, sectors: list[dict[str, Any]], prior: np.ndarray) -> dict[str, Any]:
    """Decode a 638-bit record using separate X/Z probe MWPM graphs.

    S18 convention: pauli = lz | (lx << 1), where lz is the Z-probe observable
    and lx is the X-probe observable.  Therefore Z-probe MWPM supplies bit0 (X
    frame) and X-probe MWPM supplies bit1 (Z frame).

    The two probe graphs describe the same detector evidence with different observable
    annotations, so their path weights should be close.  We use their arithmetic mean as
    a sector-path diagnostic to avoid double-counting the same syndrome evidence, and
    report the X/Z weight disagreement explicitly.
    """
    if int(record)==0:
        return {
            "record_hex":"0x0","predicted_pauli_int":0,"predicted_pauli":"I",
            "gap":float("inf"),"best_sector":0,"best_downstream_pauli":"I",
            "zero_structural_shortcut":True,"four_final_paulis_covered":True,
        }
    by_final: dict[int,list[dict[str,Any]]]=defaultdict(list)
    failures=0
    weight_disagreements=[]
    for r in sectors:
        s=int(r["steane_syndrome"])
        sig=int(r.get("detector_638_hex",r.get("detector_470_hex")),16)
        residual=int(record)^sig
        syn=_int_to_syndrome(residual)
        try:
            # X probe -> lx -> Pauli bit1 (Z frame).
            lx,wx=_decode_one_probe(xmatching,syn)
            # Z probe -> lz -> Pauli bit0 (X frame).
            lz,wz=_decode_one_probe(zmatching,syn)
        except Exception:
            failures+=1
            continue
        downstream=(int(lz)&1)|((int(lx)&1)<<1)
        sector_pauli=int(r["raw_terminal_pauli_int"])
        final=int(downstream)^int(sector_pauli)
        wmean=0.5*(float(wx)+float(wz))
        wdiff=abs(float(wx)-float(wz))
        weight_disagreements.append(wdiff)
        score=math.log(max(float(prior[s]),1e-300))-wmean
        by_final[final].append({
            "sector":s,"score":float(score),"mwpm_weight_mean":float(wmean),
            "mwpm_weight_Xprobe":float(wx),"mwpm_weight_Zprobe":float(wz),
            "mwpm_weight_abs_difference":float(wdiff),
            "xprobe_lx":int(lx),"zprobe_lz":int(lz),
            "downstream_pauli_int":int(downstream),"downstream_pauli":PAULI[int(downstream)],
            "sector_pauli_int":int(sector_pauli),"sector_pauli":PAULI[int(sector_pauli)],
            "residual_weight":int(residual.bit_count()),
        })
    covered=sorted(by_final)
    if not covered:
        return {"record_hex":hex(int(record)),"decode_success":False,"sector_decode_failures":failures}
    best_by_final={q:max(v,key=lambda z:z["score"]) for q,v in by_final.items()}
    ordered=sorted(best_by_final,key=lambda q:best_by_final[q]["score"],reverse=True)
    q0=int(ordered[0]); q1=int(ordered[1]) if len(ordered)>1 else None
    gap=float(best_by_final[q0]["score"]-best_by_final[q1]["score"]) if q1 is not None else 0.0
    return {
        "record_hex":hex(int(record)),"decode_success":True,
        "predicted_pauli_int":q0,"predicted_pauli":PAULI[q0],
        "gap_minpath_diagnostic":gap,
        "four_final_paulis_covered":len(covered)==4,
        "covered_final_paulis":[PAULI[q] for q in covered],
        "best_sector":int(best_by_final[q0]["sector"]),
        "best_downstream_pauli":best_by_final[q0]["downstream_pauli"],
        "best_sector_pauli":best_by_final[q0]["sector_pauli"],
        "best_mwpm_weight_mean":float(best_by_final[q0]["mwpm_weight_mean"]),
        "best_score":float(best_by_final[q0]["score"]),
        "best_per_final_pauli":{PAULI[q]:best_by_final[q] for q in sorted(best_by_final)},
        "sector_decode_failures":int(failures),"sector_count_tested":len(sectors),
        "probe_weight_abs_difference_max":float(max(weight_disagreements)) if weight_disagreements else None,
        "probe_weight_abs_difference_median":float(np.median(weight_disagreements)) if weight_disagreements else None,
        "zero_structural_shortcut":False,
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


def _preflight_required_sources(root: Path) -> dict[str, str]:
    """Resolve every external source needed before any expensive replay starts."""
    required = [
        "direct_cs_s19a3_lite_hirano3_precision.py",
        "direct_cs_s19a2a_exact3_independent_highstat.py",
        "direct_cs_s11a1_exact3_stratified_pilot.py",
        "direct_cs_s18b1_638_d2_calibration.py",
        "direct_cs_s18b3_firewalled_fresh_validation.py",
        "direct_cs_s18b2r_low_order_firewall.py",
        "direct_cs_s17c_d2_logical_coset_list_decoder.py",
        "direct_cs_s17c_d_hybrid_bposd_abstention.py",
        "direct_cs_s10_complete_physical_factory_mc.py",
        "direct_cs_s17b_v13d_joint_postselection_certificate.py",
        "direct_cs_s18m_modular_noisy_d13_certificate.py",
        "direct_cs_s18m2_terminal_truth_closure.py",
        "direct_cs_s17c_a_sector_to_470_map.py",
        "zxz_core_boundary_signature_map.py",
        "direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py",
        "direct_cs_s17b_v11_complete_missing_basis.py",
        "direct_cs_s18a_noisy_d13_boundary_certificate.py",
    ]
    found = {}
    missing = []
    for name in required:
        try:
            found[name] = str(_find(root, [name]))
        except FileNotFoundError:
            missing.append(name)
    if missing:
        raise FileNotFoundError(("S-MWPM-2 preflight missing required project source files", missing))
    try:
        import stim, pymatching
        found["stim_version"] = str(getattr(stim, "__version__", None))
        found["pymatching_version"] = str(getattr(pymatching, "__version__", None))
    except Exception as e:
        raise RuntimeError("S-MWPM-2 preflight requires stim and pymatching in this Python environment") from e
    return found


def _preflight_required_inputs(root: Path, args) -> dict[str, str]:
    """Check all result files and external resource paths up front."""
    dirs = {
        "s18m": _resolve_dir(root, args.s18m_result_dir),
        "s18m2": _resolve_dir(root, args.s18m2_result_dir),
        "b2r": _resolve_dir(root, args.b2r_result_dir),
        "b3": _resolve_dir(root, args.b3_result_dir),
        "a2a": _resolve_dir(root, args.s19a2a_result_dir),
        "a3": _resolve_dir(root, args.s19a3_result_dir),
    }
    required_files = {
        "s18m_sector_map": dirs["s18m"] / "s18m_sector_map_638.json",
        "s18m2_x_truth": dirs["s18m2"] / "s18m2_X_truth_closed.stim",
        "s18m2_z_truth": dirs["s18m2"] / "s18m2_Z_truth_closed.stim",
        "b3_sector_map": dirs["b3"] / "s18b3_sector_map_638_truth.json",
        "b3_priors": dirs["b3"] / "s18b3_sector_priors.json",
        "a2a_summary": dirs["a2a"] / "s19a2a_summary.json",
        "a3_summary": dirs["a3"] / "s19a3_lite_summary.json",
        "a3_plan": dirs["a3"] / "s19a3_lite_plan.json",
        "a3_rows": dirs["a3"] / "s19a3_lite_rows.json",
        "a3_firewall_cache": dirs["a3"] / "s19a3_lite_firewall_cache.jsonl",
        "a3_bp_d2_cache": dirs["a3"] / "s19a3_lite_d2_cache_0p0002.jsonl",
    }
    missing=[f"{k}: {v}" for k,v in required_files.items() if not v.exists()]
    # These resource locations are needed by the exact A3 replay/model rebuild.
    try:
        hirano = _find(root, ["msc-ls/src"])
    except FileNotFoundError:
        hirano = None; missing.append("resource: msc-ls/src")
    try:
        sahay = _find(root, ["reference/fd3_baseline_ckt.stim"])
    except FileNotFoundError:
        sahay = None; missing.append("resource: reference/fd3_baseline_ckt.stim")
    if missing:
        raise FileNotFoundError(("S-MWPM-2 preflight missing required inputs; fix these before replay", missing))
    out={k:str(v) for k,v in required_files.items()}
    out["hirano_src"] = str(hirano)
    out["sahay_circuit"] = str(sahay)
    return out


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
    ap.add_argument("--preflight-only", action="store_true", help="Check every required source/result/resource and exit before replay.")
    args = ap.parse_args()

    print(f"S-MWPM-2 script version: {SMWPM2_SCRIPT_VERSION}", flush=True)

    st = _self_test()
    if args.self_test_only:
        print(json.dumps(st, indent=2, sort_keys=True)); return
    if not st["passed"]: raise RuntimeError(st)
    if abs(float(args.decoder_p) - DEFAULT_DECODER_P) > 1e-15:
        raise SystemExit("S-MWPM-2 primary paired rescore is frozen to --decoder-p 0.0002. Use a separate later ablation for other priors.")

    root = args.search_root.resolve()
    out = _resolve_dir(root, args.out_dir); out.mkdir(parents=True, exist_ok=True)
    _dump(out / "smwpm2_self_test.json", st)

    preflight = _preflight_required_sources(root)
    inputs = _preflight_required_inputs(root, args)
    _dump(out / "smwpm2_preflight.json", {"sources":preflight,"inputs":inputs})
    print(f"S-MWPM-2 preflight PASS: {len([k for k in preflight if k.endswith('.py')])} project sources + {len(inputs)} required inputs; stim={preflight['stim_version']}, pymatching={preflight['pymatching_version']}", flush=True)

    if args.preflight_only:
        print(json.dumps({"passed":True,"script_version":SMWPM2_SCRIPT_VERSION,"sources":preflight,"inputs":inputs}, indent=2, sort_keys=True), flush=True)
        return

    # Import the completed A3-Lite/A2A implementations. MWPM helpers are embedded above.
    a3 = _import("smwpm2_a3", Path(preflight["direct_cs_s19a3_lite_hirano3_precision.py"]))
    a2 = _import("smwpm2_a2", Path(preflight["direct_cs_s19a2a_exact3_independent_highstat.py"]))
    a1 = _import("smwpm2_a1", Path(preflight["direct_cs_s11a1_exact3_stratified_pilot.py"]))
    b1mod = _import("smwpm2_b1", Path(preflight["direct_cs_s18b1_638_d2_calibration.py"]))
    b3mod = _import("smwpm2_b3", Path(preflight["direct_cs_s18b3_firewalled_fresh_validation.py"]))
    b2rmod = _import("smwpm2_b2r", Path(preflight["direct_cs_s18b2r_low_order_firewall.py"]))
    d2 = _import("smwpm2_d2", Path(preflight["direct_cs_s17c_d2_logical_coset_list_decoder.py"]))
    d1 = _import("smwpm2_d1", Path(preflight["direct_cs_s17c_d_hybrid_bposd_abstention.py"]))
    s10mc = _import("smwpm2_s10", Path(preflight["direct_cs_s10_complete_physical_factory_mc.py"]))
    cert = _import("smwpm2_cert", Path(preflight["direct_cs_s17b_v13d_joint_postselection_certificate.py"]))
    s18m = _import("smwpm2_s18m", Path(preflight["direct_cs_s18m_modular_noisy_d13_certificate.py"]))
    m2 = _import("smwpm2_m2", Path(preflight["direct_cs_s18m2_terminal_truth_closure.py"]))
    ca = _import("smwpm2_ca", Path(preflight["direct_cs_s17c_a_sector_to_470_map.py"]))
    s8b = _import("smwpm2_s8b", Path(preflight["zxz_core_boundary_signature_map.py"]))
    v7 = _import("smwpm2_v7", Path(preflight["direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"]))
    v11 = _import("smwpm2_v11", Path(preflight["direct_cs_s17b_v11_complete_missing_basis.py"]))
    s18a = _import("smwpm2_s18a", Path(preflight["direct_cs_s18a_noisy_d13_boundary_certificate.py"]))
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
    hirano = Path(inputs["hirano_src"]); sahay = Path(inputs["sahay_circuit"])
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
    prior, prior_meta = _load_b3_prior(prior_path,float(args.decoder_p))

    print(f"S-MWPM-2: building graphlike-only X/Z MWPM at decoder-p={args.decoder_p:g} ...", flush=True)
    # Saved S18-M circuits are p=1e-3, so for p=2e-4 we deliberately rebuild isolated channels.
    xc,_xclean,_xold,_xm = s18m._build_extended_channel(v7,v11,s18a,hirano,sahay,"X",float(args.decoder_p))
    zc,_zclean,_zold,_zm = s18m._build_extended_channel(v7,v11,s18a,hirano,sahay,"Z",float(args.decoder_p))
    xmatching,zmatching,dem_meta = _build_separate_matchings(xc,zc,out,bool(args.write_dem))
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
                    dd = _mwpm_sector_decode(rec,xmatching,zmatching,sectors,prior)
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
