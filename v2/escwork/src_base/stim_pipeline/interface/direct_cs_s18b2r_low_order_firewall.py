#!/usr/bin/env python3
"""S18-B2R: exact query-time <=2-fault firewall for the 638-bit boundary decoder.

Purpose
-------
S18-B2 v1 independent validation exposed two truth-closed, total-fault<=2
samples that were routed to the higher-order D2 decoder and mis-framed.  This
module installs an exact *query-time* low-order firewall before D2.

For each patch record q, the firewall computes the set of raw terminal Pauli
frames compatible with the inherited interface contract at total interface
weight <=2:

  * incoming canonical Steane sector alone, canonical weight 0,1,2;
  * incoming sector of canonical weight <=1 + one downstream physical fault;
  * clean incoming sector + two downstream physical faults at distinct
    physical locations.

Terminal-truth bits remain scoring-only and are NEVER decoder inputs.  They are
used offline only when constructing the certificate to restrict the Pauli set
to *truth-closed* low-order realizations.  Therefore the live decoder still
uses only q (638 operational bits):

  one possible low-order Pauli  -> exact frame
  multiple possible Paulis      -> reject/abstain
  no low-order realization      -> higher-order D2

The two-fault join is evaluated exactly for the queried record without
materializing O(N^2) pairs.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OP_DETECTORS = 638
TRUTH_BITS = 168
TOTAL_LOCATIONS = 3535
PAULI_NAMES = ["I", "X", "Z", "Y"]

# B2-v1 witnesses that independently exposed the missing low-order firewall.
B2_V1_WITNESSES = [
    int("2000000000000000000004000000001000000000000000000104000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000", 16),
    int("2000010010002000000000600000000020000000000100040000000000001800000000000000000000000000000000000000000000000000000000", 16),
]


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


def _mask_labels(mask: int) -> list[str]:
    return [PAULI_NAMES[p] for p in range(4) if mask & (1 << p)]


@dataclass(frozen=True)
class ConcreteClass:
    op: int
    truth: int
    pauli: int
    locations: frozenset[int]


def _aggregate_concrete_classes(model: dict[str, Any]) -> list[ConcreteClass]:
    """Aggregate concrete one-location mechanisms without losing truth or location."""
    agg: dict[tuple[int, int, int], set[int]] = {}
    for loc, choices in enumerate(model["loc_choices"]):
        for c in choices:
            k = (int(c.op), int(c.truth), int(c.pauli))
            agg.setdefault(k, set()).add(int(loc))
    return [ConcreteClass(op=k[0], truth=k[1], pauli=k[2], locations=frozenset(v))
            for k, v in sorted(agg.items(), key=lambda kv: kv[0])]


def _compatible_distinct_locations(a: ConcreteClass, b: ConcreteClass) -> tuple[bool, tuple[int, int] | None]:
    """Whether mechanisms from a,b can be assigned to two distinct locations."""
    if not a.locations or not b.locations:
        return False, None
    la = next(iter(a.locations))
    # Fast common case: if either class spans >1 location, a distinct assignment
    # exists unless both sets are exactly the same singleton (handled below).
    if len(a.locations) == 1 and len(b.locations) == 1:
        lb = next(iter(b.locations))
        return (la != lb), ((la, lb) if la != lb else None)
    for x in a.locations:
        for y in b.locations:
            if x != y:
                return True, (int(x), int(y))
    return False, None


def build_firewall_index(sectors: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    """Build O(N) low-order index.  No downstream pair Cartesian product."""
    classes = _aggregate_concrete_classes(model)
    truth0 = [c for c in classes if c.truth == 0]

    # All truth-closed <=1-downstream realizations are materialized cheaply.
    base_masks: dict[int, int] = {}
    base_min_budget: dict[int, int] = {}

    def add(q: int, p: int, budget: int) -> None:
        q = int(q); p = int(p); budget = int(budget)
        base_masks[q] = int(base_masks.get(q, 0)) | (1 << p)
        old = base_min_budget.get(q)
        base_min_budget[q] = budget if old is None else min(old, budget)

    for r in sectors:
        rs = int(r["detector_638_hex"], 16)
        ps = int(r["raw_terminal_pauli_int"])
        bw = int(r["canonical_weight"])
        if bw <= 2:
            add(rs, ps, bw)
        if bw <= 1:
            for c in truth0:
                add(rs ^ c.op, ps ^ c.pauli, bw + 1)

    # Exact two-downstream join uses equal truth masks so truth_a XOR truth_b=0.
    by_truth_syndrome: dict[int, dict[int, list[ConcreteClass]]] = {}
    for c in classes:
        by_truth_syndrome.setdefault(c.truth, {}).setdefault(c.op, []).append(c)

    clean = next((r for r in sectors if int(r["steane_syndrome"]) == 0), None)
    if clean is None or int(clean["canonical_weight"]) != 0:
        raise RuntimeError("Missing clean sector-0 row")
    if int(clean["detector_638_hex"], 16) != 0 or int(clean["raw_terminal_pauli_int"]) != 0:
        raise RuntimeError("Sector 0 must be zero operational record with identity raw frame")

    return {
        "classes": classes,
        "truth0_classes": truth0,
        "base_masks": base_masks,
        "base_min_budget": base_min_budget,
        "by_truth_syndrome": by_truth_syndrome,
        "meta": {
            "concrete_classes": len(classes),
            "truth0_classes": len(truth0),
            "truth_groups": len(by_truth_syndrome),
            "base_records": len(base_masks),
            "policy": "truth-closed exact interface weight<=2; query-time distinct-location pair join",
            "truth_bits_decoder_visible": False,
        },
    }


def _pair_mask_for_query(q: int, idx: dict[str, Any], max_examples: int = 6) -> tuple[int, list[dict[str, Any]]]:
    """Exact clean-sector + two-downstream truth-closed Pauli mask for q."""
    q = int(q)
    mask = 0
    examples: list[dict[str, Any]] = []
    by_truth_syndrome = idx["by_truth_syndrome"]
    for truth, smap in by_truth_syndrome.items():
        # Use syndrome ordering to avoid symmetric duplicate class-pair work.
        for s1, g1 in smap.items():
            s2 = q ^ int(s1)
            g2 = smap.get(s2)
            if g2 is None or int(s1) > int(s2):
                continue
            for i, a in enumerate(g1):
                for j, b in enumerate(g2):
                    if int(s1) == int(s2) and j < i:
                        continue
                    ok, loc_pair = _compatible_distinct_locations(a, b)
                    if not ok:
                        continue
                    p = int(a.pauli) ^ int(b.pauli)
                    mask |= 1 << p
                    if len(examples) < max_examples:
                        examples.append({
                            "truth_hex": hex(int(truth)),
                            "effect_A": {"op": hex(int(a.op)), "pauli": PAULI_NAMES[a.pauli], "location_count": len(a.locations)},
                            "effect_B": {"op": hex(int(b.op)), "pauli": PAULI_NAMES[b.pauli], "location_count": len(b.locations)},
                            "example_distinct_locations": list(loc_pair) if loc_pair else None,
                            "combined_pauli": PAULI_NAMES[p],
                        })
                    if mask == 0b1111 and len(examples) >= max_examples:
                        return mask, examples
    return int(mask), examples


def resolve_query(q: int, idx: dict[str, Any], max_examples: int = 6) -> dict[str, Any]:
    """Resolve one live 638-bit record against the exact <=2 closed-boundary contract."""
    q = int(q)
    base_mask = int(idx["base_masks"].get(q, 0))
    pair_mask, pair_examples = _pair_mask_for_query(q, idx, max_examples=max_examples)
    mask = base_mask | pair_mask
    labels = _mask_labels(mask)
    if len(labels) == 1:
        status = "exact"
        pauli = labels[0]
        pauli_int = PAULI_NAMES.index(pauli)
    elif len(labels) > 1:
        status = "ambiguous_reject"
        pauli = None
        pauli_int = None
    else:
        status = "higher_order_only"
        pauli = None
        pauli_int = None
    return {
        "record": hex(q),
        "status": status,
        "pauli": pauli,
        "pauli_int": pauli_int,
        "possible_paulis": labels,
        "possible_pauli_mask": int(mask),
        "base_possible_paulis": _mask_labels(base_mask),
        "pair_possible_paulis": _mask_labels(pair_mask),
        "base_min_budget": idx["base_min_budget"].get(q),
        "pair_examples": pair_examples,
    }


def resolve_queries(queries: set[int] | list[int], idx: dict[str, Any], max_examples: int = 3) -> tuple[dict[int, int], set[int], dict[int, dict[str, Any]]]:
    exact: dict[int, int] = {}
    amb: set[int] = set()
    detail: dict[int, dict[str, Any]] = {}
    for q in sorted(set(map(int, queries))):
        r = resolve_query(q, idx, max_examples=max_examples)
        detail[q] = r
        if r["status"] == "exact":
            exact[q] = int(r["pauli_int"])
        elif r["status"] == "ambiguous_reject":
            amb.add(q)
    return exact, amb, detail


def _build_project_model(root: Path, args):
    b1mod = _import("s18b2r_b1", _find(root,["direct_cs_s18b1_638_d2_calibration.py"],("stim_pipeline/interface",)))
    cert = _import("s18b2r_v13d", _find(root,["direct_cs_s17b_v13d_joint_postselection_certificate.py"],("stim_pipeline/interface",)))
    s18m = _import("s18b2r_s18m", _find(root,["direct_cs_s18m_modular_noisy_d13_certificate.py"],("stim_pipeline/interface",)))
    m2 = _import("s18b2r_m2", _find(root,["direct_cs_s18m2_terminal_truth_closure.py"],("stim_pipeline/interface",)))
    ca = _import("s18b2r_ca", _find(root,["direct_cs_s17c_a_sector_to_470_map.py"],("stim_pipeline/interface",)))
    s8b = _import("s18b2r_s8b", _find(root,["zxz_core_boundary_signature_map.py"],("stim_pipeline/interface",)))
    v7 = _import("s18b2r_v7", _find(root,["direct_cs_s17b_continuous_handoff_growth_pymatching_v7.py"],("stim_pipeline/interface",)))
    v11 = _import("s18b2r_v11", _find(root,["direct_cs_s17b_v11_complete_missing_basis.py"],("stim_pipeline/interface",)))
    s18a = _import("s18b2r_s18a", _find(root,["direct_cs_s18a_noisy_d13_boundary_certificate.py"],("stim_pipeline/interface",)))

    s18mr = (root/args.s18m_result_dir).resolve() if not args.s18m_result_dir.is_absolute() else args.s18m_result_dir.resolve()
    s18m2r = (root/args.s18m2_result_dir).resolve() if not args.s18m2_result_dir.is_absolute() else args.s18m2_result_dir.resolve()
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
    return b1mod, model, sectors, smeta


def _selftest() -> dict[str, Any]:
    # Synthetic distinct-location and truth-cancellation tests.
    fake = {
        "loc_choices": [
            [type("C", (), {"op":1,"truth":0,"pauli":1})()],
            [type("C", (), {"op":2,"truth":0,"pauli":0})()],
            [type("C", (), {"op":4,"truth":5,"pauli":2})()],
            [type("C", (), {"op":8,"truth":5,"pauli":0})()],
        ]
    }
    sectors=[{"steane_syndrome":0,"detector_638_hex":"0x0","raw_terminal_pauli_int":0,"canonical_weight":0}]
    idx=build_firewall_index(sectors,fake)
    r12=resolve_query(1^2,idx)
    r48=resolve_query(4^8,idx)
    # Same nonzero truth cancels across two faults, so second pair is closed.
    passed=(r12["status"]=="exact" and r12["pauli"]=="X" and r48["status"]=="exact" and r48["pauli"]=="Z")
    return {"passed":bool(passed),"stage":"S18-B2R exact <=2 firewall","operational_bits":638,
            "truth_bits_decoder_visible":False,"synthetic_zero_truth_pair":r12,"synthetic_equal_nonzero_truth_pair":r48}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--search-root",type=Path,default=Path("."))
    ap.add_argument("--s18m-result-dir",type=Path,default=Path("stim_pipeline/interface/s18m_results"))
    ap.add_argument("--s18m2-result-dir",type=Path,default=Path("stim_pipeline/interface/s18m2_truth_results"))
    ap.add_argument("--signature-p",type=float,default=0.001)
    ap.add_argument("--query",action="append",default=[],help="638-bit record, e.g. 0x1234; may repeat")
    ap.add_argument("--self-test-only",action="store_true")
    ap.add_argument("--out-dir",type=Path,default=Path("stim_pipeline/interface/s18b2r_firewall_results"))
    args=ap.parse_args()
    if args.self_test_only:
        print(json.dumps(_selftest(),indent=2)); return
    root=args.search_root.resolve(); out=(root/args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    print("S18-B2R: rebuilding 638-bit truth-closed mechanism model...",flush=True)
    _b1,model,sectors,smeta=_build_project_model(root,args)
    print("S18-B2R: building exact <=2 firewall index...",flush=True)
    idx=build_firewall_index(sectors,model)
    queries=B2_V1_WITNESSES.copy()
    for q in args.query:
        queries.append(int(q,0))
    # Always audit zero record too.
    queries=[0]+queries
    results=[]
    for q in queries:
        print(f"S18-B2R: exact join for {hex(q)} ...",flush=True)
        results.append(resolve_query(q,idx,max_examples=8))
    zero=results[0]
    witness=results[1:1+len(B2_V1_WITNESSES)]
    gate={
        "zero_record_closed_boundary_identity_only": zero["status"]=="exact" and zero["pauli"]=="I",
        "b2_v1_witnesses_intercepted_before_D2": all(r["status"] in {"exact","ambiguous_reject"} for r in witness),
        "b2_v1_witnesses_not_misframed_as_identity": all(not (r["status"]=="exact" and r["pauli"]=="I") for r in witness),
        "truth_bits_decoder_visible":False,
    }
    gate["passed"]=bool(gate["zero_record_closed_boundary_identity_only"] and gate["b2_v1_witnesses_intercepted_before_D2"] and gate["b2_v1_witnesses_not_misframed_as_identity"])
    summary={"stage":"S18-B2R exact query-time <=2 firewall audit","firewall_meta":idx["meta"],"sector_meta":smeta,
             "queries":results,"validation_gate":gate,
             "paper_interpretation":"Live decoder still sees only 638 operational bits. Scoring-only terminal truth is used offline to construct the exact <=2 closed-boundary Pauli contract; D2 is called only when the query has no <=2 closed-boundary realization."}
    _dump(out/"s18b2r_summary.json",summary)
    _dump(out/"s18b2r_witness_audit.json",witness)
    print(json.dumps(gate,indent=2),flush=True)
    if not gate["passed"]:
        raise RuntimeError(("S18-B2R firewall audit failed",gate))

if __name__=="__main__":
    main()
