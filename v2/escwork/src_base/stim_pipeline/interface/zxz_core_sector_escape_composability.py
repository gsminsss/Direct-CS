#!/usr/bin/env python3
"""S8-C: exact operational composability of a coherent Steane core sector with k=2 ZXZ live handoff.

Why this stage exists
=====================
S8-A showed that the live direct-|CS> core boundary can be a coherent
superposition of Steane syndrome sectors. S8-B then proved that all 64
canonical Steane syndrome sectors have distinct 171-bit operational detector
signatures under

    ZXZ + 2 faultable surface-QEC rounds -> live handoff.

Therefore the ZXZ detector measurement dephases the physical-syndrome
coherence and each sector can be treated as a classical branch while retaining
its arbitrary logical amplitudes.

However, the S7-B4 unsafe-syndrome table was certified only for a CLEAN Steane
input. It is not automatically valid when an incoming Steane syndrome
signature is XORed with ZXZ fault signatures. S8-C closes exactly that gap.

Boundary-distance audit
=======================
For one Steane block, every syndrome has a canonical representative E_s of
weight 0, 1, or 2 (inventory 1/21/42). We audit all interface events with

    boundary_weight(E_s) + number_of_ZXZ_fault_locations <= 2.

Concretely:
  * weight-0 input + 0/1/2 ZXZ faults;
  * weight-1 input + 0/1 ZXZ fault;
  * weight-2 input + 0 ZXZ faults.

This is the exact live-handoff analogue of the earlier composability contract
"incoming single + escape fault" and "incoming weight-2 + fault-free escape".
It is deliberately an INTERFACE-weight audit; it does not claim that canonical
Steane representative weight equals the number of faults that occurred in the
upstream direct-CS core.

For every reachable 195-bit recovery syndrome, collect the reachable logical
frame labels I/X/Z/Y. If more than one frame is reachable, that full recovery
syndrome is ambiguous. Project all ambiguous full syndromes onto the 171-bit
prefix physically available at live handoff. An implementable conservative
policy rejects exactly those 171-bit prefixes.

The critical outputs are:
  * boundary_operational_certified_policy_exists
  * zero_actual_syndrome_is_safe
  * clean_s7b4_policy_remains_certified_for_boundary_budget_le2
  * exact_accepted_boundary_failures_le2 (=0 by construction when policy exists)

If the clean S7-B4 table is insufficient, S8-C emits the expanded boundary-safe
unsafe-prefix table for the E2E decoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ZXZ_DIR = HERE.parent / "zxz"
SRC_CURRENT = ROOT / "src_current"

# IMPORTANT: do not import interface modules by bare module name here.
# Older S8 hotfixes may have left copies in src_current/, and prepending
# src_current to sys.path can silently import the stale copy.  Load the
# exact sibling files used by this package instead.
def _load_local_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

for p in (ZXZ_DIR, HERE, SRC_CURRENT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

s8b = _load_local_module("s8c_local_s8b", HERE / "zxz_core_boundary_signature_map.py")
s7b3 = _load_local_module("s8c_local_s7b3", HERE / "zxz_faultable_handoff_recovery_audit.py")
s7b2 = _load_local_module("s8c_local_s7b2", HERE / "zxz_ideal_recovery_handoff_audit.py")
s7b4 = _load_local_module("s8c_local_s7b4", HERE / "zxz_operational_handoff_policy_audit.py")

FAULTABLE_ROUNDS = 2
IDEAL_TAIL_ROUNDS = 1
ACTUAL_BITS = 171
FULL_BITS = 195
EXPECTED_INPUT_SECTORS = 64
EXPECTED_INPUT_WEIGHT_COUNTS = {0: 1, 1: 21, 2: 42}
EXPECTED_ESCAPE_LOCATIONS = 1028
EXPECTED_ESCAPE_SINGLE_CLASSES = 9750
EXPECTED_ESCAPE_PAIR_CLASSES = 47_460_960
FRAME_LABELS = ("I", "X", "Z", "Y")


def _prefix(full: int) -> int:
    return int(full) & ((1 << ACTUAL_BITS) - 1)


def _frames(flags: int) -> list[int]:
    return [f for f in range(4) if (int(flags) >> f) & 1]


def _frame_label_set(flags: int) -> list[str]:
    return [FRAME_LABELS[f] for f in _frames(flags)]


def _add(table: dict[int, int], syndrome: int, frame: int) -> None:
    table[int(syndrome)] = int(table.get(int(syndrome), 0)) | (1 << int(frame))


def _class_coeff(r: Any) -> float:
    return 1.0 / 15.0 if r.gate == "CX" else 1.0


@dataclass(frozen=True)
class SectorRow:
    syndrome: int
    weight: int
    actual: int
    full: int
    frame: int
    pauli: str


def _sector_rows(hirano_src: Path, signature_p: float) -> tuple[list[SectorRow], dict[str, Any]]:
    summary, rows, _singles = s8b.build_map(hirano_src, signature_p)
    if not summary["actual_signature_injective"]:
        raise AssertionError("S8-B injectivity is required before S8-C")
    if not summary["coherent_sector_dephasing_by_actual_detector_record_certified"]:
        raise AssertionError("S8-B dephasing gate failed")
    if int(summary["actual_signature_unique_count"]) != EXPECTED_INPUT_SECTORS:
        raise AssertionError(summary["actual_signature_unique_count"])
    out: list[SectorRow] = []
    for r in rows:
        out.append(
            SectorRow(
                syndrome=int(r["steane_syndrome"]),
                weight=int(r["canonical_weight"]),
                actual=int(r["actual_171_syndrome_hex"], 16),
                full=int(r["full_195_syndrome_hex"], 16),
                frame=int(r["logical_frame_int"]),
                pauli=str(r["canonical_pauli"]),
            )
        )
    wc = Counter(r.weight for r in out)
    if dict(sorted(wc.items())) != EXPECTED_INPUT_WEIGHT_COUNTS:
        raise AssertionError((wc, EXPECTED_INPUT_WEIGHT_COUNTS))
    z = [r for r in out if r.weight == 0]
    if len(z) != 1 or z[0].syndrome != 0 or z[0].actual != 0 or z[0].full != 0 or z[0].frame != 0:
        raise AssertionError(("bad zero input sector", z))
    return out, summary


def _escape_catalogue(hirano_src: Path, signature_p: float):
    xrows, xloc, xmeta = s7b3._build_channel_rows(
        hirano_src, "X", FAULTABLE_ROUNDS, IDEAL_TAIL_ROUNDS, signature_p
    )
    zrows, zloc, zmeta = s7b3._build_channel_rows(
        hirano_src, "Z", FAULTABLE_ROUNDS, IDEAL_TAIL_ROUNDS, signature_p
    )
    if len(xloc) != len(zloc):
        raise AssertionError((len(xloc), len(zloc)))
    paired, mismatches = s7b2._pair_complementary(xrows, zrows)
    if mismatches:
        raise AssertionError(("X/Z recovery syndrome mismatch", mismatches))
    if len(xloc) != EXPECTED_ESCAPE_LOCATIONS:
        raise AssertionError((len(xloc), EXPECTED_ESCAPE_LOCATIONS))
    if len(paired) != EXPECTED_ESCAPE_SINGLE_CLASSES:
        raise AssertionError((len(paired), EXPECTED_ESCAPE_SINGLE_CLASSES))
    pair_count = s7b3.s7b._pair_denominator(xloc)
    if pair_count != EXPECTED_ESCAPE_PAIR_CLASSES:
        raise AssertionError((pair_count, EXPECTED_ESCAPE_PAIR_CLASSES))
    if len(xmeta["recovery_ids"]) != FULL_BITS:
        raise AssertionError((len(xmeta["recovery_ids"]), FULL_BITS))
    clean_analysis = s7b2._analyze_exact_le2(paired)
    clean_policy = s7b4.derive_policy(paired, clean_analysis, ACTUAL_BITS)
    return paired, clean_analysis, clean_policy, xmeta


def _event_source_table(
    sectors: list[SectorRow], paired: list[Any], clean_analysis: dict[str, Any]
) -> tuple[dict[int, int], dict[tuple[int, int], str], dict[str, int]]:
    """Build all frame reachability for boundary_weight + escape_fault_order <=2.

    The expensive weight0 + two-escape-fault part is already exactly compressed
    in clean_analysis['state_table'], so no 47M class-pair loop is needed here.
    """
    table: dict[int, int] = {int(s): int(flags) for s, flags in clean_analysis["state_table"].items()}
    source: dict[tuple[int, int], str] = {}
    for syn, flags in table.items():
        for f in _frames(flags):
            source.setdefault((int(syn), int(f)), "input_w0+escape_le2")

    counts = Counter()
    # weight-1 input, no escape and + exactly one escape fault class
    w1 = [s for s in sectors if s.weight == 1]
    for s in w1:
        _add(table, s.full, s.frame)
        source.setdefault((s.full, s.frame), f"input_w1(s={s.syndrome})+escape0")
        counts["input_w1_escape0_events"] += 1
        for r in paired:
            full = s.full ^ int(r.recovery_syndrome)
            frame = s.frame ^ int(r.frame)
            _add(table, full, frame)
            source.setdefault((full, frame), f"input_w1(s={s.syndrome})+escape1(fault_id={r.fault_id})")
            counts["input_w1_escape1_class_events"] += 1

    # weight-2 input, no escape
    for s in sectors:
        if s.weight != 2:
            continue
        _add(table, s.full, s.frame)
        source.setdefault((s.full, s.frame), f"input_w2(s={s.syndrome})+escape0")
        counts["input_w2_escape0_events"] += 1

    return table, source, dict(counts)


def _policy_from_table(table: dict[int, int]) -> dict[str, Any]:
    ambiguous_full = {int(s): int(flags) for s, flags in table.items() if int(flags).bit_count() > 1}
    unsafe_actual = {_prefix(s) for s in ambiguous_full}
    return {
        "ambiguous_full_recovery_syndromes_boundary_le2": len(ambiguous_full),
        "unsafe_actual_syndrome_count_boundary_le2": len(unsafe_actual),
        "zero_actual_syndrome_is_safe": 0 not in unsafe_actual,
        "boundary_operational_certified_policy_exists": 0 not in unsafe_actual,
        "unsafe_actual": unsafe_actual,
        "ambiguous_full": ambiguous_full,
    }


def _representative_ambiguities(
    table: dict[int, int], source: dict[tuple[int, int], str], limit: int = 30
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for syn, flags in sorted(table.items()):
        if int(flags).bit_count() <= 1:
            continue
        fr = _frames(flags)
        out.append(
            {
                "full_recovery_syndrome_hex": hex(int(syn)),
                "actual_171_prefix_hex": hex(_prefix(int(syn))),
                "frames": [FRAME_LABELS[f] for f in fr],
                "sources": {FRAME_LABELS[f]: source.get((int(syn), f), "compressed_escape_le2") for f in fr},
            }
        )
        if len(out) >= limit:
            break
    return out


def audit(hirano_src: Path, signature_p: float) -> tuple[dict[str, Any], set[int]]:
    sectors, s8b_summary = _sector_rows(hirano_src, signature_p)
    paired, clean_analysis, clean_policy, xmeta = _escape_catalogue(hirano_src, signature_p)
    table, source, event_counts = _event_source_table(sectors, paired, clean_analysis)
    pol = _policy_from_table(table)

    clean_unsafe = {int(h, 16) for h in clean_policy["unsafe_actual_syndromes_hex"]}
    boundary_unsafe: set[int] = pol["unsafe_actual"]
    additional = boundary_unsafe - clean_unsafe
    removed = clean_unsafe - boundary_unsafe

    # Does the clean-input policy miss any boundary ambiguity?
    missed_ambiguous = [
        int(s) for s in pol["ambiguous_full"] if _prefix(int(s)) not in clean_unsafe
    ]

    # Acceptance of canonical incoming sectors with fault-free escape.
    sector_by_weight: dict[int, dict[str, Any]] = {}
    for w in (0, 1, 2):
        rr = [s for s in sectors if s.weight == w]
        rej = [s for s in rr if s.actual in boundary_unsafe]
        sector_by_weight[w] = {
            "total": len(rr),
            "accepted": len(rr) - len(rej),
            "rejected": len(rej),
            "rejected_syndromes": [s.syndrome for s in rej],
        }

    # Escape singles with clean input under the expanded boundary policy.
    rej_escape_singles = [r for r in paired if _prefix(int(r.recovery_syndrome)) in boundary_unsafe]
    acc_escape_singles = [r for r in paired if _prefix(int(r.recovery_syndrome)) not in boundary_unsafe]
    reject_coeff_escape = sum(_class_coeff(r) for r in rej_escape_singles)

    # Exact mixed incoming-weight1 + one-escape-fault class counts. These are
    # only ~205k and are useful for interface comparison with historical S4-B.
    mixed_total = mixed_rej = mixed_acc = 0
    for s in sectors:
        if s.weight != 1:
            continue
        for r in paired:
            mixed_total += 1
            now = s.actual ^ _prefix(int(r.recovery_syndrome))
            if now in boundary_unsafe:
                mixed_rej += 1
            else:
                mixed_acc += 1
                flags = int(table[s.full ^ int(r.recovery_syndrome)])
                if flags.bit_count() != 1:
                    raise AssertionError("accepted mixed event maps to ambiguous full syndrome")

    # Accepted weight-2 input + fault-free escape must also be unique by policy.
    for s in sectors:
        if s.weight == 2 and s.actual not in boundary_unsafe:
            flags = int(table[s.full])
            if flags.bit_count() != 1:
                raise AssertionError("accepted weight2 input maps to ambiguous full syndrome")

    summary: dict[str, Any] = {
        "stage": "S8-C",
        "faultable_handoff_rounds": FAULTABLE_ROUNDS,
        "ideal_tail_rounds": IDEAL_TAIL_ROUNDS,
        "actual_detector_bits": ACTUAL_BITS,
        "full_recovery_detector_bits": FULL_BITS,
        "s8b_actual_signature_injective": bool(s8b_summary["actual_signature_injective"]),
        "s8b_actual_signature_unique_count": int(s8b_summary["actual_signature_unique_count"]),
        "canonical_input_sector_count": len(sectors),
        "canonical_input_weight_counts": {str(k): v for k, v in EXPECTED_INPUT_WEIGHT_COUNTS.items()},
        "escape_physical_locations": EXPECTED_ESCAPE_LOCATIONS,
        "escape_single_fault_classes": len(paired),
        "escape_two_fault_class_pairs": EXPECTED_ESCAPE_PAIR_CLASSES,
        "boundary_budget_definition": "canonical_input_weight + ZXZ_fault_location_count <= 2",
        "boundary_event_counts_added_beyond_clean_escape_table": event_counts,
        "full_recovery_syndrome_states_boundary_le2": len(table),
        "ambiguous_full_recovery_syndromes_boundary_le2": int(pol["ambiguous_full_recovery_syndromes_boundary_le2"]),
        "unsafe_actual_syndrome_count_boundary_le2": len(boundary_unsafe),
        "zero_actual_syndrome_is_safe": bool(pol["zero_actual_syndrome_is_safe"]),
        "boundary_operational_certified_policy_exists": bool(pol["boundary_operational_certified_policy_exists"]),
        "exact_accepted_boundary_failures_le2": 0 if pol["boundary_operational_certified_policy_exists"] else None,
        "clean_s7b4_unsafe_actual_syndrome_count": len(clean_unsafe),
        "additional_unsafe_actual_prefixes_vs_clean_s7b4": len(additional),
        "clean_unsafe_prefixes_removed_by_boundary_rebuild": len(removed),
        "clean_policy_missed_ambiguous_full_syndromes": len(missed_ambiguous),
        "clean_s7b4_policy_remains_certified_for_boundary_budget_le2": len(missed_ambiguous) == 0,
        "input_sector_fault_free_escape_acceptance_by_canonical_weight": {str(k): v for k, v in sector_by_weight.items()},
        "expanded_policy_rejected_escape_single_fault_classes_clean_input": len(rej_escape_singles),
        "expanded_policy_accepted_escape_single_fault_classes_clean_input": len(acc_escape_singles),
        "expanded_policy_escape_single_reject_coefficient_clean_input": float(reject_coeff_escape),
        "mixed_weight1_input_plus_escape1_class_events": mixed_total,
        "mixed_weight1_input_plus_escape1_accepted": mixed_acc,
        "mixed_weight1_input_plus_escape1_rejected": mixed_rej,
        "representative_boundary_ambiguities": _representative_ambiguities(table, source),
        "location_counts": xmeta["location_counts"],
    }
    return summary, boundary_unsafe


def _synthetic_self_test() -> None:
    # Prefix ambiguity projection: full syndromes 0b001 and 0b101 have same low
    # two-bit prefix 01. If one full syndrome is frame-ambiguous, prefix 01 must
    # be unsafe.
    global ACTUAL_BITS
    old = ACTUAL_BITS
    try:
        ACTUAL_BITS = 2
        t = {0: 1, 0b001: (1 << 0) | (1 << 1), 0b101: 1 << 0}
        p = _policy_from_table(t)
        assert 0b01 in p["unsafe_actual"]
        assert p["zero_actual_syndrome_is_safe"]
    finally:
        ACTUAL_BITS = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirano-src", required=True, type=Path)
    ap.add_argument("--signature-p", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=Path("stim_pipeline/interface/s8c_results"))
    args = ap.parse_args()

    _synthetic_self_test()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, unsafe = audit(args.hirano_src.resolve(), args.signature_p)

    (args.out_dir / "s8c_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "s8c_boundary_unsafe_actual_syndromes.txt").write_text(
        "\n".join(hex(x) for x in sorted(unsafe)) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["boundary_operational_certified_policy_exists"]:
        if summary["clean_s7b4_policy_remains_certified_for_boundary_budget_le2"]:
            print("S8-C PASS: clean S7-B4 live-handoff policy remains certified with all canonical Steane input sectors through boundary budget <=2")
        else:
            print("S8-C PASS: expanded sector-aware live-handoff policy is certified through boundary budget <=2; clean S7-B4 table must be replaced")
    else:
        print("S8-C RESULT: no operational policy can accept the zero syndrome while certifying all audited core-sector + ZXZ boundary events through budget <=2")


if __name__ == "__main__":
    main()
