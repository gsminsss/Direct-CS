S8-C — core Steane sector + operational ZXZ boundary composability
==================================================================

Prerequisites
-------------
- Python 3.11
- stim
- the SAME yutakahirano/msc-ls experiment-2025-sep checkout used for S1-S8B
- merge this ZIP into the project root so that src_current/ and stim_pipeline/ align.

Run
---
PowerShell from the project root:

py -3.11 stim_pipeline\interface\zxz_core_sector_escape_composability.py `
  --hirano-src ".\msc-ls\src" `
  --signature-p 1e-3 `
  --out-dir ".\stim_pipeline\interface\s8c_results"

Output
------
- s8c_summary.json
- s8c_boundary_unsafe_actual_syndromes.txt

Most important fields
---------------------
boundary_operational_certified_policy_exists
zero_actual_syndrome_is_safe
clean_s7b4_policy_remains_certified_for_boundary_budget_le2
clean_policy_missed_ambiguous_full_syndromes
additional_unsafe_actual_prefixes_vs_clean_s7b4
exact_accepted_boundary_failures_le2
input_sector_fault_free_escape_acceptance_by_canonical_weight
expanded_policy_escape_single_reject_coefficient_clean_input
mixed_weight1_input_plus_escape1_class_events
mixed_weight1_input_plus_escape1_accepted
mixed_weight1_input_plus_escape1_rejected

Interpretation
--------------
PASS with clean policy remains certified:
  S7-B4 unsafe table can be reused for E2E.

PASS but clean policy NOT certified:
  Use the NEW s8c_boundary_unsafe_actual_syndromes.txt table for E2E.
  This is not a failure of the architecture; it means incoming core syndrome
  sectors require a larger operational reject table.

No operational policy:
  If zero_actual_syndrome_is_safe=false, k=2 is insufficient for the full
  sector-aware core->ZXZ boundary and the handoff depth/policy must be revisited.

Scientific scope
----------------
The audit budget is
  canonical Steane input representative weight + ZXZ fault-location count <=2.
This exactly generalizes the earlier interface tests incoming-single+escape1
and incoming-weight2+escape0. It does NOT identify representative weight with
upstream direct-CS core fault count.
