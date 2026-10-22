S11-A5 v2 — certified live-171 PREFIX whitelist + full-195 residual A3 audit

IMPORTANT
---------
Do not use the original A5 script that assumed `171 prefix -> unique frame`.
Use:

  stim_pipeline/interface/direct_cs_s11a5v2_certified_live171_prefix_whitelist.py

The 171-bit record controls only ACCEPT/REJECT. The Pauli frame is decoded from the 195-bit full recovery syndrome, exactly as in S7-B4/S8-C/S11-A3.

1) Self-test
------------
py -3.11 stim_pipeline\interface\direct_cs_s11a5v2_certified_live171_prefix_whitelist.py `
  --self-test `
  --out-dir ".\stim_pipeline\interface\s11a5v2_selftest"

Expected:
  S11-A5 SELF-TEST PASS

The synthetic test intentionally puts two different full-195 frames behind the same 171-bit prefix and verifies that this is legal.

2) Scientific run
-----------------
py -3.11 stim_pipeline\interface\direct_cs_s11a5v2_certified_live171_prefix_whitelist.py `
  --hirano-src ".\msc-ls\src" `
  --s11a3-strata ".\stim_pipeline\interface\s11a3_results\s11a3_strata.json" `
  --a3-seed 20260823 `
  --max-examples-per-stratum 20 `
  --out-dir ".\stim_pipeline\interface\s11a5v2_results"

If interrupted, rerun with --resume.

3) Hard certificate gates
-------------------------
In s11a5_live171_whitelist_certificate.json verify:

  exact_no_new_le2_acceptance_penalty = true
  safe_unsafe_prefix_overlap = 0
  safe_full_states_missed_by_whitelist = 0
  frame_decoding_uses_full195_not_live171 = true

`safe_prefixes_with_multiple_full195_frames` may be large. That is expected and records precisely why the first A5 draft failed.

4) Scientific outputs
---------------------
Main summary:
  s11a5_summary.json

Key aggregate fields:
  A3_certified_live171_prefix_whitelist_resolved_lower_same_samples
  A3_certified_live171_prefix_whitelist_unresolved_as_failure_upper_same_samples
  A3_certified_live171_prefix_whitelist_interval_width_same_samples
  B3_additional_live171_prefix_whitelist_rejection_same_samples_estimate
  retained_unresolved_samples
  residual_harmful_samples

Interpretation:
- The prefix whitelist itself is exactly certified against all <=2 boundary states.
- Any remaining A3 interval is from order-3 events that alias an already-safe 171 prefix and remain ambiguous at full-195 decoder level.
- Do not call the A3 number paper-final until independent seeds/high statistics or exactification closes the residual interval.
