S11-A4 — exact escape-overlap closure + selective-rejection diagnostics
=====================================================================

Run from the project root after merging this package's src_current/ and stim_pipeline/ trees.

0. Required input
-----------------
Use the exact S11-A3 strata file from the completed run:

  .\stim_pipeline\interface\s11a3_results\s11a3_strata.json

The A4 replay assumes the S11-A3 global run used seed 20260823, as in the recommended command.
If you used another seed, pass it with --a3-seed.

1. Self-test (no Hirano checkout needed)
---------------------------------------
py -3.11 .\stim_pipeline\interface\direct_cs_s11a4_exact_overlap_selective_rejection.py `
  --self-test `
  --out-dir ".\stim_pipeline\interface\s11a4_selftest"

Expected:
  S11-A4 SELF-TEST PASS

This checks the compressed distinct-location E3 algorithm against literal brute force on a synthetic fault catalogue.

2. Plan only (no Hirano checkout needed)
----------------------------------------
py -3.11 .\stim_pipeline\interface\direct_cs_s11a4_exact_overlap_selective_rejection.py `
  --plan-only `
  --s11a3-strata ".\stim_pipeline\interface\s11a3_results\s11a3_strata.json" `
  --target-cumulative-width 1.0 `
  --out-dir ".\stim_pipeline\interface\s11a4_plan"

Expected with the supplied S11-A3 result:
  11 target strata
  target width fraction = 1.0

3. Full same-sample A4 exactification
-------------------------------------
py -3.11 .\stim_pipeline\interface\direct_cs_s11a4_exact_overlap_selective_rejection.py `
  --hirano-src ".\msc-ls\src" `
  --s11a3-strata ".\stim_pipeline\interface\s11a3_results\s11a3_strata.json" `
  --a3-seed 20260823 `
  --target-cumulative-width 1.0 `
  --target-batch-size 256 `
  --pair-progress 500 `
  --max-examples-per-stratum 20 `
  --out-dir ".\stim_pipeline\interface\s11a4_results"

Important: A4 intentionally replays the exact same S11-A3 random trajectories in the targeted strata.
If the accepted or unresolved counts differ from S11-A3, the script stops with a replay fingerprint error.

4. Main outputs
---------------
s11a4_summary.json
s11a4_replacements.json
s11a4_examples.json
s11a4_exact_flags.json
s11a4_pair_catalogue_meta.json
s11a4_self_test.json
s11a4_plan.json
checkpoints\<stratum>.json

5. Fields to inspect first
--------------------------
aggregate.A3_a4_exactified_lower_same_A3_samples
aggregate.A3_a4_exactified_upper_same_A3_samples
aggregate.A3_a4_exactified_interval_width
aggregate.A3_a4_sampled_live171_rejection_upper
aggregate.A3_a4_sampled_live171_rejection_interval_width

For each replacement row:
  old_unresolved
  a4_newly_resolved
  a4_still_unresolved
  a4_actual_frame_not_in_structured_union  <-- MUST be 0
  a4_live171_rejectable_unresolved_samples
  a4_live171_prefix_collision_unresolved_samples

6. Scientific interpretation
-----------------------------
A4 changes only the previously unresolved part of the same S11-A3 samples.
Already-resolved S11-A3 A3 contributions are left untouched.

A4 is still NOT the paper-final A3 because:
- only sampled unresolved order-3 trajectories are exactified;
- the current 195-bit record includes one future noiseless 24-detector audit round outside the factory fault budget;
- the live 171-bit rejection result is a sampled candidate policy, not an exhaustive order-3 certificate.

If the remaining interval is small, S11-A5 should use independent seeds and larger statistics.
If the remaining interval is dominated by 171-bit prefix collisions with accepted <=2 states, the next problem is observability / additional recovery depth, not pair-overlap bookkeeping.
