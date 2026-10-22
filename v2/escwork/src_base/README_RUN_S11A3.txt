S11-A3 RUN GUIDE
================

Prerequisite: same project tree used successfully for S10/S11-A2, including .\msc-ls\src.
Run from the project source-code root.

1) Self-test
------------
py -3.11 stim_pipeline\interface\direct_cs_s11a3_priority_decoder_a3.py `
  --self-test `
  --out-dir ".\stim_pipeline\interface\s11a3_selftest"

Expected: S11-A3 SELF-TEST PASS

2) Dominant-only smoke / scientific preview
--------------------------------------------
py -3.11 stim_pipeline\interface\direct_cs_s11a3_priority_decoder_a3.py `
  --hirano-src ".\msc-ls\src" `
  --only-dominant `
  --dominant-shots-per-stratum 1000 `
  --core-post3-shots 10000 `
  --seed 20260823 `
  --out-dir ".\stim_pipeline\interface\s11a3_smoke"

3) Recommended global adaptive run
----------------------------------
py -3.11 stim_pipeline\interface\direct_cs_s11a3_priority_decoder_a3.py `
  --hirano-src ".\msc-ls\src" `
  --base-shots-per-stratum 200 `
  --dominant-shots-per-stratum 10000 `
  --core-post3-shots 100000 `
  --seed 20260823 `
  --max-examples-per-stratum 8 `
  --out-dir ".\stim_pipeline\interface\s11a3_results"

4) Resume after interruption
----------------------------
Repeat the same command and add:
  --resume

Each stratum checkpoint is saved in s11a3_results\checkpoints immediately after completion.

UPLOAD FIRST
------------
s11a3_summary.json
s11a3_strata.json

If requested, also upload:
s11a3_examples.json

HARD GATES
----------
The following must be zero in every stratum:
- actual_frame_not_in_conservative_union
- accepted_exact_le2_ambiguous_policy_violation

PRIMARY SCIENTIFIC OUTPUT
-------------------------
aggregate.A3_priority_decoder_resolved_lower_estimate
aggregate.A3_priority_decoder_unresolved_as_failure_upper_estimate
aggregate.A3_interval_width_due_to_unresolved

top_strata_by_lower_A3
top_strata_by_interval_width

Interpretation:
- lower is an observed/estimated decoder-aware leading coefficient from resolved events.
- upper pessimistically calls every unresolved accepted event a logical failure.
- if the interval remains wide, S11-A4 exactifies only the dominant ambiguity-width strata/syndromes.
