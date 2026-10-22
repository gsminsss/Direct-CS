S11-A1 exact-three-fault stratified pilot
========================================

Run from the project Source-code root after extracting/merging this package.
Requires the same local Hirano checkout used for S8-S10.

1) Static/self-test only
------------------------
py -3.11 stim_pipeline\interface\direct_cs_s11a1_exact3_stratified_pilot.py `
  --self-test `
  --out-dir ".\stim_pipeline\interface\s11a1_selftest"

Expected:
S11-A1 SELF-TEST PASS

2) Small scientific smoke: 20 shots per stratum = 2,400 exact-three samples
----------------------------------------------------------------------------
py -3.11 stim_pipeline\interface\direct_cs_s11a1_exact3_stratified_pilot.py `
  --hirano-src ".\msc-ls\src" `
  --shots-per-stratum 20 `
  --seed 20260820 `
  --out-dir ".\stim_pipeline\interface\s11a1_smoke"

3) Recommended S11-A1 pilot: 200 shots per stratum = 24,000 samples
------------------------------------------------------------------------
py -3.11 stim_pipeline\interface\direct_cs_s11a1_exact3_stratified_pilot.py `
  --hirano-src ".\msc-ls\src" `
  --shots-per-stratum 200 `
  --seed 20260820 `
  --max-witnesses 100 `
  --out-dir ".\stim_pipeline\interface\s11a1_results"

Upload these two files first:
  stim_pipeline\interface\s11a1_results\s11a1_summary.json
  stim_pipeline\interface\s11a1_results\s11a1_strata.json

Optional:
  s11a1_malignant_examples.json

What to inspect
---------------
Primary:
  aggregate.A3_identity_fallback_infidelity_estimate
  aggregate.A3_identity_fallback_standard_error_naive
  aggregate.A3_identity_fallback_approx_95pct_interval

Cross-checks:
  aggregate.A3_certified_lookup_resolved_lower_estimate
  aggregate.A3_certified_lookup_unresolved_as_failure_upper_estimate
  aggregate.A3_intrinsic_pauli_orbit_infidelity_estimate

Per stratum:
  identity_binary_failures
  legacy_unresolved_accepted
  A3_identity_contribution

Important
---------
S11-A1 is a pilot.  Do not quote its A3 as paper-final merely because a numerical confidence interval is printed.  Strata with zero observed rare events require S11-A2 targeted statistics or an exact reduction.
