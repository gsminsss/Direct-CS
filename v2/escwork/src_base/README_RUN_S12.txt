S12 — finite-p paired complete-factory audit
============================================

Purpose
-------
Run the same complete S10 physical trajectory under two policies:
  OLD: frozen S7-B4/S8-C live policy.
  NEW: OLD + exact <=2-derived live-171 safe-prefix whitelist.

The run directly measures finite-p acceptance/yield and compares it with the
S11-A6b cubic asymptotic coefficients. The 171-bit record is accept/reject only.
Frame decoding continues to use the deferred 195-bit signature (171 live +
ideal/noiseless future 24-bit audit context), exactly as in S8-C/S11.

Self-test
---------
py -3.11 stim_pipeline\interface\direct_cs_s12_finite_p_whitelist_audit.py ^
  --self-test ^
  --out-dir ".\stim_pipeline\interface\s12_selftest"

Scientific run — recommended first pass
---------------------------------------
The low-p points are primarily for acceptance/yield and cubic-consistency.
Direct logical failures are rare there. Allocate more shots at p=1e-3 and 2e-3.

py -3.11 stim_pipeline\interface\direct_cs_s12_finite_p_whitelist_audit.py ^
  --hirano-src ".\msc-ls\src" ^
  --a6b-summary ".\stim_pipeline\interface\s11a6b_results\s11a6b_summary.json" ^
  --p-values 0.0001 0.0002 0.0005 0.001 0.002 ^
  --shots-per-p 20000 30000 50000 100000 50000 ^
  --seed 20260828 ^
  --out-dir ".\stim_pipeline\interface\s12_results"

PowerShell uses backticks instead of ^ if preferred.

Resume
------
Append --resume. Each completed p point is checkpointed separately.

Outputs
-------
s12_self_test.json
s12_live171_whitelist_certificate.json
s12_sweep.json
s12_summary.json
checkpoints/p_XX.json

Primary quantities
------------------
For each p:
  old_policy.acceptance
  live171_whitelist_policy.acceptance
  paired_whitelist_effect.additional_rejection_probability_per_attempt
  live171_whitelist_policy.attempts_per_accepted_output
  live171_whitelist_policy.logical_infidelity_per_attempt_lower/upper
  live171_whitelist_policy.conditional_logical_infidelity_lower/upper
  live171_whitelist_policy.unresolved_accepted
  cubic_asymptotic_comparison.observed_additional_rejection_over_B3p3

Interpretation
--------------
* Compare B3 p^3 with the DIRECT observed additional whitelist rejection.
* Compare A3 p^3 with logical infidelity PER ATTEMPT, not conditional logical error.
* Use conditional logical error together with attempts_per_accepted_output for
  the later resource comparison.
* At p=1e-3, B3 p^3 ~0.315, so finite-p saturation/higher-order terms are expected;
  do not interpret the cubic term itself as the exact rejection probability.
* Exact d_f=3 and exact no-new-<=2-rejection are inherited and are not Monte Carlo claims.

Noise-model boundary
--------------------
Same as S10/S11: local active-location stochastic Pauli faults. This does not
include idle errors, leakage, coherent/calibration errors, or correlated
multi-location CZZ faults beyond that model.
