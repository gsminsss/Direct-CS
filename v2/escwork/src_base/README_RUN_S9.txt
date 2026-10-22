DIRECT-|CS> S9 — PHYSICAL MIDPOINT ED
=====================================
Run from the project root after merging this package's src_current and stim_pipeline folders.
Python 3.11 recommended.

0) Optional stale-path sanity
-----------------------------
Test-Path ".\src_current\direct_cs_step9_physical_midpoint_ed.py"
Test-Path ".\stim_pipeline\interface\direct_cs_s9b_physical_midpoint_le2_certificate.py"

1) S9-A exact physical ED self-test
------------------------------------
py -3.11 .\src_current\direct_cs_step9_physical_midpoint_ed.py `
  --self-test `
  --out-dir ".\stim_pipeline\interface\s9a_results"

Expected final line:
S9-A PASS: explicit physical Steane-ED accept instrument matches the preserved Step-2 contract

2) S9-B global <=2 certificate — recommended staged run
---------------------------------------------------------
The staged form preserves intermediate results and makes progress obvious.

2a. ED contracts + residual embedding:
py -3.11 .\stim_pipeline\interface\direct_cs_s9b_physical_midpoint_le2_certificate.py `
  --mode contracts `
  --out-dir ".\stim_pipeline\interface\s9b_results"

Expected: S9-B CONTRACTS PASS

2b. Exact first-half-core + ED cross audit:
py -3.11 .\stim_pipeline\interface\direct_cs_s9b_physical_midpoint_le2_certificate.py `
  --mode cross `
  --progress 100 `
  --cross-workers 1 `
  --out-dir ".\stim_pipeline\interface\s9b_results"

Expected reference result:
- 1431/1431 events complete
- tested_nonzero_effect_pairs = 13292
- original_class_pair_multiplicity_represented = 59464
- accepted_final_code_branches = 3528
- malignant_count = 0
- max_final_code_logical_infidelity ≈ 4.44e-16
- S9-B CROSS PASS

2c. Combine:
py -3.11 .\stim_pipeline\interface\direct_cs_s9b_physical_midpoint_le2_certificate.py `
  --mode combine `
  --contracts-json ".\stim_pipeline\interface\s9b_results\s9b_contracts.json" `
  --cross-json ".\stim_pipeline\interface\s9b_results\s9b_cross.json" `
  --out-dir ".\stim_pipeline\interface\s9b_results"

Expected final lines:
S9-B GLOBAL LOWER-BOUND PASS: explicit physical midpoint ED preserves d_f >= 3
S9-B + existing escape malignant triple: physical-midpoint integrated model has d_f = 3

You can also use --mode all, but staged mode is recommended on Windows.

3) S9-C integrated performance smoke
--------------------------------------
py -3.11 .\stim_pipeline\interface\direct_cs_s9_physical_midpoint_dual_escape_mc.py `
  --hirano-src ".\msc-ls\src" `
  --p-values 1e-3 `
  --shots 500 `
  --seed 20260820 `
  --out-dir ".\stim_pipeline\interface\s9c_smoke"

4) S9-C main diagnostic sweep
------------------------------
py -3.11 .\stim_pipeline\interface\direct_cs_s9_physical_midpoint_dual_escape_mc.py `
  --hirano-src ".\msc-ls\src" `
  --p-values 2e-4 5e-4 1e-3 `
  --shots 5000 `
  --seed 20260820 `
  --out-dir ".\stim_pipeline\interface\s9c_results"

Important S9-C fields:
- core_midED_acceptance
- integrated_acceptance
- conditional_escape_acceptance_given_core_midED
- resolved_fraction_among_accepted
- conditional_logical_infidelity_lower_bound
- conditional_logical_infidelity_upper_bound
- resolved_shots_with_infidelity_gt_1e_10
- mean_midED_fault_locations_a / b

SCOPE WARNING
-------------
S9 closes the ideal midpoint-projection abstraction. It does NOT yet integrate physical encoded-|+> preparation or CAT7 preparation/verification into the exact end-to-end state-vector model. The six CAT readouts retain the parity model q=[1-(1-2p)^7]/2.
