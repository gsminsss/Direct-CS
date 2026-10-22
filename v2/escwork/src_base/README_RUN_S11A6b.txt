S11-A6b — targeted paper-final A3 closure
Date: 2026-08-21

PURPOSE
1) Add independent statistics only to the A6 variance-dominant strata.
2) Pool A6 + A6b within each stratum.
3) Replay retained full-195 ambiguities.
4) Exactify pure ESC3 ambiguities with exact distinct-location E3 reachability.
5) For CORE_POST1+ESC2, use conservative ALL64-sector + exact-E2 candidates.
   Singleton resolution is safe; multi-frame cases remain unresolved.

IMPORTANT
- Run this script from the SAME stim_pipeline/interface directory that contains
  the previous S11 scripts (A1, A2, A3, A4, A5v2, A6) and the S8/S10 support
  scripts. This package contains the new A6b driver, not the entire dependency
  tree.
- Do not overwrite the original S11-A6 output directory.
- Default A6 seed is 20260826. A6b uses independent seed 20260827.

EXPECTED DEFAULT PLAN FROM THE USER'S COMPLETED A6 RUN
variance cumulative selected = 0.8786891075
selected strata = 5
additional shots = 169141
predicted pooled lower-A3 SE = 39999.9

Expected additions:
  ESC_A3                       old 62826  add 64710  pooled 127536
  ESC_B3                       old 59013  add 59154  pooled 118167
  CORE_POST1+ESC_A2            old 27013  add 19783  pooled 46796
  CORE_POST1+ESC_B2            old 24203  add 16572  pooled 40775
  MIDED1+CORE_POST1+ESC_B1     old  5465  add  8922  pooled 14387

STEP 1 — self-test
py -3.11 stim_pipeline\interface\direct_cs_s11a6b_paper_final_closure.py `
  --self-test `
  --out-dir ".\stim_pipeline\interface\s11a6b_selftest"

STEP 2 — fixed plan only
py -3.11 stim_pipeline\interface\direct_cs_s11a6b_paper_final_closure.py `
  --plan-only `
  --a6-strata ".\stim_pipeline\interface\s11a6v2_results\s11a6_strata.json" `
  --a6-summary ".\stim_pipeline\interface\s11a6v2_results\s11a6_summary.json" `
  --target-pooled-a3-se 40000 `
  --variance-cumulative 0.85 `
  --seed 20260827 `
  --out-dir ".\stim_pipeline\interface\s11a6b_plan"

STEP 3 — scientific run
py -3.11 stim_pipeline\interface\direct_cs_s11a6b_paper_final_closure.py `
  --hirano-src ".\msc-ls\src" `
  --a6-strata ".\stim_pipeline\interface\s11a6v2_results\s11a6_strata.json" `
  --a6-summary ".\stim_pipeline\interface\s11a6v2_results\s11a6_summary.json" `
  --a6-seed 20260826 `
  --target-pooled-a3-se 40000 `
  --variance-cumulative 0.85 `
  --seed 20260827 `
  --pair-progress 500 `
  --out-dir ".\stim_pipeline\interface\s11a6b_results"

If interrupted, rerun STEP 3 with --resume.

OUTPUTS TO UPLOAD
- s11a6b_summary.json
- s11a6b_plan.json
- s11a6b_additional_strata.json
- s11a6b_pooled_strata_postclosure.json
- s11a6b_unresolved_closure.json
- s11a6b_self_test.json

HARD GATES / INTERPRETATION
- A3_lower_SE_target_met should be true.
- closure.actual_frame_not_in_candidate_union must be 0.
- pure ESC3 closure is exact distinct-location reachability.
- CORE_POST1+ESC2 closure is conservative ALL64+exact-E2, not an exhaustive
  physical-core-single proof. Remaining ambiguity stays in the upper bound.
- paper_final_numeric_freeze_recommended becomes true only if lower SE <= target
  and unresolved width <=5% of lower A3.
