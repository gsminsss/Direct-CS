S11-A6 v2 — corrected global independent high-statistics A3/B3 audit

IMPORTANT
---------
Do NOT reuse the old v1 `s11a6_plan.json`. The v1 plan accidentally treated 2946 as the size of EACH escape block.
`a1._build_groups()` expects escape_n=1028; the total complete factory is 2946.
Use a fresh output directory (`s11a6v2_*`) below.

1) Self-test
------------
py -3.11 stim_pipeline\interface\direct_cs_s11a6_global_independent_highstat_a3.py `
  --self-test `
  --pilot-active ".\stim_pipeline\interface\s11a5v2_results\s11a5_active_strata.json" `
  --out-dir ".\stim_pipeline\interface\s11a6v2_selftest"

Expected hard inventory fields:
- escape_locations_per_block = 1028
- complete_factory_locations = 2946
- complete_factory_exact3_location_triples = 4257009280

2) Rebuild the fixed plan only
------------------------------
py -3.11 stim_pipeline\interface\direct_cs_s11a6_global_independent_highstat_a3.py `
  --plan-only `
  --pilot-active ".\stim_pipeline\interface\s11a5v2_results\s11a5_active_strata.json" `
  --target-a3-se 50000 `
  --inactive-floor-shots 2000 `
  --active-floor-shots 5000 `
  --seed 20260826 `
  --out-dir ".\stim_pipeline\interface\s11a6v2_plan"

The total shot count may remain close to the prior printed value because allocation comes from A5 pilot variances, but the IMPORTANT correction is the per-stratum `location_triples` weights.
The plan partition must sum to 4,257,009,280 location triples.

3) Scientific run
-----------------
py -3.11 stim_pipeline\interface\direct_cs_s11a6_global_independent_highstat_a3.py `
  --hirano-src ".\msc-ls\src" `
  --pilot-active ".\stim_pipeline\interface\s11a5v2_results\s11a5_active_strata.json" `
  --target-a3-se 50000 `
  --inactive-floor-shots 2000 `
  --active-floor-shots 5000 `
  --seed 20260826 `
  --max-examples-per-stratum 12 `
  --out-dir ".\stim_pipeline\interface\s11a6v2_results"

If interrupted after checkpoints begin, rerun the same corrected v2 command with `--resume`.
Do not resume from a v1 output directory.
