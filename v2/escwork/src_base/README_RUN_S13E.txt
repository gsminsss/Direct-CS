S13-E FULL PHYSICAL 3T BASELINE
================================

Key exact public-circuit count
------------------------------
Sahay fd3_baseline_ckt.stim:
  physical d5 factory       = 283 active locations
  d5->d7 growth             = 971
  d7->d13 growth            = 408
  physical pipeline total   = 1662
  ideal decoder tail        = 1128  [excluded]
  destructive readout       = 25    [excluded]

At the target-matched G>=25 dB point:
  expected attempts/T       = 1.567447075
  active loc / accepted T   = 2605.10
  active loc / three T      = 7815.29

Direct CS at p=1e-3:
  d5 active loc / accepted  = 8812.71
  d13 lower proxy           = 11570.71

Optimistic 3T->CS conversion:
  5 transversal logical CNOTs + 3 destructive magic-patch readouts
  conversion locations      = 1352

Scenarios:
  data patches free         = 9167.29
  fresh resource generation = 13495.29

Conclusion
----------
The gate-level comparison is architecture/use-case dependent. Do not claim a
universal resource winner. The direct route wins in the stand-alone
resource-generation model used here; the optimized 3T route wins if the two
encoded output data patches are treated as pre-existing/free.

Recompute:
py -3.11 stim_pipeline\interface\direct_cs_s13e_sahay_gate_count.py ^
  --stim reference\fd3_baseline_ckt.stim ^
  --out s13e_gate_count_recomputed.json
