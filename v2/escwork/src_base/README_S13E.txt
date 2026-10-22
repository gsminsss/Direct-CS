S13-E GATE-LEVEL BASELINE PACKAGE
================================

Main results
------------
Public Sahay f=3 sample no-idle active locations: 2652
Direct-CS active locations / attempt: 2946
Empirical public-circuit gate-density rho: 0.879894
Gate-level diagnostic conversion break-even at p=1e-3: ~3061 active locations

Read first
----------
- S13E_STATUS.md
- S13E_METHOD.md
- s13e_summary.json

Reproduce the public circuit audit
----------------------------------
Download:
https://raw.githubusercontent.com/kaavyas99/MSC_foldedH/main/sample_circuits/fd3_baseline_ckt.stim

Then:
py -3.11 stim_pipeline\interface\direct_cs_s13e_sahay_gate_audit.py ^
  fd3_baseline_ckt.stim ^
  --out sahay_gate_audit_recomputed.json

Important boundary
------------------
This package DOES NOT claim an exact full d=13 3T->CS physical baseline.
The exact public circuit audit ends at the publicly supplied stabilizer-escape sample.
The missing main-paper d7->d13 unitary circuit and the optimal physical 3T->CS
conversion must be explicitly compiled before an absolute winner can be frozen.
