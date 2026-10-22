S13-D RUN GUIDE
================

Purpose
-------
Stress-test whether the unpriced 3T->CS conversion can overturn the S13-C
target-matched production comparison.

Recompute
---------
py -3.11 stim_pipeline\interface\direct_cs_s13d_conversion_break_even.py ^
  --s13c-summary .\reference\s13c_summary.json ^
  --conversion-qsteps 0 1000 2000 4000 6000 7143 8000 10000 ^
  --out .\s13d_recomputed.json

Frozen break-even
-----------------
3T production: 12544.304
Direct d13 lower proxy: 19687.031
Conversion break-even: 7142.727 qsteps

Interpretation
--------------
No architecture-independent conversion qstep value is claimed.
The exact conversion is compact (3 T/T† + 2 CNOT), but surface-code
routing/Pauli-measurement cost depends on layout and scheduling.
The recommended project stop-rule is to abandon a generic lower-STO claim
instead of building a full architecture compiler solely to chase that claim.
