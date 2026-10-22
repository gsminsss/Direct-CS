S13-B RUN GUIDE
================

Purpose
-------
Normalize the two accepted d=5 |CS> output patches to d=13 using the same
published f=3 Rot(5)->Rot(7)->Rot(13) growth-cost convention from Sahay et al.

Run
---
py -3.11 stim_pipeline\interface\direct_cs_s13b_d13_normalization.py ^
  --s13a-summary .\reference\s13a_resource_summary.json ^
  --out .\s13b_recomputed.json

Frozen growth costs
-------------------
Rot5 -> Rot7 per patch: 3*(97*8) = 2328 qsteps
Rot7 -> Rot13 per patch: 169*4 = 676 qsteps
Total per patch: 3004
Two patches per |CS>: 6008

Caution
-------
The resulting d=13 number is a hybrid LOWER resource proxy:
(1) the pre-growth 8204 uses the project's native-step convention,
(2) the added 6008 uses Sahay's published gate-step convention,
(3) faults/postselection in the appended growth are not yet simulated.
Do not publish an absolute STO ranking from this file alone.
