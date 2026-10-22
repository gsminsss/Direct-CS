S13-A RUN GUIDE
================

Purpose
-------
Compute model-internal resource metrics for the completed S12 direct-|CS> factory,
and audit whether an absolute literature STO comparison is ready.

Run
---
py -3.11 stim_pipeline\interface\direct_cs_s13a_resource_accounting.py ^
  --s12-sweep .\reference\s12_sweep.json ^
  --s7b3-results-zip <path-to-s7b3_results.zip> ^
  --out .\s13a_resource_summary_recomputed.json

Frozen main values
------------------
Complete active locations/attempt: 2946
Output: two d=5 surface-code patches
Full successful-path native qstep proxy: 8204
Native depth proxy: 154
Peak allocated-qubit proxy: 132
One k=2 escape: 1028 locations, 44 TICK layers, 66 qubit labels

Important
---------
The native qstep number is a model-specific scheduling proxy, not a paper-final,
architecture-independent STO. Do not compare it directly against Sahay Table V
until final distance and gate/idle/noise conventions are normalized.

Staging note
------------
The dual escape is charged to every core_midED_accepted attempt, before old/new 171-bit policy evaluation.
