DIRECT-|CS> S17-A — LIVE-195 physical-tail + PyMatching gate
Date: 2026-08-21

PURPOSE
-------
S12 used:
  171 live bits = present-time accept/reject
  195 full bits = 171 live + 24 noiseless future-audit bits for frame decoding

S16's raw-coset growth audit does NOT prove that the S12 target Pauli frame is
recoverable after growth. A target-frame-aware re-audit found single-fault
collisions. Therefore S17-A first physicalizes the missing 24 bits by adding
ONE MORE real d=5 surface-syndrome round (3 total after ZXZ).

The script then asks whether the resulting LIVE-195 record itself can support:
  1) exact <=2 target-Pauli-frame decoding;
  2) a graphlike Stim DEM;
  3) PyMatching MWPM with only the 195 live detector bits;
  4) finite-p logical-error estimates.

IMPORTANT SCORING SEMANTICS
---------------------------
An ideal destructive X/Z probe is appended only to label the true logical
observable. Its 12 terminal DETECTOR annotations are REMOVED before the
PyMatching DEM is built. OBSERVABLE_INCLUDE is retained as ground truth.
Therefore PyMatching sees exactly 195 detector bits and zero ideal future
syndrome bits.

WHERE TO PUT THE SCRIPT
-----------------------
Copy:
  stim_pipeline\interface\direct_cs_s17a_live195_pymatching_gate.py
into the same source tree where you previously ran S12.

Your existing environment should already contain:
  Python 3.11
  numpy
  stim <= 1.13 (the project-pinned convention)
  pymatching

FIRST RUN — EXACT GATE + PYMATCHING SINGLE-FAULT AUDIT ONLY
----------------------------------------------------------
From the root of your Direct-CS source tree in PowerShell:

py -3.11 stim_pipeline\interface\direct_cs_s17a_live195_pymatching_gate.py `
  --hirano-src ".\msc-ls\src" `
  --signature-p 0.001 `
  --skip-mc `
  --out-dir ".\stim_pipeline\interface\s17a_results"

This is the most important first run. Inspect:
  s17a_results\s17a_summary.json
  s17a_results\s17a_live195_exact_le2_summary.json
  s17a_results\s17a_pymatching_signature_summary.json

FULL FINITE-p RUN — SAME p GRID STYLE AS YOUR S12 COMMAND
---------------------------------------------------------
py -3.11 stim_pipeline\interface\direct_cs_s17a_live195_pymatching_gate.py `
  --hirano-src ".\msc-ls\src" `
  --signature-p 0.001 `
  --p-values 0.0001 0.0002 0.0005 0.001 0.002 `
  --shots-per-p 20000 30000 50000 100000 50000 `
  --seed 20260828 `
  --out-dir ".\stim_pipeline\interface\s17a_results"

OPTIONAL STRICT EXIT CODE
-------------------------
Add:
  --require-strong-le2-unique

Then the script exits with code 2 if any LIVE-195 syndrome is associated with
multiple target Pauli frames through <=2 physical faults.

EXPECTED FINGERPRINTS BEFORE RUNNING
------------------------------------
After ZXZ + 3 physical d=5 syndrome rounds, per patch:
  live detectors              = 195
  terminal scoring detectors  = 12 (removed from PyMatching input)
  physical fault locations    = 1156
  single physical Pauli classes = 10998

Relative to S12's live-171 handoff:
  added active locations = 128 / patch
                         = 256 / two-|CS>-output patches

HOW TO INTERPRET THE GATES
--------------------------
Strong gate:
  strong_live195_unique_le2_gate_pass = true
means every reachable <=2-fault LIVE-195 syndrome has one unique Pauli frame.
This would allow S17-B to replace the ideal future 24-bit tail with a real
physical round without selective ambiguity rejection at <=2.

Selective gate:
  selective_live195_df3_gate_pass = true
means exact <=2 accepted-distance-3 can still be preserved by rejecting the
ambiguous LIVE-195 keys, even if the strong gate fails.

PyMatching single gate:
  pymatching_all_single_faults_corrected = true
is a separate decoder-performance statement. It does NOT replace the exact
ambiguity proof.

DO NOT CLAIM YET
----------------
This S17-A script is NOT yet the complete factory + d=5->13 growth finite-p
result. It is the operational handoff gate needed before S17-B.

In particular, do NOT reuse the earlier S16 no-live-rejection resource proxy
(~9037.8 locations/output). The target-frame-aware precheck invalidated that
unconditional interpretation.
