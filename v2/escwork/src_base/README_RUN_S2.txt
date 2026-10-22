DIRECT-|CS> — S2 Hirano ZXZ DETECTOR/OBSERVABLE audit
======================================================

Prerequisite:
- S1 runtime fingerprint already passed.
- Python >= 3.10 (3.11 recommended).
- msc-ls branch experiment-2025-sep checked out locally.
- stim/numpy and Hirano branch dependencies installed.

From the project root (the directory containing stim_pipeline and msc-ls):

py -3.11 stim_pipeline/zxz/zxz_detector_observable_audit.py `
  --hirano-src ".\\msc-ls\\src" `
  --shots 256 `
  --out-dir ".\\stim_pipeline\\zxz\\s2_results"

One-line PowerShell version:

py -3.11 stim_pipeline/zxz/zxz_detector_observable_audit.py --hirano-src ".\\msc-ls\\src" --shots 256 --out-dir ".\\stim_pipeline\\zxz\\s2_results"

Expected final line:

S2 PASS: detector determinism + logical observable semantics

Expected semantic contract for BOTH channels:
- num_detectors = 135
- pre-readout detector IDs = 0..122 (123 detectors)
- terminal destructive-readout detector IDs = 123..134 (12 detectors)
- all 135 are postselection detectors when full_post_selection=True
- num_observables = 1
- baseline: all detector bits 0; observable bit 0
- commuting logical Pauli: all detector bits 0; observable bit 0
- anticommuting logical Pauli: all detector bits 0; observable bit 1

Logical perturbations used:
- X-transfer channel: commuting X_L=X^7, anticommuting Z_L=Z^7
- Z-transfer channel: commuting Z_L=Z^7, anticommuting X_L=X^7

The script also exports:
- s2_X_annotation_dependencies.json
- s2_Z_annotation_dependencies.json
- s2_X_baseline_tagged.stim
- s2_Z_baseline_tagged.stim

The JSON dependency files resolve every DETECTOR/OBSERVABLE_INCLUDE rec[-k]
target into absolute measurement IDs.  Keep these files as reproducibility
artifacts.

If an AssertionError/Traceback appears, do NOT disable assertions for the
research run. Send the full terminal output back for diagnosis.
