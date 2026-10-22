DIRECT-CS S17-B — RUN FROM THE EXISTING SOURCE PROJECT ROOT

Prerequisites (same VS Code Python 3.11 environment used for S17-A):
  stim 1.16.x
  pymatching 2.4.x
  numpy

Put/copy:
  src/direct_cs_s17b_continuous_handoff_growth_pymatching.py
into:
  stim_pipeline/interface/direct_cs_s17b_continuous_handoff_growth_pymatching.py

The provided integrated source ZIP already has it in the correct location.

STEP 1 — precheck only (recommended first)

py -3.11 stim_pipeline\interface\direct_cs_s17b_continuous_handoff_growth_pymatching.py `
  --hirano-src ".\msc-ls\src" `
  --sahay-stim ".\reference\fd3_baseline_ckt.stim" `
  --signature-p 0.001 `
  --skip-mc `
  --out-dir ".\stim_pipeline\interface\s17b_results"

Important outputs:
  stim_pipeline\interface\s17b_results\s17b_summary.json
  stim_pipeline\interface\s17b_results\s17b_X_signature_summary.json
  stim_pipeline\interface\s17b_results\s17b_Z_signature_summary.json
  stim_pipeline\interface\s17b_results\s17b_X_signature_circuit.stim
  stim_pipeline\interface\s17b_results\s17b_Z_signature_circuit.stim
  stim_pipeline\interface\s17b_results\s17b_X_graphlike.dem
  stim_pipeline\interface\s17b_results\s17b_Z_graphlike.dem

Primary gate:
  first_order_pymatching_signature_gate_pass = true

If false, upload the three JSON summaries before running the large sweep.  The
failure witnesses determine whether the issue is decoder weighting/decomposition
or a structural stitched-circuit problem.

STEP 2 — finite-p sweep, only after reviewing the precheck

py -3.11 stim_pipeline\interface\direct_cs_s17b_continuous_handoff_growth_pymatching.py `
  --hirano-src ".\msc-ls\src" `
  --sahay-stim ".\reference\fd3_baseline_ckt.stim" `
  --signature-p 0.001 `
  --p-values 0.0001 0.0002 0.0005 0.001 0.002 `
  --shots-per-p 20000 30000 50000 100000 50000 `
  --seed 20260902 `
  --out-dir ".\stim_pipeline\interface\s17b_results"

Decoder-information firewall:
  PyMatching sees only 470 detector bits = 171 Hirano live + 131 growth +
  168 ideal final-d13 syndrome bits.  The destructive terminal measurement is
  used only for the truth OBSERVABLE and contributes no detector bits.

The sweep reports separate complementary logical components:
  X-channel -> residual logical-Z component
  Z-channel -> residual logical-X component
and a conservative single-patch any-Pauli bracket
  max(P_X_component, P_Z_component) <= P_any <= P_X_component + P_Z_component.

Scope:
  This is the Clifford escape+growth module, not yet the non-Clifford core E2E.
