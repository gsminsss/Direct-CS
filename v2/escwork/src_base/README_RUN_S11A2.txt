S11-A2 decoder-aware exact-three calibration
=============================================

1) Self-test
py -3.11 stim_pipeline\interface\direct_cs_s11a2_decoder_aware_exact3.py `
  --self-test `
  --out-dir ".\stim_pipeline\interface\s11a2_selftest"

2) Recommended scientific run
py -3.11 stim_pipeline\interface\direct_cs_s11a2_decoder_aware_exact3.py `
  --hirano-src ".\msc-ls\src" `
  --escape-shots 5000 `
  --core-post3-shots 50000 `
  --seed 20260821 `
  --max-examples 30 `
  --out-dir ".\stim_pipeline\interface\s11a2_results"

Upload s11a2_summary.json first.

Hard gates:
- actual_frame_not_in_candidate_set == 0 on ESC_A3 and ESC_B3
- conservative_le3_unique_frame_mismatches == 0 on ESC_A3 and ESC_B3

HOTFIX NOTE (2026-08-20): JSON serialization now handles nested NumPy arrays and checkpoints the three expensive component audits as soon as each finishes. If a later final-summary write fails for any unrelated reason, completed component results remain in the output directory.
