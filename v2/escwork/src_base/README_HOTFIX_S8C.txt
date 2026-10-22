S8-C import-path hotfix (2026-08-20)
====================================

Cause
-----
The previous S8-C prepended src_current/ to sys.path. If an older copy of
zxz_core_boundary_signature_map.py remained in src_current/, Python imported
that stale copy instead of stim_pipeline/interface/zxz_core_boundary_signature_map.py.
The stale copy then computed the project root incorrectly and searched for
  <...>/5. Magic CS state/src_current/direct_cs_step8_live_boundary.py
instead of
  <...>/5. Magic CS state/2. Source code/src_current/direct_cs_step8_live_boundary.py

Fix
---
1. S8-C now loads its sibling interface modules by explicit file path.
2. S8-B now locates src_current/direct_cs_step8_live_boundary.py from robust candidates.
3. S8-A now locates direct_cs_step7_core_mc.py robustly.
4. A stale src_current/zxz_core_boundary_signature_map.py can no longer hijack S8-C.

Install
-------
Extract this ZIP directly into the project root (2. Source code) and merge/overwrite
stim_pipeline/ and src_current/.

Optional cleanup (recommended):
  Remove-Item ".\src_current\zxz_core_boundary_signature_map.py" -ErrorAction SilentlyContinue

Sanity check:
  py -3.11 stim_pipeline\interface\zxz_core_sector_escape_composability.py --help

Then rerun:
  py -3.11 stim_pipeline\interface\zxz_core_sector_escape_composability.py `
    --hirano-src ".\msc-ls\src" `
    --signature-p 1e-3 `
    --out-dir ".\stim_pipeline\interface\s8c_results"
