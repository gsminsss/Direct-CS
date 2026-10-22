# Reproducibility guide (v2, current manuscript)

## Fast path
```bash
python -m pip install -r requirements.txt
python scripts/validate_archive_v2.py
cd v2 && python make_figs_v2.py && python escwork/final_compare.py
```

## Re-running the v2 simulations
Run all commands from `v2/`. The engine root (`SRC`) is `escwork/src_base`.

1. **Distance-3 interface and order-2 audit:**
   - `python escwork/drivers/run_general.py escwork/src_base escwork/gen/g3w0 3 0 53 316` (OPT3B: escape distance 3, no wait rounds, 53 live escape detectors, 316 escape locations). This was re-run from the repository on 2026-10-22: PASS, and the regenerated circuits are byte-identical to the archived ones.
   - `python escwork/drivers/run_c0.py escwork/src_base escwork/gen/g3w0 escwork/gen/opt3c0_g3w0` (OPT3C0 firewall and matrix audit).
2. **Finite p, all thresholds:**
   - `python escwork/drivers/run_c1_multigap.py escwork/src_base escwork/gen/g3w0 escwork/gen/opt3c0_g3w0 OUT --p-values ... --prior-seed ... --evaluation-seed ...`
   - Seeds are recorded in each `s21opt3c1_summary.json`. Reporting runs: `c1rep_g3w0`, `c1rep_lowA`, `c1rep_lowB`. Calibration run: `c1cal_g3w0`.
3. **Order three:**
   - `escwork/neyman_alloc.py` sets the allocation (`escwork/gen/alloc_*.json`).
   - `escwork/drivers/run_opt4b_resumable.py` draws the 300,001 samples.
   - `escwork/drivers/run_opt4b_multigap.py` scores them at all gaps (`multigap_order3.json`).
4. **Threshold selection:** `python escwork/select_gap.py` reads calibration data only and follows `escwork/GAP_POLICY_PLAN.md`.
5. **Comparator:**
   - `python gapdec/run_gap.py p 0 notail0 shots seed gapdec/runs/out.json`. This needs the third-party circuit; see `snap/.../reference/README.md`.
   - Ledger: `gapdec/ledger2.py`. Abort cost: `abort_cost.py`. Spacetime: `spacetime_abort.py`.
6. **Final comparison:** `python escwork/final_compare.py`, then `escwork/fig7_d3.py`.
7. **Idle noise:**
   - `python idle/idle_spec.py` builds the core idle specification.
   - `python idle/down_idle.py g3w0 g3w0_idle_lp lp` builds the downstream circuits.
   - Apply `idle/src_idle_patch.diff` to a copy of `escwork/src_base` (`src_idle`), then run `IDLE_SPEC=idle/idle_spec_lpu.json python escwork/drivers/run_c1_idle_exact.py src_idle escwork/gen/g3w0_idle_lp escwork/gen/opt3c0_g3w0 OUT --p-values ...`.
   - Comparator: `gapdec/run_gap_sched.py p lp notail0 ...` and `idle/abort_lp.py`.
   - Comparison: `idle/idle_compare.py`.

## Omitted large files
Files larger than 5 MB are not archived:
- BP+OSD decision caches (`*_d2_cache_*.jsonl`) and the `*_cached` run directories;
- raw order-3 sample caches.

They are regenerated deterministically from the recorded seeds. The archived summaries, `multigap*.json` tables and logs contain every number used in the manuscript.

---

# v1 (distance-5 interface, archived)

# Reproducibility guide

## Fast path
```bash
python -m pip install -r requirements.txt
python scripts/validate_archive.py
python scripts/reproduce_data_figures.py --out-dir reproduced_figures
```

This validates the detector/matrix/resource contracts and regenerates compact data-driven versions of the main resource/A3/same-p figures from the archived final tables.

## Preserved late-stage source
`code/pipeline/` contains archived source for S21-OPT3B, OPT3C0, OPT3C1, OPT4D, OPT5C and historical S10/S17/S18 dependencies.

Frozen decoder: gap 10; 8 sector candidates; 30 minimum-sum BP iterations; scale 0.625; OSDcs order 2; max correction effects 8; no threshold sweep.

## Archive limitations
The standalone monolithic **OPT5A 10-point runner** was not preserved. Its final summary/audit/matrix/preflight artifacts are archived in `data/final_results/`, while the underlying frozen decoder implementation is preserved in OPT3C1/OPT5C source. No reconstructed replacement is mislabeled as the original.

`direct_cs_s10_complete_physical_factory_mc.py` also references an earlier project-local physical frontend module that was not present in the final handoff. Therefore this is a **paper-supporting code/result archive**, not a guaranteed one-command rebuild of every historical intermediate from an empty directory.

A proposed 100k-per-point high-statistics finite-p run is not part of the final manuscript dataset and is intentionally excluded.
