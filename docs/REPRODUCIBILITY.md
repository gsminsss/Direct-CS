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
