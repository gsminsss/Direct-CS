DIRECT-|CS> S10 — RUN GUIDE
===========================
Run from the project/source root. Python 3.11 is recommended.

1) Front-end self-test
py -3.11 src_current\direct_cs_step10_physical_frontend.py
Expected: S10-A FRONT-END SELF-TEST PASS

2) Complete <=2 certificate
py -3.11 stim_pipeline\interface\direct_cs_s10b_complete_factory_le2_certificate.py `
  --out-dir ".\stim_pipeline\interface\s10b_results"
Expected:
S10-B GLOBAL LOWER-BOUND PASS: complete active-location physical factory has d_f >= 3
S10-B + existing escape malignant triple: complete active-location physical factory has d_f = 3

3) Integrated MC self-test
py -3.11 stim_pipeline\interface\direct_cs_s10_complete_physical_factory_mc.py `
  --self-test `
  --out-dir ".\stim_pipeline\interface\s10_selftest"
Expected: S10 SELF-TEST PASS: complete physical front end integrated

4) 500-shot smoke
py -3.11 stim_pipeline\interface\direct_cs_s10_complete_physical_factory_mc.py `
  --hirano-src ".\msc-ls\src" `
  --p-values 1e-3 `
  --shots 500 `
  --seed 20260820 `
  --out-dir ".\stim_pipeline\interface\s10c_smoke"

5) Main sweep
py -3.11 stim_pipeline\interface\direct_cs_s10_complete_physical_factory_mc.py `
  --hirano-src ".\msc-ls\src" `
  --p-values 2e-4 5e-4 1e-3 `
  --shots 5000 `
  --seed 20260820 `
  --out-dir ".\stim_pipeline\interface\s10c_results"

Upload s10_sweep.json and s10_meta.json.

LOCATION INVENTORY
physical preparation + initial ED, two blocks : 194
six CAT7 prep/verification gadgets             : 168
six CAT X readouts                             : 42
six-check physical core                        : 322
physical midpoint ED, two blocks               : 164
dual live ZXZ escape                            : 2056
TOTAL                                           : 2946
