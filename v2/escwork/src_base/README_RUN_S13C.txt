S13-C RUN GUIDE
================

Recompute the target-error-matched Sahay 3T baseline:

py -3.11 stim_pipeline\interface\direct_cs_s13c_target_error_matched_3t.py ^
  --sahay-csv .\reference\sahay_all_paper_data_v2.csv ^
  --cs-target-error 0.00085255317239522595 ^
  --out .\s13c_recomputed.json

Frozen public-data result at p=1e-3
-----------------------------------
Gap threshold: 25 dB
T LER: 0.000194133806281
Expected attempts/T: 1.56744707482
3T leading CS error: 0.000582401418843
Digitized no-gap STO/T: 3970.7
Matched expected STO/T: 4181.4
Three-T production STO: 12544.3

Caution
-------
The direct-CS d13 number is still a lower hybrid proxy because the appended
d5->d13 growth is not physically sampled. The 3T->CS conversion/routing cost
is also not included. S13-C therefore answers whether the T-production part
alone is already competitive; it is not a final architecture-level ranking.
