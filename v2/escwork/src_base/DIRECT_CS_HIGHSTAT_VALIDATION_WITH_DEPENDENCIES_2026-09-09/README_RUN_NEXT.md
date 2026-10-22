# Direct-CS high-stat validation: dependency-complete runner package

The five newly uploaded historical dependencies have been normalized to the
canonical filenames expected by S21-OPT3C1. Their bytes were not edited.
The uploaded `direct_cs_s17c_d2b_logical_coset_list_decoder(1).py` is exposed
as `direct_cs_s17c_d2_logical_coset_list_decoder.py` because OPT3C1 searches
that exact canonical filename; the required D2 classes/functions are present.

Recommended execution:
1. Run `CHECK_ENVIRONMENT.ps1`.
2. Run `RUN_PILOT_10K.ps1`.
3. Check that the main gate passes and sampled harmful <=2 remains zero.
4. Without changing any decoder parameter, run `RUN_FINAL_100K.ps1`.
5. Return `highstat_binary_ler_validation_summary.json`,
   `s21opt3c1_summary.json`, and `s21opt3c1_decoder_audit.json`.

The final manuscript validation metric is attempt-level binary closed-boundary
harmful probability, not the conditional harmful fraction among truth-closed
accepted samples.
