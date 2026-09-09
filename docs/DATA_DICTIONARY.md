# Data dictionary

- `direct_cs_finite_p_10point.csv`: final OPT5A acceptance/resource sweep, 5,000 evaluation shots per point.
- `a3_binary_vs_infidelity_4point.csv`: final reconstructed binary and infidelity-weighted A3 values, plus sample-wise oracle residual.
- `matched_ler_binary_resource.csv`: primary binary-LER-matched Direct-vs-three-T active-location comparison.
- `matched_ler_binary_a3_uncertainty.csv`: pointwise propagation of Direct binary-A3 estimator uncertainty through gap reselection; not a complete baseline/model uncertainty interval.
- `same_p_binary_ler.csv`: supporting same-p diagnostic, not the primary matched-resource observable.
- `downstream_schedule.csv`: final per-patch d=5->13 schedule and the distinct 18,330-per-probe vs 19,590-joint mechanism counts.
- `data/final_results/`: frozen structural, decoder, exact-three, and crossover artifacts. Large OPT5C D2 caches are omitted from Git to keep the repository compact.
