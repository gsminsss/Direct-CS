# Data dictionary

- `direct_cs_finite_p_10point.csv`: final OPT5A acceptance/resource sweep, 5,000 evaluation shots per point.
- `a3_binary_vs_infidelity_4point.csv`: final reconstructed binary and infidelity-weighted A3 values, plus sample-wise oracle residual.
- `matched_ler_binary_resource.csv`: primary binary-LER-matched Direct-vs-three-T active-location comparison.
- `matched_ler_binary_a3_uncertainty.csv`: pointwise propagation of Direct binary-A3 estimator uncertainty through gap reselection; not a complete baseline/model uncertainty interval.
- `same_p_binary_ler.csv`: supporting same-p diagnostic, not the primary matched-resource observable.
- `downstream_schedule.csv`: final per-patch d=5->13 schedule and the distinct 18,330-per-probe vs 19,590-joint mechanism counts.
- `data/final_results/`: frozen structural, decoder, exact-three, and crossover artifacts. Large OPT5C D2 caches are omitted from Git to keep the repository compact.


## v2 files
| Path | Content |
|---|---|
| `v2/escwork/sweep_d3.json` | Ten-point sweep, per (p, gap): acceptance, cost, locations per attempt, attempts, residual fraction, records, D2 records, median gap, harmful counts |
| `v2/escwork/final_compare.json` | Direct vs three-T per (p, gap): costs, ratios, storage, spacetime (same rule and least favorable), break-even |
| `v2/escwork/gap_selection.json` | Threshold-selection objective on calibration data |
| `v2/escwork/reach_fractions.json` | Front-end and core reach fractions per attempt (spacetime) |
| `v2/escwork/gen/*/s21opt3c1_summary.json` | Finite-p evaluation rows, prior calibrations, matrix audit, seeds |
| `v2/escwork/gen/*/multigap.json` | The same trajectories rescored at gaps {4,...,10,12} |
| `v2/escwork/gen/a3*/multigap_order3.json` | Per-stratum order-3 estimates (A3_bin, A3_infid, R3, D3, SEs) at every gap |
| `v2/escwork/gen/g3w0/*.stim` | Distance-3 interface X/Z probe circuits (noisy and clean) |
| `v2/escwork/gen/opt3c0_g3w0/` | Exact order-2 firewall and sector map (136 operational bits) |
| `v2/gapdec/runs/r_<p>_i<idle>_<mode>.json` | Comparator gap histograms: I (kept), X (strict failures), XC (closed failures), R (detectable residuals), XF (continuation-aware), per 0.1-dB gap bin |
| `v2/gapdec/runs/r_<p>_s<sched>_notail0.json` | Comparator with idle noise under the common rule; sched = orig (public schedule) or lp (retimed) |
| `v2/idle/idle_compare.json` | Idle comparison per (p, gap, A3 factor k) |
| `v2/idle/idle_spec_lpu.json` | Idle locations inserted into the core simulator (LP schedule, uniform gadgets) |
| `v2/figure_data/` | Data behind Figs. 4-7 and S2 |
