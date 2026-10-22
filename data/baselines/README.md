# Three-T comparator provenance

The indirect comparator uses `CS = (T_A ⊗ T_B) CNOT_{A→B} (I ⊗ T_B†) CNOT_{A→B}`.

The public comparator data are from K. Sahay *et al.*, *Fold-transversal surface code cultivation*, PRX Quantum **7**, 033006 (2026). The analysis uses exact public anchors at `p=5e-4` and `p=1e-3`; intermediate fixed-gap LER and total T-preparation attempts are interpolated geometrically (linearly in log space), followed by least-cost feasible gap reselection.

The complete third-party raw CSV is intentionally not redistributed. The transformed values used in the manuscript are archived in `../tables/matched_ler_binary_resource.csv`, `../tables/matched_ler_binary_a3_uncertainty.csv`, and `../tables/same_p_binary_ler.csv`.

Ready-patch convention: three T-state preparation paths, 1,662 active locations per T path times expected attempts, 1,352 active locations for the d=13 conversion, pre-existing output patches, and no routing/storage/idle/extra close-out QEC charge.

> **Superseded in v2.** The current manuscript re-decodes the comparator circuit under common conventions (1,400 active locations per T path, 676 for the conversion, early abort, same output boundary, no idle noise in the main comparison). See `v2/gapdec/` and `docs/CHANGES_v1_to_v2.md`.
