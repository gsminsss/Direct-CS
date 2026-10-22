# Direct cultivation of entangled |CS> magic states

Code, frozen numerical artifacts, processed data, and paper figures for **"Direct Cultivation of Entangled |CS> Magic States"** by Gunsik Min and Jun Heo (arXiv:2609.09993).

The project builds a direct fault-tolerant cultivation architecture for `|CS> = CS|++>`. It uses commuting Clifford involutions, a minimal six-bit `[6,2,4]` logical-record code, verified CAT7 readout, Steane error detection, native-CZZ-assisted controlled checks, and an escape into a surface code followed by direct growth to distance 13.

**The current manuscript (v2) uses a distance-3 escape interface.** Its material is in `v2/`. The first version (v1) used a distance-5 interface; its material is kept unchanged in `code/`, `data/`, `figures/v1/` and `archive/v1/`.

## v2 contracts (current manuscript)
- **Potential active locations:** 3,762 = 890 (front end and six-check core) + 2 × 1,436 (Steane→d=3 escape 316 + one faultable d=3→13 growth round 1,120).
- **Detector interface per patch:** 136 operational + 168 future-validation bits. Validation bits are never decoder-visible.
  - Decoder matrix `H`: 136 × 1,067, rank 136; augmented rank 138.
  - 1,071 raw first-order effect classes.
- **Certificate:** no accepted closed-boundary logical failure through total fault order 2; explicit order-3 failures exist.
- **Leading coefficient:** `A3_bin` at `p_dec = 5e-4` is 9.2(1.6)e5 at the default gap 10 and 1.70(0.22)e6 at the calibrated gap g* = 4.
  - g* = 4 was selected on independent calibration seeds with a rule fixed beforehand (`v2/escwork/GAP_POLICY_PLAN.md`).
- **Accepted-output cost** (active locations): 5,056 / 5,904 / 5,953 / 6,341 at p = 5e-4, 8e-4, 9e-4, 1e-3 (g*).
- **Comparison with a re-decoded three-T comparator** (same conventions, matched to `A3_bin p^3`):
  - cost ratio 0.98 / 1.16 / 1.20 / 1.28 without T-state storage;
  - 0.47–0.67 with synchronous three-factory assembly.
- **Idle-memory noise** (common rule, idle-minimizing LP schedule for both routes): ratio 1.08 (5e-4) and 1.46 (1e-3).

**Active locations are operation counts, not spacetime volume.**

## Layout
```text
v2/                      current manuscript (distance-3 interface)
  escwork/src_base/      simulation, certificate and decoder engine (generator, OPT3B/C0/C1, OPT4B, S10 core MC)
  escwork/drivers/       drivers that run the engine on generalized escape variants (resumable / multi-threshold)
  escwork/variants/      generators of the generalized escape circuits and OPT4B
  escwork/*.py, *.json   threshold selection, sweep collection, final comparison, Fig. 7
  escwork/gen/           run outputs: circuits, audits, finite-p summaries, order-3 tables, idle runs
  gapdec/                complementary-gap re-decoding of the three-T comparator, ledger, runs/
  idle/                  idle-minimizing retiming LP, idle specs, idle comparison, simulator patch
  figure_data/           data behind Figs. 4-7 and S2
  make_figs_v2.py        regenerates Figs. 4, 5, 6 and S2
  snap/.../reference/    place the third-party comparator circuit here (see README there)
figures/paper/           v2 paper figures
code/, data/             v1 archive (distance-5 interface), unchanged
figures/v1/, archive/v1/ v1 figures and original release zip
docs/                    scope, data dictionary, reproducibility notes, v1 to v2 changes
scripts/                 archive validation (v1 and v2) and v1 figure regeneration
```

## Quick start
```bash
python -m pip install -r requirements.txt
python scripts/validate_archive_v2.py     # v2 contracts
python scripts/validate_archive.py        # v1 contracts
cd v2 && python make_figs_v2.py           # writes v2/reproduced_figures/
cd v2 && python escwork/final_compare.py  # rebuilds the main comparison table
```

To re-run simulations, see `docs/REPRODUCIBILITY.md`. All v2 scripts locate the `v2/` root automatically; override with `DIRECT_CS_WORK`.

## Third-party material
- `v2/escwork/src_base/msc-ls/` is vendored under its own Apache License 2.0 (`LICENSE` in that folder).
- The three-T comparator circuit and raw data of Sahay *et al.*, PRX Quantum **7**, 033006 (2026), are not redistributed. Only transformed results are archived.

## Citation and license
Use `CITATION.cff`. Choose a license for the authors' code and data before public release (see `docs/PUBLIC_RELEASE_CHECKLIST.md`).
