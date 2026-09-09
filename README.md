# Direct cultivation of entangled |CS> magic states

Supporting code, frozen numerical artifacts, processed data, and paper figures for **“Direct Cultivation of Entangled |CS> Magic States”** by Gun Sik Min and Jun Heo.

The project constructs a direct fault-tolerant cultivation architecture for `|CS> = CS|++>`, using commuting Clifford involutions, a minimal six-bit `[6,2,4]` logical-record code, verified CAT7 readout, Steane error detection, native-CZZ-assisted controlled checks, and direct `d=5 -> 13` surface-code expansion.

## Main archived contracts
- six-check core: **42 CZZ + 56 CX + 224 composite 1Q = 322 active locations**;
- optimized full potential-location inventory: **5,154**;
- downstream schedule: **2,132 active locations per output patch**;
- detector interface: **263 operational + 168 future-truth bits**;
- `H`: `263 x 3339`, rank 263; augmented rank 265;
- no accepted closed-boundary harmful mechanism through total fault order 2; explicit order-3 witnesses exist;
- binary `A3` at `p_dec=5e-4`: approximately **2.365e6**;
- Direct accepted-output active-location costs: **6,651**, **8,068**, **8,941**, **9,836** at `p=5e-4, 8e-4, 9e-4, 1e-3`;
- binary-LER-matched ready-patch comparison: Direct wins at `5e-4` and `8e-4`; ordering reverses between `8e-4` and `9e-4`.

**Active locations are operation counts, not spacetime volume.** Future-truth bits are never decoder-visible.

## Layout
```text
code/foundation/       early exact/foundation checks
code/pipeline/         late-stage factory/decoder/certificate/exact-three source
data/final_results/    frozen final artifacts
data/tables/           compact paper-facing tables under the final binary metric contract
data/baselines/        comparator provenance and conventions
figures/paper/         paper data figures and supporting diagnostics
docs/                  scope, data dictionary, reproducibility notes, release checklist
scripts/               archive validation and compact figure regeneration
```

## Quick start
```bash
python -m pip install -r requirements.txt
python scripts/validate_archive.py
python scripts/reproduce_data_figures.py --out-dir reproduced_figures
```

See `docs/REPRODUCIBILITY.md` for the preserved-source scope and two documented historical source gaps. A proposed high-statistics finite-p run is intentionally excluded because it is not part of the final manuscript dataset.

## Three-T baseline
The matched comparator uses public data from K. Sahay *et al.*, *Fold-transversal surface code cultivation*, PRX Quantum **7**, 033006 (2026). Only transformed values used in this manuscript are archived here; the full third-party raw dataset is not redistributed.

## Citation and license
Use `CITATION.cff` and update it with the final repository/article DOI. No license has been selected automatically; choose one before public release if reuse permissions should be granted.
