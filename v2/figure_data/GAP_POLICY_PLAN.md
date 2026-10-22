# Gap-threshold policy study — plan frozen before any calibration outcome (2026-10-21)

Circuit: d3 escape -> d13 (1,436 locations/patch). Everything except the D2 gap
threshold is unchanged (firewall order, ambiguity rejection, BP+OSD settings).

Candidate thresholds G = {4, 5, 6, 7, 8, 9, 10, 12}.

Calibration data (fresh seeds, never used for reporting):
  * finite-p: OPT3C1, p = 5e-4 and 1e-3, 5,000 shots each, prior seed 30270901,
    evaluation seed 30270921, 30,000 prior-calibration shots;
  * order-3: resumable OPT4B, seed 30281021, ~100k samples, Neyman allocation
    scaled from the main run (alloc_g3w0_cal100k.json).

Selection rule: choose g* in G minimizing
    R(g) = mean_{p in {5e-4, 1e-3}} C_Direct(g, p) / C_3T(A3_bin(g) p^3, p),
with C_3T from the re-decoded idle-free 3T comparator (convention D) and A3_bin(g)
from the order-3 calibration sample. Constraint: no sampled closed-boundary harmful
event with total fault order <= 2 (certificate regression) at g*. Ties -> larger g.

Reporting data (generated before this plan, independent of calibration):
  * finite-p: OPT3C1 rerun with the original seeds 20270901 / 20270921 (reproduces
    the published gap-10 run exactly), rescored at g*;
  * order-3: the existing 300k-sample d3 ensemble (seed 20281021), rescored at g*.
