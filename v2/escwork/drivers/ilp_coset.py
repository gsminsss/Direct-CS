"""Exact minimum-weight (maximum-likelihood single explanation) solver for
H_aug x = t (mod 2), used to augment the BP+OSD candidate list of the
four-coset decoder.  Weights w_j = log((1-q_j)/q_j) from the same effect
parity priors the BP+OSD decoder uses.  HiGHS via scipy.optimize.milp."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, hstack, diags
import time

class ILPSolver:
    def __init__(self, h_aug, priors, extra_solutions=0, time_limit=5.0):
        H = csr_matrix(h_aug).astype(np.int64)
        self.H = H; self.m, self.n = H.shape
        q = np.clip(np.asarray(priors, float), 1e-15, 0.5 - 1e-12)
        self.w = np.log((1 - q) / q)
        rw = np.asarray(H.sum(axis=1)).ravel()
        self.zmax = np.floor(rw / 2.0)
        self.A = hstack([H, diags(-2.0 * np.ones(self.m))]).tocsr()
        self.c = np.concatenate([self.w, np.zeros(self.m)])
        self.integrality = np.ones(self.n + self.m)
        self.bounds = Bounds(np.zeros(self.n + self.m), np.concatenate([np.ones(self.n), self.zmax]))
        self.extra = int(extra_solutions); self.tl = float(time_limit)
        self.stats = dict(calls=0, solved=0, infeasible=0, seconds=0.0)

    def solve(self, target):
        t = np.asarray(target, float).ravel()
        self.stats["calls"] += 1; t0 = time.time()
        cons = [LinearConstraint(self.A, t, t)]
        out = []
        for k in range(1 + self.extra):
            r = milp(self.c, constraints=cons, integrality=self.integrality, bounds=self.bounds,
                     options=dict(time_limit=self.tl, disp=False))
            if r.x is None or r.status not in (0,):
                if k == 0: self.stats["infeasible"] += 1
                break
            x = np.rint(r.x[:self.n]).astype(np.uint8)
            out.append(x)
            # no-good cut: exclude this exact support
            s = np.flatnonzero(x); row = np.zeros(self.n + self.m); row[s] = 1.0; row[[j for j in range(self.n) if x[j] == 0]] = -1.0
            cons = cons + [LinearConstraint(csr_matrix(row), -np.inf, len(s) - 1)]
        if out: self.stats["solved"] += 1
        self.stats["seconds"] += time.time() - t0
        return out
