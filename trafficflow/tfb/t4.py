"""Task 4: path flows from link counts and the weak prior.

Methods (all solved per panel and split):
  ridge   the official baseline, min ||Af-c||^2 + 0.05||f-b||^2, f >= 0
  l2      L2 projection of the prior onto the counts: min ||f-b||^2 s.t. Af = c, f >= 0
  kl      max-entropy projection: min KL(f || b) s.t. Af = c  (f = b * exp(A^T lam))
  skl     kl after rescaling the prior by one scalar fitted to the counts
The released counts are noise-free, so the count constraints are imposed exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from .data import REL, panels


def operator(panel: str):
    net = REL / "corridors" / panel / "network"
    paths = pd.read_csv(net / "path_set.csv", dtype=str)
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    pids = paths.path_id.tolist()
    lids = inc.link_id.drop_duplicates().tolist()
    pi = {x: i for i, x in enumerate(pids)}
    li = {x: i for i, x in enumerate(lids)}
    A = np.zeros((len(lids), len(pids)))
    A[inc.link_id.map(li).to_numpy(), inc.path_id.map(pi).to_numpy()] = 1.0
    return pids, lids, A, paths


def problem(panel: str, split: str):
    pids, lids, A, paths = operator(panel)
    c = pd.read_csv(REL / "task4" / panel / split / "synthetic_link_counts.csv", dtype={"link_id": str})
    b = pd.read_csv(REL / "task4" / panel / split / "synthetic_weak_prior.csv", dtype={"path_id": str})
    li = {x: i for i, x in enumerate(lids)}
    Am = A[[li[x] for x in c.link_id]]
    bv = b.set_index("path_id").reindex(pids).path_flow.fillna(0).to_numpy(float)
    return Am, c["count"].to_numpy(float), bv, b


def solve_ridge(A, c, b, lam=0.05):
    aa = np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])])
    return nnls(aa, np.concatenate([c, np.sqrt(lam) * b]), maxiter=50 * A.shape[1])[0]


def solve_l2(A, c, b):
    """Dual of min 0.5||f-b||^2 s.t. Af=c, f>=0: f = max(0, b + A^T lam)."""
    s = np.maximum(A.sum(1), 1)
    An = A / s[:, None]
    cn = c / s

    def dual(lam):
        f = np.maximum(0.0, b + An.T @ lam)
        return 0.5 * f @ f - cn @ lam, An @ f - cn
    r = minimize(dual, np.zeros(A.shape[0]), jac=True, method="L-BFGS-B",
                 options={"maxiter": 50000, "gtol": 1e-10, "ftol": 1e-16, "maxcor": 50})
    return np.maximum(0.0, b + An.T @ r.x)


def solve_kl(A, c, b):
    """Dual of min sum f log(f/b) - f s.t. Af=c; paths with b=0 stay 0."""
    s = np.maximum(A.sum(1), 1)
    An = A / s[:, None]
    cn = c / s

    def dual(lam):
        e = b * np.exp(np.clip(An.T @ lam, -50, 50))
        return e.sum() - cn @ lam, An @ e - cn
    r = minimize(dual, np.zeros(A.shape[0]), jac=True, method="L-BFGS-B",
                 options={"maxiter": 20000, "gtol": 1e-9, "ftol": 1e-15})
    return b * np.exp(np.clip(An.T @ r.x, -50, 50))


def solve_skl(A, c, b):
    lb = A @ b
    scale = float(np.sum(c * lb) / max(np.sum(lb * lb), 1e-9))
    return solve_kl(A, c, b * scale)


METHODS = {"ridge": solve_ridge, "l2": solve_l2, "kl": solve_kl, "skl": solve_skl}


def s_link(A, c, f):
    return max(0.0, 1 - np.abs(A @ f - c).sum() / c.sum())


def build(method: str = "l2", splits=("validation", "private")) -> pd.DataFrame:
    out = []
    for p in panels():
        for s in splits:
            A, c, b, prior = problem(p, s)
            f = METHODS[method](A, c, b)
            f = np.where(np.isfinite(f), np.maximum(f, 0), 0)
            print(f"{p:11s} {s:10s} {method}: S_link={s_link(A, c, f):.5f} "
                  f"dev/prior={np.abs(f - b).sum() / b.sum():.3f} sum f/b={f.sum() / b.sum():.3f}", flush=True)
            fr = prior[["panel", "departure_time", "path_id", "origin_zone", "destination_zone"]].copy()
            fr["path_flow"] = f
            out.append(fr)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    import sys
    from .data import ROOT
    method = sys.argv[1] if len(sys.argv) > 1 else "l2"
    outdir = ROOT / "out" / "tfb"
    outdir.mkdir(parents=True, exist_ok=True)
    build(method).to_csv(outdir / f"odme_{method}.csv", index=False)
