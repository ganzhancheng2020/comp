"""T4：先验 = 重力结构 × 独立噪声（跨 split 残差相关 ~0），故先平滑再投影。"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .t4 import METHODS, problem


def gravity_design(prior_frame):
    zi = prior_frame.origin_zone.str[1:].astype(int).to_numpy()
    zj = prior_frame.destination_zone.str[1:].astype(int).to_numpy()
    d = zj - zi
    n = max(zi.max(), zj.max()) + 1
    D = d.max() + 1
    m = len(zi)
    r = np.arange(m)
    X = sp.hstack([sp.csr_matrix((np.ones(m), (r, zi)), shape=(m, n)),
                   sp.csr_matrix((np.ones(m), (r, zj)), shape=(m, n)),
                   sp.csr_matrix((np.ones(m), (r, d)), shape=(m, D))]).tocsr()
    return X


def gravity_fit(prior_frame, priors: list[np.ndarray]) -> np.ndarray:
    """exp(log-linear fit) jointly on the given priors (same design)."""
    X = gravity_design(prior_frame)
    Xs = sp.vstack([X] * len(priors)).tocsr()
    y = np.concatenate([np.log(np.maximum(b, 1e-6)) for b in priors])
    coef = lsqr(Xs, y, atol=1e-10, btol=1e-10)[0]
    return np.exp(X @ coef)


def estimate(panel: str, split: str, how: str) -> np.ndarray:
    """how = '<prior>_<method>': prior in {raw, geo, grav, grav3}, method in METHODS."""
    A, c, b, pr = problem(panel, split)
    kind, method = how.split("_", 1)
    if kind == "raw":
        g = b
    elif kind == "geo":
        g = np.exp(np.mean([np.log(np.maximum(problem(panel, s)[2], 1e-6))
                            for s in ("train", "validation", "private")], 0))
    elif kind == "grav":
        g = gravity_fit(pr, [b])
    elif kind == "grav3":
        g = gravity_fit(pr, [problem(panel, s)[2] for s in ("train", "validation", "private")])
    else:
        raise ValueError(kind)
    return METHODS[method](A, c, g)
