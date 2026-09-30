"""Task 1 baseline: per-link linear interpolation in time, spatial fallback."""
from __future__ import annotations

import numpy as np


def interp_axis1(a: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs along axis 1 of a (D, T, L) array, nearest at the edges.

    Rows with no observation at all stay NaN.
    """
    D, T, L = a.shape
    x = np.arange(T, dtype=np.float64)
    b = np.moveaxis(a, 1, 2).reshape(-1, T)          # (D*L, T)
    out = b.copy()
    m = ~np.isnan(b)
    # index of previous / next observation for every slot
    idx = np.where(m, np.arange(T)[None, :], -1)
    prev = np.maximum.accumulate(idx, axis=1)
    idx2 = np.where(m, np.arange(T)[None, :], T)
    nxt = np.minimum.accumulate(idx2[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(b.shape[0])[:, None]
    has_p, has_n = prev >= 0, nxt < T
    vp = b[rows, np.clip(prev, 0, T - 1)]
    vn = b[rows, np.clip(nxt, 0, T - 1)]
    w = np.where(has_p & has_n, (x[None, :] - prev) / np.maximum(nxt - prev, 1), 0.0)
    val = np.where(has_p & has_n, vp + w * (vn - vp), np.where(has_p, vp, vn))
    val = np.where(has_p | has_n, val, np.nan)
    out[~m] = val[~m]
    return np.moveaxis(out.reshape(D, L, T), 2, 1)


def reconstruct(m_speed: np.ndarray, m_flow: np.ndarray):
    s = interp_axis1(m_speed)
    f = interp_axis1(m_flow)
    s2 = np.swapaxes(interp_axis1(np.swapaxes(m_speed, 1, 2)), 1, 2)
    f2 = np.swapaxes(interp_axis1(np.swapaxes(m_flow, 1, 2)), 1, 2)
    s = np.where(np.isnan(s), s2, s)
    f = np.where(np.isnan(f), f2, f)
    return s, f
