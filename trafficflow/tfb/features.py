"""Task 1 features, computed on dense (D, T, L) arrays of one panel and gathered at target cells."""
from __future__ import annotations

import numpy as np
import pandas as pd

import warnings

from .data import CACHE, load, network

warnings.filterwarnings("ignore", category=RuntimeWarning)

CH = ("speed", "flow", "occ")


def prev_next(a: np.ndarray, axis: int = 1):
    """For NaN-holed a, return (lin, v_prev, v_next, d_prev, d_next) along `axis`.

    d_* are distances in slots (inf when absent); at observed cells lin == a and d == 0.
    """
    a = np.moveaxis(a, axis, -1)
    shp = a.shape
    T = shp[-1]
    b = a.reshape(-1, T)
    m = ~np.isnan(b)
    ar = np.arange(T)
    prev = np.maximum.accumulate(np.where(m, ar, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(m, ar, T)[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(b.shape[0])[:, None]
    hp, hn = prev >= 0, nxt < T
    vp = np.where(hp, b[rows, np.clip(prev, 0, T - 1)], np.nan)
    vn = np.where(hn, b[rows, np.clip(nxt, 0, T - 1)], np.nan)
    dp = np.where(hp, ar - prev, np.inf).astype(np.float32)
    dn = np.where(hn, nxt - ar, np.inf).astype(np.float32)
    w = np.where(hp & hn, dp / np.maximum(dp + dn, 1), 0.0)
    lin = np.where(hp & hn, vp + w * (vn - vp), np.where(hp, vp, vn))
    res = [x.reshape(shp) for x in (lin, vp, vn, dp, dn)]
    return [np.moveaxis(x, -1, axis).astype(np.float32) for x in res]


def shift_l(a: np.ndarray, k: int) -> np.ndarray:
    """Value at link l+k (NaN outside the corridor)."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[..., :-k] = a[..., k:]
    elif k < 0:
        out[..., -k:] = a[..., :k]
    else:
        out[:] = a
    return out


def profile(panel: str) -> dict[str, np.ndarray]:
    """Mean observed value per (weekend flag, slot, link) over the train masked layer."""
    path = CACHE / f"{panel}_profile.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    z = load(panel, "train")
    we = pd.to_datetime(z["dates"]).dayofweek.to_numpy() >= 5
    out = {}
    for c, k in (("speed", "m_speed"), ("flow", "m_flow"), ("occ", "m_occ")):
        a = z[k]
        out[c] = np.stack([np.nanmean(a[~we], 0), np.nanmean(a[we], 0)]).astype(np.float32)
    np.savez(path, **out)
    return out


def build(panel: str, z: dict, idx: tuple, prof: dict | None = None) -> pd.DataFrame:
    """Features for cells idx = (d, t, l) arrays. z holds the masked arrays of the split."""
    d, t, l = idx
    net = network(panel)
    prof = prof if prof is not None else profile(panel)
    dates = pd.to_datetime(z["dates"])
    dow = dates.dayofweek.to_numpy()
    we = (dow >= 5).astype(int)
    F: dict[str, np.ndarray] = {}
    L = len(net)
    F["tod"] = t.astype(np.float32)
    F["dow"] = dow[d]
    F["lpos"] = (l / max(L - 1, 1)).astype(np.float32)
    F["lanes"] = net.lanes.to_numpy(np.float32)[l]
    F["vfree"] = net.free_speed_kmh.to_numpy(np.float32)[l]
    F["cap"] = net.capacity_vph.to_numpy(np.float32)[l]
    F["len"] = net.length_km.to_numpy(np.float32)[l]
    F["pct"] = z["pct"][d, t, l]
    F["rate"] = pd.Series(z["regime"]).map({"R1": .2, "R2": .3, "R3": .5}).to_numpy(np.float32)[d]
    for k in (-1, 1):
        F[f"pct_n{k}"] = shift_l(z["pct"], k)[d, t, l]
    for c in CH:
        a = z[f"m_{c}"]
        lin, vp, vn, dp, dn = prev_next(a, 1)
        F[f"{c}_lin"], F[f"{c}_vp"], F[f"{c}_vn"] = lin[d, t, l], vp[d, t, l], vn[d, t, l]
        if c == "speed":
            F["dp"], F["dn"] = dp[d, t, l], dn[d, t, l]
        F[f"{c}_prof"] = prof[c][we[d], t, l]
        # spatial linear interpolation at the same slot
        slin, _, _, sdp, sdn = prev_next(a, 2)
        F[f"{c}_slin"] = slin[d, t, l]
        if c == "speed":
            F["sdp"], F["sdn"] = sdp[d, t, l], sdn[d, t, l]
        for k in (-2, -1, 1, 2):
            nb = shift_l(a, k)
            F[f"{c}_n{k}"] = nb[d, t, l]
            F[f"{c}_nlin{k}"] = shift_l(lin, k)[d, t, l]
            if abs(k) == 1:
                # difference to the neighbour, interpolated in time, added back to the neighbour's value now
                diff_lin = prev_next(a - nb, 1)[0]
                F[f"{c}_dest{k}"] = (nb + diff_lin)[d, t, l]
                F[f"{c}_dnlin{k}"] = (shift_l(lin, k) + diff_lin)[d, t, l]
        # local level relative to the profile over the day so far/after (+-1h), from observed cells
        rel = a - prof[c][we]
        csum = np.nancumsum(np.nan_to_num(rel, nan=0.0), axis=1)
        ccnt = np.cumsum(~np.isnan(rel), axis=1)
        lo, hi = np.clip(t - 12, 0, 287), np.clip(t + 12, 0, 287)
        num = csum[d, hi, l] - csum[d, lo, l]
        den = ccnt[d, hi, l] - ccnt[d, lo, l]
        F[f"{c}_rel1h"] = np.where(den > 0, num / np.maximum(den, 1), np.nan).astype(np.float32)
    df = pd.DataFrame(F)
    # derived
    df["k_lin"] = df.flow_lin / df.speed_lin.clip(lower=1)
    df["k_prof"] = df.flow_prof / df.speed_prof.clip(lower=1)
    return df
