"""Task 1 features, computed on dense (D, T, L) arrays of one panel and gathered at target cells."""
from __future__ import annotations

import numpy as np
import pandas as pd

import warnings

from .data import CACHE, REL, load, network

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


def ramp_link_flows(panel: str, z: dict, net: pd.DataFrame):
    """Sum of attached on-/off-ramp flows per mainline link, (D, T, L); NaN if any attached ramp is
    missing or below 75% observed, 0 where no ramp of that type is attached."""
    li = {x: i for i, x in enumerate(net.link_id)}
    rmap = pd.read_csv(REL / "corridors" / panel / "network" / "ramp_attachment_map.csv", dtype=str)
    rid = {x: i for i, x in enumerate(z["ramps"].tolist())}
    D, T = z["m_flow"].shape[:2]
    L = len(net)
    on = np.zeros((D, T, L), np.float32)
    off = np.zeros((D, T, L), np.float32)
    rf = np.where(z["ramp_pct"] >= 75, z["ramp_flow"], np.nan)
    for r in rmap.itertuples():
        if r.nearest_mainline_link_id not in li or r.ramp_link_id not in rid:
            continue
        tgt = on if r.ramp_type.upper() in ("OR", "ON") else off
        tgt[:, :, li[r.nearest_mainline_link_id]] += rf[:, :, rid[r.ramp_link_id]]
    return on, off


def plateau(panel: str) -> np.ndarray:
    """Per-link free-flow speed plateau: median of observed train speeds above 0.85 x the link median.
    The generator's free-flow branch is flat (speed = plateau + noise at any flow), so speed - plateau
    at a free-flowing cell is pure measurement noise."""
    path = CACHE / f"{panel}_plateau.npy"
    if path.exists():
        return np.load(path)
    v = load(panel, "train")["m_speed"]
    L = v.shape[2]
    med = np.nanmedian(v.reshape(-1, L), 0)
    out = np.array([np.nanmedian(v[:, :, l][v[:, :, l] > 0.85 * med[l]]) for l in range(L)], np.float32)
    np.save(path, out)
    return out


def _window_mean(a: np.ndarray, k: int) -> np.ndarray:
    """NaN-mean over links l-k..l+k (excluding l itself), along the last axis."""
    v = np.nan_to_num(a, nan=0.0)
    c = (~np.isnan(a)).astype(np.float32)
    pad = [(0, 0)] * (a.ndim - 1) + [(k + 1, k)]
    cv, cc = np.cumsum(np.pad(v, pad), -1), np.cumsum(np.pad(c, pad), -1)
    L = a.shape[-1]
    s = cv[..., 2 * k + 1:2 * k + 1 + L] - cv[..., :L] - v
    n = cc[..., 2 * k + 1:2 * k + 1 + L] - cc[..., :L] - c
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)


def common_factors(panel: str, z: dict, net: pd.DataFrame) -> dict[str, np.ndarray]:
    """Measurement noise shared by all links at the same 5-minute slot (corridor-wide common mode).

    Free-flow speed noise has a corridor-wide component (corr ~0.3 between links 10+ apart): at a masked
    cell it is estimated from the other links observed at the same slot. Flow uses second-difference
    residuals q_t - (q_{t-1} + q_{t+1}) / 2 per lane (white noise is not smoothable; a shared component is)."""
    pl = plateau(panel)
    v = z["m_speed"]
    ffm = v > 0.85 * pl
    e = np.where(ffm, v - pl, np.nan)
    er = np.where(ffm, v / pl - 1, np.nan)
    F = {"cf_sp": np.nanmean(e, 2), "cf_sp_rel": np.nanmean(er, 2), "cf_sp_n": np.isfinite(e).sum(2).astype(np.float32)}
    F = {k: np.broadcast_to(x[..., None], v.shape) for k, x in F.items()}
    F["cf_sp_loc"] = _window_mean(e, 5)
    q = z["m_flow"] / net.lanes.to_numpy(np.float32)
    r2 = np.full_like(q, np.nan)
    r2[:, 1:-1] = q[:, 1:-1] - (q[:, :-2] + q[:, 2:]) / 2
    F["cf_fl"] = np.broadcast_to(np.nanmean(r2, 2)[..., None], v.shape)
    F["cf_fl_loc"] = _window_mean(r2, 3)
    F["plat"] = np.broadcast_to(pl, v.shape)
    return F


CF = __import__("os").environ.get("TFB_CF", "0") == "1"
RAMP_RATIO = __import__("os").environ.get("TFB_RAMP_RATIO", "1") == "1"


def ramp_profile(panel: str) -> np.ndarray:
    """Mean train ramp flow per (weekend flag, slot, ramp)."""
    path = CACHE / f"{panel}_ramp_profile.npy"
    if path.exists():
        return np.load(path)
    z = load(panel, "train")
    we = pd.to_datetime(z["dates"]).dayofweek.to_numpy() >= 5
    rf = np.where(z["ramp_pct"] >= 75, z["ramp_flow"], np.nan)
    out = np.stack([np.nanmean(rf[~we], 0), np.nanmean(rf[we], 0)]).astype(np.float32)
    np.save(path, out)
    return out


def ramp_ratio_features(panel: str, z: dict, net: pd.DataFrame) -> dict[str, np.ndarray]:
    """Ramp flows relative to their time-of-day profile, as a congestion sensor.

    On-ramp inflow collapses (5-70% of its profile) while the mainline link it feeds is queued, and off-ramp flow rises
    ~10%. The ramp layer is published at every slot, including the 90-minute mainline blackout after each Task 2 origin,
    so it locates the queue inside that blackout (Task 1 is offline reconstruction: every published observation of the
    split is an input)."""
    prof = ramp_profile(panel)
    we = (pd.to_datetime(z["dates"]).dayofweek.to_numpy() >= 5).astype(int)
    rf = np.where(z["ramp_pct"] >= 75, z["ramp_flow"], np.nan)
    ratio = rf / np.maximum(prof[we], 1.0)                     # (D, T, n_ramps)
    rid = {x: i for i, x in enumerate(z["ramps"].tolist())}
    D, T = rf.shape[:2]
    L = len(net)
    out = {}
    for kind, col in (("on", "on_ramp_link_ids"), ("off", "off_ramp_link_ids")):
        num = np.zeros((D, T, L), np.float32)
        cnt = np.zeros((D, T, L), np.float32)
        for l, ids in enumerate(net[col].fillna("")):
            for x in str(ids).split(";"):
                if x in rid:
                    v = ratio[:, :, rid[x]]
                    ok = np.isfinite(v)
                    num[:, :, l] += np.where(ok, v, 0)
                    cnt[:, :, l] += ok
        r = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan).astype(np.float32)
        rt = r.copy()                                           # +-1 slot temporal mean (ramp loops miss ~20%)
        rt[:, 1:-1] = np.nanmean(np.stack([r[:, :-2], r[:, 1:-1], r[:, 2:]]), 0)
        out[f"r{kind}"] = r
        out[f"r{kind}_t"] = rt
        out[f"r{kind}_w2"] = _window_mean(rt, 2)
        out[f"r{kind}_w5"] = _window_mean(rt, 5)
    low = np.where(np.isfinite(out["ron_t"]), (out["ron_t"] < 0.6).astype(np.float32), np.nan)
    out["ron_low_w3"] = _window_mean(low, 3)
    out["ron_low_w8"] = _window_mean(low, 8)
    return out


def build(panel: str, z: dict, idx: tuple, prof: dict | None = None, ramps: bool = False) -> pd.DataFrame:
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
    if ramps and RAMP_RATIO:
        for k, a in ramp_ratio_features(panel, z, net).items():
            F[k] = a[d, t, l]
    if ramps:
        on, off = ramp_link_flows(panel, z, net)
        for k in (-1, 0, 1):
            F[f"ron{k}"] = shift_l(on, k)[d, t, l]
            F[f"roff{k}"] = shift_l(off, k)[d, t, l]
        fm = z["m_flow"]
        F["cons_up"] = (shift_l(fm, -1) + on - off)[d, t, l]
        F["cons_dn"] = (shift_l(fm, 1) - shift_l(on, 1) + shift_l(off, 1))[d, t, l]
    if CF:
        for k, a in common_factors(panel, z, net).items():
            F[k] = a[d, t, l]
    df = pd.DataFrame(F)
    # derived
    df["k_lin"] = df.flow_lin / df.speed_lin.clip(lower=1)
    df["k_prof"] = df.flow_prof / df.speed_prof.clip(lower=1)
    return df
