"""Task 2 onset: per-link probability that the queue appears at T+30, then expected-IoU decoding."""
from __future__ import annotations

import numpy as np
import pandas as pd

import warnings

from .data import CACHE, REL, load, network

warnings.filterwarnings("ignore", category=RuntimeWarning)
from .t2_events import onset_events, queue_truth

HIST = 13  # slots T-60 .. T


def candidates(events, min_count=3):
    from collections import Counter
    c = Counter(x for e in events for x in e[2].tolist())
    return sorted(k for k, v in c.items() if v >= min_count)


def link_features(hist_speed, hist_flow, vcut, cap, T, links, dow):
    """hist_*: (13, L) visible history. One row per candidate link."""
    L = hist_speed.shape[1]
    r = hist_speed / vcut[None, :]
    rf = pd.DataFrame(r).ffill().bfill().to_numpy()
    ff = pd.DataFrame(hist_flow).ffill().bfill().to_numpy() / cap[None, :]
    rows = []
    for l in links:
        lo, hi = max(0, l - 3), min(L, l + 4)
        f = {"link": l, "tod": T, "dow": dow,
             "r0": rf[-1, l], "r1": rf[-2, l], "r3": rf[-4, l], "r6": rf[-7, l], "r12": rf[0, l],
             "rmin3": np.nanmin(rf[-3:, l]), "rtrend": rf[-1, l] - rf[-7, l],
             "q0": ff[-1, l], "q3": ff[-4, l], "qmean": np.nanmean(ff[:, l]),
             "nb_rmin": np.nanmin(rf[-3:, lo:hi]), "nb_rmean": np.nanmean(rf[-1, lo:hi]),
             "nb_q": np.nanmean(ff[-1, lo:hi]),
             "dn_r": rf[-1, min(L - 1, l + 1)], "up_r": rf[-1, max(0, l - 1)],
             "cor_rmin": np.nanmin(rf[-1]), "cor_q": np.nanmean(ff[-1])}
        rows.append(f)
    return pd.DataFrame(rows)


def train_frame(panel: str):
    z = load(panel, "train")
    Q, vcut = queue_truth(panel, z=z)
    net = network(panel)
    cap = net.capacity_vph.to_numpy()
    ev = onset_events(Q)
    cand = candidates(ev)
    dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
    parts = []
    for d, s, ls in ev:
        T = s - 6
        if T - 12 < 0:
            continue
        hs, hf = z["speed"][d, T - 12:T + 1], z["flow"][d, T - 12:T + 1]
        X = link_features(hs, hf, vcut, cap, T, cand, dow[d])
        X["y"] = X.link.isin(ls).astype(int)
        X["d"], X["s"], X["panel"] = d, s, panel
        X["n_true"] = len(ls)
        parts.append(X)
    return pd.concat(parts, ignore_index=True), cand


def decode(p: np.ndarray, links: np.ndarray, n_mc: int = 400, rng=None) -> list[int]:
    """Choose the top-k links (by probability) maximising Monte-Carlo expected IoU."""
    rng = rng or np.random.default_rng(0)
    order = np.argsort(-p)
    samp = rng.random((n_mc, len(p))) < p[None, :]
    best, best_k = -1.0, 1
    for k in range(1, len(p) + 1):
        S = np.zeros(len(p), bool)
        S[order[:k]] = True
        inter = (samp & S).sum(1)
        union = (samp | S).sum(1)
        v = np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0))
        if v > best:
            best, best_k = v, k
    return links[order[:best_k]].tolist()


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.8, verbose=-1, num_threads=2)
T2P = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]


def fit_all():
    import lightgbm as lgb
    path = CACHE / "t2_onset.parquet"
    df = pd.read_parquet(path) if path.exists() else pd.concat([train_frame(p)[0] for p in T2P], ignore_index=True)
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    feats = [c for c in df.columns if c not in ("y", "d", "s", "panel", "n_true")]
    m = lgb.train(PARAMS, lgb.Dataset(df[feats], df.y, categorical_feature=["pid"]), 400)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    return m, feats, cands


def predict_windows(split: str, model=None):
    """{window_id: list of link indices queued at T+30} for the onset windows of a split."""
    m, feats, cands = model or fit_all()
    out = {}
    for p in T2P:
        net = network(p)
        li = {x: i for i, x in enumerate(net.link_id)}
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        w = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
        h = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet")
        for r in w[w.condition == "queue_onset"].itertuples():
            g = h[h.window_id == r.window_id]
            T0 = pd.Timestamp(r.forecast_origin)
            k = ((T0 - pd.to_datetime(g.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
            hs = np.full((13, len(net)), np.nan)
            hf = np.full((13, len(net)), np.nan)
            ll = g.link_id.map(li).to_numpy()
            hs[12 - k, ll] = g.speed_kmh.to_numpy()
            hf[12 - k, ll] = g.flow_vph.to_numpy()
            T = T0.hour * 12 + T0.minute // 5
            X = link_features(hs, hf, vcut, cap, T, cands[p], T0.dayofweek)
            X["pid"] = T2P.index(p)
            prob = m.predict(X[feats])
            out[r.window_id] = decode(prob, X.link.to_numpy())
    return out
