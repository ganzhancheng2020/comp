"""Ongoing with kinematic-wave (LWR) tail features: for the queue run nearest to each cell, the shock speed at its tail
w = (q_up - q_queue) / (k_up - k_queue), k = q/v (last visible row, T-5), and the cell's distance in km to the tail
extrapolated k steps ahead. Paired comparison without/with these features: in scenario (train 2-fold, masked-layer
input) and out of scenario (val/private masked-layer windows, evaluation only). 30% window subsample of
t2_ongoing_parts_vis, like t2_ongoing_idshift. Usage: python -m tfb.t2_ongoing_lwr
"""
from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import t2_ongoing as og
from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_events import visible
from .t2_ongoing_idshift import PAR, frame
from .t2_ongoing_shift import iou, mined_windows

LWR = ["lwr_w", "lwr_dtail", "lwr_in"]


def lwr_arrays(hs, hf, vcut, length):
    """Per link: shock speed of the nearest queue run's tail, that tail's position (km), the run's head index."""
    L = hs.shape[1]
    v = pd.DataFrame(hs[:12]).ffill().to_numpy()[-1]
    q = pd.DataFrame(hf[:12]).ffill().to_numpy()[-1]
    k = q / np.clip(v, 1.0, None)
    qm = v <= vcut
    c = np.cumsum(length) - length / 2
    w = np.full(L, np.nan)
    xt = np.full(L, np.nan)
    hd = np.full(L, -1)
    runs, a = [], None
    for l in range(L + 1):
        if l < L and qm[l] and a is None:
            a = l
        elif (l == L or not qm[l]) and a is not None:
            runs.append((a, l - 1))
            a = None
    if not runs:
        return w, xt, hd, c
    rw = []
    for a, b in runs:
        up = [j for j in range(max(0, a - 3), a) if np.isfinite(k[j])]
        qu = [j for j in range(a, min(b, a + 2) + 1) if np.isfinite(k[j])]
        ws = np.nan
        if up and qu:
            dk = np.mean(k[up]) - np.mean(k[qu])
            if abs(dk) > 1e-3:
                ws = np.clip((np.mean(q[up]) - np.mean(q[qu])) / dk, -30, 30)
        rw.append(ws)
    mids = np.array([(a + b) / 2 for a, b in runs])
    for l in range(L):
        i = int(np.argmin(np.abs(mids - l)))
        w[l], xt[l], hd[l] = rw[i], c[runs[i][0]], runs[i][1]
    return w, xt, hd, c


def add_lwr(X, hs, hf, vcut, length):
    w, xt, hd, c = lwr_arrays(hs, hf, vcut, length)
    lk, kk = X.link.to_numpy(), X.k.to_numpy()
    xt_k = xt[lk] + w[lk] * kk / 12.0
    X = X.assign(lwr_w=w[lk], lwr_dtail=c[lk] - xt_k)
    return X.assign(lwr_in=((X.lwr_dtail.to_numpy() >= 0) & (lk <= hd[lk])).astype(np.float32))


def train_frame_lwr(df):
    parts = []
    for p in T2P:
        net = network(p)
        vc = 0.6 * net.free_speed_kmh.to_numpy()
        ln = net.length_km.to_numpy(dtype=float)
        z = load(p, "train")
        g = df[df.panel == p] if "panel" in df else df[df.pid == T2P.index(p)]
        for (d, T), gg in g.groupby(["d", "T"]):
            parts.append(add_lwr(gg, visible(z["speed"][d], T), visible(z["flow"][d], T), vc, ln))
        print("lwr", p, flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    df = train_frame_lwr(frame())
    gc.collect()
    base = [c for c in df.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out") + tuple(LWR)]
    variants = {"full": base, "lwr": base + LWR}
    models = {}
    for name, feats in variants.items():
        fit = lambda m: lgb.train(PAR, lgb.Dataset(df.loc[m, feats].to_numpy(np.float32), df.y.to_numpy()[m],
                                                   feature_name=feats, categorical_feature=["pid"]), 500)
        dd = df.d.to_numpy()
        models[name] = (feats, {0: fit(dd % 2 != 0), 1: fit(dd % 2 != 1), "all": fit(np.ones(len(df), bool))})
        print("fitted", name, flush=True)
    del df
    gc.collect()
    bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
    rows = []
    rng = np.random.default_rng(0)
    for p in T2P:
        net = network(p)
        vc = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        ln = net.length_km.to_numpy(dtype=float)
        bneck = sorted(bn[bn.panel == p].link)
        for s in ("train", "validation", "private"):
            z = load(p, s)
            dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
            for d, T, tru, e, pers in mined_windows(p, s, z, rng):
                hs, hf = visible(z["m_speed"][d], T), visible(z["m_flow"][d], T)
                X = og.cell_features(hs, hf, vc, cap, T, dow[d], margin=12, bneck=bneck,
                                     early=og.early_features(z["m_speed"][d], T, vc))
                r = dict(panel=p, split=s, pers=iou(pers, tru, e))
                if X is not None:
                    X = add_lwr(X.assign(pid=T2P.index(p)), hs, hf, vc, ln)
                for name, (feats, ms) in models.items():
                    mp = np.zeros_like(tru, dtype=np.float32)
                    if X is not None:
                        m = ms[d % 2] if s == "train" else ms["all"]
                        mp[X.k.to_numpy() - 1, X.link.to_numpy()] = m.predict(X[feats].to_numpy(np.float32))
                    r[name] = iou(mp > 0.5, tru, e)
                rows.append(r)
            print(p, s, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_ongoing_lwr.csv", index=False)
    print(r.groupby(["split", "panel"])[["pers", "full", "lwr"]].mean().groupby("split").mean().T.round(4))
    r["dd"] = r.lwr - r.full
    print(r.groupby(["panel", "split"]).dd.mean().unstack().round(4).to_string())
    o = r[r.split != "train"]
    print("lwr - full: oos event-level %+.4f se %.4f" % (o.dd.mean(), o.dd.std() / np.sqrt(len(o))),
          r.groupby("split").dd.mean().round(4).to_dict())


if __name__ == "__main__":
    main()
