"""LightGBM + CNN ongoing 融合的大样本评估（同一测试窗口、2 折按日奇偶）。"""
import gc
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import t2_cnn as C
from .data import CACHE, load
from .t2 import T2_PANELS as T2P
from .t2_events import queue_truth
from .t2_ongoing import decode

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
wins_all, wins_test = {}, {}
for p in T2P:
    w = pq.read_table(CACHE / "t2_ongoing_parts" / f"{p}.parquet", columns=["d", "T"]).to_pandas().drop_duplicates()
    wins_all[p] = w
    wins_test[p] = w.sample(min(len(w), 1500), random_state=0)
store = {}  # (panel, d, T) -> dict(lgb=map, cnn=map)
for f in (0, 1):
    t0 = time.time()
    # ---- LightGBM (memory-lean: per-panel read, keep only this fold's train rows as float32)
    Xs, ys, feats = [], [], None
    for p in T2P:
        t = pq.read_table(CACHE / "t2_ongoing_parts" / f"{p}.parquet").to_pandas()
        t["pid"] = np.int32(T2P.index(p))
        if feats is None:
            feats = [c for c in t.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")]
        m_ = t.d.to_numpy() % 2 != f
        Xs.append(t.loc[m_, feats].to_numpy(np.float32))
        ys.append(t.y.to_numpy()[m_])
        del t
        gc.collect()
    X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys
    ds = lgb.Dataset(X, y, feature_name=feats, categorical_feature=["pid"], free_raw_data=True)
    m = lgb.train(PAR, ds, 1000)
    del X, y, ds
    gc.collect()
    for p in T2P:
        t = pq.read_table(CACHE / "t2_ongoing_parts" / f"{p}.parquet").to_pandas()
        t["pid"] = np.int32(T2P.index(p))
        wt = wins_test[p][wins_test[p].d % 2 == f]
        g = t.merge(wt, on=["d", "T"])
        del t
        g = g.assign(pr=m.predict(g[feats].to_numpy(np.float32)))
        L = len(load(p, "validation")["links"])
        for (d, T), gg in g.groupby(["d", "T"]):
            mp = np.zeros((6, L), np.float32)
            mp[gg.k.to_numpy() - 1, gg.link.to_numpy()] = gg.pr.to_numpy()
            store[(p, d, T)] = {"lgb": mp}
    del m
    gc.collect()
    print("fold", f, "lgb done", round(time.time() - t0), "s", flush=True)
    # ---- CNN
    data = {}
    rng = np.random.default_rng(f)
    for p in T2P:
        X, Y, E = C.panel_windows(p, wins_all[p][wins_all[p].d % 2 != f])
        perm = rng.permutation(len(X))
        data[p] = (X[perm], Y[perm], E[perm])
    net = C.train_model(data, epochs=EPOCHS, seed=f)
    del data
    gc.collect()
    for p in T2P:
        wt = wins_test[p][wins_test[p].d % 2 == f]
        X, _, _ = C.panel_windows(p, wt)
        P = C.predict(net, X)
        for i, (d, T) in enumerate(zip(wt.d.to_numpy(), wt["T"].to_numpy())):
            store.setdefault((p, d, T), {"lgb": np.zeros_like(P[i])})["cnn"] = P[i]
    print("fold", f, "cnn done", round(time.time() - t0), "s", flush=True)
import pickle
pickle.dump(store, open(CACHE / "t2_blend_store.pkl", "wb"))
rows = []
for p in T2P:
    z = load(p, "train")
    Q, vc = queue_truth(p, z=z)
    el = z["elig"] == 1
    cov = np.isfinite(z["speed"])
    for (pp, d, T), s in store.items():
        if pp != p or "cnn" not in s:
            continue
        if cov[d, T - 12:T + 1].mean() < 0.7:
            continue
        e = el[d, T + 1:T + 7]
        tru = Q[d, T + 1:T + 7] & e
        if not tru.any():
            continue
        last = pd.DataFrame(z["speed"][d, T - 12:T + 1] / vc).ffill().to_numpy()[-1] <= 1
        pers = np.repeat(last[None], 6, 0) & e

        def sc(pred):
            pred = pred & e
            u = (pred | tru).sum()
            return (pred & tru).sum() / u if u else 1.0
        if sc(pers) > 0.9:
            continue
        row = dict(panel=p, pers=sc(pers))
        cand = s["lgb"] > 0
        for w in (0.0, 0.3, 0.5, 0.7, 1.0):
            pr = np.where(cand, w * s["cnn"] + (1 - w) * s["lgb"], s["cnn"] if w > 0 else 0.0)
            row[f"w{w}"] = sc(decode(pr.ravel()).reshape(pr.shape))
            row[f"w{w}_thr"] = sc(pr > 0.5)
        rows.append(row)
r = pd.DataFrame(rows)
pm = r.drop(columns="panel").groupby(r.panel).mean()
print(pm.round(3))
print("BLEND", pm.mean().round(4).to_dict(), len(r))
