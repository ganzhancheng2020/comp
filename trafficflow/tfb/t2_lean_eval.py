"""省内存的 ongoing LightGBM 大样本评估：逐走廊读取 parts、只保留本折训练行为 float32。用法: python -m tfb.t2_lean_eval <parts_dir>"""
import gc
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import CACHE, load
from .t2 import T2_PANELS as T2P
from .t2_events import queue_truth

PARTS = CACHE / (sys.argv[1] if len(sys.argv) > 1 else "t2_ongoing_parts")
TEST = CACHE / (sys.argv[2] if len(sys.argv) > 2 else PARTS.name)   # test rows may come from another frame
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
wins_test = {}
for p in T2P:
    w = pq.read_table(TEST / f"{p}.parquet", columns=["d", "T"]).to_pandas().drop_duplicates()
    wins_test[p] = w.sample(min(len(w), 1500), random_state=0)
maps = {}
for f in (0, 1):
    Xs, ys, feats = [], [], None
    for p in T2P:
        t = pq.read_table(PARTS / f"{p}.parquet").to_pandas()
        t["pid"] = np.int32(T2P.index(p))
        if feats is None:
            feats = [c for c in t.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")]
        m_ = t.d.to_numpy() % 2 != f
        Xs.append(t.loc[m_, feats].to_numpy(np.float32)); ys.append(t.y.to_numpy()[m_])
        del t; gc.collect()
    X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys; gc.collect()
    m = lgb.train(PAR, lgb.Dataset(X, y, feature_name=feats, categorical_feature=["pid"], free_raw_data=True), 1000)
    del X, y; gc.collect()
    for p in T2P:
        t = pq.read_table(TEST / f"{p}.parquet").to_pandas()
        t["pid"] = np.int32(T2P.index(p))
        for c in feats:
            if c not in t.columns:
                t[c] = np.nan
        g = t.merge(wins_test[p][wins_test[p].d % 2 == f], on=["d", "T"]); del t
        g = g.assign(pr=m.predict(g[feats].to_numpy(np.float32)))
        L = len(load(p, "validation")["links"])
        for (d, T), gg in g.groupby(["d", "T"]):
            mp = np.zeros((6, L), np.float32)
            mp[gg.k.to_numpy() - 1, gg.link.to_numpy()] = gg.pr.to_numpy()
            maps[(p, d, T)] = mp
    del m; gc.collect()
    print("fold", f, "done", flush=True)
import pickle  # noqa: E402

pickle.dump(maps, open(CACHE / f"t2_lean_maps_{PARTS.name}__{TEST.name}.pkl", "wb"))
rows = []
for p in T2P:
    z = load(p, "train"); Q, vc = queue_truth(p, z=z); el = z["elig"] == 1; cov = np.isfinite(z["speed"])
    for (pp, d, T), mp in maps.items():
        if pp != p or cov[d, T - 12:T + 1].mean() < 0.7:
            continue
        e = el[d, T + 1:T + 7]; tru = Q[d, T + 1:T + 7] & e
        if not tru.any():
            continue
        last = pd.DataFrame(z["speed"][d, T - 12:T] / vc).ffill().to_numpy()[-1] <= 1
        pers = np.repeat(last[None], 6, 0) & e
        # official-style persistence (build_task2_persistence_submission): last published row (T-5), eligible, no ffill
        v5 = z["speed"][d, T - 1]
        lo = (v5 <= vc) & (z["elig"][d, T - 1] == 1)
        pers_off = np.repeat(lo[None], 6, 0) & e

        def sc(pr):
            pr = pr & e; u = (pr | tru).sum(); return (pr & tru).sum() / u if u else 1.0
        rows.append(dict(panel=p, pers=sc(pers), pers_off=sc(pers_off), lgb_thr=sc(mp > 0.5)))
r = pd.DataFrame(rows)
for name, sel in (("ffill-filter", r.pers <= 0.9), ("official-filter", r.pers_off <= 0.9), ("no-filter", r.pers >= 0)):
    rr = r[sel]
    pm = rr.drop(columns="panel").groupby(rr.panel).mean()
    print("LEAN", name, "train", PARTS.name, "test", TEST.name, pm.mean().round(4).to_dict(), len(rr), flush=True)
