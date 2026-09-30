"""在已保存的 LightGBM + CNN(40ep) 预测上，再加一个 CNN(seed 2, 25ep)，评估 CNN 种子集成的融合效果。"""
import gc
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import t2_cnn as C
from .data import CACHE, load
from .t2 import T2_PANELS as T2P
from .t2_events import queue_truth

store = pickle.load(open(CACHE / "t2_blend_store.pkl", "rb"))
wins_all, wins_test = {}, {}
for p in T2P:
    w = pq.read_table(CACHE / "t2_ongoing_parts" / f"{p}.parquet", columns=["d", "T"]).to_pandas().drop_duplicates()
    wins_all[p] = w
    wins_test[p] = w.sample(min(len(w), 1500), random_state=0)
for f in (0, 1):
    data = {}
    rng = np.random.default_rng(10 + f)
    for p in T2P:
        X, Y, E = C.panel_windows(p, wins_all[p][wins_all[p].d % 2 != f])
        perm = rng.permutation(len(X))
        data[p] = (X[perm], Y[perm], E[perm])
    net = C.train_model(data, epochs=25, seed=2 + f)
    del data
    gc.collect()
    for p in T2P:
        wt = wins_test[p][wins_test[p].d % 2 == f]
        X, _, _ = C.panel_windows(p, wt)
        P = C.predict(net, X)
        for i, (d, T) in enumerate(zip(wt.d.to_numpy(), wt["T"].to_numpy())):
            store[(p, d, T)]["cnn2"] = P[i]
    print("fold", f, "done", flush=True)
pickle.dump(store, open(CACHE / "t2_blend_store2.pkl", "wb"))
rows = []
for p in T2P:
    z = load(p, "train")
    Q, vc = queue_truth(p, z=z)
    el = z["elig"] == 1
    cov = np.isfinite(z["speed"])
    for (pp, d, T), s in store.items():
        if pp != p or "cnn2" not in s:
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
        cand = s["lgb"] > 0
        cavg = 0.5 * (s["cnn"] + s["cnn2"])
        row = dict(panel=p)
        for name, c_ in (("cnn1", s["cnn"]), ("cnn2", s["cnn2"]), ("cnnavg", cavg)):
            row[name] = sc(c_ > 0.5)
            for w in (0.5, 0.6):
                pr = np.where(cand, w * c_ + (1 - w) * s["lgb"], c_)
                row[f"{name}_w{w}"] = sc(pr > 0.5)
        rows.append(row)
r = pd.DataFrame(rows)
pm = r.drop(columns="panel").groupby(r.panel).mean()
print("ENS2", pm.mean().round(4).to_dict(), len(r))
