"""按起点排队规模分层的 ongoing 融合 IoU（大评估集，含 4 种阈值）。
验证/私有集的 ongoing 窗口排队规模约为 train 组织方窗口的 60%，看小排队窗口的表现与阈值。"""
import pickle

import numpy as np
import pandas as pd

from .data import CACHE, load
from .t2 import T2_PANELS as T2P
from .t2_events import queue_truth

store = pickle.load(open(CACHE / "t2_blend_store2.pkl", "rb"))
rows = []
for p in T2P:
    z = load(p, "train")
    Q, vc = queue_truth(p, z=z)
    el = z["elig"] == 1
    cov = np.isfinite(z["speed"])
    for (pp, d, T), s in store.items():
        if pp != p or "cnn2" not in s or cov[d, T - 12:T + 1].mean() < 0.7:
            continue
        e = el[d, T + 1:T + 7]
        tru = Q[d, T + 1:T + 7] & e
        if not tru.any():
            continue
        ff = pd.DataFrame(z["speed"][d, T - 12:T + 1] / vc).ffill().to_numpy()[-1] <= 1
        pers = np.repeat(ff[None], 6, 0) & e

        def sc(pr):
            pr = pr & e
            u = (pr | tru).sum()
            return (pr & tru).sum() / u if u else 1.0
        if sc(pers) > 0.9:
            continue
        cand = s["lgb"] > 0
        c = 0.5 * (s["cnn"] + s["cnn2"])
        pr = np.where(cand, 0.5 * c + 0.5 * s["lgb"], c)
        row = dict(panel=p, qff=int(ff.sum()), ntrue=int(tru.sum()))
        for t in (0.3, 0.4, 0.5, 0.6):
            row[f"t{t}"] = sc(pr > t)
        rows.append(row)
r = pd.DataFrame(rows)
r["bucket"] = pd.cut(r.qff, [-1, 5, 10, 20, 40, 1000], labels=["0-5", "6-10", "11-20", "21-40", ">40"])
cols = ["t0.3", "t0.4", "t0.5", "t0.6"]
print(r.groupby("bucket", observed=True)[cols].mean().round(4).assign(n=r.groupby("bucket", observed=True).size()))
print("overall", r[cols].mean().round(4).to_dict(), "mean qff", round(r.qff.mean(), 1))
r.to_parquet(CACHE / "t2_blend_rows_bucket.parquet")
