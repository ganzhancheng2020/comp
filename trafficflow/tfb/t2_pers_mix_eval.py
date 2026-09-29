"""本地：ongoing 融合概率与持续性（前向填充的最后可见状态）混合 p=(1-a)·p_model + a·pers 的 IoU 变化。
（使用 t2_blend_store2 的 LGB + 2×CNN 概率图；持续性按线上口径只用 T−60…T−5）"""
import pickle

import numpy as np
import pandas as pd

from .data import CACHE, load
from .t2 import T2_PANELS as T2P
from .t2_events import queue_truth

store = pickle.load(open(CACHE / "t2_blend_store2.pkl", "rb"))
AS = (0.0, 0.2, 0.3, 0.5)
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
        last = pd.DataFrame(z["speed"][d, T - 12:T] / vc).ffill().to_numpy()[-1] <= 1
        pers = np.repeat(last[None], 6, 0)

        def sc(pr):
            pr = pr & e
            u = (pr | tru).sum()
            return (pr & tru).sum() / u if u else 1.0
        if sc(pers) > 0.9:
            continue
        c = 0.5 * (s["cnn"] + s["cnn2"])
        pm = np.where(s["lgb"] > 0, 0.5 * c + 0.5 * s["lgb"], c)
        row = dict(panel=p)
        for a in AS:
            row[f"a{a}"] = sc((1 - a) * pm + a * pers > 0.5)
        rows.append(row)
r = pd.DataFrame(rows)
pm_ = r.drop(columns="panel").groupby(r.panel).mean()
print("PERSMIX", pm_.mean().round(4).to_dict(), len(r))
