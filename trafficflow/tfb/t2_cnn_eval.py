"""CNN ongoing 的大样本评估（与 run_ongoing 相同的测试窗口、筛选与 IoU 规则），2 折按日奇偶。"""
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import t2_cnn as C
from .data import CACHE, load
from .t2 import T2_PANELS as T2P
from .t2_events import queue_truth
from .t2_ongoing import decode

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
wins_all, wins_test = {}, {}
for p in T2P:
    w = pq.read_table(CACHE / "t2_ongoing_parts" / f"{p}.parquet", columns=["d", "T"]).to_pandas().drop_duplicates()
    wins_all[p] = w
    wins_test[p] = w.sample(min(len(w), 1500), random_state=0)
rows = []
for f in (0, 1):
    t0 = time.time()
    data = {}
    rng = np.random.default_rng(f)
    for p in T2P:
        w = wins_all[p][wins_all[p].d % 2 != f]
        X, Y, E = C.panel_windows(p, w)
        perm = rng.permutation(len(X))
        data[p] = (X[perm], Y[perm], E[perm])
    print("fold", f, "data", round(time.time() - t0), "s", flush=True)
    net = C.train_model(data, epochs=EPOCHS, seed=f)
    del data
    print("fold", f, "trained", round(time.time() - t0), "s", flush=True)
    for p in T2P:
        z = load(p, "train")
        Q, vc = queue_truth(p, z=z)
        el = z["elig"] == 1
        cov = np.isfinite(z["speed"])
        w = wins_test[p][wins_test[p].d % 2 == f]
        X, Y, E = C.panel_windows(p, w)
        P = C.predict(net, X)
        for i, (d, T) in enumerate(zip(w.d.to_numpy(), w["T"].to_numpy())):
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
            pr = P[i]
            rows.append(dict(panel=p, pers=sc(pers), cnn=sc(decode(pr.ravel()).reshape(pr.shape)), cnn_thr=sc(pr > 0.5)))
r = pd.DataFrame(rows)
pm = r.drop(columns="panel").groupby(r.panel).mean()
print(pm.round(3))
print("CNN ongoing", pm.mean().round(4).to_dict(), len(r))
r.to_parquet(CACHE / "t2_cnn_eval_rows.parquet")
