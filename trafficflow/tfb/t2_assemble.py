"""组装 queue 文件：onset 取自给定 queue 文件，ongoing = w·LGB + (1-w)·CNN 种子均值，阈值 0.5。
用法: python -m tfb.t2_assemble <lgb_maps.pkl> <cnn_maps.pkl> <onset_from.csv> <out.csv> [w]"""
import pickle
import sys

import numpy as np
import pandas as pd

from .data import CACHE, REL
from .submit import OUT
from .t2 import predict

lgb_pkl, cnn_pkl, onset_from, out = sys.argv[1:5]
w = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
alpha = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0   # persistence mix: p = (1-a)·p_model + a·pers
L = pickle.load(open(CACHE / lgb_pkl, "rb"))["lgb"]
Cm = pickle.load(open(CACHE / cnn_pkl, "rb"))["cnn"]
pers = {}
if alpha > 0:
    from .data import network
    from .t2 import T2_PANELS
    from . import t2_ongoing as og
    for p in T2_PANELS:
        net = network(p)
        li = {x: i for i, x in enumerate(net.link_id)}
        vc = 0.6 * net.free_speed_kmh.to_numpy()
        for s in ("validation", "private"):
            wi = pd.read_csv(REL / "task2" / p / s / "window_index.csv")
            h = pd.read_parquet(REL / "task2" / p / s / "window_history.parquet")
            for r in wi[wi.condition == "queue_ongoing"].itertuples():
                hs, _ = og.history_arrays(h[h.window_id == r.window_id], pd.Timestamp(r.forecast_origin), li, len(net))
                last = pd.DataFrame(hs / vc).ffill().to_numpy()[-1] <= 1   # last visible state (T-5 or earlier)
                pers[r.window_id] = np.repeat(last[None], 6, 0).astype(np.float32)
ongoing = {}
for wid, pl in L.items():
    pc = np.mean([Cm[sd][wid] for sd in Cm], axis=0)
    pm = np.where(pl > 0, w * pl + (1 - w) * pc, pc)
    if alpha > 0:
        pm = (1 - alpha) * pm + alpha * pers[wid]
    ongoing[wid] = pm > 0.5
q = pd.concat([predict(s, ongoing_windows=ongoing) for s in ("validation", "private")], ignore_index=True)
src = pd.read_csv(OUT / onset_from)
assert (src.window_id.values == q.window_id.values).all() and (src.link_id.values == q.link_id.values).all()
idx = pd.concat([pd.read_csv(p) for p in REL.glob("task2/*/*/window_index.csv")])
ons = q.window_id.map(dict(zip(idx.window_id, idx.condition))) == "queue_onset"
q.loc[ons, "queue_pred"] = src.loc[ons, "queue_pred"].values
q.to_csv(OUT / out, index=False)
print("assembled", out, "cnn seeds", len(Cm), "onset cells", int(q.loc[ons, "queue_pred"].sum()),
      "ongoing cells", int(q.loc[~ons, "queue_pred"].sum()),
      "agree with onset source %.4f" % (q.queue_pred.values == src.queue_pred.values).mean())
