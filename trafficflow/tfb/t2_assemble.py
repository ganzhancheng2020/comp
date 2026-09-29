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
L = pickle.load(open(CACHE / lgb_pkl, "rb"))["lgb"]
Cm = pickle.load(open(CACHE / cnn_pkl, "rb"))["cnn"]
ongoing = {}
for wid, pl in L.items():
    pc = np.mean([Cm[sd][wid] for sd in Cm], axis=0)
    ongoing[wid] = np.where(pl > 0, w * pl + (1 - w) * pc, pc) > 0.5
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
