"""生产：onset v2 + ongoing = 0.5·LightGBM + 0.5·CNN（阈值 0.5），输出 val/private 的 queue 文件。"""
import gc
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import t2_cnn as C
from . import t2_onset2 as o2
from . import t2_ongoing as og
from .data import CACHE, REL, network
from .t2 import T2_PANELS as T2P, predict
from .submit import OUT

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SPLITS = ("validation", "private")


def window_inputs(p, split):
    net = network(p)
    li = {x: i for i, x in enumerate(net.link_id)}
    w = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
    h = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet")
    out = []
    for r in w[w.condition == "queue_ongoing"].itertuples():
        T0 = pd.Timestamp(r.forecast_origin)
        hs, hf = og.history_arrays(h[h.window_id == r.window_id], T0, li, len(net))
        out.append((r.window_id, T0, hs, hf))
    return out


# ---- LightGBM probabilities on candidate cells
m, feats = og.fit_all()
bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
plgb = {}
for p in T2P:
    net = network(p)
    vcut = 0.6 * net.free_speed_kmh.to_numpy()
    cap = net.capacity_vph.to_numpy()
    bneck = sorted(bn[bn.panel == p].link)
    for s in SPLITS:
        for wid, T0, hs, hf in window_inputs(p, s):
            mp = np.zeros((6, len(net)), np.float32)
            X = og.cell_features(hs, hf, vcut, cap, T0.hour * 12 + T0.minute // 5, T0.dayofweek, margin=12, bneck=bneck)
            if X is not None:
                X["pid"] = T2P.index(p)
                mp[X.k.to_numpy() - 1, X.link.to_numpy()] = m.predict(X[feats].astype(np.float32))
            plgb[wid] = mp
del m
gc.collect()
print("lgb done", len(plgb), flush=True)
# ---- CNN on all train windows
data = {}
rng = np.random.default_rng(0)
for p in T2P:
    w = pq.read_table(CACHE / "t2_ongoing_parts" / f"{p}.parquet", columns=["d", "T"]).to_pandas().drop_duplicates()
    X, Y, E = C.panel_windows(p, w)
    perm = rng.permutation(len(X))
    data[p] = (X[perm], Y[perm], E[perm])
net_c = C.train_model(data, epochs=EPOCHS, seed=0)
del data
gc.collect()
import torch
torch.save(net_c.state_dict(), CACHE / "t2_cnn_full.pt")
ongoing = {}
for p in T2P:
    net = network(p)
    vcut = 0.6 * net.free_speed_kmh.to_numpy()
    cap = net.capacity_vph.to_numpy()
    bmask = np.zeros(len(net), np.float32)
    bmask[bn[bn.panel == p].link.unique()] = 1
    for s in SPLITS:
        for wid, T0, hs, hf in window_inputs(p, s):
            x = C.window_tensor(hs, hf, vcut, cap, T0.hour * 12 + T0.minute // 5, T0.dayofweek, bmask, T2P.index(p))
            pc = C.predict(net_c, x[None])[0]
            cand = plgb[wid] > 0
            pr = np.where(cand, 0.5 * pc + 0.5 * plgb[wid], pc)
            ongoing[wid] = pr > 0.5
print("cnn done", flush=True)
parts = [predict(s, onset_windows=o2.predict_windows(s), ongoing_windows=ongoing) for s in SPLITS]
q = pd.concat(parts)
q.to_csv(OUT / "queue_models5.csv", index=False)
p4 = pd.read_csv(OUT / "queue_models4.csv")
print("agree with v7c queue %.4f" % (p4.queue_pred.values == q.queue_pred.values).mean(), p4.queue_pred.sum(), q.queue_pred.sum())
