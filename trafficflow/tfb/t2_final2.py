"""v9 生产：LightGBM 概率图 + 多种子 CNN 概率图，保存到 cache，再以 0.5/0.5 融合（CNN 取种子均值）。"""
import gc
import pickle
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from . import t2_cnn as C
from . import t2_onset2 as o2
from . import t2_ongoing as og
from .data import CACHE, REL, network
from .t2 import T2_PANELS as T2P, predict
from .submit import OUT

SPLITS = ("validation", "private")
SEEDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3").split(",")]
MAPS = CACHE / ("t2_final_maps_early.pkl" if C.EARLY else "t2_final_maps.pkl")


def window_inputs(p, split):
    net = network(p)
    li = {x: i for i, x in enumerate(net.link_id)}
    w = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
    h = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet")
    return [(r.window_id, pd.Timestamp(r.forecast_origin),
             *og.history_arrays(h[h.window_id == r.window_id], pd.Timestamp(r.forecast_origin), li, len(net)))
            for r in w[w.condition == "queue_ongoing"].itertuples()]


maps = pickle.load(open(MAPS, "rb")) if MAPS.exists() else {"lgb": {}, "cnn": {}}
bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
inputs = {(p, s): window_inputs(p, s) for p in T2P for s in SPLITS}
if not maps["lgb"]:
    m, feats = og.fit_all()
    for p in T2P:
        net = network(p)
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        bneck = sorted(bn[bn.panel == p].link)
        for s in SPLITS:
            for wid, T0, hs, hf in inputs[(p, s)]:
                mp = np.zeros((6, len(net)), np.float32)
                X = og.cell_features(hs, hf, vcut, cap, T0.hour * 12 + T0.minute // 5, T0.dayofweek, margin=12, bneck=bneck)
                if X is not None:
                    X["pid"] = T2P.index(p)
                    mp[X.k.to_numpy() - 1, X.link.to_numpy()] = m.predict(X[feats].astype(np.float32))
                maps["lgb"][wid] = mp
    del m
    gc.collect()
    pickle.dump(maps, open(MAPS, "wb"))
    print("lgb maps saved", flush=True)


def cnn_maps(net_c, seed):
    out = {}
    for p in T2P:
        net = network(p)
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        bmask = np.zeros(len(net), np.float32)
        bmask[bn[bn.panel == p].link.unique()] = 1
        for s in SPLITS:
            if C.EARLY:
                from .data import load as _load
                zs = _load(p, s)
                di = {d: i for i, d in enumerate(zs["dates"].tolist())}
            for wid, T0, hs, hf in inputs[(p, s)]:
                T = T0.hour * 12 + T0.minute // 5
                early = None
                if C.EARLY:  # causal: masked layer of the same (published) day before T-60
                    early = C.early_channels(zs["m_speed"][di[T0.strftime("%Y-%m-%d")]], T, vcut)
                x = C.window_tensor(hs, hf, vcut, cap, T, T0.dayofweek, bmask, T2P.index(p), early)
                out[wid] = C.predict(net_c, x[None])[0]
    maps["cnn"][seed] = out
    pickle.dump(maps, open(MAPS, "wb"))
    print("cnn seed", seed, "maps saved", flush=True)


if 0 not in maps["cnn"] and (CACHE / "t2_cnn_full.pt").exists():
    n0 = C.Net()
    n0.load_state_dict(torch.load(CACHE / "t2_cnn_full.pt"))
    cnn_maps(n0, 0)
for sd in SEEDS:
    if sd in maps["cnn"]:
        continue
    data = {}
    rng = np.random.default_rng(100 + sd)
    for p in T2P:
        w = pq.read_table(CACHE / "t2_ongoing_parts" / f"{p}.parquet", columns=["d", "T"]).to_pandas().drop_duplicates()
        X, Y, E = C.panel_windows(p, w)
        perm = rng.permutation(len(X))
        data[p] = (X[perm], Y[perm], E[perm])
    net_c = C.train_model(data, epochs=25, seed=sd)
    del data
    gc.collect()
    torch.save(net_c.state_dict(), CACHE / f"t2_cnn_full_s{sd}.pt")
    cnn_maps(net_c, sd)
# combine
ongoing = {}
for wid, pl in maps["lgb"].items():
    pc = np.mean([maps["cnn"][sd][wid] for sd in maps["cnn"]], axis=0)
    ongoing[wid] = np.where(pl > 0, 0.5 * pc + 0.5 * pl, pc) > 0.5
q = pd.concat([predict(s, onset_windows=o2.predict_windows(s), ongoing_windows=ongoing) for s in SPLITS])
q.to_csv(OUT / "queue_models6.csv", index=False)
p5 = pd.read_csv(OUT / "queue_models5.csv")
print("n cnn seeds", len(maps["cnn"]), "agree with v8 queue %.4f" % (p5.queue_pred.values == q.queue_pred.values).mean(),
      p5.queue_pred.sum(), q.queue_pred.sum(), flush=True)
