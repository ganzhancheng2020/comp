"""T 行修复后的生产：用 t2_ongoing_parts_vis（去掉起点行、含当天早时段特征）全量训练 LightGBM（省内存），
输出 val/private ongoing 窗口的概率图到 cache/t2_final_maps_vis.pkl（键 "lgb"）。
推理输入与线上一致：window_history（T−60…T−5）+ 同日已发布 masked 层中 T−60 之前的早时段特征。"""
import gc
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import t2_ongoing as og
from .data import CACHE, REL, load, network
from .t2 import T2_PANELS as T2P

PARTS = CACHE / "t2_ongoing_parts_vis"
MAPS = CACHE / "t2_final_maps_vis.pkl"
SPLITS = ("validation", "private")
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)

if __name__ == "__main__":
    Xs, ys, feats = [], [], None
    for p in T2P:
        t = pq.read_table(PARTS / f"{p}.parquet").to_pandas()
        t["pid"] = np.int32(T2P.index(p))
        if feats is None:
            feats = [c for c in t.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")]
        Xs.append(t[feats].to_numpy(np.float32))
        ys.append(t.y.to_numpy())
        del t
        gc.collect()
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    gc.collect()
    m = lgb.train(PAR, lgb.Dataset(X, y, feature_name=feats, categorical_feature=["pid"], free_raw_data=True), 1000)
    del X, y
    gc.collect()
    m.save_model(str(CACHE / "t2_lgb_vis.txt"))
    bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
    maps = pickle.load(open(MAPS, "rb")) if MAPS.exists() else {"lgb": {}, "cnn": {}}
    maps["lgb"] = {}
    for p in T2P:
        net = network(p)
        li = {x: i for i, x in enumerate(net.link_id)}
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        bneck = sorted(bn[bn.panel == p].link)
        for s in SPLITS:
            zs = load(p, s)
            di = {d: i for i, d in enumerate(zs["dates"].tolist())}
            w = pd.read_csv(REL / "task2" / p / s / "window_index.csv")
            h = pd.read_parquet(REL / "task2" / p / s / "window_history.parquet")
            for r in w[w.condition == "queue_ongoing"].itertuples():
                T0 = pd.Timestamp(r.forecast_origin)
                T = T0.hour * 12 + T0.minute // 5
                hs, hf = og.history_arrays(h[h.window_id == r.window_id], T0, li, len(net))
                early = og.early_features(zs["m_speed"][di[T0.strftime("%Y-%m-%d")]], T, vcut)
                mp = np.zeros((6, len(net)), np.float32)
                Xw = og.cell_features(hs, hf, vcut, cap, T, T0.dayofweek, margin=12, bneck=bneck, early=early)
                if Xw is not None:
                    Xw["pid"] = T2P.index(p)
                    mp[Xw.k.to_numpy() - 1, Xw.link.to_numpy()] = m.predict(Xw[feats].to_numpy(np.float32))
                maps["lgb"][r.window_id] = mp
    pickle.dump(maps, open(MAPS, "wb"))
    print("LGB_VIS maps saved", len(maps["lgb"]), flush=True)
