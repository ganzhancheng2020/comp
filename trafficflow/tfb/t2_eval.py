"""本地 T2 评估器：用组织方在 train 中选出的真实窗口（每走廊 5 onset + 5 ongoing），
真值 = train 观测速度 <= 0.6*v_free（时间插值补缺），只计 eligible 格，聚合与 score_task2.py 一致：
window -> (panel, condition) -> panel -> family -> overall，全部等权。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import REL, families, load, network
from .t2_events import queue_truth

T2P = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]


def windows(panel: str):
    """[(window_id, condition, day, T, truth (6,L) bool, eligible (6,L) bool, hist_speed, hist_flow)]"""
    z = load(panel, "train")
    Q, vcut = queue_truth(panel, z=z)
    di = {d: i for i, d in enumerate(z["dates"])}
    w = pd.read_csv(REL / "task2" / panel / "train" / "window_index.csv")
    h = pd.read_parquet(REL / "task2" / panel / "train" / "window_history.parquet")
    net = network(panel)
    li = {x: i for i, x in enumerate(net.link_id)}
    out = []
    for r in w.itertuples():
        d = di[r.date]
        T0 = pd.Timestamp(r.forecast_origin)
        T = T0.hour * 12 + T0.minute // 5
        g = h[h.window_id == r.window_id]
        k = ((T0 - pd.to_datetime(g.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
        hs = np.full((13, len(net)), np.nan)
        hf = np.full((13, len(net)), np.nan)
        hs[12 - k, g.link_id.map(li).to_numpy()] = g.speed_kmh.to_numpy()
        hf[12 - k, g.link_id.map(li).to_numpy()] = g.flow_vph.to_numpy()
        el = z["elig"][d, T + 1:T + 7] == 1
        out.append(dict(window_id=r.window_id, condition=r.condition, d=d, T=T, T0=T0,
                        truth=Q[d, T + 1:T + 7], elig=el, hs=hs, hf=hf, vcut=vcut))
    return out


def iou(pred, truth, elig):
    p, t = pred[elig], truth[elig]
    u = (p | t).sum()
    return 1.0 if u == 0 else (p & t).sum() / u


def aggregate(rows: pd.DataFrame, col: str = "iou") -> dict:
    fam = families()
    pc = rows.groupby(["panel", "condition"])[col].mean().reset_index()
    pm = pc.groupby("panel")[col].mean()
    by_cond = pc.groupby("condition")[col].mean().to_dict()
    total = pm.groupby(pm.index.map(fam)).mean().mean()
    return {"S_queue": round(float(total), 4), **{k: round(v, 4) for k, v in by_cond.items()}}


def eval_models(verbose=True, onset_rounds=400, ongoing_rounds=600, ongoing_params=None, decode_fn=None):
    """2 折（按日奇偶）训练 onset / ongoing 模型，在组织方 train 窗口上打分。"""
    import lightgbm as lgb
    from . import t2_onset as on, t2_ongoing as og
    from .data import CACHE
    from .t2 import static_sets
    S = static_sets()
    don = pd.read_parquet(CACHE / "t2_onset.parquet")
    don["pid"] = don.panel.map({p: i for i, p in enumerate(T2P)})
    fon = [c for c in don.columns if c not in ("y", "d", "s", "panel", "n_true")]
    dog = pd.read_parquet(CACHE / "t2_ongoing.parquet")
    dog["pid"] = dog.panel.map({p: i for i, p in enumerate(T2P)})
    fog = [c for c in dog.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")]
    par = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2)
    par.update(ongoing_params or {})
    mon, mog = {}, {}
    for f in (0, 1):
        a = don[don.d % 2 != f]
        mon[f] = lgb.train(on.PARAMS, lgb.Dataset(a[fon], a.y, categorical_feature=["pid"]), onset_rounds)
        b = dog[dog.d % 2 != f]
        mog[f] = lgb.train(par, lgb.Dataset(b[fog], b.y, categorical_feature=["pid"]), ongoing_rounds)
    cands = {p: sorted(don[don.panel == p].link.unique().tolist()) for p in T2P}
    rows = []
    for p in T2P:
        net = network(p)
        cap = net.capacity_vph.to_numpy()
        for w in windows(p):
            f = w["d"] % 2
            last = pd.DataFrame(w["hs"]).ffill().to_numpy()[-1] <= w["vcut"]
            pers = np.repeat(last[None], 6, 0)
            model = np.zeros_like(pers)
            if w["condition"] == "queue_onset":
                st = np.zeros_like(pers)
                st[5, S[p]] = True
                X = on.link_features(w["hs"], w["hf"], w["vcut"], cap, w["T"], cands[p], w["T0"].dayofweek)
                X["pid"] = T2P.index(p)
                model[5, on.decode(mon[f].predict(X[fon]), X.link.to_numpy())] = True
            else:
                st = pers
                X = og.cell_features(w["hs"], w["hf"], w["vcut"], cap, w["T"], w["T0"].dayofweek)
                if X is not None:
                    X["pid"] = T2P.index(p)
                    pr = mog[f].predict(X[fog])
                    sel = (decode_fn or og.decode)(pr)
                    model[X.k.to_numpy()[sel] - 1, X.link.to_numpy()[sel]] = True
            rows.append(dict(panel=p, condition=w["condition"], window_id=w["window_id"],
                             pers=iou(pers, w["truth"], w["elig"]), static=iou(st, w["truth"], w["elig"]),
                             model=iou(model, w["truth"], w["elig"])))
    r = pd.DataFrame(rows)
    if verbose:
        for c in ("pers", "static", "model"):
            print(c, aggregate(r, c), flush=True)
    return r
