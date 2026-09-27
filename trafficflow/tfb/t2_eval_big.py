"""大样本、与线上对齐的 onset 评估：挖掘的全部 onset 事件 + 选择器约束（horizon 内有 eligible 排队格、
历史覆盖率 >= 0.7），IoU 只算 eligible 格（T+30 这一步）。模型按日奇偶 2 折。"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import t2_onset as on
from .data import CACHE, load
from .t2 import T2_PANELS as T2P, static_sets


def event_table():
    rows = []
    for p in T2P:
        z = load(p, "train")
        el = z["elig"] == 1
        cov = np.isfinite(z["speed"])
        df = pd.read_parquet(CACHE / "t2_onset.parquet")
        df = df[df.panel == p]
        for (d, s), g in df.groupby(["d", "s"]):
            T = s - 6
            c = cov[d, T - 12:T + 1].mean()
            e = el[d, s, g.link.to_numpy()]
            rows.append(dict(panel=p, d=d, s=s, hcov=c, elig_links=g.link.to_numpy()[e],
                             true=set(g.link.to_numpy()[(g.y.to_numpy() == 1) & e])))
    ev = pd.DataFrame(rows)
    return ev[(ev.hcov >= 0.7) & (ev.true.map(len) > 0)]


def iou_elig(pred: set, true: set, elig) -> float:
    pe = pred & set(elig.tolist())
    u = len(pe | true)
    return len(pe & true) / u if u else 1.0


def run(params=None, rounds=400, decode=None, feat_drop=()):
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    feats = [c for c in df.columns if c not in ("y", "d", "s", "panel", "n_true") and c not in feat_drop]
    ev = event_table()
    S = static_sets()
    par = dict(on.PARAMS, **(params or {}))
    out = []
    for f in (0, 1):
        tr = df[df.d % 2 != f]
        m = lgb.train(par, lgb.Dataset(tr[feats], tr.y, categorical_feature=["pid"]), rounds)
        te = df[df.d % 2 == f].copy()
        te["p"] = m.predict(te[feats])
        grp = dict(tuple(te.groupby(["panel", "d", "s"])))
        for r in ev[ev.d % 2 == f].itertuples():
            g = grp[(r.panel, r.d, r.s)]
            sel = set((decode or on.decode)(g.p.to_numpy(), g.link.to_numpy()))
            out.append(dict(panel=r.panel, static=iou_elig(set(S[r.panel]), r.true, r.elig_links),
                            model=iou_elig(sel, r.true, r.elig_links)))
    res = pd.DataFrame(out)
    pm = res.groupby("panel")[["static", "model"]].mean()
    return pm, pm.mean()


if __name__ == "__main__":
    pm, m = run()
    print(pm.round(3)); print(m.round(4).to_dict())


def run_ongoing(params=None, rounds=600, decode=None, feat_drop=(), max_windows=None, train_filter=None):
    """挖掘的 ongoing 窗口（t2_ongoing.parquet）+ 选择器约束：历史覆盖率 >= 0.7、horizon 有 eligible 排队格、
    持续性 IoU（eligible）<= 0.9。IoU 只算 eligible 格，含候选集外的真值格。2 折按日奇偶。"""
    from . import t2_ongoing as og
    from .t2_events import queue_truth
    df = og.load_frame()
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    feats = [c for c in df.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out") and c not in feat_drop]
    par = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
    par.update(params or {})
    models = {}
    for f in (0, 1):
        tr = df[df.d % 2 != f]
        if train_filter is not None:
            tr = train_filter(tr)
        models[f] = lgb.train(par, lgb.Dataset(tr[feats], tr.y, categorical_feature=["pid"]), rounds)
    out = []
    for p in T2P:
        z = load(p, "train")
        Q, _ = queue_truth(p, z=z)
        el = z["elig"] == 1
        cov = np.isfinite(z["speed"])
        g_all = df[df.panel == p]
        wins = g_all[["d", "T"]].drop_duplicates()
        if max_windows:
            wins = wins.sample(min(len(wins), max_windows), random_state=0)
        g_all = g_all.merge(wins, on=["d", "T"])
        for f in (0, 1):
            g_f = g_all[g_all.d % 2 == f]
            if g_f.empty:
                continue
            pr = models[f].predict(g_f[feats])
            g_f = g_f.assign(p=pr)
            for (d, T), g in g_f.groupby(["d", "T"]):
                if cov[d, T - 12:T + 1].mean() < 0.7:
                    continue
                e = el[d, T + 1:T + 7]
                tru = Q[d, T + 1:T + 7] & e
                if not tru.any():
                    continue
                k, l = g.k.to_numpy() - 1, g.link.to_numpy()
                pers = np.zeros_like(tru)
                pers[k[g.lastq.to_numpy() > .5], l[g.lastq.to_numpy() > .5]] = True
                pers &= e

                def sc(pred):
                    pred = pred & e
                    u = (pred | tru).sum()
                    return (pred & tru).sum() / u if u else 1.0
                sp = sc(pers)
                if sp > 0.9:
                    continue
                sel = (decode or og.decode)(g.p.to_numpy())
                mp = np.zeros_like(tru)
                mp[k[sel], l[sel]] = True
                out.append(dict(panel=p, pers=sp, model=sc(mp)))
    res = pd.DataFrame(out)
    pm = res.groupby("panel")[["pers", "model"]].mean()
    return pm, pm.mean(), len(res)
