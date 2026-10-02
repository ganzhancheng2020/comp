"""Onset v2：簇级 + 簇内条件的两层模型，联合蒙特卡洛解码（保留同簇相关性）。"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CACHE
from .t2 import T2_PANELS as T2P
from . import t2_onset as on

CPAR = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
            feature_fraction=0.8, verbose=-1, num_threads=4)


def clusters(links: list[int], gap: int = 3) -> dict[int, int]:
    """Group sorted candidate links into clusters of neighbours at most `gap` apart."""
    out, cid, prev = {}, -1, None
    for l in sorted(links):
        if prev is None or l - prev > gap:
            cid += 1
        out[l] = cid
        prev = l
    return out


def add_cluster(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for p, g in df.groupby("panel"):
        cm = clusters(sorted(g.link.unique()))
        parts.append(g.assign(cl=g.link.map(cm)))
    return pd.concat(parts)


def cluster_frame(df: pd.DataFrame, link_feats: list[str]) -> pd.DataFrame:
    agg = {c: ["mean", "min", "max"] for c in ("r0", "r1", "r3", "r6", "rmin3", "rtrend", "q0", "q3", "qmean",
                                                "nb_rmin", "nb_q", "up_r", "dn_r")}
    agg.update({c: ["mean", "max"] for c in df.columns if c.startswith("rp_")})   # optional ramp features
    agg.update({c: ["mean", "max", "min"] for c in df.columns if c.startswith("e_")})   # optional same-day early features
    c = df.groupby(["panel", "d", "s", "cl"]).agg(agg)
    c.columns = ["_".join(x) for x in c.columns]
    first = df.groupby(["panel", "d", "s", "cl"])[["tod", "dow", "pid", "cor_rmin", "cor_q"]].first()
    y = df.groupby(["panel", "d", "s", "cl"]).y.max().rename("y")
    nl = df.groupby(["panel", "d", "s", "cl"]).size().rename("n_links")
    return pd.concat([c, first, y, nl], axis=1).reset_index()


def fit(df: pd.DataFrame, rounds: int = 400):
    lf = [c for c in df.columns if c not in ("y", "d", "s", "panel", "n_true", "cl")]
    cf = cluster_frame(df, lf)
    cfeat = [c for c in cf.columns if c not in ("panel", "d", "s", "y")] 
    cf["cid"] = cf.pid * 100 + cf.cl
    mc = lgb.train(CPAR, lgb.Dataset(cf[cfeat + ["cid"]], cf.y, categorical_feature=["pid", "cid"]), rounds)
    act = df.merge(cf[["panel", "d", "s", "cl", "y"]].rename(columns={"y": "cy"}), on=["panel", "d", "s", "cl"])
    act = act[act.cy == 1]
    ml = lgb.train(on.PARAMS, lgb.Dataset(act[lf], act.y, categorical_feature=["pid"]), rounds)
    return mc, ml, lf, cfeat


def predict_event(g: pd.DataFrame, mc, ml, lf, cfeat, n_mc: int = 600, rng=None):
    """g: link rows of one event (with cl). Returns chosen link list."""
    rng = rng or np.random.default_rng(0)
    cf = cluster_frame(g.assign(y=0), lf)
    cf["cid"] = cf.pid * 100 + cf.cl
    pc = dict(zip(cf.cl, mc.predict(cf[cfeat + ["cid"]])))
    pl = ml.predict(g[lf])
    cl = g.cl.to_numpy()
    ucl = np.array(sorted(pc))
    act = rng.random((n_mc, len(ucl))) < np.array([pc[c] for c in ucl])[None, :]
    cidx = np.searchsorted(ucl, cl)
    samp = act[:, cidx] & (rng.random((n_mc, len(pl))) < pl[None, :])
    marg = samp.mean(0)
    order = np.argsort(-marg)
    best, bk = -1, 1
    for k in range(1, len(pl) + 1):
        S = np.zeros(len(pl), bool)
        S[order[:k]] = True
        inter = (samp & S).sum(1)
        union = (samp | S).sum(1)
        v = np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0))
        if v > best:
            best, bk = v, k
    return g.link.to_numpy()[order[:bk]].tolist()


def evaluate(rounds: int = 400):
    from .t2_eval_big import event_table, iou_elig
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = add_cluster(df)
    ev = event_table()
    out = []
    for f in (0, 1):
        mc, ml, lf, cfeat = fit(df[df.d % 2 != f], rounds)
        te = df[df.d % 2 == f]
        grp = dict(tuple(te.groupby(["panel", "d", "s"])))
        for r in ev[ev.d % 2 == f].itertuples():
            sel = set(predict_event(grp[(r.panel, r.d, r.s)], mc, ml, lf, cfeat))
            out.append(dict(panel=r.panel, model2=iou_elig(sel, r.true, r.elig_links)))
    res = pd.DataFrame(out)
    pm = res.groupby("panel").model2.mean()
    return pm, pm.mean()


if __name__ == "__main__":
    pm, m = evaluate()
    print(pm.round(3).to_dict())
    print("onset v2 (cluster-level) mean IoU %.4f" % m)


def predict_windows(split: str) -> dict:
    """{window_id: links queued at T+30} for onset windows, trained on all train events."""
    from .data import REL, network
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = add_cluster(df)
    mc, ml, lf, cfeat = fit(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    out = {}
    for p in T2P:
        net = network(p)
        li = {x: i for i, x in enumerate(net.link_id)}
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        cm = clusters(cands[p])
        w = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
        h = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet")
        for r in w[w.condition == "queue_onset"].itertuples():
            g = h[h.window_id == r.window_id]
            T0 = pd.Timestamp(r.forecast_origin)
            k = ((T0 - pd.to_datetime(g.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
            hs = np.full((13, len(net)), np.nan)
            hf = np.full((13, len(net)), np.nan)
            ll = g.link_id.map(li).to_numpy()
            hs[12 - k, ll] = g.speed_kmh.to_numpy()
            hf[12 - k, ll] = g.flow_vph.to_numpy()
            X = on.link_features(hs, hf, vcut, cap, T0.hour * 12 + T0.minute // 5, cands[p], T0.dayofweek)
            X["pid"] = T2P.index(p)
            X["panel"], X["d"], X["s"] = p, 0, 0
            X["cl"] = X.link.map(cm)
            out[r.window_id] = predict_event(X, mc, ml, lf, cfeat)
    return out


def predict_event_mix(g, mc, ml, lf, cfeat, p_old, w_old=0.5, n_mc=800, rng=None):
    """Decode over a mixture: (1-w_old) joint cluster samples + w_old independent samples from the old link model."""
    rng = rng or np.random.default_rng(0)
    cf = cluster_frame(g.assign(y=0), lf)
    cf["cid"] = cf.pid * 100 + cf.cl
    pc = dict(zip(cf.cl, mc.predict(cf[cfeat + ["cid"]])))
    pl = ml.predict(g[lf])
    ucl = np.array(sorted(pc))
    n2 = int(n_mc * (1 - w_old))
    act = rng.random((n2, len(ucl))) < np.array([pc[c] for c in ucl])[None, :]
    cidx = np.searchsorted(ucl, g.cl.to_numpy())
    s1 = act[:, cidx] & (rng.random((n2, len(pl))) < pl[None, :])
    s2 = rng.random((n_mc - n2, len(pl))) < p_old[None, :]
    samp = np.vstack([s1, s2])
    order = np.argsort(-samp.mean(0))
    best, bk = -1, 1
    for k in range(1, len(pl) + 1):
        S = np.zeros(len(pl), bool)
        S[order[:k]] = True
        inter = (samp & S).sum(1)
        union = (samp | S).sum(1)
        v = np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0))
        if v > best:
            best, bk = v, k
    return g.link.to_numpy()[order[:bk]].tolist()


def evaluate_mix(weights=(0.0, 0.3, 0.5, 1.0)):
    import lightgbm as lgb
    from .t2_eval_big import event_table, iou_elig
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = add_cluster(df)
    lf_old = [c for c in df.columns if c not in ("y", "d", "s", "panel", "n_true", "cl")]
    ev = event_table()
    out = []
    for f in (0, 1):
        tr = df[df.d % 2 != f]
        mc, ml, lf, cfeat = fit(tr)
        mo = lgb.train(on.PARAMS, lgb.Dataset(tr[lf_old], tr.y, categorical_feature=["pid"]), 400)
        te = df[df.d % 2 == f]
        grp = dict(tuple(te.groupby(["panel", "d", "s"])))
        for r in ev[ev.d % 2 == f].itertuples():
            g = grp[(r.panel, r.d, r.s)]
            p_old = mo.predict(g[lf_old])
            row = dict(panel=r.panel)
            for w in weights:
                row[f"w{w}"] = iou_elig(set(predict_event_mix(g, mc, ml, lf, cfeat, p_old, w)), r.true, r.elig_links)
            out.append(row)
    res = pd.DataFrame(out)
    pm = res.drop(columns="panel").groupby(res.panel).mean()
    return pm, pm.mean()


def predict_windows_mix(split: str, w_old: float = 0.3) -> dict:
    """Production onset: mixture of the cluster-level v2 and the old independent link model (weight w_old)."""
    import lightgbm as lgb
    from .data import REL, network
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = add_cluster(df)
    lf_old = [c for c in df.columns if c not in ("y", "d", "s", "panel", "n_true", "cl")]
    mc, ml, lf, cfeat = fit(df)
    mo = lgb.train(on.PARAMS, lgb.Dataset(df[lf_old], df.y, categorical_feature=["pid"]), 400)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    out = {}
    for p in T2P:
        net = network(p)
        li = {x: i for i, x in enumerate(net.link_id)}
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        cm = clusters(cands[p])
        w = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
        h = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet")
        for r in w[w.condition == "queue_onset"].itertuples():
            g = h[h.window_id == r.window_id]
            T0 = pd.Timestamp(r.forecast_origin)
            k = ((T0 - pd.to_datetime(g.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
            hs = np.full((13, len(net)), np.nan)
            hf = np.full((13, len(net)), np.nan)
            ll = g.link_id.map(li).to_numpy()
            hs[12 - k, ll] = g.speed_kmh.to_numpy()
            hf[12 - k, ll] = g.flow_vph.to_numpy()
            X = on.link_features(hs, hf, vcut, cap, T0.hour * 12 + T0.minute // 5, cands[p], T0.dayofweek)
            X["pid"] = T2P.index(p)
            X["panel"], X["d"], X["s"] = p, 0, 0
            X["cl"] = X.link.map(cm)
            out[r.window_id] = predict_event_mix(X, mc, ml, lf, cfeat, mo.predict(X[lf_old]), w_old)
    return out
