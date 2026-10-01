"""Paired A/B for the ongoing LightGBM: origin row missing (vis, V10a) vs origin row T from the masked layer (vt).

In scenario: train windows, 2-fold by day parity, official-style selector filter (official persistence IoU <= 0.9).
Out of scenario: ongoing windows mined from the validation/private masked layer (evaluation only; inputs are the
masked layer through T, exactly what each variant would see). Same windows for both variants.
Usage: python -m tfb.t2_vt_ab [frac_windows=0.35] [rounds=600]
"""
import gc
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import t2_ongoing as og
from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_events import queue_truth, visible
from .t2_ongoing_shift import mined_windows, iou

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
DROP = ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")


def windows(p):
    w = pq.read_table(CACHE / "t2_ongoing_parts_vis" / f"{p}.parquet", columns=["d", "T"]).to_pandas().drop_duplicates()
    rng = np.random.default_rng(1)
    w["tr"] = rng.random(len(w)) < FRAC
    return w


def load_variant(name, W):
    Xs, ys, keys, feats = [], [], [], None
    for p in T2P:
        path = CACHE / f"t2_ongoing_parts_{name}" / f"{p}.parquet"
        key = pq.read_table(path, columns=["d", "T"]).to_pandas()
        sel = key.merge(W[p].assign(_i=1), on=["d", "T"], how="left")._i.notna().to_numpy()
        tab = pq.read_table(path).filter(pa.array(sel))      # row subset in Arrow, before pandas
        t = tab.to_pandas(); del tab, key
        f32 = t.select_dtypes("float64").columns
        t[f32] = t[f32].astype(np.float32)
        t = t.merge(W[p], on=["d", "T"])
        t["pid"] = np.int32(T2P.index(p))
        if feats is None:
            feats = [c for c in t.columns if c not in DROP + ("tr",)]
        Xs.append(t[feats].to_numpy(np.float32)); ys.append(t.y.to_numpy())
        keys.append(t[["d", "T", "k", "link", "tr"]].assign(panel=p))
        del t; gc.collect()
    return np.concatenate(Xs), np.concatenate(ys), pd.concat(keys, ignore_index=True), feats


def main():
    W = {p: windows(p) for p in T2P}
    test = {p: W[p][~W[p].tr].sample(min(1500, int((~W[p].tr).sum())), random_state=0) for p in T2P}
    W = {p: pd.concat([W[p][W[p].tr], test[p]]) for p in T2P}   # only training windows + the test sample in memory
    res, full = {}, {}
    for name in ("vis", "vt"):
        X, y, K, feats = load_variant(name, W)
        pr = np.full(len(y), np.nan, np.float32)
        for f in (0, 1):
            trm = K.tr.to_numpy() & (K.d.to_numpy() % 2 != f)
            m = lgb.train(PAR, lgb.Dataset(X[trm], y[trm], feature_name=feats, categorical_feature=["pid"]), ROUNDS)
            tem = (~K.tr.to_numpy()) & (K.d.to_numpy() % 2 == f)
            pr[tem] = m.predict(X[tem])
            del m; gc.collect()
        trm = K.tr.to_numpy()
        full[name] = (lgb.train(PAR, lgb.Dataset(X[trm], y[trm], feature_name=feats, categorical_feature=["pid"]), ROUNDS), feats)
        K["pr"] = pr
        res[name] = K[~K.tr.to_numpy()]
        del X, y; gc.collect()
        print(name, "trained", flush=True)
    rows = []
    for p in T2P:
        z = load(p, "train"); Q, vc = queue_truth(p, z=z); el = z["elig"] == 1; L = len(vc)
        g = {n: {k: gg for k, gg in res[n][res[n].panel == p].groupby(["d", "T"])} for n in res}
        for r in test[p].itertuples():
            d, T = int(r.d), int(r.T)
            e = el[d, T + 1:T + 7]; tru = Q[d, T + 1:T + 7] & e
            if not tru.any():
                continue
            lo = (z["speed"][d, T - 1] <= vc) & (z["elig"][d, T - 1] == 1)
            pers = np.repeat(lo[None], 6, 0) & e
            if iou(pers, tru, e) > 0.9:
                continue
            out = dict(panel=p, split="train", pers=iou(pers, tru, e))
            for n in res:
                mp = np.zeros((6, L), np.float32)
                gg = g[n].get((d, T))
                if gg is not None:
                    mp[gg.k.to_numpy() - 1, gg.link.to_numpy()] = gg.pr.to_numpy()
                out[n] = iou(mp > 0.5, tru, e)
            rows.append(out)
    bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
    rng = np.random.default_rng(0)
    for p in T2P:
        net = network(p); vc = 0.6 * net.free_speed_kmh.to_numpy(); cap = net.capacity_vph.to_numpy()
        bneck = sorted(bn[bn.panel == p].link)
        for s in ("validation", "private"):
            z = load(p, s); dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
            for d, T, tru, e, pers in mined_windows(p, s, z, rng):
                ef = og.early_features(z["m_speed"][d], T, vc)
                out = dict(panel=p, split=s, pers=iou(pers, tru, e))
                for n, (m, feats) in full.items():
                    if n == "vis":
                        hs, hf = visible(z["m_speed"][d], T), visible(z["m_flow"][d], T)
                    else:
                        hs, hf = z["m_speed"][d, T - 12:T + 1].astype(float), z["m_flow"][d, T - 12:T + 1].astype(float)
                    Xw = og.cell_features(hs, hf, vc, cap, T, dow[d], margin=12, bneck=bneck, early=ef)
                    mp = np.zeros((6, len(net)), np.float32)
                    if Xw is not None:
                        Xw["pid"] = T2P.index(p)
                        for c in feats:
                            if c not in Xw.columns:
                                Xw[c] = np.nan
                        mp[Xw.k.to_numpy() - 1, Xw.link.to_numpy()] = m.predict(Xw[feats].to_numpy(np.float32))
                    out[n] = iou(mp > 0.5, tru, e)
                rows.append(out)
        print(p, "out-of-scenario done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_vt_ab.csv", index=False)
    print(r.groupby(["split", "panel"])[["pers", "vis", "vt"]].mean().groupby("split").mean().round(4))
    dd = r.vt - r.vis
    for s, g in dd.groupby(r.split):
        print(f"vt - vis [{s}]: {g.mean():+.4f} ± {g.std() / np.sqrt(len(g)):.4f}  (n={len(g)})")
    pm = (r.assign(dd=dd).groupby(["split", "panel"]).dd.mean().unstack("split")).round(4)
    print(pm)


if __name__ == "__main__":
    main()
