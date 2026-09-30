"""Ongoing: in-scenario vs out-of-scenario on the SAME input type (masked-layer history), and whether dropping the
identity/time features (link, lpos, tod, dow, pid, is_bneck) helps out of scenario. LightGBM on a 30% window subsample
of t2_ongoing_parts_vis; train windows scored 2-fold by day parity, validation/private by the all-train model.
Evaluation-only use of the val/private masked layer. Usage: python -m tfb.t2_ongoing_idshift
"""
from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import t2_ongoing as og
from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_events import visible
from .t2_ongoing_shift import iou, mined_windows

PAR = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
DROPS = {"full": (), "no_id": ("link", "lpos", "tod", "dow"), "no_id_pid": ("link", "lpos", "tod", "dow", "pid", "is_bneck")}


def frame(frac=0.3, seed=0):
    parts = []
    for p in T2P:
        t = pq.read_table(CACHE / "t2_ongoing_parts_vis" / f"{p}.parquet").to_pandas()
        w = t[["d", "T"]].drop_duplicates().sample(frac=frac, random_state=seed)
        t = t.merge(w, on=["d", "T"])
        t["pid"] = np.int32(T2P.index(p))
        f32 = t.select_dtypes("float64").columns
        t[f32] = t[f32].astype(np.float32)
        parts.append(t)
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def main():
    df = frame()
    base = [c for c in df.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")]
    print("rows", len(df), flush=True)
    models = {}
    for name, drop in DROPS.items():
        feats = [c for c in base if c not in drop]
        cat = ["pid"] if "pid" in feats else []
        fit = lambda m: lgb.train(PAR, lgb.Dataset(df.loc[m, feats].to_numpy(np.float32), df.y.to_numpy()[m],
                                                   feature_name=feats, categorical_feature=cat), 500)
        models[name] = (feats, {0: fit(df.d.to_numpy() % 2 != 0), 1: fit(df.d.to_numpy() % 2 != 1),
                                "all": fit(np.ones(len(df), bool))})
        print("fitted", name, flush=True)
    del df
    gc.collect()
    bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
    rows = []
    rng = np.random.default_rng(0)
    for p in T2P:
        net = network(p)
        vc = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        bneck = sorted(bn[bn.panel == p].link)
        for s in ("train", "validation", "private"):
            z = load(p, s)
            dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
            for d, T, tru, e, pers in mined_windows(p, s, z, rng):
                X = og.cell_features(visible(z["m_speed"][d], T), visible(z["m_flow"][d], T), vc, cap, T, dow[d],
                                     margin=12, bneck=bneck, early=og.early_features(z["m_speed"][d], T, vc))
                r = dict(panel=p, split=s, pers=iou(pers, tru, e))
                for name, (feats, ms) in models.items():
                    mp = np.zeros_like(tru, dtype=np.float32)
                    if X is not None:
                        m = ms[d % 2] if s == "train" else ms["all"]
                        Xm = X.assign(pid=T2P.index(p))
                        mp[Xm.k.to_numpy() - 1, Xm.link.to_numpy()] = m.predict(Xm[feats].to_numpy(np.float32))
                    r[name] = iou(mp > 0.5, tru, e)
                rows.append(r)
            print(p, s, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_ongoing_idshift.csv", index=False)
    cols = ["pers"] + list(DROPS)
    print(r.groupby(["split", "panel"])[cols].mean().groupby("split").mean().T.round(4))
    for name in list(DROPS)[1:]:
        dd = r[name] - r["full"]
        o = r.split != "train"
        print(name, "- full: oos %+.4f se %.4f" % (dd[o].mean(), dd[o].std() / np.sqrt(o.sum())),
              dd.groupby(r.split).mean().round(4).to_dict())


if __name__ == "__main__":
    main()
