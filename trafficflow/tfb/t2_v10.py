"""v10 queue file: onset = pooled ensemble of the v9b onset model and its no-tod/dow variant (weight w on the latter),
ongoing = P1 (origin-row-fixed LightGBM + old 4-seed CNN, rows copied from queue_models8_p1.csv).
Usage: python -m tfb.t2_v10 <w> <out.csv>   (w=0 must reproduce the v9b onset rows exactly)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from . import t2_onset as on
from . import t2_onset2 as o2
from .data import CACHE, REL, network
from .submit import OUT
from .t2 import T2_PANELS as T2P, predict
from .t2_onset_shift import fit_general
from .t2_onset_shift_cal import probs
from .t2_onset_shift_ens import decode_pool


def onset_windows(split, models, cands, w):
    out = {}
    for p in T2P:
        net = network(p)
        li = {x: i for i, x in enumerate(net.link_id)}
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        cm = o2.clusters(cands[p])
        wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
        h = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet")
        for r in wi[wi.condition == "queue_onset"].itertuples():
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
            es = [probs(m, X) for m in models]
            out[r.window_id] = sorted(decode_pool(es, [1 - w, w]))
    return out


def main():
    w = float(sys.argv[1])
    out_name = sys.argv[2]
    df = pd.read_parquet(CACHE / os.environ.get("TFB_ONSET_FRAME", "t2_onset.parquet"))   # t2_onset_withT.parquet = v9b
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    models = [fit_general(df), fit_general(df, drop=("tod", "dow"))]
    q = pd.concat([predict(s, onset_windows=onset_windows(s, models, cands, w)) for s in ("validation", "private")],
                  ignore_index=True)
    src = pd.read_csv(OUT / "queue_models8_p1.csv")
    assert (src.window_id.values == q.window_id.values).all() and (src.link_id.values == q.link_id.values).all()
    idx = pd.concat([pd.read_csv(p) for p in REL.glob("task2/*/*/window_index.csv")])
    ons = q.window_id.map(dict(zip(idx.window_id, idx.condition))) == "queue_onset"
    q.loc[~ons, "queue_pred"] = src.loc[~ons, "queue_pred"].values
    q.to_csv(OUT / out_name, index=False)
    ref = pd.read_csv(OUT / "queue_models7.csv")
    print("wrote", out_name, "onset cells", int(q.queue_pred[ons].sum()),
          "onset cells differing from v9b", int((q.queue_pred[ons].values != ref.queue_pred[ons].values).sum()),
          "ongoing cells differing from v9b", int((q.queue_pred[~ons].values != ref.queue_pred[~ons].values).sum()))


if __name__ == "__main__":
    main()
