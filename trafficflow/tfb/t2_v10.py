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

RAMP = os.environ.get("TFB_ONSET_RAMP", "0") == "1"   # onset with ramp-demand features (t2_onset_ramp)
PRIOR = os.environ.get("TFB_ONSET_PRIOR", "0") == "1"  # Round 11: label-shift prior correction from earlier days
PRIOR_W, PRIOR_A, PRIOR_B = 12, 1.0, 0.5


def prior_ratio(pc, co_src, co, d, slot):
    """rate_tgt / rate_src per cluster (t2_onset_prior): onsets within +-W slots on all earlier days of the split."""
    from .t2_onset_prior import EPS, near
    r = {}
    for c in pc:
        rs = near(co_src[c], slot, PRIOR_W).mean() + EPS
        n = near(co[c][:d], slot, PRIOR_W).sum() if d else 0.0
        r[c] = ((n + PRIOR_A * rs) / (d + PRIOR_A)) / rs
    return r


def onset_windows(split, models, cands, w):
    out = {}
    for p in T2P:
        net = network(p)
        li = {x: i for i, x in enumerate(net.link_id)}
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        cm = o2.clusters(cands[p])
        if RAMP:
            from .data import load
            from .t2_onset_ramp import ramp_feats_arr, ramp_index
            zs = load(p, split)
            ridx = ramp_index(p, zs)
            di = {str(dd): i for i, dd in enumerate(zs["dates"].tolist())}
        if PRIOR:
            from .data import load
            from .t2_onset_timing import cluster_onsets
            zp = load(p, split)
            co_src, co = cluster_onsets(load(p, "train"), vcut, cm), cluster_onsets(zp, vcut, cm)
            dix = {str(dd): i for i, dd in enumerate(zp["dates"].tolist())}
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
            if RAMP:   # ramp flows over T-60..T-5 (published, pre-origin) + last visible mainline flow of the window
                d = di[T0.strftime("%Y-%m-%d")]
                T = T0.hour * 12 + T0.minute // 5
                mf = pd.DataFrame(hf[:12]).ffill().to_numpy()[-1]
                X = pd.concat([X.reset_index(drop=True),
                               ramp_feats_arr(zs["ramp_flow"][d, T - 12:T], mf, X.link.to_numpy(), ridx)], axis=1)
            es = [probs(m, X) for m in models]
            if PRIOR:   # data before the window's day only (Task 2 ruling: timestamps <= T)
                from .t2_onset_prior import adjust
                T = T0.hour * 12 + T0.minute // 5
                r = prior_ratio(es[0]["pc"], co_src, co, dix[T0.strftime("%Y-%m-%d")], T + 6)
                es = [adjust(e, cm, r, PRIOR_B) for e in es]
            out[r.window_id] = sorted(decode_pool(es, [1 - w, w]))
    return out


def main():
    w = float(sys.argv[1])
    out_name = sys.argv[2]
    df = pd.read_parquet(CACHE / os.environ.get("TFB_ONSET_FRAME", "t2_onset.parquet"))   # t2_onset_withT.parquet = v9b
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    if RAMP:
        from .t2_onset_ramp import ramp_frame
        df = ramp_frame(df)
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
