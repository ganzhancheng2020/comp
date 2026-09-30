"""Onset under scenario shift: evaluation-only events mined from the published masked layer of every split.

Validation and private are different traffic scenarios from train (random scenario seed): bottleneck clusters
switch on or off. This scores onset variants on onset events mined from the masked layer (train holdout by
day parity, validation, private). The masked layer only serves as an evaluation set; no prediction reads it
beyond the event's own visible history (T-60..T-5, origin row missing), exactly as at inference.
Usage: python -m tfb.t2_onset_shift
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import t2_onset as on
from . import t2_onset2 as o2
from .data import CACHE, load, network
from .t1_interp import interp_axis1
from .t2 import T2_PANELS as T2P, static_sets
from .t2_eval_big import iou_elig
from .t2_events import onset_events, visible

SPLITS = ("train", "validation", "private")


def mined_events(p, split, z=None):
    """(d, s, true eligible queued links at s, eligible observed links at s) from the masked layer."""
    z = z if z is not None else load(p, split)
    net = network(p)
    vc = 0.6 * net.free_speed_kmh.to_numpy()
    v = z["m_speed"]
    Q = interp_axis1(v) <= vc
    obs = np.isfinite(v) & (z["pct"] >= 75)
    out = []
    for d, s, _ in onset_events(Q):
        T = s - 6
        if T < 12 or s >= v.shape[1]:
            continue
        rows = np.isfinite(v[d, T - 12:s + 1]).mean(1)
        if rows.min() == 0 or np.isfinite(v[d, T - 12:T]).mean() < 0.5:   # skip T2 blackouts and thin history
            continue
        el = np.where(obs[d, s])[0]
        tru = set(np.where(Q[d, s] & obs[d, s])[0].tolist())
        if tru:
            out.append((d, s, tru, el))
    return out


def event_rows(p, z, d, T, cands, cm):
    net = network(p)
    vc = 0.6 * net.free_speed_kmh.to_numpy()
    hs, hf = visible(z["m_speed"][d], T), visible(z["m_flow"][d], T)
    dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()[d]
    X = on.link_features(hs, hf, vc, net.capacity_vph.to_numpy(), T, cands, dow)
    X["pid"] = T2P.index(p)
    X["panel"], X["d"], X["s"] = p, d, T + 6
    X["cl"] = X.link.map(cm)
    return X


class _Sel:
    """Model wrapper that picks its own columns, so callers can pass wider frames."""
    def __init__(self, m, cols):
        self.m, self.cols = m, cols

    def predict(self, X):
        return self.m.predict(X[self.cols])


def fit_general(df, drop=(), use_cid=True):
    """Onset v9b structure (cluster model + in-cluster link model + old link model), minus the `drop` features."""
    base = [c for c in df.columns if c not in ("y", "d", "s", "panel", "n_true", "cl")]
    lf = [c for c in base if c not in drop]
    cf = o2.cluster_frame(df, base)
    cf["cid"] = cf.pid * 100 + cf.cl
    call = [c for c in cf.columns if c not in ("panel", "d", "s", "y", "cid")]
    cfeat = [c for c in call if c not in drop] + (["cid"] if use_cid else [])
    cat = [c for c in ("pid", "cid") if c in cfeat]
    mc = lgb.train(o2.CPAR, lgb.Dataset(cf[cfeat], cf.y, categorical_feature=cat), 400)
    act = df.merge(cf[["panel", "d", "s", "cl", "y"]].rename(columns={"y": "cy"}), on=["panel", "d", "s", "cl"])
    act = act[act.cy == 1]
    lcat = ["pid"] if "pid" in lf else []
    ml = lgb.train(on.PARAMS, lgb.Dataset(act[lf], act.y, categorical_feature=lcat), 400)
    mo = lgb.train(on.PARAMS, lgb.Dataset(df[lf], df.y, categorical_feature=lcat), 400)
    return _Sel(mc, cfeat), _Sel(ml, lf), base, call, _Sel(mo, lf), base


def predict(models, X, w_old=0.3):
    mc, ml, lf, cfeat, mo, lf_old = models
    return set(o2.predict_event_mix(X, mc, ml, lf, cfeat, mo.predict(X[lf_old]), w_old))


VARIANTS = {
    "v9b_re": dict(),
    "no_link": dict(drop=("link",)),
    "no_link_cid": dict(drop=("link",), use_cid=False),
    "no_tod": dict(drop=("tod", "dow")),
    "no_id_tod": dict(drop=("link", "tod", "dow"), use_cid=False),
}


def main():
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    S = static_sets()
    Z = {(p, s): load(p, s) for p in T2P for s in SPLITS}
    EV = {(p, s): mined_events(p, s, Z[(p, s)]) for p in T2P for s in SPLITS}
    print("events", {s: sum(len(EV[(p, s)]) for p in T2P) for s in SPLITS}, flush=True)
    rows = []
    for name, kw in VARIANTS.items():
        full = fit_general(df, **kw)
        folds = {f: fit_general(df[df.d % 2 != f], **kw) for f in (0, 1)}
        for (p, s), evs in EV.items():
            for d, ss, tru, el in evs:
                m = folds[d % 2] if s == "train" else full
                X = event_rows(p, Z[(p, s)], d, ss - 6, cands[p], cms[p])
                r = dict(variant=name, panel=p, split=s, iou=iou_elig(predict(m, X), tru, el))
                rows.append(r)
                if name == "v9b_re":
                    rows.append(dict(variant="static", panel=p, split=s, iou=iou_elig(set(S[p]), tru, el)))
        r = pd.DataFrame(rows)
        t = r.groupby(["variant", "split", "panel"]).iou.mean().groupby(["variant", "split"]).mean().unstack()
        print(t.round(4), flush=True)
    r.to_csv(CACHE / "t2_onset_shift.csv", index=False)
    print(r.groupby(["variant", "split", "panel"]).iou.mean().unstack("panel").round(3).to_string())


if __name__ == "__main__":
    main()
