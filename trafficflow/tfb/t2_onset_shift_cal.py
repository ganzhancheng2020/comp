"""Onset decoding under scenario shift: temperature on the v9b probabilities, and shrinking cluster activation
toward the train base rate. Evaluation events come from `t2_onset_shift.mined_events` (evaluation only).
Usage: python -m tfb.t2_onset_shift_cal
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, load
from .t2 import T2_PANELS as T2P
from .t2_eval_big import iou_elig
from .t2_onset_shift import SPLITS, event_rows, fit_general, mined_events


def probs(models, X):
    mc, ml, lf, cfeat, mo, lf_old = models
    cf = o2.cluster_frame(X.assign(y=0), lf)
    cf["cid"] = cf.pid * 100 + cf.cl
    pc = dict(zip(cf.cl, mc.predict(cf)))
    return dict(cl=X.cl.to_numpy(), link=X.link.to_numpy(), pc=pc, pl=ml.predict(X), po=mo.predict(X))


def temp(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / t))


def decode(e, t=1.0, a=0.0, base=None, w_old=0.3, n_mc=800, rng=None):
    rng = rng or np.random.default_rng(0)
    ucl = np.array(sorted(e["pc"]))
    pc = temp(np.array([e["pc"][c] for c in ucl]), t)
    if a > 0:
        pc = (1 - a) * pc + a * np.array([base.get(c, 0.0) for c in ucl])
    pl, po = temp(e["pl"], t), temp(e["po"], t)
    n2 = int(n_mc * (1 - w_old))
    act = rng.random((n2, len(ucl))) < pc[None, :]
    s1 = act[:, np.searchsorted(ucl, e["cl"])] & (rng.random((n2, len(pl))) < pl[None, :])
    s2 = rng.random((n_mc - n2, len(pl))) < po[None, :]
    samp = np.vstack([s1, s2])
    order = np.argsort(-samp.mean(0))
    best, bk = -1, 1
    for k in range(1, len(pl) + 1):
        S = np.zeros(len(pl), bool)
        S[order[:k]] = True
        v = np.mean(np.where((samp | S).sum(1) > 0, (samp & S).sum(1) / np.maximum((samp | S).sum(1), 1), 1.0))
        if v > best:
            best, bk = v, k
    return set(e["link"][order[:bk]].tolist())


def main():
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    cy = df.groupby(["panel", "d", "s", "cl"]).y.max().groupby(["panel", "cl"]).mean()
    base = {p: cy[p].to_dict() for p in T2P}
    path = CACHE / "t2_onset_shift_probs.pkl"
    if path.exists():
        E = pickle.load(open(path, "rb"))
    else:
        full = fit_general(df)
        folds = {f: fit_general(df[df.d % 2 != f]) for f in (0, 1)}
        E = []
        for p in T2P:
            for s in SPLITS:
                z = load(p, s)
                for d, ss, tru, el in mined_events(p, s, z):
                    m = folds[d % 2] if s == "train" else full
                    e = probs(m, event_rows(p, z, d, ss - 6, cands[p], cms[p]))
                    E.append(dict(e, panel=p, split=s, tru=tru, el=el))
        pickle.dump(E, open(path, "wb"))
    rows = []
    grid = [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (3.0, 0.0), (1.0, 0.3), (1.0, 0.5), (1.5, 0.3), (2.0, 0.3)]
    for e in E:
        r = dict(panel=e["panel"], split=e["split"])
        for t, a in grid:
            sel = decode(e, t, a, base[e["panel"]])
            r[f"t{t}_a{a}"] = iou_elig(sel, e["tru"], e["el"])
            r[f"n_t{t}_a{a}"] = len(sel)
        rows.append(r)
    r = pd.DataFrame(rows)
    sc = [c for c in r.columns if c.startswith("t")]
    print(r.groupby(["split", "panel"])[sc].mean().groupby("split").mean().T.round(4))
    print("mean set size", r.groupby("split")[[c for c in r.columns if c.startswith("n_")]].mean().T.round(2))
    print("true size", pd.Series([len(e["tru"]) for e in E]).groupby(r.split).mean().round(2).to_dict())


if __name__ == "__main__":
    main()
