"""Onset + scenario timing evidence (causal): per cluster, its activation slots on the previous k days of the same split
(masked layer), k drawn from 0..4 per event to mimic the Task 2 windows (days 1-5 of each split => 0-4 earlier days).
Features (cluster-level, prefixed e_ so the cluster model aggregates them): distance to the nearest prior activation,
share of prior days with an activation within +-30 / +-60 min of the event slot, number of prior days.
Paired A/B, in scenario (train 2-fold) and out of scenario (validation/private mined events). Usage: python -m tfb.t2_onset_timing_ab"""
import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_eval_big import iou_elig
from .t2_onset_shift import SPLITS, event_rows, fit_general, mined_events, predict
from .t2_onset_timing import cluster_onsets

EC = ["e_tdist", "e_tnear6", "e_tnear12", "e_kdays"]


def timing_feats(co, cm, d, slot, k):
    out = {}
    for c in co:
        days = range(max(0, d - k), d)
        prior = [[t for t in co[c][dd]] for dd in days]
        flat = [t for ts in prior for t in ts]
        n = len(prior)
        out[c] = dict(e_tdist=min([abs(t - slot) for t in flat], default=np.nan) if n else np.nan,
                      e_tnear6=np.mean([any(abs(t - slot) <= 6 for t in ts) for ts in prior]) if n else np.nan,
                      e_tnear12=np.mean([any(abs(t - slot) <= 12 for t in ts) for ts in prior]) if n else np.nan,
                      e_kdays=float(n))
    return out


def add(X, tf, cm):
    cl = X.link.map(cm).to_numpy()
    for c in EC:
        X[c] = [tf[k][c] if k in tf else np.nan for k in cl]
    return X


def main():
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    Z = {(p, s): load(p, s) for p in T2P for s in SPLITS}
    CO = {(p, s): cluster_onsets(Z[(p, s)], 0.6 * network(p).free_speed_kmh.to_numpy(), cms[p]) for p in T2P for s in SPLITS}
    rng = np.random.default_rng(5)
    parts = []
    for (p, d, s), g in df.groupby(["panel", "d", "s"]):
        k = int(rng.integers(0, 5))
        parts.append(add(g.copy(), timing_feats(CO[(p, "train")], cms[p], d, s, k), cms[p]))
    dft = pd.concat(parts, ignore_index=True)
    EV = {(p, s): mined_events(p, s, Z[(p, s)]) for p in T2P for s in SPLITS}
    ks = {key: rng.integers(0, 5, len(v)) for key, v in EV.items()}
    rows = []
    for name, d_ in (("base", dft.drop(columns=EC)), ("timing", dft)):
        full = fit_general(d_)
        folds = {f: fit_general(d_[d_.d % 2 != f]) for f in (0, 1)}
        for (p, s), evs in EV.items():
            for i, (dd, ss, tru, el) in enumerate(evs):
                m = folds[dd % 2] if s == "train" else full
                X = event_rows(p, Z[(p, s)], dd, ss - 6, cands[p], cms[p])
                if name == "timing":
                    X = add(X, timing_feats(CO[(p, s)], cms[p], dd, ss, int(ks[(p, s)][i])), cms[p])
                rows.append(dict(variant=name, panel=p, split=s, i=i, k=int(ks[(p, s)][i]),
                                 iou=iou_elig(predict(m, X), tru, el)))
        print(name, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_onset_timing_ab.csv", index=False)
    print(r.groupby(["variant", "split", "panel"]).iou.mean().groupby(["variant", "split"]).mean().unstack().round(4))
    w = r.pivot_table(index=["split", "panel", "i", "k"], columns="variant", values="iou").reset_index()
    w["dd"] = w.timing - w.base
    for s, g in w.groupby("split"):
        print(f"[{s}] {g.dd.mean():+.4f} ± {g.dd.std() / np.sqrt(len(g)):.4f} n={len(g)}  by k:",
              g.groupby("k").dd.mean().round(3).to_dict())
    print(w.groupby(["split", "panel"]).dd.mean().unstack("split").round(3))


if __name__ == "__main__":
    main()
