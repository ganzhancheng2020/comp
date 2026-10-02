"""Onset with same-day early evidence (masked layer before T-60: queued-slot count, slots since last queue, 3-hour queued
fraction per candidate link; cluster aggregates), judged OUT OF SCENARIO. In scenario the train priors already know which
cluster queues when; out of scenario they do not, and the same day's earlier queues are the only causal scenario evidence.
Usage: python -m tfb.t2_onset_early_ab"""
import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_eval_big import iou_elig
from .t2_ongoing import early_features
from .t2_onset_shift import SPLITS, event_rows, fit_general, mined_events, predict

EC = ["e_q_cnt", "e_since", "e_frac3h"]


def add_early(X, mday, T, vc):
    e = early_features(mday, T, vc)
    l = X.link.to_numpy()
    for c in EC:
        X[c] = e[c][l]
    return X


def train_frame():
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    parts = []
    for p, g in df.groupby("panel"):
        z = load(p, "train")
        vc = 0.6 * network(p).free_speed_kmh.to_numpy()
        for (d, s), gg in g.groupby(["d", "s"]):
            parts.append(add_early(gg.copy(), z["m_speed"][d], s - 6, vc))
    return o2.add_cluster(pd.concat(parts, ignore_index=True))


def main():
    df = train_frame()
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    Z = {(p, s): load(p, s) for p in T2P for s in SPLITS}
    EV = {(p, s): mined_events(p, s, Z[(p, s)]) for p in T2P for s in SPLITS}
    rows = []
    for name, d in (("base", df.drop(columns=EC)), ("early", df)):
        full = fit_general(d)
        folds = {f: fit_general(d[d.d % 2 != f]) for f in (0, 1)}
        for (p, s), evs in EV.items():
            vc = 0.6 * network(p).free_speed_kmh.to_numpy()
            for i, (dd, ss, tru, el) in enumerate(evs):
                m = folds[dd % 2] if s == "train" else full
                X = event_rows(p, Z[(p, s)], dd, ss - 6, cands[p], cms[p])
                if name == "early":
                    X = add_early(X, Z[(p, s)]["m_speed"][dd], ss - 6, vc)
                rows.append(dict(variant=name, panel=p, split=s, i=i, iou=iou_elig(predict(m, X), tru, el)))
        print(name, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_onset_early_ab.csv", index=False)
    print(r.groupby(["variant", "split", "panel"]).iou.mean().groupby(["variant", "split"]).mean().unstack().round(4))
    w = r.pivot_table(index=["split", "panel", "i"], columns="variant", values="iou").reset_index()
    w["dd"] = w.early - w.base
    for s, g in w.groupby("split"):
        print(f"[{s}] {g.dd.mean():+.4f} ± {g.dd.std() / np.sqrt(len(g)):.4f} n={len(g)}")
    print(w.groupby(["split", "panel"]).dd.mean().unstack("split").round(3))


if __name__ == "__main__":
    main()
