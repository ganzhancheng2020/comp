"""Feasibility: can earlier days of the same split (strictly before the event day: causal) correct the onset cluster?
Mined events on days 2..7 of validation/private (the window period). For each event: the model's top cluster, the true
cluster(s), and how often each cluster queued (any queued slot on a member link, masked layer) on earlier days of the
split vs its train queue-day rate. Evaluation-only use of the masked layer.
Usage: python -m tfb.t2_onset_prior_days"""
import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, load, network
from .t1_interp import interp_axis1
from .t2 import T2_PANELS as T2P
from .t2_onset_shift import event_rows, fit_general, mined_events
from .t2_onset_shift_cal import probs


def cluster_day_rates(z, vc, cm):
    Q = interp_axis1(z["m_speed"]) <= vc
    cls = sorted(set(cm.values()))
    return {c: Q[:, :, [l for l, k in cm.items() if k == c]].any((1, 2)) for c in cls}   # (D,) bool per cluster


def main():
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    m = fit_general(df)
    rows = []
    for p in T2P:
        vc = 0.6 * network(p).free_speed_kmh.to_numpy()
        ztr = load(p, "train")
        rtr = {c: v.mean() for c, v in cluster_day_rates(ztr, vc, cms[p]).items()}
        for s in ("validation", "private"):
            z = load(p, s)
            rd = cluster_day_rates(z, vc, cms[p])
            for d, ss, tru, el in mined_events(p, s, z):
                if not 1 <= d <= 6:
                    continue
                X = event_rows(p, z, d, ss - 6, cands[p], cms[p])
                e = probs(m, X)
                top = max(e["pc"], key=e["pc"].get)
                tcl = {cms[p][l] for l in tru if l in cms[p]}
                for c in e["pc"]:
                    rows.append(dict(panel=p, split=s, d=d, c=c, pc=e["pc"][c], is_top=c == top, is_true=c in tcl,
                                     prior_days=d, seen=int(rd[c][:d].sum()), r_train=rtr[c]))
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_onset_prior_days.csv", index=False)
    ev = r.groupby(["panel", "split", "d"])
    wrong = ev.apply(lambda g: not g[g.is_top].is_true.any())
    print("events on days 2..7:", len(wrong), "wrong top cluster:", int(wrong.sum()))
    w = r.merge(wrong.rename("wrong").reset_index(), on=["panel", "split", "d"])
    w = w[w.wrong & (w.is_top | w.is_true)]
    print(w[["panel", "split", "d", "c", "pc", "is_top", "is_true", "seen", "prior_days", "r_train"]].round(3).to_string())


if __name__ == "__main__":
    main()
