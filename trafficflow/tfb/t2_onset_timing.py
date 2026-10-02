"""Feasibility: does the cluster's activation TIME on earlier days of the same split (causal) separate the true cluster
from the model's wrong pick? (The day-rate check could not: most clusters queue every day; the scenario shift is in when.)
Mined onset events of validation/private with >= 1 earlier day; per cluster, onset slots on earlier days of the split
(cluster turns queued after >= 18 unqueued slots, masked layer); distance = min |onset slot - event slot|.
Usage: python -m tfb.t2_onset_timing"""
import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, load, network
from .t1_interp import interp_axis1
from .t2 import T2_PANELS as T2P
from .t2_onset_shift import event_rows, fit_general, mined_events
from .t2_onset_shift_cal import probs


def cluster_onsets(z, vc, cm):
    Q = interp_axis1(z["m_speed"]) <= vc
    out = {}
    for c in sorted(set(cm.values())):
        q = Q[:, :, [l for l, k in cm.items() if k == c]].any(2)          # (D, T)
        ons = []
        for d in range(q.shape[0]):
            s = [t for t in range(18, q.shape[1]) if q[d, t] and not q[d, t - 18:t].any()]
            ons.append(s)
        out[c] = ons
    return out


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
        for s in ("validation", "private"):
            z = load(p, s)
            co = cluster_onsets(z, vc, cms[p])
            for d, ss, tru, el in mined_events(p, s, z):
                if d < 1:
                    continue
                X = event_rows(p, z, d, ss - 6, cands[p], cms[p])
                e = probs(m, X)
                top = max(e["pc"], key=e["pc"].get)
                tcl = {cms[p][l] for l in tru if l in cms[p]}
                for c in e["pc"]:
                    prior = [t for dd in range(d) for t in co[c][dd]]
                    dist = min([abs(t - ss) for t in prior], default=288)
                    rows.append(dict(panel=p, split=s, d=d, s=ss, c=c, pc=e["pc"][c], top=c == top, true=c in tcl,
                                     dist=dist, n_prior=len(prior)))
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_onset_timing.csv", index=False)
    ev = r.groupby(["panel", "split", "d", "s"])
    wrong = ev.apply(lambda g: not g[g.top].true.any()).rename("wrong").reset_index()
    print("events (day >= 2):", len(wrong), "wrong top cluster:", int(wrong.wrong.sum()))
    w = r.merge(wrong, on=["panel", "split", "d", "s"])
    ww = w[w.wrong]
    cmp = []
    for k, g in ww.groupby(["panel", "split", "d", "s"]):
        dt, dp = g[g.true].dist.min(), g[g.top].dist.min()
        cmp.append((dt, dp))
    cmp = np.array(cmp)
    print("wrong events: true-cluster prior-onset distance < predicted-cluster distance:", int((cmp[:, 0] < cmp[:, 1]).sum()),
          "equal:", int((cmp[:, 0] == cmp[:, 1]).sum()), "greater:", int((cmp[:, 0] > cmp[:, 1]).sum()))
    print("median distance (slots): true", np.median(cmp[:, 0]), "predicted", np.median(cmp[:, 1]))
    # in correct events, how often would the timing rule flip a correct pick?
    ok = w[~w.wrong]
    flips = 0
    for k, g in ok.groupby(["panel", "split", "d", "s"]):
        if g[g.top].dist.min() > g.dist.min() + 6:
            flips += 1
    print("correct events where another cluster is >30 min closer in prior timing:", flips, "of", ok.groupby(["panel", "split", "d", "s"]).ngroups)


if __name__ == "__main__":
    main()
