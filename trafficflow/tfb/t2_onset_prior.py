"""Round 11: onset under scenario shift as label-shift adaptation (Saerens et al. 2002; Alexandari et al. 2020).

The onset model's cluster probabilities carry the TRAIN scenario's prior of which bottleneck cluster switches on near a
given time of day. Validation/private redraw that schedule. Instead of feeding scenario evidence to the model as features
(rounds 2 and 9: in-scenario training teaches the model to ignore them), the posterior odds are corrected post hoc by
the prior ratio, estimated causally from the split's own earlier days (all data before the window's day; allowed under
the Task 2 ruling, timestamps < T):

    rate_src(c, t) = share of train days with an onset of cluster c within +-W slots of t
    rate_tgt(c, t) = (n_c + a * rate_src) / (k + a)        n_c of the k most recent earlier days of the split
    logit p'(c)    = logit p(c) + b * log(rate_tgt / rate_src)     (same shift for the old link model's links of c)

k is drawn from the real Task 2 window-day distribution (window day - 1, capped by the event's day), so the evidence
matches what real windows have. Judged on onset events mined from the validation/private masked layer with the
production model and decoder (V13: fit_general on t2_onset.parquet, decode_pool, w = 0); paired against b = 0.
Usage: python -m tfb.t2_onset_prior
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, REL, load, network
from .t2 import T2_PANELS as T2P
from .t2_eval_big import iou_elig
from .t2_onset_shift import event_rows, fit_general, mined_events
from .t2_onset_shift_cal import probs
from .t2_onset_shift_ens import decode_pool
from .t2_onset_timing import cluster_onsets

SPLITS = ("validation", "private")
GRID = dict(W=(6, 12), a=(1.0, 3.0), b=(0.5, 1.0))
EPS = 0.02


def window_k():
    """Earlier days available to the real Task 2 windows (day of month - 1), both splits."""
    wi = pd.concat([pd.read_csv(p) for p in REL.glob("task2/*/*/window_index.csv")])
    return (pd.to_datetime(wi.forecast_origin, utc=True).dt.day - 1).to_numpy()


def near(ons_days, t, W):
    """Per day: 1 if the cluster has an onset within +-W slots of t."""
    return np.array([any(abs(x - t) <= W for x in day) for day in ons_days], float)


def adjust(e, cm, r, b):
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    sg = lambda x: 1 / (1 + np.exp(-x))
    sh = {c: b * np.log(r[c]) for c in e["pc"]}
    out = dict(e)
    out["pc"] = {c: float(sg(lg(p) + sh[c])) for c, p in e["pc"].items()}
    out["po"] = sg(lg(e["po"]) + np.array([sh.get(c, 0.0) for c in e["cl"]]))
    return out


def main():
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    model = fit_general(df)                                   # the production onset model (V13, w = 0)
    ks = window_k()
    rng = np.random.default_rng(11)
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    rows = []
    for p in T2P:
        vc = 0.6 * network(p).free_speed_kmh.to_numpy()
        co_src = cluster_onsets(load(p, "train"), vc, cms[p])
        for s in SPLITS:
            z = load(p, s)
            co = cluster_onsets(z, vc, cms[p])
            for i, (d, ss, tru, el) in enumerate(mined_events(p, s, z)):
                e = probs(model, event_rows(p, z, d, ss - 6, cands[p], cms[p]))
                k = int(min(d, rng.choice(ks)))
                base = iou_elig(decode_pool([e], [1.0]), tru, el)
                rec = dict(panel=p, split=s, d=d, s=ss, k=k, base=base)
                for cfg in configs:
                    W, a, b = cfg["W"], cfg["a"], cfg["b"]
                    r = {}
                    for c in e["pc"]:
                        rs = near(co_src[c], ss, W).mean() + EPS
                        n = near(co[c][d - k:d], ss, W).sum() if k else 0.0
                        r[c] = ((n + a * rs) / (k + a)) / rs
                    rec[f"W{W}_a{a:g}_b{b:g}"] = iou_elig(decode_pool([adjust(e, cms[p], r, b)], [1.0]), tru, el)
                rows.append(rec)
        print(p, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_onset_prior.csv", index=False)
    report(r)


def report(r):
    cols = [c for c in r.columns if c.startswith("W")]
    print("events", r.groupby("split").size().to_dict(), "base IoU", r.groupby("split").base.mean().round(4).to_dict())
    out = []
    for c in cols:
        rec = {"cfg": c}
        for s, g in r.groupby("split"):
            dd = g[c] - g.base
            pan = g.assign(dd=dd).groupby("panel").dd.mean()
            rec[s[:3]] = f"{dd.mean():+.4f} ± {dd.std() / np.sqrt(len(dd)):.4f} ({int((pan > 0).sum())}+/{int((pan < 0).sum())}-)"
        out.append(rec)
    print(pd.DataFrame(out).to_string(index=False))
    for s, g in r.groupby("split"):
        print(s, "by k (k>0 only matters):", g.groupby(np.minimum(g.k, 5)).size().to_dict())


if __name__ == "__main__":
    main()
