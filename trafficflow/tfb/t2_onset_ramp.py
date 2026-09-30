"""Onset with ramp-demand features (first principles: a bottleneck activates when mainline + on-ramp demand exceeds
its capacity, and ramp flows are published observations that do not depend on the scenario's link identities).
Per candidate link, from the visible history T-60..T-5 only: on-ramp flow within 2 links upstream (last, mean, trend),
off-ramp flow within 2 links downstream (last, mean), and demand ratio (mainline + on-ramp) / capacity.
Compares v9b structure without/with these features, in scenario (train 2-fold) and out of scenario (val/private masked
-layer events, evaluation only). Usage: python -m tfb.t2_onset_ramp
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_eval_big import iou_elig
from .t2_onset_shift import SPLITS, event_rows, fit_general, mined_events, predict

RP = ["rp_on_last", "rp_on_mean", "rp_on_trend", "rp_off_last", "rp_off_mean", "rp_dem"]


def ramp_index(p, z):
    net = network(p)
    ri = {r: i for i, r in enumerate(z["ramps"])}
    on = [[ri[x] for x in str(s).split(";") if x in ri] for s in net.on_ramp_link_ids.fillna("")]
    off = [[ri[x] for x in str(s).split(";") if x in ri] for s in net.off_ramp_link_ids.fillna("")]
    return on, off, net.capacity_vph.to_numpy()


def ramp_feats(z, d, T, links, idx, flow_key):
    on, off, cap = idx
    L = len(on)
    rf = z["ramp_flow"][d, T - 12:T]                         # visible history rows only (T-60..T-5)
    mf = pd.DataFrame(z[flow_key][d, T - 12:T]).ffill().to_numpy()[-1]
    def agg(sets, lo_off, hi_off, l):
        ids = [i for j in range(max(0, l + lo_off), min(L, l + hi_off + 1)) for i in sets[j]]
        if not ids:
            return np.zeros(12)
        return np.nansum(rf[:, ids], axis=1)
    rows = []
    for l in links:
        a = agg(on, -2, 0, l)
        b = agg(off, 0, 2, l)
        rows.append({"rp_on_last": a[-1], "rp_on_mean": a.mean(), "rp_on_trend": a[-3:].mean() - a[:3].mean(),
                     "rp_off_last": b[-1], "rp_off_mean": b.mean(),
                     "rp_dem": (np.nan_to_num(mf[l]) + a[-1]) / max(cap[l], 1.0)})
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    parts = []
    for p in T2P:
        z = load(p, "train")
        idx = ramp_index(p, z)
        g = df[df.panel == p]
        for (d, s), gg in g.groupby(["d", "s"]):
            parts.append(pd.concat([gg.reset_index(drop=True),
                                    ramp_feats(z, d, s - 6, gg.link.to_numpy(), idx, "flow")], axis=1))
    dr = pd.concat(parts, ignore_index=True)
    print("frame", len(dr), dr[RP].describe().loc[["mean", "std"]].round(1).to_dict(), flush=True)
    variants = {"base": dr.drop(columns=RP), "ramp": dr}
    models = {k: (fit_general(v), {f: fit_general(v[v.d % 2 != f]) for f in (0, 1)}) for k, v in variants.items()}
    print("fitted", flush=True)
    rows = []
    for p in T2P:
        for s in SPLITS:
            z = load(p, s)
            idx = ramp_index(p, z)
            for d, ss, tru, el in mined_events(p, s, z):
                X = event_rows(p, z, d, ss - 6, cands[p], cms[p])
                Xr = pd.concat([X.reset_index(drop=True), ramp_feats(z, d, ss - 6, X.link.to_numpy(), idx, "m_flow")], axis=1)
                r = dict(panel=p, split=s)
                for k, (full, folds) in models.items():
                    m = folds[d % 2] if s == "train" else full
                    r[k] = iou_elig(predict(m, Xr if k == "ramp" else X), tru, el)
                rows.append(r)
        print(p, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_onset_ramp.csv", index=False)
    print(r.groupby(["split", "panel"])[["base", "ramp"]].mean().groupby("split").mean().T.round(4))
    dd = r.ramp - r.base
    o = r.split != "train"
    print("ramp - base: oos %+.4f se %.4f" % (dd[o].mean(), dd[o].std() / np.sqrt(o.sum())),
          dd.groupby(r.split).mean().round(4).to_dict())


if __name__ == "__main__":
    main()
