"""Paired A/B for onset: origin row missing (vis, V10a) vs origin row T from the masked layer (vt).
Same mined events as t2_onset_shift (train 2-fold by day parity; validation/private by the all-train model).
Usage: python -m tfb.t2_onset_vt_ab"""
import numpy as np
import pandas as pd

from . import t2_onset as on
from . import t2_onset2 as o2
from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_eval_big import iou_elig
from .t2_onset_shift import SPLITS, fit_general, mined_events, predict
from .t2_events import visible


def rows_for(p, z, d, T, cands, cm, t_row):
    net = network(p)
    vc = 0.6 * net.free_speed_kmh.to_numpy()
    if t_row:
        hs, hf = z["m_speed"][d, T - 12:T + 1].astype(float), z["m_flow"][d, T - 12:T + 1].astype(float)
    else:
        hs, hf = visible(z["m_speed"][d], T), visible(z["m_flow"][d], T)
    dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()[d]
    X = on.link_features(hs, hf, vc, net.capacity_vph.to_numpy(), T, cands, dow)
    X["pid"] = T2P.index(p)
    X["panel"], X["d"], X["s"] = p, d, T + 6
    X["cl"] = X.link.map(cm)
    return X


def main():
    Z = {(p, s): load(p, s) for p in T2P for s in SPLITS}
    EV = {(p, s): mined_events(p, s, Z[(p, s)]) for p in T2P for s in SPLITS}
    out = []
    for name, frame, t_row in (("vis", "t2_onset.parquet", False), ("vt", "t2_onset_vt.parquet", True)):
        df = pd.read_parquet(CACHE / frame)
        df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
        df = o2.add_cluster(df)
        cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
        cms = {p: o2.clusters(cands[p]) for p in T2P}
        full = fit_general(df)
        folds = {f: fit_general(df[df.d % 2 != f]) for f in (0, 1)}
        for (p, s), evs in EV.items():
            for i, (d, ss, tru, el) in enumerate(evs):
                m = folds[d % 2] if s == "train" else full
                X = rows_for(p, Z[(p, s)], d, ss - 6, cands[p], cms[p], t_row)
                out.append(dict(variant=name, panel=p, split=s, i=i, iou=iou_elig(predict(m, X), tru, el)))
        print(name, "done", flush=True)
    r = pd.DataFrame(out)
    r.to_csv(CACHE / "t2_onset_vt_ab.csv", index=False)
    print(r.groupby(["variant", "split", "panel"]).iou.mean().groupby(["variant", "split"]).mean().unstack().round(4))
    w = r.pivot_table(index=["split", "panel", "i"], columns="variant", values="iou").reset_index()
    w["dd"] = w.vt - w.vis
    pm = w.groupby(["split", "panel"]).dd.mean()
    print("panel-mean delta vt - vis:", pm.groupby("split").mean().round(4).to_dict())
    for s, g in w.groupby("split"):
        print(f"[{s}] event-level {g.dd.mean():+.4f} ± {g.dd.std() / np.sqrt(len(g)):.4f} n={len(g)}")
    print(pm.unstack("split").round(3))


if __name__ == "__main__":
    main()
