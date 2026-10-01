"""Out-of-scenario Task 1 evaluator: hide extra observed cells of the validation/private masked layer and predict them.

Validation/private are different scenarios (per-link free-flow plateaus shift by sd ~1.3 km/h, ramp demand changes), so
the train-day holdout cannot see scenario shift. Here: 3% of the observed eligible cells of each split are hidden
(evaluation only; never used for fitting), features are rebuilt with them hidden, and models are scored against the
published values. Also simulates Task 2 blackouts (all links blank for 18 slots) at random origins to score the gap model.
Usage: TFB_CF=1 python -m tfb.t1_shift_eval
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, load, panels
from .t1_model import add_gaps, feat_cols
from .t1_train import PARAMS, targets_of

SPLITS = ("validation", "private")


def hidden_frame(p, s, frac=0.03, seed=0, gaps=False):
    z = load(p, s)
    rng = np.random.default_rng(seed)
    obs = np.isfinite(z["m_speed"]) & np.isfinite(z["m_flow"]) & (z["pct"] >= 75)
    D, T, L = obs.shape
    if gaps:      # synthetic blackouts: 18 slots on every link after random origins (one per day), cells inside scored
        orig = [(d, int(rng.integers(12 * 6, 288 - 30))) for d in range(D)]
        zz, in_gap = add_gaps(z, orig)
        m = obs & in_gap[:, :, None]
        d, t, l = np.where(m)
        k = rng.choice(len(d), min(len(d), 20000), replace=False)
    else:
        d, t, l = np.where(obs & ~np.isnan(z["m_speed"]).all(2)[:, :, None])
        k = rng.choice(len(d), int(len(d) * frac), replace=False)
        zz = dict(z)
        for key in ("m_speed", "m_flow", "m_occ"):
            a = z[key].copy(); a[d[k], t[k], l[k]] = np.nan; zz[key] = a
    d, t, l = d[k], t[k], l[k]
    o = np.lexsort((l, t, d)); d, t, l = d[o], t[o], l[o]
    X = FE.build(p, zz, (d, t, l), ramps=gaps)
    X["y_speed"], X["y_flow"] = z["m_speed"][d, t, l], z["m_flow"][d, t, l]
    X["panel"], X["split"] = p, s
    return X


def score(df, s, f):
    out = {}
    for (sp, p), g in df.assign(es=(s - df.y_speed) ** 2, ef=((f - df.y_flow) / df.lanes) ** 2).groupby(["split", "panel"]):
        out[(sp, p)] = (np.sqrt(g.es.mean()), np.sqrt(g.ef.mean()))
    r = pd.DataFrame(out, index=["rmse_speed", "rmse_flow"]).T
    return r.groupby(level=0).mean()


CFC = ["cf_sp", "cf_sp_rel", "cf_sp_n", "cf_sp_loc", "cf_fl", "cf_fl_loc", "plat"]


def base_speed_model():
    path = CACHE / "t1p_speed_base.txt"
    if not path.exists():
        tr = pd.read_parquet(CACHE / "t1_full_cf_300000.parquet")
        cols = [c for c in feat_cols(tr) if c not in CFC]
        m = lgb.train(PARAMS, lgb.Dataset(tr[cols], targets_of(tr)["speed"]), 4000)
        m.save_model(str(path))
        del tr
    return lgb.Booster(model_file=str(path))


def pred(m, X, kind):
    v = m.predict(X[m.feature_name()])
    return X.speed_lin.to_numpy() + v if kind == "speed" else X.flow_lin.to_numpy() + v * X.lanes.to_numpy()


def build_all(gaps, scen, prof):
    FE.SCEN, FE.PROF_SCEN = scen, prof
    return pd.concat([hidden_frame(p, s, gaps=gaps) for p in panels() for s in SPLITS], ignore_index=True)


def main():
    mb = base_speed_model()
    mc = lgb.Booster(model_file=str(CACHE / "t1p_speed.txt"))
    mf = lgb.Booster(model_file=str(CACHE / "t1p_flow.txt"))
    rows = {}
    for tag, scen, prof in (("trainplat", "train", False), ("splitplat", "split", False), ("splitprof", "split", True)):
        X = build_all(False, scen, prof)
        f = pred(mf, X, "flow")
        for name, m in (("base", mb), ("cf", mc)):
            r = score(X, pred(m, X, "speed"), f)
            rows[(tag, name)] = r
            print(tag, name, r.round(4).to_dict(), flush=True)
    print("== hidden-cell RMSE (mean over panels), validation / private")
    for k, r in rows.items():
        print(k, " ".join(f"{s}: speed {r.loc[s, 'rmse_speed']:.4f} flow {r.loc[s, 'rmse_flow']:.3f}" for s in r.index))
    # gap specialist under scenario shift
    g_old = {k: lgb.Booster(model_file=str(CACHE / f"t1p_gap_{k}.txt")) for k in ("speed", "flow")}
    g_new = {k: lgb.Booster(model_file=str(CACHE / f"t1p_gapr_{k}.txt")) for k in ("speed", "flow")}
    for tag, scen in (("trainprof", "train"), ("splitprof", "split")):
        X = build_all(True, scen, False)
        for name, g in (("gap_old", g_old), ("gap_ramp", g_new)):
            r = score(X, pred(g["speed"], X, "speed"), pred(g["flow"], X, "flow"))
            print("GAP", tag, name, " ".join(f"{s}: speed {r.loc[s, 'rmse_speed']:.3f} flow {r.loc[s, 'rmse_flow']:.2f}" for s in r.index), flush=True)


if __name__ == "__main__":
    main()
