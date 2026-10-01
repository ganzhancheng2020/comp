"""Out-of-scenario Task 1 evaluator: hide extra observed cells of the validation/private masked layer and predict them.

Validation/private are new scenarios (per-link free-flow plateaus move by sd ~1.3 km/h, profiles move), so the train-day
holdout cannot see scenario shift. 3% of the observed eligible cells of each split are hidden (evaluation only), and
every split statistic the features use (plateau, profile) is recomputed from the layer WITH those cells hidden (as in
production, where targets are blank). Synthetic blackouts (18 slots on every link) score the gap specialist.
Results are cached per (variant, panel, split), so the run survives container restarts.
Usage: TFB_CF=1 python -m tfb.t1_shift_eval
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, load, panels
from .t1_model import add_gaps, feat_cols
from .t1_train import PARAMS, targets_of

SPLITS = ("validation", "private")
CFC = ["cf_sp", "cf_sp_rel", "cf_sp_n", "cf_sp_loc", "cf_fl", "cf_fl_loc", "plat"]
RES = CACHE / "t1_shift_res.json"


def hidden(p, s, gaps, seed=0, frac=0.03):
    z = load(p, s)
    rng = np.random.default_rng(seed)
    obs = np.isfinite(z["m_speed"]) & np.isfinite(z["m_flow"]) & (z["pct"] >= 75)
    if gaps:
        orig = [(d, int(rng.integers(72, 258))) for d in range(obs.shape[0])]
        zz, in_gap = add_gaps(z, orig)
        d, t, l = np.where(obs & in_gap[:, :, None])
        k = rng.choice(len(d), min(len(d), 20000), replace=False)
    else:
        d, t, l = np.where(obs & ~np.isnan(z["m_speed"]).all(2)[:, :, None])
        k = rng.choice(len(d), int(len(d) * frac), replace=False)
        zz = dict(z)
        for key in ("m_speed", "m_flow", "m_occ"):
            a = z[key].copy(); a[d[k], t[k], l[k]] = np.nan; zz[key] = a
    d, t, l = d[k], t[k], l[k]
    o = np.lexsort((l, t, d))
    return z, zz, (d[o], t[o], l[o])


def frame(p, s, gaps, plat_mode, prof_mode):
    z, zz, idx = hidden(p, s, gaps)
    if plat_mode == "split":
        zz["_plat"] = FE.plateau_of(zz["m_speed"])            # leak-free: hidden cells excluded
    else:
        zz["_plat"] = FE.plateau(p, "train")
    prof = FE.profile_of(zz) if prof_mode == "split" else FE.profile(p, "train")
    FE.SCEN = "split" if plat_mode == "split" else "train"      # ramp profile follows the plateau mode
    X = FE.build(p, zz, idx, prof=prof, ramps=gaps)
    X["y_speed"], X["y_flow"] = z["m_speed"][idx], z["m_flow"][idx]
    return X


def base_speed_model():
    path = CACHE / "t1p_speed_base.txt"
    if not path.exists():
        tr = pd.read_parquet(CACHE / "t1_full_cf_300000.parquet")
        cols = [c for c in feat_cols(tr) if c not in CFC]
        lgb.train(PARAMS, lgb.Dataset(tr[cols], targets_of(tr)["speed"]), 4000).save_model(str(path))
    return lgb.Booster(model_file=str(path))


def pred(m, X, kind):
    v = m.predict(X[m.feature_name()])
    return X.speed_lin.to_numpy() + v if kind == "speed" else X.flow_lin.to_numpy() + v * X.lanes.to_numpy()


def main():
    res = json.loads(RES.read_text()) if RES.exists() else {}
    M = {"base": base_speed_model(), "cf": lgb.Booster(model_file=str(CACHE / "t1p_speed.txt")),
         "flow": lgb.Booster(model_file=str(CACHE / "t1p_flow.txt")),
         "g_old_s": lgb.Booster(model_file=str(CACHE / "t1p_gap_speed.txt")),
         "g_old_f": lgb.Booster(model_file=str(CACHE / "t1p_gap_flow.txt")),
         "g_new_s": lgb.Booster(model_file=str(CACHE / "t1p_gapr_speed.txt")),
         "g_new_f": lgb.Booster(model_file=str(CACHE / "t1p_gapr_flow.txt"))}
    variants = [("train", "train", False), ("split", "train", False), ("split", "split", False),
                ("train", "train", True), ("split", "train", True), ("split", "split", True)]
    for plat, prof, gaps in variants:
        for p in panels():
            for s in SPLITS:
                key = f"{'gap' if gaps else 'main'}|{plat}plat|{prof}prof|{p}|{s}"
                if key in res:
                    continue
                X = frame(p, s, gaps, plat, prof)
                ys, yf, ln = X.y_speed.to_numpy(), X.y_flow.to_numpy(), X.lanes.to_numpy()
                r = {}
                pairs = ((("old", "g_old_s", "g_old_f"), ("ramp", "g_new_s", "g_new_f")) if gaps
                         else (("base", "base", "flow"), ("cf", "cf", "flow")))
                for name, ms, mf in pairs:
                    sp, fl = pred(M[ms], X, "speed"), pred(M[mf], X, "flow")
                    r[name] = [float(np.mean((sp - ys) ** 2)), float(np.mean(((fl - yf) / ln) ** 2)), len(ys)]
                res[key] = r
                RES.write_text(json.dumps(res))
                print(key, {k: (round(v[0] ** .5, 3), round(v[1] ** .5, 2)) for k, v in r.items()}, flush=True)
    report(res)


def report(res):
    rows = []
    for key, r in res.items():
        kind, plat, prof, p, s = key.split("|")
        for name, (ms, mf, n) in r.items():
            rows.append(dict(kind=kind, variant=f"{plat}/{prof}", model=name, panel=p, split=s,
                             speed=ms ** .5, flow=mf ** .5))
    d = pd.DataFrame(rows)
    print(d.groupby(["kind", "variant", "model", "split"])[["speed", "flow"]].mean().unstack("split").round(4).to_string())


if __name__ == "__main__":
    main()
