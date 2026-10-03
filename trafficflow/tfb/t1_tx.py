"""Round 10: transductive Task 1 (domain adaptation on the split's own published cells), judged out of scenario.

Task 1 is offline and may use any released observation of its split (organizer ruling, forum topic 742068). Task 1
targets are Bernoulli cells (rate 0.2/0.3/0.5 by day regime), so hiding further random observed cells of the
validation/private masked layer gives in-scenario training rows with the same structure: features rebuilt without the
hidden cells, label = the published value. A correction booster is fitted on top of the V13 main model (its predictions
as init_score) and judged on the evaluator's 3% hidden cells E (`t1_shift_eval.hidden`, seed 0). E stays hidden in
every training frame and in the split statistics, so E never enters a feature or a label.

Variants per target (speed, flow): correction trained on validation frames (M_val) or private frames (M_pri), each
scored on both splits -> own split (transductive) and the other split (generic adaptation to a new scenario).
Usage: TFB_CF=1 TFB_SCEN=split python -m tfb.t1_tx [frames|fit|eval|repro]
"""
import gc
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, panels
from .t1_shift_eval import hidden

SPLITS = ("validation", "private")
DIR = CACHE / "t1_tx"
FRAC, SEEDS = 0.06, (1, 2)
LR, ROUNDS, CKPTS = 0.03, 1500, (100, 300, 600, 1000, 1500)
BASE = {"speed": "t1a_speed", "flow": "t1a_flow"}
RES = CACHE / "t1_tx_res.json"


def _blank(z, d, t, l):
    zz = dict(z)
    for key in ("m_speed", "m_flow", "m_occ"):
        a = z[key].copy(); a[d, t, l] = np.nan; zz[key] = a
    return zz


def eval_frame(p, s):
    f = DIR / f"E_{p}_{s}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    z, zz, idx = hidden(p, s, False)
    zz["_plat"] = FE.plateau_of(zz["m_speed"])
    X = FE.build(p, zz, idx, prof=FE.profile(p, "train"), ramps=False)
    X["y_speed"], X["y_flow"] = z["m_speed"][idx], z["m_flow"][idx]
    _f32(X).to_parquet(f)
    return X


def train_frame(p, s, seed):
    f = DIR / f"H{seed}_{p}_{s}.parquet"
    if f.exists():
        return
    z, zz, _ = hidden(p, s, False)                       # E blank, exactly as in the evaluator
    rng = np.random.default_rng(1000 + seed)
    obs = np.isfinite(zz["m_speed"]) & np.isfinite(zz["m_flow"]) & (zz["pct"] >= 75)
    d, t, l = np.where(obs & ~np.isnan(zz["m_speed"]).all(2)[:, :, None])
    k = np.sort(rng.choice(len(d), int(len(d) * FRAC), replace=False))
    d, t, l = d[k], t[k], l[k]
    zh = _blank(zz, d, t, l)
    zh["_plat"] = FE.plateau_of(zh["m_speed"])          # hidden labels never enter their own statistics
    X = FE.build(p, zh, (d, t, l), prof=FE.profile(p, "train"), ramps=False)
    X["y_speed"], X["y_flow"] = z["m_speed"][d, t, l], z["m_flow"][d, t, l]
    X["panel"] = p
    _f32(X).to_parquet(f)
    print("frame", f.name, len(X), flush=True)


def _f32(X):
    f64 = X.select_dtypes("float64").columns
    X[f64] = X[f64].astype(np.float32)
    return X


def frames():
    DIR.mkdir(exist_ok=True)
    FE.SCEN = "split"
    for s in SPLITS:
        for p in panels():
            eval_frame(p, s)
            for seed in SEEDS:
                train_frame(p, s, seed)
            gc.collect()


def _target(X, k, base):
    pred = base.predict(X[base.feature_name()])
    y = (X.y_speed - X.speed_lin) if k == "speed" else (X.y_flow - X.flow_lin) / X.lanes
    return y.to_numpy(np.float32), pred.astype(np.float32)


def fit():
    from .t1_train import PARAMS
    for k in ("speed", "flow"):
        base = lgb.Booster(model_file=str(CACHE / f"{BASE[k]}.txt"))
        cols = base.feature_name()
        for s in SPLITS:
            path = CACHE / f"t1tx_{k}_{s}.txt"
            if path.exists():
                continue
            X = pd.concat([pd.read_parquet(f) for f in sorted(DIR.glob(f"H*_*_{s}.parquet"))], ignore_index=True)
            y, init = _target(X, k, base)
            ds = lgb.Dataset(X[cols].to_numpy(np.float32), y, init_score=init, feature_name=cols, free_raw_data=True)
            del X; gc.collect()
            m = lgb.train({**PARAMS, "learning_rate": LR}, ds, ROUNDS)
            m.save_model(str(path))
            print("fit", k, s, len(y), flush=True)
            del ds, m; gc.collect()


def evaluate():
    res = json.loads(RES.read_text()) if RES.exists() else {}
    B = {k: lgb.Booster(model_file=str(CACHE / f"{BASE[k]}.txt")) for k in BASE}
    C = {(k, s): lgb.Booster(model_file=str(CACHE / f"t1tx_{k}_{s}.txt")) for k in BASE for s in SPLITS}
    for s in SPLITS:
        for p in panels():
            key = f"{p}|{s}"
            if key in res:
                continue
            X = eval_frame(p, s)
            ln = X.lanes.to_numpy()
            ys, yf = X.y_speed.to_numpy(), X.y_flow.to_numpy()
            bs = X.speed_lin.to_numpy() + B["speed"].predict(X[B["speed"].feature_name()])
            bf = X.flow_lin.to_numpy() + B["flow"].predict(X[B["flow"].feature_name()]) * ln
            r = {"base": [float(np.mean((bs - ys) ** 2)), float(np.mean(((bf - yf) / ln) ** 2))]}
            for src in SPLITS:
                tag = "own" if src == s else "cross"
                for n in CKPTS:
                    cs = C[("speed", src)].predict(X[C[("speed", src)].feature_name()], num_iteration=n)
                    cf = C[("flow", src)].predict(X[C[("flow", src)].feature_name()], num_iteration=n)
                    r[f"{tag}{n}"] = [float(np.mean((bs + cs - ys) ** 2)), float(np.mean(((bf + cf * ln - yf) / ln) ** 2))]
            res[key] = r
            RES.write_text(json.dumps(res))
            print(key, {n: (round(v[0] ** .5, 3), round(v[1] ** .5, 2)) for n, v in r.items()}, flush=True)
    report(res)


def repro():
    """Production all-data models (t1a) vs the ones rebuilt by reproduce.sh (copied to t1r_*), on the same cells E."""
    out = CACHE / "t1_repro_res.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    M = {n: {k: lgb.Booster(model_file=str(CACHE / f"{pre}_{k}.txt")) for k in BASE} for n, pre in (("prod", "t1a"), ("repro", "t1r"))}
    for s in SPLITS:
        for p in panels():
            key = f"{p}|{s}"
            if key in res:
                continue
            X = eval_frame(p, s)
            ln = X.lanes.to_numpy()
            r = {}
            for n, m in M.items():
                sp = X.speed_lin.to_numpy() + m["speed"].predict(X[m["speed"].feature_name()])
                fl = X.flow_lin.to_numpy() + m["flow"].predict(X[m["flow"].feature_name()]) * ln
                r[n] = [float(np.mean((sp - X.y_speed.to_numpy()) ** 2)), float(np.mean(((fl - X.y_flow.to_numpy()) / ln) ** 2))]
            res[key] = r
            out.write_text(json.dumps(res))
    report(res)


def report(res):
    d = pd.DataFrame([dict(panel=k.split("|")[0], split=k.split("|")[1], model=n, speed=v[0] ** .5, flow=v[1] ** .5)
                      for k, r in res.items() for n, v in r.items()])
    print(d.groupby(["model", "split"])[["speed", "flow"]].mean().unstack("split").round(4).to_string())
    w = d.pivot_table(index=["panel", "split"], columns="model", values=["speed", "flow"])
    for n in sorted(set(d.model) - {"base"}):
        print(n, "panels better: speed", int((w["speed"][n] < w["speed"]["base"]).sum()), "flow",
              int((w["flow"][n] < w["flow"]["base"]).sum()), "/", len(w))


if __name__ == "__main__":
    {"frames": frames, "fit": fit, "eval": evaluate, "repro": repro}[sys.argv[1]]()
