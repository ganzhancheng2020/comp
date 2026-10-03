"""Round 10b: transductive gap specialist — correction trained on synthetic blackouts placed on the split's own layer.

Same idea as `tfb/t1_tx.py` for the Task 1 cells inside the Task 2 blackouts (every link blank for 18 slots). The
evaluator's blackouts (`t1_shift_eval.hidden(..., gaps=True)`: one per day, seed 0) stay blank in every training frame;
training blackouts are placed on the same days at origins that do not overlap them, the split profiles are recomputed
with all blackouts blank, and the label is the published value of the observed cells inside the training blackouts.
A correction booster is fitted on top of the production gap specialist (`t1p_gapr_*`) per split and judged on the
evaluator's cells (own split and other split, as in t1_tx).
Usage: TFB_CF=1 TFB_SCEN=split python -m tfb.t1_txgap [frames|fit|eval]
"""
import gc
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, panels
from .t1_model import GAP, add_gaps
from .t1_shift_eval import hidden
from .t1_tx import SPLITS, _f32, report

DIR = CACHE / "t1_txgap"
SEEDS = (1, 2, 3, 4)
LR, ROUNDS, CKPTS = 0.03, 1000, (100, 200, 400, 700, 1000)
BASE = {"speed": "t1p_gapr_speed", "flow": "t1p_gapr_flow"}
RES = CACHE / "t1_txgap_res.json"


def _eval_origins(p, s):
    """The evaluator's blackout origins, regenerated exactly as in t1_shift_eval.hidden(gaps=True)."""
    from .data import load
    z = load(p, s)
    rng = np.random.default_rng(0)
    return [(d, int(rng.integers(72, 258))) for d in range(z["m_speed"].shape[0])]


def _build(p, zz, idx):
    zz["_plat"] = FE.plateau_of(zz["m_speed"])
    return FE.build(p, zz, idx, prof=FE.profile_of(zz), ramps=True)


def eval_frame(p, s):
    f = DIR / f"E_{p}_{s}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    z, zz, idx = hidden(p, s, True)
    X = _build(p, zz, idx)
    X["y_speed"], X["y_flow"] = z["m_speed"][idx], z["m_flow"][idx]
    _f32(X).to_parquet(f)
    return X


def train_frame(p, s, seed):
    f = DIR / f"H{seed}_{p}_{s}.parquet"
    if f.exists():
        return
    z, zz, _ = hidden(p, s, True)                        # evaluator blackouts blank
    ev = dict(_eval_origins(p, s))
    rng = np.random.default_rng(2000 + seed)
    orig = []
    for d in range(zz["m_speed"].shape[0]):
        ok = [T for T in range(72, 258) if abs(T - ev[d]) > GAP]
        orig.append((d, int(rng.choice(ok))))
    zh, in_gap = add_gaps(zz, orig)
    obs = np.isfinite(zz["m_speed"]) & np.isfinite(zz["m_flow"]) & (zz["pct"] >= 75)
    d, t, l = np.where(obs & in_gap[:, :, None])
    X = _build(p, zh, (d, t, l))
    X["y_speed"], X["y_flow"] = z["m_speed"][d, t, l], z["m_flow"][d, t, l]
    _f32(X).to_parquet(f)
    print("frame", f.name, len(X), flush=True)


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
            path = CACHE / f"t1txg_{k}_{s}.txt"
            if path.exists():
                continue
            X = pd.concat([pd.read_parquet(DIR / f"H{seed}_{p}_{s}.parquet") for seed in SEEDS for p in panels()],
                          ignore_index=True)
            y, init = _target(X, k, base)
            ds = lgb.Dataset(X[cols].to_numpy(np.float32), y, init_score=init, feature_name=cols, free_raw_data=True)
            del X; gc.collect()
            lgb.train({**PARAMS, "learning_rate": LR}, ds, ROUNDS).save_model(str(path))
            print("fit", k, s, len(y), flush=True)


def evaluate():
    res = json.loads(RES.read_text()) if RES.exists() else {}
    B = {k: lgb.Booster(model_file=str(CACHE / f"{BASE[k]}.txt")) for k in BASE}
    C = {(k, s): lgb.Booster(model_file=str(CACHE / f"t1txg_{k}_{s}.txt")) for k in BASE for s in SPLITS}
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


if __name__ == "__main__":
    {"frames": frames, "fit": fit, "eval": evaluate}[sys.argv[1]]()
