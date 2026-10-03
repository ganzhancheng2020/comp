"""Production transductive Task 1 correction (Round 10): per split, a correction booster on top of the all-data main
models, trained on the split's own published cells (random observed cells hidden, features rebuilt without them,
label = published value). `t1_predict` adds it to the main-model rows when TFB_T1_TX=1.
Usage: TFB_CF=1 TFB_SCEN=split python -m tfb.t1_tx_prod [frames|fit]
"""
import gc
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, load, panels
from .t1_tx import BASE, FRAC, LR, _blank, _f32

SPLITS = ("validation", "private")
PASSES = int(os.environ.get("TFB_TX_PASSES", "6"))
ROUNDS = int(os.environ.get("TFB_TX_ROUNDS", "300"))
DIR = CACHE / "t1_txp"


def frame(p, s, seed):
    f = DIR / f"H{seed}_{p}_{s}.parquet"
    if f.exists():
        return
    z = load(p, s)
    rng = np.random.default_rng(1000 + seed)
    obs = np.isfinite(z["m_speed"]) & np.isfinite(z["m_flow"]) & (z["pct"] >= 75)
    d, t, l = np.where(obs & ~np.isnan(z["m_speed"]).all(2)[:, :, None])
    k = np.sort(rng.choice(len(d), int(len(d) * FRAC), replace=False))
    d, t, l = d[k], t[k], l[k]
    zh = _blank(z, d, t, l)
    zh["_plat"] = FE.plateau_of(zh["m_speed"])
    X = FE.build(p, zh, (d, t, l), prof=FE.profile(p, "train"), ramps=False)
    X["y_speed"], X["y_flow"] = z["m_speed"][d, t, l], z["m_flow"][d, t, l]
    _f32(X).to_parquet(f)
    print("frame", f.name, len(X), flush=True)


def frames():
    DIR.mkdir(exist_ok=True)
    FE.SCEN = "split"
    for s in SPLITS:
        for p in panels():
            for seed in range(1, PASSES + 1):
                frame(p, s, seed)
            gc.collect()


def fit():
    from .t1_train import PARAMS
    for k in ("speed", "flow"):
        base = lgb.Booster(model_file=str(CACHE / f"{BASE[k]}.txt"))
        cols = base.feature_name()
        for s in SPLITS:
            path = CACHE / f"t1txp_{k}_{s}.txt"
            if path.exists():
                continue
            X = pd.concat([pd.read_parquet(DIR / f"H{seed}_{p}_{s}.parquet") for seed in range(1, PASSES + 1)
                           for p in panels()], ignore_index=True)
            y = ((X.y_speed - X.speed_lin) if k == "speed" else (X.y_flow - X.flow_lin) / X.lanes).to_numpy(np.float32)
            init = base.predict(X[cols]).astype(np.float32)
            ds = lgb.Dataset(X[cols].to_numpy(np.float32), y, init_score=init, feature_name=cols, free_raw_data=True)
            del X; gc.collect()
            lgb.train({**PARAMS, "learning_rate": LR}, ds, ROUNDS).save_model(str(path))
            print("fit", k, s, len(y), flush=True)
            del ds; gc.collect()


if __name__ == "__main__":
    {"frames": frames, "fit": fit}[sys.argv[1]]()
