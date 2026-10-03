"""Production transductive gap correction (Round 10b): per split, a correction booster on top of the gap specialist
(t1p_gapr_*), trained on synthetic blackouts (all links blank for 18 slots, one per day per pass, random origin) placed
on the split's own published layer; split profiles recomputed with the blackouts blank. `t1_predict` adds it to the
gap rows when TFB_T1_TXG=1.
Usage: TFB_CF=1 TFB_SCEN=split python -m tfb.t1_txgap_prod [frames|fit]
"""
import gc
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, load, panels
from .t1_model import add_gaps
from .t1_tx import _f32
from .t1_txgap import BASE, LR, SEEDS, _build

SPLITS = ("validation", "private")
ROUNDS = 200
DIR = CACHE / "t1_txgp"


def frame(p, s, seed):
    f = DIR / f"H{seed}_{p}_{s}.parquet"
    if f.exists():
        return
    z = load(p, s)
    rng = np.random.default_rng(2000 + seed)
    orig = [(d, int(rng.integers(72, 258))) for d in range(z["m_speed"].shape[0])]
    zh, in_gap = add_gaps(z, orig)
    obs = np.isfinite(z["m_speed"]) & np.isfinite(z["m_flow"]) & (z["pct"] >= 75)
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
            for seed in SEEDS:
                frame(p, s, seed)
            gc.collect()


def fit():
    from .t1_train import PARAMS
    for k in ("speed", "flow"):
        base = lgb.Booster(model_file=str(CACHE / f"{BASE[k]}.txt"))
        cols = base.feature_name()
        for s in SPLITS:
            path = CACHE / f"t1txgp_{k}_{s}.txt"
            if path.exists():
                continue
            X = pd.concat([pd.read_parquet(DIR / f"H{seed}_{p}_{s}.parquet") for seed in SEEDS for p in panels()],
                          ignore_index=True)
            y = ((X.y_speed - X.speed_lin) if k == "speed" else (X.y_flow - X.flow_lin) / X.lanes).to_numpy(np.float32)
            init = base.predict(X[cols]).astype(np.float32)
            ds = lgb.Dataset(X[cols].to_numpy(np.float32), y, init_score=init, feature_name=cols, free_raw_data=True)
            del X; gc.collect()
            lgb.train({**PARAMS, "learning_rate": LR}, ds, ROUNDS).save_model(str(path))
            print("fit", k, s, len(y), flush=True)


if __name__ == "__main__":
    {"frames": frames, "fit": fit}[sys.argv[1]]()
