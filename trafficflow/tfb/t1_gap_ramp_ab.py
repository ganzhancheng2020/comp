"""A/B for the Task 1 gap specialist (cells inside synthetic Task 2 blackouts): without vs with ramp features.
Frames: train days d%4!=0 (60k gap cells per panel) / d%4==0 (30k), built with TFB_CF=1 and ramps=True.
Usage: TFB_CF=1 python -m tfb.t1_gap_ramp_ab"""
import numpy as np
import pandas as pd

from .data import CACHE, panels
from .t1_model import panel_frame
from .t1_train import evaluate, targets_of, PARAMS
from .t1_model import feat_cols
import lightgbm as lgb

RAMPC = lambda cols: [c for c in cols if c.startswith(("ron", "roff", "cons_"))]

if __name__ == "__main__":
    ptr, pva = CACHE / "t1_gap_ramp_tr.parquet", CACHE / "t1_gap_ramp_va.parquet"
    if not ptr.exists():
        tr, va = [], []
        for p in panels():
            tr.append(panel_frame(p, "train", lambda d: d % 4 != 0, 60_000, seed=3, gaps=True, ramps=True))
            va.append(panel_frame(p, "train", lambda d: d % 4 == 0, 30_000, seed=4, gaps=True, ramps=True))
            print(p, len(tr[-1]), len(va[-1]), flush=True)
        pd.concat(tr, ignore_index=True).to_parquet(ptr)
        pd.concat(va, ignore_index=True).to_parquet(pva)
    tr, va = pd.read_parquet(ptr), pd.read_parquet(pva)
    ys, yf, ln = va.y_speed.to_numpy(), va.y_flow.to_numpy(), va.lanes.to_numpy()
    res = {}
    for name in ("base", "ramp"):
        cols = feat_cols(tr)
        if name == "base":
            cols = [c for c in cols if c not in RAMPC(cols)]
        y, yv = targets_of(tr), targets_of(va)
        pred = {}
        for k in ("speed", "flow"):
            m = lgb.train(PARAMS, lgb.Dataset(tr[cols], y[k]), 3000, valid_sets=[lgb.Dataset(va[cols], yv[k])],
                          callbacks=[lgb.log_evaluation(500), lgb.early_stopping(100)])
            pred[k] = m.predict(va[cols])
        s = va.speed_lin.to_numpy() + pred["speed"]
        f = va.flow_lin.to_numpy() + pred["flow"] * ln
        eN = np.abs(va.len.to_numpy() * (f / np.maximum(s, 1) - yf / np.maximum(ys, 1))).sum()
        print(f"{name}: gap speed RMSE {np.sqrt(np.mean((s - ys) ** 2)):.3f}  flow/lane RMSE "
              f"{np.sqrt(np.mean(((f - yf) / ln) ** 2)):.2f}  sum|N err| {eN:.0f}", flush=True)
        evaluate(va, s, f, name)
