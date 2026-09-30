"""Train / evaluate the Task 1 residual models on the cached feature frames."""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CACHE, families
from .score import state_score
from .t1_model import feat_cols

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
              num_threads=4)


def targets_of(df):
    return {"speed": df.y_speed - df.speed_lin, "flow": (df.y_flow - df.flow_lin) / df.lanes}


def apply(models, df):
    X = df[feat_cols(df)]
    s = df.speed_lin + models["speed"].predict(X)
    f = df.flow_lin + models["flow"].predict(X) * df.lanes
    return s.to_numpy(), f.to_numpy()


def evaluate(df, s, f, label=""):
    fam = families()
    rows = []
    for p, g in df.groupby("panel"):
        m = df.panel.to_numpy() == p
        r = state_score(s[m], f[m], g.y_speed.to_numpy(), g.y_flow.to_numpy(), g.lanes.to_numpy(),
                        g.regime.to_numpy())
        rows.append((p, fam[p], r["S_state"]))
    r = pd.DataFrame(rows, columns=["panel", "family", "S"])
    total = r.groupby("family").S.mean().mean()
    print(f"{label:24s} S_state={total:.5f}  " + " ".join(f"{p[-6:]}={v:.4f}" for p, _, v in rows), flush=True)
    return total


def train(tr, va=None, rounds=2000):
    X = tr[feat_cols(tr)]
    y = targets_of(tr)
    models = {}
    for k in ("speed", "flow"):
        dtr = lgb.Dataset(X, y[k])
        valid = []
        if va is not None:
            valid = [lgb.Dataset(va[feat_cols(va)], targets_of(va)[k], reference=dtr)]
        models[k] = lgb.train(PARAMS, dtr, rounds, valid_sets=valid,
                              callbacks=[lgb.log_evaluation(250)] + ([lgb.early_stopping(100)] if valid else []))
    return models


if __name__ == "__main__":
    tr = pd.read_parquet(CACHE / "t1_tr.parquet")
    va = pd.read_parquet(CACHE / "t1_va.parquet")
    evaluate(va, va.speed_lin.to_numpy(), va.flow_lin.to_numpy(), "linear interp")
    models = train(tr, va, int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
    s, f = apply(models, va)
    evaluate(va, s, f, "lgbm residual")
    for k, m in models.items():
        m.save_model(str(CACHE / f"t1_{k}.txt"))
        imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
        print(k, (imp / imp.sum()).head(15).round(3).to_dict())
