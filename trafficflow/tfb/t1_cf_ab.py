"""A/B: Task 1 residual model with vs without the corridor common-mode features (cf_*), same rows.
Usage: python -m tfb.t1_cf_ab   (needs t1_tr/t1_va frames built with TFB_CF=1)"""
import numpy as np
import pandas as pd

from .data import CACHE
from . import t1_train as tt
from .t1_model import feat_cols

CFC = ["cf_sp", "cf_sp_rel", "cf_sp_n", "cf_sp_loc", "cf_fl", "cf_fl_loc", "plat"]

if __name__ == "__main__":
    tr = pd.read_parquet(CACHE / "t1_tr.parquet")
    va = pd.read_parquet(CACHE / "t1_va.parquet")
    tt.evaluate(va, va.speed_lin.to_numpy(), va.flow_lin.to_numpy(), "linear interp")
    res = {}
    for name, drop in (("base", CFC), ("cf", [])):
        a, b = tr.drop(columns=drop), va.drop(columns=drop)
        models = tt.train(a, b, 3000)
        s, f = tt.apply(models, b)
        res[name] = (s, f)
        tt.evaluate(b, s, f, name)
        np.save(CACHE / f"t1ab_{name}.npy", np.stack([s, f]))
    ys, yf, ln = va.y_speed.to_numpy(), va.y_flow.to_numpy(), va.lanes.to_numpy()
    for name, (s, f) in res.items():
        print(name, "RMSE speed", np.sqrt(np.mean((s - ys) ** 2)).round(4), "flow/lane", np.sqrt(np.mean(((f - yf) / ln) ** 2)).round(3))
