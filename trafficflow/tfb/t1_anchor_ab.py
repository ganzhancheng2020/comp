"""Speed-only A/B on the cf frames: residual over speed_lin (current) vs residual over a physics anchor.
Anchor: plateau + corridor common mode (cf_sp) where the temporal interpolation looks free-flowing
(speed_lin > 0.85 plateau), else speed_lin. The generator's free-flow branch is flat, so the anchor removes the
neighbour noise that speed_lin carries and leaves the model the idiosyncratic noise plus regime corrections.
Usage: python -m tfb.t1_anchor_ab"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CACHE
from .t1_model import feat_cols
from .t1_train import PARAMS


def anchor(df):
    ff = df.speed_lin.to_numpy() > 0.85 * df.plat.to_numpy()
    a = df.plat.to_numpy() + np.nan_to_num(df.cf_sp.to_numpy())
    return np.where(ff, a, df.speed_lin.to_numpy())


if __name__ == "__main__":
    tr = pd.read_parquet(CACHE / "t1_tr.parquet")
    va = pd.read_parquet(CACHE / "t1_va.parquet")
    for d in (tr, va):
        d["anc"] = anchor(d)
        d["lin_m_anc"] = d.speed_lin - d.anc
    ys = va.y_speed.to_numpy()
    cong = ys < 0.8 * va.plat.to_numpy()
    print("anchor alone RMSE", np.sqrt(np.mean((va.anc - ys) ** 2)).round(4),
          "free", np.sqrt(np.mean((va.anc - ys)[~cong] ** 2)).round(4))
    for name, base in (("lin", "speed_lin"), ("anchor", "anc")):
        X, Xv = tr[feat_cols(tr)], va[feat_cols(va)]
        m = lgb.train(PARAMS, lgb.Dataset(X, tr.y_speed - tr[base]), 3000,
                      valid_sets=[lgb.Dataset(Xv, va.y_speed - va[base])],
                      callbacks=[lgb.log_evaluation(500), lgb.early_stopping(100)])
        s = va[base].to_numpy() + m.predict(Xv)
        np.save(CACHE / f"t1anc_{name}.npy", s)
        m.save_model(str(CACHE / f"t1anc_{name}.txt"))
        r = lambda msk: np.sqrt(np.mean((s[msk] - ys[msk]) ** 2))
        print(f"{name}: speed RMSE all {r(np.ones(len(s), bool)):.4f} free {r(~cong):.4f} congested {r(cong):.4f}", flush=True)
