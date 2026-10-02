"""Round 8a: does averaging LightGBM fits of different sizes help out of scenario? (all-data t1a, 1.2M t1c, 600k t1b)
Hidden cells of validation/private as in t1_shift_eval (split plateau, train profile). Usage: TFB_CF=1 python -m tfb.t1_ens_eval"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, panels
from .t1_shift_eval import SPLITS, hidden

OUT = CACHE / "t1_ens_res.json"
W = (0.0, 0.25, 0.5)            # weight on the second model

if __name__ == "__main__":
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    M = {pre: (lgb.Booster(model_file=str(CACHE / f"{pre}_speed.txt")), lgb.Booster(model_file=str(CACHE / f"{pre}_flow.txt")))
         for pre in ("t1a", "t1c", "t1b")}
    FE.SCEN = "split"
    for p in panels():
        for s in SPLITS:
            key = f"{p}|{s}"
            if key in res:
                continue
            z, zz, idx = hidden(p, s, False)
            zz["_plat"] = FE.plateau_of(zz["m_speed"])
            X = FE.build(p, zz, idx, prof=FE.profile(p, "train"))
            ys, yf, ln = z["m_speed"][idx], z["m_flow"][idx], X.lanes.to_numpy()
            P = {pre: (X.speed_lin.to_numpy() + ms.predict(X[ms.feature_name()]),
                       X.flow_lin.to_numpy() + mf.predict(X[mf.feature_name()]) * ln) for pre, (ms, mf) in M.items()}
            r = {}
            for other in ("t1c", "t1b"):
                for w in W:
                    sp = (1 - w) * P["t1a"][0] + w * P[other][0]
                    fl = (1 - w) * P["t1a"][1] + w * P[other][1]
                    r[f"{other}_{w}"] = [float(np.mean((sp - ys) ** 2)), float(np.mean(((fl - yf) / ln) ** 2))]
            res[key] = r
            OUT.write_text(json.dumps(res))
            print(key, {k: (round(v[0] ** .5, 3), round(v[1] ** .5, 2)) for k, v in r.items() if k.endswith(("0.0", "0.5"))}, flush=True)
    d = pd.DataFrame([dict(split=k.split("|")[1], model=n, speed=v[0] ** .5, flow=v[1] ** .5) for k, r in res.items() for n, v in r.items()])
    print(d.groupby(["model", "split"])[["speed", "flow"]].mean().unstack("split").round(4))
