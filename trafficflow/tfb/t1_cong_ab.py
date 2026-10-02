"""Round 6: split-own queued-speed plateau per link (plat_cong, cong_ratio) for the T1 speed model, paired A/B on one
frame (300k rows/panel, 3000 rounds), judged out of scenario (hidden cells; plateaus recomputed without them).
Usage: TFB_CF=1 TFB_CONG=1 python -m tfb.t1_cong_ab [frame|train|eval]"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, panels
from .t1_model import feat_cols, panel_frame
from .t1_train import PARAMS, targets_of
from .t1_shift_eval import SPLITS, hidden

FR = CACHE / "t1_cong_300k.parquet"
OUT = CACHE / "t1_cong_res.json"
CONGC = ["plat_cong", "cong_ratio"]


def frame():
    if not FR.exists():
        parts = []
        for p in panels():
            df = panel_frame(p, "train", None, 300_000, seed=61)
            f64 = df.select_dtypes("float64").columns
            df[f64] = df[f64].astype(np.float32)
            parts.append(df)
        pd.concat(parts, ignore_index=True).to_parquet(FR)


def train():
    tr = pd.read_parquet(FR)
    y = targets_of(tr)["speed"]
    for name in ("base", "cong"):
        path = CACHE / f"t1cong_{name}_speed.txt"
        if path.exists():
            continue
        cols = feat_cols(tr) if name == "cong" else [c for c in feat_cols(tr) if c not in CONGC]
        lgb.train(PARAMS, lgb.Dataset(tr[cols].to_numpy(np.float32), y.to_numpy(), feature_name=cols), 3000).save_model(str(path))
        print("trained", name, flush=True)


def evaluate():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    M = {n: lgb.Booster(model_file=str(CACHE / f"t1cong_{n}_speed.txt")) for n in ("base", "cong")}
    FE.SCEN = "split"
    for p in panels():
        for s in SPLITS:
            key = f"{p}|{s}"
            if key in res:
                continue
            z, zz, idx = hidden(p, s, False)
            zz["_plat"] = FE.plateau_of(zz["m_speed"])
            zz["_platc"] = FE.cong_plateau_of(zz["m_speed"], zz["_plat"])
            X = FE.build(p, zz, idx, prof=FE.profile(p, "train"))
            ys = z["m_speed"][idx]
            q = ys < 0.5 * zz["_plat"][idx[2]]
            r = {}
            for n, m in M.items():
                e = X.speed_lin.to_numpy() + m.predict(X[m.feature_name()]) - ys
                r[n] = [float(np.mean(e ** 2)), float(np.mean(e[q] ** 2)) if q.any() else 0.0]
            res[key] = r
            OUT.write_text(json.dumps(res))
            print(key, {n: (round(v[0] ** .5, 3), round(v[1] ** .5, 2)) for n, v in r.items()}, flush=True)
    d = pd.DataFrame([dict(panel=k.split("|")[0], split=k.split("|")[1], model=n, speed=v[0] ** .5, queued=v[1] ** .5)
                      for k, r in res.items() for n, v in r.items()])
    print(d.groupby(["model", "split"])[["speed", "queued"]].mean().unstack("split").round(4))
    w = d.pivot_table(index=["panel", "split"], columns="model", values="speed")
    print("panels improved (speed):", int((w["cong"] < w["base"]).sum()), "/", len(w))


if __name__ == "__main__":
    {"frame": frame, "train": train, "eval": evaluate}[sys.argv[1]]()
