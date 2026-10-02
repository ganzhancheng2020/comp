"""Round 3: bigger Task 1 main models (600k rows per panel, 5000 rounds), restart-safe (boosting continues from the
last saved 500-round chunk via init_model), judged out of scenario against production (t1_shift_eval hidden cells,
split plateau, train profile). Usage: TFB_CF=1 python -m tfb.t1_big [frame|train|eval]"""
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

N = 600_000
FR = CACHE / f"t1_full_cf_{N}.parquet"
CFC = ["cf_sp", "cf_sp_rel", "cf_sp_n", "cf_sp_loc", "cf_fl", "cf_fl_loc", "plat"]
ROUNDS, CHUNK = 5000, 500
OUT = CACHE / "t1_big_res.json"


def frame():
    if not FR.exists():
        pd.concat([panel_frame(p, "train", None, N, seed=31) for p in panels()], ignore_index=True).to_parquet(FR)


def train():
    tr = pd.read_parquet(FR)
    y = targets_of(tr)
    for k in ("speed", "flow"):
        cols = feat_cols(tr) if k == "speed" else [c for c in feat_cols(tr) if c not in CFC]
        path = CACHE / f"t1b_{k}.txt"
        done = lgb.Booster(model_file=str(path)).current_iteration() if path.exists() else 0
        ds = lgb.Dataset(tr[cols], y[k], free_raw_data=False)
        while done < ROUNDS:
            m = lgb.train(PARAMS, ds, CHUNK, init_model=str(path) if done else None, keep_training_booster=True)
            m.save_model(str(path))
            done = m.current_iteration()
            print(k, "rounds", done, flush=True)


def evaluate():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    M = {"prod_s": CACHE / "t1p_speed.txt", "prod_f": CACHE / "t1p_flow.txt", "big_s": CACHE / "t1b_speed.txt",
         "big_f": CACHE / "t1b_flow.txt"}
    M = {k: lgb.Booster(model_file=str(v)) for k, v in M.items()}
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
            r = {}
            for n in ("prod", "big"):
                sp = X.speed_lin.to_numpy() + M[f"{n}_s"].predict(X[M[f"{n}_s"].feature_name()])
                fl = X.flow_lin.to_numpy() + M[f"{n}_f"].predict(X[M[f"{n}_f"].feature_name()]) * ln
                r[n] = [float(np.mean((sp - ys) ** 2)), float(np.mean(((fl - yf) / ln) ** 2))]
            res[key] = r
            OUT.write_text(json.dumps(res))
            print(key, {n: (round(v[0] ** .5, 3), round(v[1] ** .5, 2)) for n, v in r.items()}, flush=True)
    d = pd.DataFrame([dict(panel=k.split("|")[0], split=k.split("|")[1], model=n, speed=v[0] ** .5, flow=v[1] ** .5)
                      for k, r in res.items() for n, v in r.items()])
    print(d.groupby(["model", "split"])[["speed", "flow"]].mean().unstack("split").round(4))
    w = d.pivot_table(index=["panel", "split"], columns="model", values=["speed", "flow"])
    print("panels improved speed", int((w["speed"]["big"] < w["speed"]["prod"]).sum()), "/", len(w),
          "flow", int((w["flow"]["big"] < w["flow"]["prod"]).sum()))


if __name__ == "__main__":
    {"frame": frame, "train": train, "eval": evaluate}[sys.argv[1]]()
