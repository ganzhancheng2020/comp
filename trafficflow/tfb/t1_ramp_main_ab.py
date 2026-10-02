"""T1 main model with vs without ramp congestion-sensor features, judged OUT OF SCENARIO (hidden cells of
validation/private, split plateau, split ramp profile, train speed/flow profile).
Steps (restartable): frame -> models -> eval. Usage: TFB_CF=1 python -m tfb.t1_ramp_main_ab [frame|train|eval]"""
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

FR = CACHE / "t1_ramp_main_150k.parquet"
RAMPC = lambda cols: [c for c in cols if c.startswith(("ron", "roff", "cons_"))]
OUT = CACHE / "t1_ramp_main_res.json"


def frame():
    if not FR.exists():
        pd.concat([panel_frame(p, "train", None, 150_000, seed=21, ramps=True) for p in panels()],
                  ignore_index=True).to_parquet(FR)


def train():
    tr = None
    for name in ("noramp", "ramp"):
        for k in ("speed", "flow"):
            path = CACHE / f"t1rm_{name}_{k}.txt"
            if path.exists():
                continue
            tr = tr if tr is not None else pd.read_parquet(FR)
            cols = feat_cols(tr)
            if name == "noramp":
                cols = [c for c in cols if c not in RAMPC(cols)]
            if k == "flow":
                cols = [c for c in cols if c not in ("cf_sp", "cf_sp_rel", "cf_sp_n", "cf_sp_loc", "cf_fl", "cf_fl_loc", "plat")]
            lgb.train(PARAMS, lgb.Dataset(tr[cols], targets_of(tr)[k]), 3000).save_model(str(path))
            print("trained", name, k, flush=True)


def evaluate():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    M = {f"{n}_{k}": lgb.Booster(model_file=str(CACHE / f"t1rm_{n}_{k}.txt")) for n in ("noramp", "ramp") for k in ("speed", "flow")}
    FE.SCEN = "split"
    for p in panels():
        for s in SPLITS:
            key = f"{p}|{s}"
            if key in res:
                continue
            z, zz, idx = hidden(p, s, False)
            zz["_plat"] = FE.plateau_of(zz["m_speed"])
            X = FE.build(p, zz, idx, prof=FE.profile(p, "train"), ramps=True)
            ys, yf, ln = z["m_speed"][idx], z["m_flow"][idx], X.lanes.to_numpy()
            r = {}
            for n in ("noramp", "ramp"):
                sp = X.speed_lin.to_numpy() + M[f"{n}_speed"].predict(X[M[f"{n}_speed"].feature_name()])
                fl = X.flow_lin.to_numpy() + M[f"{n}_flow"].predict(X[M[f"{n}_flow"].feature_name()]) * ln
                r[n] = [float(np.mean((sp - ys) ** 2)), float(np.mean(((fl - yf) / ln) ** 2))]
            res[key] = r
            OUT.write_text(json.dumps(res))
            print(key, {n: (round(v[0] ** .5, 3), round(v[1] ** .5, 2)) for n, v in r.items()}, flush=True)
    d = pd.DataFrame([dict(panel=k.split("|")[0], split=k.split("|")[1], model=n, speed=v[0] ** .5, flow=v[1] ** .5)
                      for k, r in res.items() for n, v in r.items()])
    print(d.groupby(["model", "split"])[["speed", "flow"]].mean().unstack("split").round(4))
    w = d.pivot_table(index=["panel", "split"], columns="model", values=["speed", "flow"])
    print("panels improved (speed):", int((w["speed"]["ramp"] < w["speed"]["noramp"]).sum()), "/", len(w),
          " (flow):", int((w["flow"]["ramp"] < w["flow"]["noramp"]).sum()))


if __name__ == "__main__":
    {"frame": frame, "train": train, "eval": evaluate}[sys.argv[1]]()
