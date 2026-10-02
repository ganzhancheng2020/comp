"""Round 4: Task 1 capacity, restart-safe, judged out of scenario.
  main: N rows/panel (default 1.2M), speed (cf) + flow (no cf), ROUNDS rounds  -> cache/t1c_{speed,flow}.txt
  gap : ramp-aware gap specialist on N_GAP synthetic-blackout cells per panel  -> cache/t1cg_{speed,flow}.txt
Usage: TFB_CF=1 python -m tfb.t1_cap [frame|train|eval|gframe|gtrain|geval]"""
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

N, N_GAP, ROUNDS, CHUNK = 1_200_000, 400_000, 5000, 500
CFC = ["cf_sp", "cf_sp_rel", "cf_sp_n", "cf_sp_loc", "cf_fl", "cf_fl_loc", "plat"]
FR, GFR = CACHE / f"t1_full_cf_{N}.parquet", CACHE / f"t1_gap{N_GAP // 1000}k_cfr.parquet"


def _frame(path, **kw):
    if path.exists():
        return
    parts = []
    for p in panels():
        df = panel_frame(p, "train", None, **kw)
        f64 = df.select_dtypes("float64").columns
        df[f64] = df[f64].astype(np.float32)
        parts.append(df)
        print("frame", p, len(df), flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(path)


def _train(path_fr, prefix, rounds, drop_cf_for_flow=True):
    tr = pd.read_parquet(path_fr)
    y = targets_of(tr)
    for k in ("speed", "flow"):
        cols = feat_cols(tr)
        if k == "flow" and drop_cf_for_flow:
            cols = [c for c in cols if c not in CFC]
        path = CACHE / f"{prefix}_{k}.txt"
        done = lgb.Booster(model_file=str(path)).current_iteration() if path.exists() else 0
        if done >= rounds:
            continue
        ds = lgb.Dataset(tr[cols].to_numpy(np.float32), y[k].to_numpy(), feature_name=cols, free_raw_data=True)

        def ckpt(env, path=path, k=k):
            if (env.iteration + 1) % CHUNK == 0:
                env.model.save_model(str(path))
                print(prefix, k, "rounds", env.model.current_iteration(), flush=True)
        m = lgb.train(PARAMS, ds, rounds - done, init_model=str(path) if done else None, callbacks=[ckpt])
        m.save_model(str(path))
        print(prefix, k, "done", m.current_iteration(), flush=True)
        del ds


def _eval(gaps, models, out):
    res = json.loads(out.read_text()) if out.exists() else {}
    M = {n: (lgb.Booster(model_file=str(CACHE / f"{pre}_speed.txt")), lgb.Booster(model_file=str(CACHE / f"{pre}_flow.txt")))
         for n, pre in models.items()}
    FE.SCEN = "split"
    for p in panels():
        for s in SPLITS:
            key = f"{p}|{s}"
            if key in res:
                continue
            z, zz, idx = hidden(p, s, gaps)
            zz["_plat"] = FE.plateau_of(zz["m_speed"])
            prof = FE.profile_of(zz) if gaps else FE.profile(p, "train")
            X = FE.build(p, zz, idx, prof=prof, ramps=gaps)
            ys, yf, ln = z["m_speed"][idx], z["m_flow"][idx], X.lanes.to_numpy()
            r = {}
            for n, (ms, mf) in M.items():
                sp = X.speed_lin.to_numpy() + ms.predict(X[ms.feature_name()])
                fl = X.flow_lin.to_numpy() + mf.predict(X[mf.feature_name()]) * ln
                r[n] = [float(np.mean((sp - ys) ** 2)), float(np.mean(((fl - yf) / ln) ** 2))]
            res[key] = r
            out.write_text(json.dumps(res))
            print(key, {n: (round(v[0] ** .5, 3), round(v[1] ** .5, 2)) for n, v in r.items()}, flush=True)
    d = pd.DataFrame([dict(panel=k.split("|")[0], split=k.split("|")[1], model=n, speed=v[0] ** .5, flow=v[1] ** .5)
                      for k, r in res.items() for n, v in r.items()])
    print(d.groupby(["model", "split"])[["speed", "flow"]].mean().unstack("split").round(4))
    names = list(models)
    w = d.pivot_table(index=["panel", "split"], columns="model", values=["speed", "flow"])
    print("panels improved (last vs first model): speed", int((w["speed"][names[-1]] < w["speed"][names[0]]).sum()),
          "/", len(w), "flow", int((w["flow"][names[-1]] < w["flow"][names[0]]).sum()))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "frame":
        _frame(FR, max_rows=N, seed=41)
    elif cmd == "train":
        _train(FR, "t1c", ROUNDS)
    elif cmd == "eval":
        _eval(False, {"big600": "t1b", "cap1200": "t1c"}, CACHE / "t1_cap_res.json")
    elif cmd == "gframe":
        _frame(GFR, max_rows=N_GAP, seed=43, gaps=True, ramps=True)
    elif cmd == "gtrain":
        _train(GFR, "t1cg", 3500, drop_cf_for_flow=False)
    elif cmd == "geval":
        _eval(True, {"gapr200": "t1p_gapr", "gap400": "t1cg"}, CACHE / "t1_capgap_res.json")
