"""Restart-safe Task 1 prediction: per (panel, split) cached parquet, then one state file.
Main models: t1p_speed (common-mode features, plateau from the split's own masked layer: features.SCEN="split") and
t1p_flow (no cf features); profiles stay train (split profiles made flow worse out of scenario, t1_shift_eval).
Gap rows (Task 2 blackouts) are patched with the ramp-aware gap specialist t1p_gapr_*, fed the split's own ramp and
speed/flow profiles (out of scenario, t1_shift_eval: gap speed RMSE 6.32 old / 9.78 train ramp profile (V12) / 4.92).
Usage: TFB_CF=1 TFB_SCEN=split python -m tfb.t1_predict <name>    -> out/tfb/state_<name>.csv"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .data import CACHE, load, panels, targets
from .submit import OUT
from .t1_model import gap_rows, panel_frame

if __name__ == "__main__":
    assert FE.CF
    name = sys.argv[1]
    cdir = CACHE / f"t1pred_{name}"
    cdir.mkdir(exist_ok=True)
    main = __import__("os").environ.get("TFB_T1_MAIN", "t1p")   # t1p (V12b, 300k rows) | t1c (1.2M) | t1a (all data)
    ms = lgb.Booster(model_file=str(CACHE / f"{main}_speed.txt"))
    mf = lgb.Booster(model_file=str(CACHE / f"{main}_flow.txt"))
    print("main models", main, flush=True)
    gs = lgb.Booster(model_file=str(CACHE / "t1p_gapr_speed.txt"))
    gf = lgb.Booster(model_file=str(CACHE / "t1p_gapr_flow.txt"))
    parts = []
    for p in panels():
        for s in ("validation", "private"):
            f = cdir / f"{p}_{s}.parquet"
            if not f.exists():
                df = panel_frame(p, s, with_truth=False)
                sp = df.speed_lin.to_numpy() + ms.predict(df[ms.feature_name()], num_threads=4)
                fl = df.flow_lin.to_numpy() + mf.predict(df[mf.feature_name()], num_threads=4) * df.lanes.to_numpy()
                out = pd.DataFrame({"row": df.row.to_numpy(), "speed_kmh": np.clip(sp, 1.0, None),
                                    "flow_vph": np.clip(fl, 0.0, None)}).set_index("row").sort_index()
                rows = gap_rows(p, s)
                if len(rows):
                    FE.PROF_SCEN = True     # gap specialist: split speed/flow profiles help out of scenario
                    g = panel_frame(p, s, with_truth=False, row_filter=set(rows.tolist()), ramps=True)
                    FE.PROF_SCEN = False
                    out.loc[g.row.to_numpy(), "speed_kmh"] = np.clip(g.speed_lin.to_numpy() + gs.predict(g[gs.feature_name()]), 1.0, None)
                    out.loc[g.row.to_numpy(), "flow_vph"] = np.clip(g.flow_lin.to_numpy() + gf.predict(g[gf.feature_name()]) * g.lanes.to_numpy(), 0.0, None)
                out.to_parquet(f)
                print("done", p, s, len(out), "gap rows", len(rows), flush=True)
            o = pd.read_parquet(f)
            t = targets(p, s)
            assert len(o) == len(t) and (o.index.to_numpy() == t.index.to_numpy()).all()
            fr = t[["timestamp", "station_id", "link_id", "mask_regime"]].copy()
            fr.insert(0, "panel", p)
            fr["speed_kmh"] = np.round(o.speed_kmh.to_numpy(), 4)
            fr["flow_vph"] = np.round(o.flow_vph.to_numpy(), 3)
            parts.append(fr)
    pd.concat(parts, ignore_index=True).to_csv(OUT / f"state_{name}.csv", index=False)
    print("wrote", f"state_{name}.csv", flush=True)
