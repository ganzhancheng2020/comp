"""Production gap specialist with ramp congestion-sensor features; patches the blackout-slot rows of a state file.
Usage: TFB_CF=1 python -m tfb.t1_gap_prod <src_state.csv> <dst_state.csv> [rounds_speed] [rounds_flow]
       src "-" trains the models only (t1_predict applies them)."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features
from .data import CACHE, panels
from .submit import OUT
from .t1_model import feat_cols, gap_rows, panel_frame
from .t1_train import PARAMS, targets_of

if __name__ == "__main__":
    assert features.CF and features.RAMP_RATIO
    src, dst = sys.argv[1], sys.argv[2]
    rs = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    rf = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    gpath = CACHE / "t1_gap200_cfr.parquet"
    if not gpath.exists():
        pd.concat([panel_frame(p, "train", None, 200_000, seed=3, gaps=True, ramps=True) for p in panels()],
                  ignore_index=True).to_parquet(gpath)
    tr = pd.read_parquet(gpath)
    cols = feat_cols(tr)
    y = targets_of(tr)
    ms = lgb.train(PARAMS, lgb.Dataset(tr[cols], y["speed"]), rs)
    mf = lgb.train(PARAMS, lgb.Dataset(tr[cols], y["flow"]), rf)
    ms.save_model(str(CACHE / "t1p_gapr_speed.txt")); mf.save_model(str(CACHE / "t1p_gapr_flow.txt"))
    del tr
    if src == "-":
        raise SystemExit(0)
    st = pd.read_csv(OUT / src)
    off = 0
    for p in panels():
        for s in ("validation", "private"):
            npan = int(((st.panel == p).to_numpy()
                        & st.timestamp.str.startswith("2031-03" if s == "validation" else "2031-04").to_numpy()).sum())
            rows = gap_rows(p, s)
            if len(rows):
                df = panel_frame(p, s, with_truth=False, row_filter=set(rows.tolist()), ramps=True)
                X = df[cols]
                sp = np.clip(df.speed_lin.to_numpy() + ms.predict(X), 1.0, None)
                fl = np.clip(df.flow_lin.to_numpy() + mf.predict(X) * df.lanes.to_numpy(), 0.0, None)
                idx = off + df.row.to_numpy()
                assert (st.panel.to_numpy()[idx] == p).all()
                st.loc[idx, "speed_kmh"] = np.round(sp, 4)
                st.loc[idx, "flow_vph"] = np.round(fl, 3)
                print("gap", p, s, len(rows), flush=True)
            off += npan
    assert off == len(st)
    st.to_csv(OUT / dst, index=False)
    print("wrote", dst, flush=True)
