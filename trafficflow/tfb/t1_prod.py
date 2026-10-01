"""Production Task 1 (round 1): all train days, 300k rows per panel.
Speed model uses the corridor common-mode features (cf_*, plat); the flow model does not (no gain for flow in the
A/B). Then the gap specialist (synthetic Task 2 blackouts, 200k rows per panel) replaces the blackout-slot rows.
Usage: TFB_CF=1 python -m tfb.t1_prod [n_rows=300000] [rounds=4000]
Outputs: out/tfb/state_cf.csv (main model) and out/tfb/state_cf_gap.csv (gap-patched, the submission file)."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features
from .data import CACHE, panels
from .submit import OUT, state_file
from .t1_model import feat_cols, gap_rows, panel_frame, predict_rows
from .t1_train import PARAMS, targets_of

CFC = ["cf_sp", "cf_sp_rel", "cf_sp_n", "cf_sp_loc", "cf_fl", "cf_fl_loc", "plat"]


def fit(tr, rounds, flow_drop=CFC):
    X = tr[feat_cols(tr)]
    y = targets_of(tr)
    ms = lgb.train(PARAMS, lgb.Dataset(X, y["speed"]), rounds)
    Xf = X.drop(columns=[c for c in flow_drop if c in X.columns])
    mf = lgb.train(PARAMS, lgb.Dataset(Xf, y["flow"]), rounds)
    return ms, mf


def predict(ms, mf, df):
    X = df[feat_cols(df)]
    s = df.speed_lin.to_numpy() + ms.predict(X[ms.feature_name()], num_threads=4)
    f = df.flow_lin.to_numpy() + mf.predict(X[mf.feature_name()], num_threads=4) * df.lanes.to_numpy()
    return np.clip(s, 1.0, None), np.clip(f, 0.0, None)


if __name__ == "__main__":
    assert features.CF, "run with TFB_CF=1"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    path = CACHE / f"t1_full_cf_{n}.parquet"
    if not path.exists():
        pd.concat([panel_frame(p, "train", None, n, seed=11) for p in panels()], ignore_index=True).to_parquet(path)
    tr = pd.read_parquet(path)
    ms, mf = fit(tr, rounds)
    ms.save_model(str(CACHE / "t1p_speed.txt")); mf.save_model(str(CACHE / "t1p_flow.txt"))
    del tr
    print("main models trained", flush=True)

    def pred(p, s, z, t):
        df = panel_frame(p, s, with_truth=False)
        sp, fl = predict(ms, mf, df)
        order = np.argsort(df.row.to_numpy())
        return sp[order], fl[order]
    state_file(pred, "cf")
    # gap specialist: cells inside synthetic Task 2 blackouts (train has none; val/private have one per window)
    gpath = CACHE / "t1_gap200_cf.parquet"
    if not gpath.exists():
        pd.concat([panel_frame(p, "train", None, 200_000, seed=3, gaps=True) for p in panels()],
                  ignore_index=True).to_parquet(gpath)
    gtr = pd.read_parquet(gpath)
    gs, gf = fit(gtr, 2000, flow_drop=[])
    gs.save_model(str(CACHE / "t1p_gap_speed.txt")); gf.save_model(str(CACHE / "t1p_gap_flow.txt"))
    del gtr
    st = pd.read_csv(OUT / "state_cf.csv")
    off = 0
    for p in panels():
        for s in ("validation", "private"):
            npan = int(((st.panel == p).to_numpy()
                        & st.timestamp.str.startswith("2031-03" if s == "validation" else "2031-04").to_numpy()).sum())
            rows = gap_rows(p, s)
            if len(rows):
                df = panel_frame(p, s, with_truth=False, row_filter=set(rows.tolist()))
                sp, fl = predict(gs, gf, df)
                idx = off + df.row.to_numpy()
                assert (st.panel.to_numpy()[idx] == p).all()
                st.loc[idx, "speed_kmh"] = np.round(sp, 4)
                st.loc[idx, "flow_vph"] = np.round(fl, 3)
                print("gap", p, s, len(rows), flush=True)
            off += npan
    assert off == len(st)
    st.to_csv(OUT / "state_cf_gap.csv", index=False)
    print("wrote state_cf_gap.csv", flush=True)
