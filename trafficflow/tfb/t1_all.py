"""Round 5: Task 1 main models on (nearly) all train target cells: up to 2.4M rows per panel (~20M rows), frames
written per panel, the training matrix filled column by column into one preallocated float32 array (no DataFrame
copy; peak ≈ matrix + LightGBM bins). Restart-safe; judged out of scenario against the 1.2M models.
Usage: TFB_CF=1 python -m tfb.t1_all [frame|train speed|train flow|eval]"""
import gc
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import CACHE, panels
from .t1_model import panel_frame
from .t1_cap import CFC, CHUNK, _eval

N, ROUNDS = 2_400_000, 5000
DIR = CACHE / "t1_all_frames"
META = ("d", "t", "l", "regime", "row", "panel", "y_speed", "y_flow")


def frame():
    DIR.mkdir(exist_ok=True)
    for p in panels():
        f = DIR / f"{p}.parquet"
        if f.exists():
            continue
        df = panel_frame(p, "train", None, N, seed=51)
        f64 = df.select_dtypes("float64").columns
        df[f64] = df[f64].astype(np.float32)
        df.to_parquet(f)
        print("frame", p, len(df), flush=True)
        del df; gc.collect()


def matrix(cols):
    files = [DIR / f"{p}.parquet" for p in panels()]
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    X = np.empty((n, len(cols)), np.float32)
    aux = {c: np.empty(n, np.float32) for c in ("y_speed", "y_flow", "speed_lin", "flow_lin", "lanes")}
    o = 0
    for f in files:
        t = pq.read_table(f, columns=list(dict.fromkeys(cols + list(aux))))
        m = t.num_rows
        for j, c in enumerate(cols):
            X[o:o + m, j] = t.column(c).to_numpy(zero_copy_only=False)
        for c in aux:
            aux[c][o:o + m] = t.column(c).to_numpy(zero_copy_only=False)
        o += m
        del t; gc.collect()
    return X, aux


def train(k):
    from .t1_train import PARAMS
    allc = pq.read_schema(next(DIR.glob("*.parquet"))).names
    cols = [c for c in allc if c not in META]
    if k == "flow":
        cols = [c for c in cols if c not in CFC]
    path = CACHE / f"t1a_{k}.txt"
    done = lgb.Booster(model_file=str(path)).current_iteration() if path.exists() else 0
    if done >= ROUNDS:
        return
    X, a = matrix(cols)
    y = a["y_speed"] - a["speed_lin"] if k == "speed" else (a["y_flow"] - a["flow_lin"]) / a["lanes"]
    del a; gc.collect()
    ds = lgb.Dataset(X, y, feature_name=cols, free_raw_data=True)

    def ckpt(env):
        if (env.iteration + 1) % CHUNK == 0:
            env.model.save_model(str(path))
            print(k, "rounds", env.model.current_iteration(), flush=True)
    m = lgb.train(PARAMS, ds, ROUNDS - done, init_model=str(path) if done else None, callbacks=[ckpt])
    m.save_model(str(path))
    print(k, "done", m.current_iteration(), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "frame":
        frame()
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "eval":
        _eval(False, {"cap1200": "t1c", "all2400": "t1a"}, CACHE / "t1_all_res.json")
