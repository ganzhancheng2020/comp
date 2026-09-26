"""Final Task 1 model: all train days, more rows; then val/private predictions with gap patching."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CACHE, panels
from .submit import state_file
from .t1_model import panel_frame, predict_split
from .t1_train import train

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    path = CACHE / f"t1_full_{n}.parquet"
    if not path.exists():
        fr = []
        for p in panels():
            fr.append(panel_frame(p, "train", None, n, seed=11))
            print("frame", p, flush=True)
        pd.concat(fr, ignore_index=True).to_parquet(path)
    tr = pd.read_parquet(path)
    models = train(tr, None, rounds)
    for k, m in models.items():
        m.save_model(str(CACHE / f"t1f_{k}.txt"))
    del tr

    def pred(p, s, z, t):
        return predict_split(models, p, s)
    state_file(pred, "lgbfull")
