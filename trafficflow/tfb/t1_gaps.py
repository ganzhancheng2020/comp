"""Build gap-augmented Task 1 frames (cells inside synthetic Task 2 blackout spans)."""
import pandas as pd

from .data import CACHE, panels
from .t1_model import panel_frame

if __name__ == "__main__":
    tr, va = [], []
    for p in panels():
        tr.append(panel_frame(p, "train", lambda d: d % 4 != 0, 60_000, seed=3, gaps=True))
        va.append(panel_frame(p, "train", lambda d: d % 4 == 0, 30_000, seed=4, gaps=True))
        print(p, len(tr[-1]), len(va[-1]), flush=True)
    pd.concat(tr, ignore_index=True).to_parquet(CACHE / "t1_tr_gap.parquet")
    pd.concat(va, ignore_index=True).to_parquet(CACHE / "t1_va_gap.parquet")
