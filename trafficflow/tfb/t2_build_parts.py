"""Build the per-panel ongoing training parts (reproducible recipe).

Usage: python -m tfb.t2_build_parts <out_dir> [t_row=0|1] [keep]
  vis  (t_row=0): history T-60..T-5, origin row missing        (V10a production frame, t2_ongoing_parts_vis)
  vt   (t_row=1): same windows, origin row T from the masked layer (~58% of links observed, as published)
Same seed => same windows in both variants (paired comparison). margin=12, bneck = onset candidate links,
early (same-day pre-origin) features on.
"""
import sys
import time

import pandas as pd

from . import t2_ongoing as og
from .data import CACHE
from .t2 import T2_PANELS as T2P

if __name__ == "__main__":
    out = CACHE / sys.argv[1]
    t_row = len(sys.argv) > 2 and sys.argv[2] == "1"
    keep = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    out.mkdir(parents=True, exist_ok=True)
    bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
    for p in T2P:
        if (out / f"{p}.parquet").exists():
            continue
        t0 = time.time()
        df = og.train_frame(p, stride=3, seed=7, keep=keep, margin=12, bneck=sorted(bn[bn.panel == p].link),
                            use_early=True, t_row=t_row)
        df.to_parquet(out / f"{p}.parquet")
        print(p, len(df), df[["d", "T"]].drop_duplicates().shape[0], "windows", round(time.time() - t0), "s", flush=True)
