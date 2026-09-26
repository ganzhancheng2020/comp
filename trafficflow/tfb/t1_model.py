"""Task 1: LightGBM residual model on top of temporal interpolation.

Targets: speed - speed_lin and (flow - flow_lin) / lanes. Trained on train-split target cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CACHE, load, network, panels, targets
from .features import build, profile

CHUNK = 40  # days per feature chunk


def panel_frame(panel: str, split: str, day_filter=None, max_rows: int | None = None, seed: int = 0,
                with_truth: bool = True) -> pd.DataFrame:
    z = load(panel, split)
    t = targets(panel, split, z)
    if day_filter is not None:
        t = t[day_filter(t.d.to_numpy())]
    if max_rows is not None and len(t) > max_rows:
        t = t.sample(max_rows, random_state=seed)
    t = t.sort_values(["d", "t", "l"])
    prof = profile(panel)
    parts = []
    nd = len(z["dates"])
    for d0 in range(0, nd, CHUNK):
        sel = (t.d >= d0) & (t.d < d0 + CHUNK)
        if not sel.any():
            continue
        tt = t[sel]
        zc = {k: (v[d0:d0 + CHUNK] if isinstance(v, np.ndarray) and v.ndim == 3 or k in ("dates", "regime") else v)
              for k, v in z.items()}
        idx = (tt.d.to_numpy() - d0, tt.t.to_numpy(), tt.l.to_numpy())
        X = build(panel, zc, idx, prof)
        X["d"], X["t"], X["l"] = tt.d.to_numpy(), tt.t.to_numpy(), tt.l.to_numpy()
        X["regime"] = tt.mask_regime.to_numpy()
        X["row"] = tt.index.to_numpy()
        if with_truth and "speed" in z:
            X["y_speed"] = z["speed"][idx[0] + d0, idx[1], idx[2]]
            X["y_flow"] = z["flow"][idx[0] + d0, idx[1], idx[2]]
        parts.append(X)
    df = pd.concat(parts, ignore_index=True)
    df["panel"] = panel
    return df


FEATS_DROP = {"d", "t", "l", "regime", "row", "panel", "y_speed", "y_flow"}


def feat_cols(df):
    return [c for c in df.columns if c not in FEATS_DROP]


if __name__ == "__main__":
    import sys
    import time
    n_tr = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    n_va = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    tr, va = [], []
    for p in panels():
        t0 = time.time()
        tr.append(panel_frame(p, "train", lambda d: d % 4 != 0, n_tr, seed=1))
        va.append(panel_frame(p, "train", lambda d: d % 4 == 0, n_va, seed=2))
        print(p, round(time.time() - t0), "s", flush=True)
    pd.concat(tr, ignore_index=True).to_parquet(CACHE / "t1_tr.parquet")
    pd.concat(va, ignore_index=True).to_parquet(CACHE / "t1_va.parquet")
