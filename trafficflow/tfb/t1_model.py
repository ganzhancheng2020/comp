"""Task 1: LightGBM residual model on top of temporal interpolation.

Targets: speed - speed_lin and (flow - flow_lin) / lanes. Trained on train-split target cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CACHE, load, network, panels, targets
from .features import build, profile

CHUNK = 40  # days per feature chunk


GAP = 18  # Task 2 horizon (6 slots) + 60 min buffer, blank on every link in validation/private


def gap_origins(panel: str, z: dict, seed: int = 0) -> list[tuple[int, int]]:
    """Train (day, origin) pairs placed like Task 2 windows: onset origins and random ongoing slots."""
    from .t2_events import onset_events, queue_truth
    rng = np.random.default_rng(seed)
    Q, _ = queue_truth(panel, z=z)
    out = [(d, s - 6) for d, s, _ in onset_events(Q) if s - 6 >= 12]
    anyq = Q.any(2)
    for d in range(Q.shape[0]):
        ts = np.where(anyq[d, 12:288 - GAP])[0] + 12
        if len(ts):
            out.append((d, int(rng.choice(ts))))
    return out


def add_gaps(z: dict, origins) -> tuple[dict, np.ndarray]:
    """Copy of z with the masked channels blanked on T+1..T+GAP; returns (z, in_gap mask (D,T))."""
    z = dict(z)
    in_gap = np.zeros(z["m_speed"].shape[:2], bool)
    for d, T in origins:
        in_gap[d, T + 1:T + 1 + GAP] = True
    for k in ("m_speed", "m_flow", "m_occ"):
        a = z[k].copy()
        a[in_gap] = np.nan
        z[k] = a
    return z, in_gap


def panel_frame(panel: str, split: str, day_filter=None, max_rows: int | None = None, seed: int = 0,
                with_truth: bool = True, gaps: bool = False, row_filter=None) -> pd.DataFrame:
    z = load(panel, split)
    t = targets(panel, split, z)
    if gaps:
        z, in_gap = add_gaps(z, gap_origins(panel, z, seed))
        t = t[in_gap[t.d.to_numpy(), t.t.to_numpy()]]
    if day_filter is not None:
        t = t[day_filter(t.d.to_numpy())]
    if row_filter is not None:
        t = t[t.index.isin(list(row_filter))]
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


def predict_split(models: dict, panel: str, split: str):
    """(speed, flow) for the template rows of panel/split, in template order."""
    df = panel_frame(panel, split, with_truth=False)
    X = df[feat_cols(df)]
    s = df.speed_lin.to_numpy() + models["speed"].predict(X, num_threads=4)
    f = df.flow_lin.to_numpy() + models["flow"].predict(X, num_threads=4) * df.lanes.to_numpy()
    order = np.argsort(df.row.to_numpy())
    return np.clip(s[order], 1.0, None), np.clip(f[order], 0.0, None)


def gap_rows(panel: str, split: str) -> np.ndarray:
    """Template row indices whose slot is blank on every link (Task 2 blackout spans)."""
    z = load(panel, split)
    t = targets(panel, split, z)
    allblank = np.isnan(z["m_speed"]).all(2)
    return t.index.to_numpy()[allblank[t.d.to_numpy(), t.t.to_numpy()]]


def predict_rows(models: dict, panel: str, split: str, rows: np.ndarray):
    """(speed, flow) for a subset of template rows (by template index)."""
    z = load(panel, split)
    t = targets(panel, split, z)
    keep = set(rows.tolist())
    df = panel_frame(panel, split, day_filter=None, with_truth=False, row_filter=keep)
    X = df[feat_cols(df)]
    s = df.speed_lin.to_numpy() + models["speed"].predict(X)
    f = df.flow_lin.to_numpy() + models["flow"].predict(X) * df.lanes.to_numpy()
    return df.row.to_numpy(), np.clip(s, 1.0, None), np.clip(f, 0.0, None)
