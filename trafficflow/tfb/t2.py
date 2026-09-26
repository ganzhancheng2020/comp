"""Task 2 predictions.

onset    queue only at the last horizon step (T+30) on a per-panel link set chosen to
         maximise mean IoU over all train onset events (see t2_events).
ongoing  persistence of the last visible queue state per link.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import CACHE, REL, network
from .t2_events import best_static_set, onset_events, queue_truth

T2_PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]


def static_sets() -> dict[str, list[int]]:
    path = CACHE / "t2_static_sets.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {}
    for p in T2_PANELS:
        Q, _ = queue_truth(p)
        v, s = best_static_set(onset_events(Q), Q.shape[2])
        out[p] = sorted(int(x) for x in s)
        print(p, out[p], round(v, 3), flush=True)
    path.write_text(json.dumps(out))
    return out


def history_grid(panel: str, split: str):
    """Visible history speeds per window as {window_id: (13, L) array}, slots T-60..T."""
    net = network(panel)
    li = {x: i for i, x in enumerate(net.link_id)}
    w = pd.read_csv(REL / "task2" / panel / split / "window_index.csv")
    h = pd.read_parquet(REL / "task2" / panel / split / "window_history.parquet")
    out = {}
    for r in w.itertuples():
        g = h[h.window_id == r.window_id]
        T = pd.Timestamp(r.forecast_origin)
        k = ((T - pd.to_datetime(g.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
        a = np.full((13, len(net)), np.nan, np.float32)
        a[12 - k, g.link_id.map(li).to_numpy()] = g.speed_kmh.to_numpy(np.float32)
        out[r.window_id] = a
    return w, out


def predict(split: str, onset_sets=None, onset_windows=None, ongoing_windows=None) -> pd.DataFrame:
    """onset_windows: {window_id: links} overrides the static per-panel sets."""
    sets = onset_sets or static_sets()
    rows = []
    for p in T2_PANELS:
        net = network(p)
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        w, hist = history_grid(p, split)
        tmpl = pd.read_csv(REL / "task2" / p / split / "sample_submission_queue.csv")
        li = {x: i for i, x in enumerate(net.link_id)}
        for r in w.itertuples():
            T = pd.Timestamp(r.forecast_origin)
            pred = np.zeros((6, len(net)), bool)
            if r.condition == "queue_onset":
                pred[5, (onset_windows or {}).get(r.window_id, sets[p])] = True
            elif ongoing_windows is not None:
                pred = ongoing_windows[r.window_id]
            else:
                a = hist[r.window_id]
                last = pd.DataFrame(a).ffill().to_numpy()[-1]       # last visible value per link
                pred[:] = (last <= vcut)[None, :]
            g = tmpl[tmpl.window_id == r.window_id].copy()
            k = ((pd.to_datetime(g.timestamp, utc=True) - T).dt.total_seconds() // 300).astype(int).to_numpy() - 1
            g["queue_pred"] = pred[k, g.link_id.map(li).to_numpy()].astype(int)
            rows.append(g)
    return pd.concat(rows, ignore_index=True)
