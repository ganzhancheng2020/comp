"""Array versions of the official Task 1 score (same formula as score_task1.py)."""
from __future__ import annotations

import numpy as np

from .data import REGIMES

SPEED_NORM, FLOW_NORM, W_SPEED, W_FLOW = 25.0, 600.0, 0.54, 0.46


def state_score(pred_speed, pred_flow, true_speed, true_flow, lanes, regime) -> dict:
    """All arguments are 1-D arrays over target cells. Returns per-regime and mean S_state."""
    out = {}
    for r in REGIMES:
        m = regime == r
        if not m.any():
            out[r] = 0.0
            continue
        rs = np.sqrt(np.mean((pred_speed[m] - true_speed[m]) ** 2))
        rf = np.sqrt(np.mean(((pred_flow[m] - true_flow[m]) / lanes[m]) ** 2))
        out[r] = W_SPEED * max(0.0, 1 - rs / SPEED_NORM) + W_FLOW * max(0.0, 1 - rf / FLOW_NORM)
        out[r + "_rmse"] = (round(float(rs), 3), round(float(rf), 2))
    out["S_state"] = float(np.mean([out[r] for r in REGIMES]))
    return out
