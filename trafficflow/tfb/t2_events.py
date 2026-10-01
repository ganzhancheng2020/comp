"""Task 2 event mining on train truth.

Queue indicator: speed <= 0.6 * free_speed (fd_parameters), with the truth
interpolated in time where the detector was down.
Onset windows: the release puts the origin 30 min before the first queued slot,
after at least 90 min with no queue anywhere (history + horizon).
"""
from __future__ import annotations

import numpy as np

from .data import load, network
from .t1_interp import interp_axis1


def queue_truth(panel: str, split: str = "train", z=None):
    z = z if z is not None else load(panel, split)
    vcut = 0.6 * network(panel).free_speed_kmh.to_numpy()
    S = interp_axis1(z["speed"])
    return S <= vcut, vcut


def onset_events(Q: np.ndarray, quiet: int = 18):
    """(day, first queued slot s, queued links at s) for every onset in the split."""
    out = []
    anyq = Q.any(2)
    for d in range(Q.shape[0]):
        for s in range(quiet, Q.shape[1]):
            if anyq[d, s] and not anyq[d, s - quiet:s].any():
                out.append((d, s, np.where(Q[d, s])[0]))
    return out


def iou(pred: set, true: set) -> float:
    u = len(pred | true)
    return 1.0 if u == 0 else len(pred & true) / u


def best_static_set(events, n_links: int, max_size: int = 40):
    """Greedy set maximising the mean IoU over events."""
    sets = [set(e[2].tolist()) for e in events]
    cur: set = set()
    best = (np.mean([iou(cur, s) for s in sets]), set())
    cand = sorted({x for s in sets for x in s})
    while len(cur) < max_size:
        scores = [(np.mean([iou(cur | {c}, s) for s in sets]), c) for c in cand if c not in cur]
        if not scores:
            break
        v, c = max(scores)
        if v <= best[0]:
            break
        cur = cur | {c}
        best = (v, set(cur))
    return best


def visible(a_day, T):
    """History as published in window_history: slots T-60..T-5 (12 rows) plus the origin row T, which is NEVER
    published, so it is returned as NaN. Shape (13, L), same layout as before."""
    h = np.array(a_day[T - 12:T + 1], dtype=float, copy=True)
    h[-1] = np.nan
    return h


def visible_t(a_day, m_day, T):
    """Like visible(), but the origin row T comes from the published masked layer (~58% of links observed in
    validation/private: the Task 1 targets are blank there, everything else is published). Rows T-60..T-5 are the
    window history (published raw)."""
    h = np.array(a_day[T - 12:T + 1], dtype=float, copy=True)
    h[-1] = m_day[T]
    return h
