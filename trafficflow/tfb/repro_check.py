"""Compare a reproduced submission with the production one and bound the score difference.

    python -m tfb.repro_check <prod_state.csv[.gz]> <repro_state.csv> <prod_queue.csv> <repro_queue.csv> \
        <prod_odme.csv> <repro_odme.csv>

Task 1: by the triangle inequality |RMSE(a) - RMSE(b)| <= RMSE(a - b), so the state-score change is at most
0.54 * d_v / 25 + 0.46 * d_q,lane / 600 (SCORING_SPEC Task 1), whatever the truth. Task 2: share of queue cells that
differ. Task 4: L1 difference of the path flows relative to their total (bounds |dS_od| and |dS_link| via A).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .data import network

KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]


def lanes_of(panels) -> pd.DataFrame:
    rows = []
    for p in panels:
        n = network(p)
        rows.append(pd.DataFrame({"panel": p, "link_id": n.link_id.astype(str), "lanes": n.lanes.to_numpy()}))
    return pd.concat(rows, ignore_index=True).drop_duplicates(["panel", "link_id"])


def main(ps, rs, pq, rq, po, ro):
    a = pd.read_csv(ps, dtype={"link_id": str, "station_id": str})
    b = pd.read_csv(rs, dtype={"link_id": str, "station_id": str})
    m = a.merge(b, on=KEYS, suffixes=("_p", "_r"), validate="one_to_one")
    assert len(m) == len(a) == len(b), (len(a), len(b), len(m))
    m = m.merge(lanes_of(m.panel.unique()), on=["panel", "link_id"], how="left")
    lanes = m.lanes.fillna(1.0).clip(lower=1.0).to_numpy()
    dv = np.sqrt(np.mean((m.speed_kmh_p - m.speed_kmh_r) ** 2))
    dq = np.sqrt(np.mean(((m.flow_vph_p - m.flow_vph_r) / lanes) ** 2))
    bound = 0.54 * dv / 25 + 0.46 * dq / 600
    print(f"T1 rows {len(m):,}: speed RMSE(prod-repro) {dv:.4f} km/h, max {np.abs(m.speed_kmh_p - m.speed_kmh_r).max():.3f}; "
          f"flow/lane RMSE {dq:.3f} vph -> |dS_state| <= {bound:.5f}, |dS_total| <= {0.35 * bound:.5f} "
          f"(+ physics, which follows Task 1)")
    for p, g in m.groupby("panel"):
        print(f"  {p}: speed {np.sqrt(np.mean((g.speed_kmh_p - g.speed_kmh_r) ** 2)):.4f}  "
              f"flow {np.sqrt(np.mean((g.flow_vph_p - g.flow_vph_r) ** 2)):.3f}")
    qa, qb = pd.read_csv(pq), pd.read_csv(rq)
    col = "queue_pred" if "queue_pred" in qa.columns else qa.columns[-1]
    assert len(qa) == len(qb)
    nd = int((qa[col].to_numpy() != qb[col].to_numpy()).sum())
    print(f"T2 cells {len(qa):,}: differ {nd} ({nd / len(qa):.5%}); queued prod {int(qa[col].sum())} repro {int(qb[col].sum())}")
    oa, ob = pd.read_csv(po), pd.read_csv(ro)
    fc = "path_flow" if "path_flow" in oa.columns else oa.columns[-1]
    l1 = float(np.abs(oa[fc].to_numpy() - ob[fc].to_numpy()).sum() / max(oa[fc].sum(), 1e-9))
    print(f"T4 paths {len(oa):,}: relative L1 difference {l1:.2e}")


if __name__ == "__main__":
    main(*sys.argv[1:7])
