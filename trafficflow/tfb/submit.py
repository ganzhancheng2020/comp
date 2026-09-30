"""Assemble per-task files and the merged Kaggle upload."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from .data import REL, ROOT, load, panels, targets

OUT = ROOT / "out" / "tfb"
REF = ROOT / "tfb_ref"


def state_file(predict, name: str, splits=("validation", "private")) -> str:
    """predict(panel, split, data, targets) -> (speed, flow) arrays over the target rows."""
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for p in panels():
        for s in splits:
            z = load(p, s)
            t = targets(p, s, z)
            sp, fl = predict(p, s, z, t)
            assert np.isfinite(sp).all() and np.isfinite(fl).all(), (p, s)
            fr = t[["timestamp", "station_id", "link_id", "mask_regime"]].copy()
            fr.insert(0, "panel", p)
            fr["speed_kmh"] = np.round(sp, 4)
            fr["flow_vph"] = np.round(fl, 3)
            parts.append(fr)
            print("state", p, s, len(fr), flush=True)
    path = OUT / f"state_{name}.csv"
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return str(path)


def persistence_queue_file() -> str:
    path = OUT / "queue_persistence.csv"
    parts = []
    for s in ("validation", "private"):
        pth = OUT / f"queue_persistence_{s}.csv"
        subprocess.run([sys.executable, str(REF / "src/task2/build_task2_persistence_submission.py"),
                        "--release-root", str(REL), "--split", s, "--output", str(pth)], check=True)
        parts.append(pd.read_csv(pth))
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return str(path)


def merge(state: str, queue: str, odme: str, name: str) -> str:
    out = OUT / f"submission_{name}.csv"
    subprocess.run([sys.executable, str(REF / "src/merge_submissions.py"), "--state", state, "--queue", queue,
                    "--odme", odme, "--key", str(REL / "submission_key.csv"), "--output", str(out)], check=True)
    return str(out)
