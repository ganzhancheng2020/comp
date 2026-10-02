"""Load the release into dense (day, time, link) arrays and cache them as .npz.

Every panel has exactly one station per mainline link, so a cell is fully
identified by (date, 5-minute slot, link). Links are ordered along the
corridor by ``order_index`` from lwr_mainline_topology.csv (upstream first).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
REL = Path(os.environ.get("TFB_REL", ROOT / "data_tfb" / "kaggle_public"))
CACHE = Path(os.environ.get("TFB_CACHE", ROOT / "data_tfb" / "cache"))
NT = 288
REGIMES = ("R1", "R2", "R3")
SPLITS = ("train", "validation", "private")


def panels() -> list[str]:
    m = json.loads((REL / "config" / "corridors.json").read_text())
    return [p["corridor_id"] for p in m["panels"]]


def families() -> dict[str, str]:
    m = json.loads((REL / "config" / "corridors.json").read_text())
    return {p["corridor_id"]: p["family_id"] for p in m["panels"]}


def network(panel: str) -> pd.DataFrame:
    """One row per mainline link in corridor order, with FD parameters."""
    net = REL / "corridors" / panel / "network"
    topo = pd.read_csv(net / "lwr_mainline_topology.csv", dtype=str)
    topo["order_index"] = topo.order_index.astype(int)
    topo = topo.sort_values("order_index").reset_index(drop=True)
    fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str})
    out = topo[["link_id", "detector_id", "incoming_link_ids", "outgoing_link_ids",
                "on_ramp_link_ids", "off_ramp_link_ids"]].merge(fd, on="link_id", how="left")
    for c in ("incoming_link_ids", "outgoing_link_ids", "on_ramp_link_ids", "off_ramp_link_ids"):
        out[c] = out[c].fillna("")
    return out


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    return pq.read_table(path, columns=cols).to_pandas()


def _grid(frame: pd.DataFrame, link_index: dict[str, int], cols: list[str], nl: int) -> dict[str, np.ndarray]:
    ts = pd.to_datetime(frame.timestamp, utc=True)
    t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
    li = frame.link_id.map(link_index).to_numpy()
    out = {}
    for c in cols:
        a = np.full((NT, nl), np.nan, dtype=np.float32)
        a[t, li] = frame[c].to_numpy(dtype=np.float32)
        out[c] = a
    return out


def build(panel: str, split: str) -> Path:
    """Build the cache for one panel and split."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{panel}_{split}.npz"
    net = network(panel)
    links = net.link_id.tolist()
    li = {x: i for i, x in enumerate(links)}
    nl = len(links)
    base = REL / "corridors" / panel / split
    masked = {}
    for r in REGIMES:
        for p in sorted((base / "mainline_states_masked" / f"mask_regime={r}").glob("*.parquet")):
            masked[p.name] = (r, p)
    names = sorted(masked)
    dates = [n.replace("synthetic_mainline_", "").replace(".parquet", "").replace("_", "-") for n in names]
    nd = len(names)
    arr = {k: np.full((nd, NT, nl), np.nan, dtype=np.float32)
           for k in ("m_speed", "m_flow", "m_occ", "pct")}
    truth_files = {p.name: p for p in (base / "mainline_states").glob("**/*.parquet")}
    has_truth = bool(truth_files)
    if has_truth:
        for k in ("speed", "flow", "occ", "elig"):
            arr[k] = np.full((nd, NT, nl), np.nan, dtype=np.float32)
    regime = []
    for d, n in enumerate(names):
        r, p = masked[n]
        regime.append(r)
        f = _read(p, ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed"])
        g = _grid(f, li, ["speed_kmh", "flow_vph", "occupancy", "pct_observed"], nl)
        arr["m_speed"][d], arr["m_flow"][d], arr["m_occ"][d], arr["pct"][d] = (
            g["speed_kmh"], g["flow_vph"], g["occupancy"], g["pct_observed"])
        if has_truth:
            f = _read(truth_files[n], ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy",
                                        "is_score_eligible"])
            g = _grid(f, li, ["speed_kmh", "flow_vph", "occupancy", "is_score_eligible"], nl)
            arr["speed"][d], arr["flow"][d], arr["occ"][d], arr["elig"][d] = (
                g["speed_kmh"], g["flow_vph"], g["occupancy"], g["is_score_eligible"])
    # Ramp counts on the same time grid, one column per ramp link.
    rmap = pd.read_csv(REL / "corridors" / panel / "network" / "ramp_attachment_map.csv", dtype=str)
    ramps = rmap.ramp_link_id.tolist()
    ri = {x: i for i, x in enumerate(ramps)}
    ramp_flow = np.full((nd, NT, len(ramps)), np.nan, dtype=np.float32)
    ramp_pct = np.full((nd, NT, len(ramps)), np.nan, dtype=np.float32)
    rfiles = {p.name.replace("synthetic_ramp_", ""): p for p in (base / "ramp_states").glob("**/*.parquet")}
    for d, n in enumerate(names):
        p = rfiles.get(n.replace("synthetic_mainline_", ""))
        if p is None:
            continue
        f = _read(p, ["timestamp", "ramp_link_id", "flow_vph", "pct_observed"]).rename(
            columns={"ramp_link_id": "link_id"})
        g = _grid(f, ri, ["flow_vph", "pct_observed"], len(ramps))
        ramp_flow[d], ramp_pct[d] = g["flow_vph"], g["pct_observed"]
    np.savez(out, dates=np.array(dates), regime=np.array(regime), links=np.array(links),
             ramps=np.array(ramps), ramp_flow=ramp_flow, ramp_pct=ramp_pct, **arr)
    return out


def load(panel: str, split: str) -> dict[str, np.ndarray]:
    path = CACHE / f"{panel}_{split}.npz"
    if not path.exists():
        build(panel, split)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def targets(panel: str, split: str, data: dict | None = None) -> pd.DataFrame:
    """Task 1 target cells as (day index, slot, link index, regime) from the template."""
    t = pd.read_csv(REL / "task1" / panel / split / "sample_submission_state.csv",
                    usecols=["timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
    data = data if data is not None else load(panel, split)
    di = {d: i for i, d in enumerate(data["dates"].tolist())}
    li = {x: i for i, x in enumerate(data["links"].tolist())}
    t["d"] = t.timestamp.str[:10].map(di).astype(int)
    t["t"] = t.timestamp.str[11:13].astype(int) * 12 + t.timestamp.str[14:16].astype(int) // 5
    t["l"] = t.link_id.map(li).astype(int)
    return t


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(p, s) for p in panels() for s in SPLITS]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    jobs = [j for j in jobs if not (CACHE / f"{j[0]}_{j[1]}.npz").exists()]   # restartable
    if not jobs:
        raise SystemExit(0)
    with ProcessPoolExecutor(4) as ex:
        for (p, s), r in zip(jobs, ex.map(build, *zip(*jobs))):
            print(p, s, r.stat().st_size // 2**20, "MB", flush=True)
