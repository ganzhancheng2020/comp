"""Task 2 ongoing: per-cell queue probability over the horizon, expected-IoU decoding."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import CACHE, REL, load, network
from .t2_events import queue_truth

warnings.filterwarnings("ignore", category=RuntimeWarning)
T2P = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
MARGIN = 6


def is_ongoing(hq: np.ndarray) -> bool:
    """Release rule: at least two queued observations with at least two on one link."""
    return hq.sum() >= 2 and hq.sum(0).max() >= 2


def cell_features(hs, hf, vcut, cap, T, dow):
    """hs/hf: (13, L) visible history. Returns a frame over (candidate link, step k=1..6)."""
    L = hs.shape[1]
    r = hs / vcut[None, :]
    rf = pd.DataFrame(r).ffill().to_numpy()
    ff = pd.DataFrame(hf / cap[None, :]).ffill().to_numpy()
    hq = hs <= vcut[None, :]
    lastq = rf[-1] <= 1.0
    everq = hq.any(0)
    q_idx = np.where(everq | lastq)[0]
    if len(q_idx) == 0:
        return None
    cand = np.unique(np.clip(np.concatenate([q_idx + o for o in range(-MARGIN, MARGIN + 1)]), 0, L - 1))
    last_idx = np.where(lastq)[0]
    head = last_idx.max() if len(last_idx) else -1
    tail = last_idx.min() if len(last_idx) else -1
    # slots since each link was last queued in the history (13 = never)
    since = np.full(L, 13.0)
    for i in range(12, -1, -1):
        since = np.where(np.isnan(since) | (since == 13.0), np.where(hq[i], 12 - i, 13.0), since)
    nq = hq.sum(0)
    rows = []
    padr = np.pad(rf[-1], MARGIN, constant_values=np.nan)
    padr3 = np.pad(rf[-4], MARGIN, constant_values=np.nan)
    padq = np.pad(lastq.astype(float), MARGIN)
    for l in cand:
        base = {"link": l, "tod": T, "dow": dow, "lpos": l / L,
                "r0": rf[-1, l], "r1": rf[-2, l], "r3": rf[-4, l], "r6": rf[-7, l], "r12": rf[0, l],
                "q0": ff[-1, l], "q3": ff[-4, l],
                "since": since[l], "nq_hist": nq[l], "lastq": float(lastq[l]),
                "d_head": (l - head) if head >= 0 else 99, "d_tail": (l - tail) if tail >= 0 else 99,
                "n_lastq": len(last_idx), "n_everq": int(everq.sum()), "tot_hq": int(hq.sum()),
                "n_q_trend": int(hq[-1].sum()) - int(hq[-4].sum())}
        for o in (-3, -2, -1, 1, 2, 3):
            base[f"nr{o}"] = padr[l + MARGIN + o]
            base[f"nr3_{o}"] = padr3[l + MARGIN + o]
            base[f"nq{o}"] = padq[l + MARGIN + o]
        for k in range(1, 7):
            rows.append({**base, "k": k})
    return pd.DataFrame(rows)


def train_frame(panel: str, stride: int = 3, seed: int = 0, keep: float = 1.0):
    rng = np.random.default_rng(seed)
    z = load(panel, "train")
    Q, vcut = queue_truth(panel, z=z)
    cap = network(panel).capacity_vph.to_numpy()
    dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
    parts = []
    for d in range(Q.shape[0]):
        hsd = z["speed"][d]
        vis_q = hsd <= vcut[None, :]
        for T in range(12 + rng.integers(stride), 288 - 6, stride):
            hq = vis_q[T - 12:T + 1]
            fut = Q[d, T + 1:T + 7]
            if not fut.any() or not is_ongoing(hq) or rng.random() > keep:
                continue
            X = cell_features(hsd[T - 12:T + 1], z["flow"][d, T - 12:T + 1], vcut, cap, T, dow[d])
            if X is None:
                continue
            X["y"] = fut[X.k.to_numpy() - 1, X.link.to_numpy()].astype(int)
            X["d"], X["T"], X["panel"] = d, T, panel
            X["n_fut_total"] = int(fut.sum())
            X["n_fut_out"] = int(fut.sum()) - int(X.y.sum())   # queued cells outside the candidate set
            parts.append(X)
    out = pd.concat(parts, ignore_index=True)
    f32 = out.select_dtypes("float64").columns
    out[f32] = out[f32].astype(np.float32)
    return out


def decode(p: np.ndarray, extra_true: float = 0.0) -> np.ndarray:
    """Top-k by p maximising sum(p_top) / (k + sum(p_rest) + extra)."""
    order = np.argsort(-p)
    cs = np.cumsum(p[order])
    tot = p.sum() + extra_true
    k = np.arange(1, len(p) + 1)
    val = cs / (k + tot - cs)
    best = int(np.argmax(val)) + 1
    sel = np.zeros(len(p), bool)
    sel[order[:best]] = True
    return sel


def fit_all(rounds: int = 600):
    import lightgbm as lgb
    df = pd.read_parquet(CACHE / "t2_ongoing.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    feats = [c for c in df.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")]
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4),
                  lgb.Dataset(df[feats], df.y, categorical_feature=["pid"]), rounds)
    return m, feats


def history_arrays(g: pd.DataFrame, T0: pd.Timestamp, li: dict, L: int):
    k = ((T0 - pd.to_datetime(g.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
    hs = np.full((13, L), np.nan)
    hf = np.full((13, L), np.nan)
    ll = g.link_id.map(li).to_numpy()
    hs[12 - k, ll] = g.speed_kmh.to_numpy()
    hf[12 - k, ll] = g.flow_vph.to_numpy()
    return hs, hf


def predict_windows(split: str, model=None) -> dict:
    """{window_id: (6, L) bool} for the ongoing windows of a split."""
    m, feats = model or fit_all()
    out = {}
    for p in T2P:
        net = network(p)
        li = {x: i for i, x in enumerate(net.link_id)}
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        w = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
        h = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet")
        for r in w[w.condition == "queue_ongoing"].itertuples():
            T0 = pd.Timestamp(r.forecast_origin)
            hs, hf = history_arrays(h[h.window_id == r.window_id], T0, li, len(net))
            X = cell_features(hs, hf, vcut, cap, T0.hour * 12 + T0.minute // 5, T0.dayofweek)
            pred = np.zeros((6, len(net)), bool)
            if X is not None:
                X["pid"] = T2P.index(p)
                sel = decode(m.predict(X[feats]))
                pred[X.k.to_numpy()[sel] - 1, X.link.to_numpy()[sel]] = True
            out[r.window_id] = pred
    return out
