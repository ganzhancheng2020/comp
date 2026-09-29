"""Ongoing under scenario shift: v9b vs P1 vs P2 blends on ongoing windows mined from the published masked
layer of validation and private (evaluation only). Inputs are built exactly as at inference: the visible
history T-60..T-5 (origin row missing) plus the same-day pre-origin early features; the horizon truth is the
masked layer's observed eligible cells. Selector-style filter: official persistence IoU <= 0.9.
Needs cache/t2_lgb_old.txt (TFB_ONGOING_PARTS=t2_ongoing_parts TFB_LGB_OUT=t2_lgb_old.txt TFB_MODEL_ONLY=1
python -m tfb.t2_final3), t2_lgb_vis.txt and the CNN weights. Usage: python -m tfb.t2_ongoing_shift
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from . import t2_cnn as C
from . import t2_ongoing as og
from .data import CACHE, load, network
from .t1_interp import interp_axis1
from .t2 import T2_PANELS as T2P
from .t2_events import visible

N_WIN = 150


def mined_windows(p, split, z, rng):
    vc = 0.6 * network(p).free_speed_kmh.to_numpy()
    v = z["m_speed"]
    Q = interp_axis1(v) <= vc
    obs = np.isfinite(v) & (z["pct"] >= 75)
    vq = np.nan_to_num(v <= vc[None, None, :]).astype(bool)
    out = []
    for d in range(v.shape[0]):
        for T in range(12 + rng.integers(3), 288 - 6, 3):
            if not og.is_ongoing(vq[d, T - 12:T]):
                continue
            if np.isfinite(v[d, T - 12:T + 7]).mean(1).min() == 0 or np.isfinite(v[d, T - 12:T]).mean() < 0.5:
                continue
            e = obs[d, T + 1:T + 7]
            tru = Q[d, T + 1:T + 7] & e
            if not tru.any():
                continue
            lo = vq[d, T - 1] & obs[d, T - 1]
            pers = np.repeat(lo[None], 6, 0) & e
            if (pers & tru).sum() / max((pers | tru).sum(), 1) > 0.9:
                continue
            out.append((d, T, tru, e, pers))
    if len(out) > N_WIN:
        out = [out[i] for i in sorted(rng.choice(len(out), N_WIN, replace=False))]
    return out


def iou(pred, tru, e):
    pred = pred & e
    u = (pred | tru).sum()
    return (pred & tru).sum() / u if u else 1.0


def nets(files, early):
    C.EARLY, C.NCH = early, 4 * 13 + 2 + 2 + len(C.T2P) + (4 if early else 0)
    out = []
    for f in files:
        n = C.Net()
        n.load_state_dict(torch.load(CACHE / f))
        out.append(n)
    return out


def cnn_map(ns, early, x_args, early_ch):
    C.EARLY = early
    x = C.window_tensor(*x_args, early_ch if early else None)
    return np.mean([C.predict(n, x[None])[0] for n in ns], axis=0)


def blend(pl, pc, w=0.5):
    return np.where(pl > 0, w * pl + (1 - w) * pc, pc) > 0.5


def main():
    rng = np.random.default_rng(0)
    m_old = lgb.Booster(model_file=str(CACHE / "t2_lgb_old.txt"))
    m_vis = lgb.Booster(model_file=str(CACHE / "t2_lgb_vis.txt"))
    cnn_old = nets(["t2_cnn_full.pt", "t2_cnn_full_s1.pt", "t2_cnn_full_s2.pt", "t2_cnn_full_s3.pt"], False)
    cnn_new = nets(["t2_cnn_full_early_s1.pt", "t2_cnn_full_early_s2.pt"], True)
    bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
    rows = []
    for p in T2P:
        net = network(p)
        vc = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        bneck = sorted(bn[bn.panel == p].link)
        bmask = np.zeros(len(net), np.float32)
        bmask[bneck] = 1
        for s in ("validation", "private"):
            z = load(p, s)
            dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
            for d, T, tru, e, pers in mined_windows(p, s, z, rng):
                hs, hf = visible(z["m_speed"][d], T), visible(z["m_flow"][d], T)
                ef = og.early_features(z["m_speed"][d], T, vc)
                X = og.cell_features(hs, hf, vc, cap, T, dow[d], margin=12, bneck=bneck, early=ef)
                maps = {}
                for name, m in (("lgb_old", m_old), ("lgb_vis", m_vis)):
                    mp = np.zeros((6, len(net)), np.float32)
                    if X is not None:
                        Xm = X.assign(pid=T2P.index(p))
                        for c in m.feature_name():
                            if c not in Xm.columns:
                                Xm[c] = np.nan
                        mp[Xm.k.to_numpy() - 1, Xm.link.to_numpy()] = m.predict(
                            Xm[m.feature_name()].to_numpy(np.float32))
                    maps[name] = mp
                args = (hs, hf, vc, cap, T, dow[d], bmask, T2P.index(p))
                ech = C.early_channels(z["m_speed"][d], T, vc)
                maps["cnn_old"] = cnn_map(cnn_old, False, args, ech)
                maps["cnn_new"] = cnn_map(cnn_new, True, args, ech)
                preds = {"pers": pers, "lgb_old": maps["lgb_old"] > 0.5, "lgb_vis": maps["lgb_vis"] > 0.5,
                         "cnn_old": maps["cnn_old"] > 0.5, "cnn_new": maps["cnn_new"] > 0.5,
                         "v9b": blend(maps["lgb_old"], maps["cnn_old"]),
                         "P1": blend(maps["lgb_vis"], maps["cnn_old"]),
                         "P2": blend(maps["lgb_vis"], maps["cnn_new"])}
                rows.append(dict(panel=p, split=s, **{k: iou(v, tru, e) for k, v in preds.items()}))
            print(p, s, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_ongoing_shift.csv", index=False)
    cols = [c for c in r.columns if c not in ("panel", "split")]
    print(r.groupby(["split", "panel"])[cols].mean().groupby("split").mean().T.round(4))
    print("windows", r.groupby("split").size().to_dict())
    for a, b in (("P2", "v9b"), ("P1", "v9b"), ("lgb_vis", "lgb_old"), ("cnn_new", "cnn_old")):
        dd = r[a] - r[b]
        print(f"{a} - {b}: mean {dd.mean():+.4f}  paired se {dd.std() / np.sqrt(len(dd)):.4f}  "
              + " ".join(f"{s} {g.mean():+.4f}" for s, g in dd.groupby(r.split)))


if __name__ == "__main__":
    main()
