"""Onset ensemble under scenario shift: pool the Monte-Carlo samples of v9b and of the no-tod/dow variant, then
decode by expected IoU. Evaluation only (masked-layer events, see t2_onset_shift). Usage: python -m tfb.t2_onset_shift_ens
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, load
from .t2 import T2_PANELS as T2P
from .t2_eval_big import iou_elig
from .t2_onset_shift import SPLITS, event_rows, fit_general, mined_events
from .t2_onset_shift_cal import probs


def samples(e, n, rng, w_old=0.3):
    ucl = np.array(sorted(e["pc"]))
    pc = np.array([e["pc"][c] for c in ucl])
    n2 = int(n * (1 - w_old))
    act = rng.random((n2, len(ucl))) < pc[None, :]
    s1 = act[:, np.searchsorted(ucl, e["cl"])] & (rng.random((n2, len(e["pl"]))) < e["pl"][None, :])
    s2 = rng.random((n - n2, len(e["po"]))) < e["po"][None, :]
    return np.vstack([s1, s2])


def decode_pool(es, ws, n_mc=800, rng=None):
    rng = rng or np.random.default_rng(0)
    samp = np.vstack([samples(e, max(1, int(n_mc * w)), rng) for e, w in zip(es, ws) if w > 0])
    order = np.argsort(-samp.mean(0))
    best, bk = -1, 1
    for k in range(1, samp.shape[1] + 1):
        S = np.zeros(samp.shape[1], bool)
        S[order[:k]] = True
        v = np.mean(np.where((samp | S).sum(1) > 0, (samp & S).sum(1) / np.maximum((samp | S).sum(1), 1), 1.0))
        if v > best:
            best, bk = v, k
    return set(es[0]["link"][order[:bk]].tolist())


FRAME = os.environ.get("TFB_ONSET_FRAME", "t2_onset.parquet")   # t2_onset_withT.parquet = the online v9b onset
FRAME_TAG = "" if FRAME == "t2_onset.parquet" else "_" + FRAME.replace(".parquet", "")


def cached(name, df, cands, cms, drop):
    path = CACHE / f"t2_onset_shift_probs{name}{FRAME_TAG}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    full = fit_general(df, drop=drop)
    folds = {f: fit_general(df[df.d % 2 != f], drop=drop) for f in (0, 1)}
    E = []
    for p in T2P:
        for s in SPLITS:
            z = load(p, s)
            for d, ss, tru, el in mined_events(p, s, z):
                m = folds[d % 2] if s == "train" else full
                E.append(dict(probs(m, event_rows(p, z, d, ss - 6, cands[p], cms[p])), panel=p, split=s, tru=tru, el=el))
    pickle.dump(E, open(path, "wb"))
    return E


def main():
    df = pd.read_parquet(CACHE / FRAME)
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    A = cached("", df, cands, cms, ())
    B = cached("_notod", df, cands, cms, ("tod", "dow"))
    rows = []
    for a, b in zip(A, B):
        assert a["panel"] == b["panel"] and a["tru"] == b["tru"]
        r = dict(panel=a["panel"], split=a["split"])
        for w in (0.0, 0.3, 0.5, 0.7, 1.0):
            r[f"w_notod{w}"] = iou_elig(decode_pool([a, b], [1 - w, w]), a["tru"], a["el"])
        rows.append(r)
    r = pd.DataFrame(rows)
    sc = [c for c in r.columns if c.startswith("w_")]
    print(r.groupby(["split", "panel"])[sc].mean().groupby("split").mean().T.round(4))
    o = r[r.split != "train"]
    for c in sc[1:]:
        d = o[c] - o["w_notod0.0"]
        print(c, "oos paired %+.4f se %.4f" % (d.mean(), d.std() / np.sqrt(len(d))), d.groupby(o.split).mean().round(4).to_dict())


if __name__ == "__main__":
    main()
