"""Onset cluster prior from the current scenario: reweight the cluster activation probabilities by how often each cluster
queued on the EARLIER days of the same split (published masked layer, strictly before the window's day), relative to
its train rate. Causal: day d only sees days < d of its own split (splits are separate scenarios).
Evaluation on the masked-layer events of t2_onset_shift (evaluation only). Usage: python -m tfb.t2_onset_recent
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import t2_onset2 as o2
from .data import CACHE, load, network
from .t1_interp import interp_axis1
from .t2 import T2_PANELS as T2P
from .t2_eval_big import iou_elig
from .t2_onset_shift import SPLITS, mined_events


def cluster_days(p, z, cands, cm):
    """(days, n_clusters) bool: cluster had any queued candidate link that day (masked layer)."""
    vc = 0.6 * network(p).free_speed_kmh.to_numpy()
    Q = (interp_axis1(z["m_speed"]) <= vc).any(1)          # (days, L)
    ncl = max(cm.values()) + 1
    out = np.zeros((Q.shape[0], ncl), bool)
    for l in cands:
        out[:, cm[l]] |= Q[:, l]
    return out


def adjust(pc: dict, act_prev: np.ndarray, r_tr: np.ndarray, beta: float, a: float = 2.0):
    """logit(pc') = logit(pc) + beta*log(post/r_tr); post = (k + a*r_tr)/(n + a) over the n earlier days."""
    n = len(act_prev)
    if n == 0 or beta == 0:
        return pc
    k = act_prev.sum(0)
    out = {}
    for c, p in pc.items():
        post = (k[c] + a * r_tr[c]) / (n + a)
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        lg += beta * np.log(np.clip(post, 1e-3, 1) / np.clip(r_tr[c], 1e-3, 1))
        out[c] = 1 / (1 + np.exp(-lg))
    return out


def decode(e, pc, n_mc=800, w_old=0.3, rng=None):
    rng = rng or np.random.default_rng(0)
    ucl = np.array(sorted(pc))
    n2 = int(n_mc * (1 - w_old))
    act = rng.random((n2, len(ucl))) < np.array([pc[c] for c in ucl])[None, :]
    s1 = act[:, np.searchsorted(ucl, e["cl"])] & (rng.random((n2, len(e["pl"]))) < e["pl"][None, :])
    s2 = rng.random((n_mc - n2, len(e["po"]))) < e["po"][None, :]
    samp = np.vstack([s1, s2])
    order = np.argsort(-samp.mean(0))
    best, bk = -1, 1
    for k in range(1, samp.shape[1] + 1):
        S = np.zeros(samp.shape[1], bool)
        S[order[:k]] = True
        v = np.mean(np.where((samp | S).sum(1) > 0, (samp & S).sum(1) / np.maximum((samp | S).sum(1), 1), 1.0))
        if v > best:
            best, bk = v, k
    return set(e["link"][order[:bk]].tolist())


def main():
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    cms = {p: o2.clusters(cands[p]) for p in T2P}
    E = pickle.load(open(CACHE / "t2_onset_shift_probs.pkl", "rb"))   # same order as mined_events (panel, split)
    meta, act, rtr = [], {}, {}
    for p in T2P:
        for s in SPLITS:
            z = load(p, s)
            act[(p, s)] = cluster_days(p, z, cands[p], cms[p])
            meta += [(p, s, d) for d, _, _, _ in mined_events(p, s, z)]
        rtr[p] = act[(p, "train")].mean(0)
    assert len(meta) == len(E)
    betas = (0.0, 0.5, 1.0, 2.0)
    rows = []
    for (p, s, d), e in zip(meta, E):
        assert e["panel"] == p and e["split"] == s
        if s == "train" or d > 7:        # windows sit in the first week of each split
            continue
        r = dict(panel=p, split=s, d=d)
        for b in betas:
            pc = adjust(e["pc"], act[(p, s)][:d], rtr[p], b)
            r[f"b{b}"] = iou_elig(decode(e, pc), e["tru"], e["el"])
        rows.append(r)
    r = pd.DataFrame(rows)
    sc = [f"b{b}" for b in betas]
    print("events (days 1-7):", r.groupby("split").size().to_dict())
    print(r.groupby(["split", "panel"])[sc].mean().groupby("split").mean().T.round(4))
    for c in sc[1:]:
        dd = r[c] - r["b0.0"]
        print(c, "paired %+.4f se %.4f" % (dd.mean(), dd.std() / np.sqrt(len(dd))), dd.groupby(r.split).mean().round(4).to_dict())


if __name__ == "__main__":
    main()
