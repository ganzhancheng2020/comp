"""Production Task 2 (round 1).

onset    V10a: origin-row-free refit (t2_onset.parquet), cluster + in-cluster + link models, expected-IoU decoding.
ongoing  0.5 * LightGBM + 0.5 * mean of CNN seeds, threshold 0.5 (the P1 blend), both models now trained AND run with
         the origin row T from the published masked layer (~58% of links observed); rows T-60..T-5 are the window
         history. A/B (t2_vt_ab): +0.010 in scenario, +0.023 / +0.021 out of scenario (validation / private).
Usage:
  python -m tfb.t2_prod lgb                 # LightGBM on cache/t2_ongoing_parts_vt (all windows) -> cache/t2p_lgb_vt.txt
  python -m tfb.t2_prod cnn <seed> [epochs] # CNN seed on the same windows, vt inputs -> cache/t2p_cnn_vt_s<seed>.pt
  python -m tfb.t2_prod assemble <name>     # out/tfb/queue_<name>.csv
"""
import gc
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import t2_onset2 as o2
from . import t2_ongoing as og
from .data import CACHE, REL, load, network
from .submit import OUT
from .t2 import T2_PANELS as T2P, predict
from .t2_events import queue_truth, visible_t

PARTS = CACHE / "t2_ongoing_parts_vt"
DROP = ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
SPLITS = ("validation", "private")


def bnecks():
    bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"]).drop_duplicates()
    return {p: sorted(bn[bn.panel == p].link) for p in T2P}


def train_lgb():
    Xs, ys, feats = [], [], None
    for p in T2P:
        t = pq.read_table(PARTS / f"{p}.parquet").to_pandas()
        t["pid"] = np.int32(T2P.index(p))
        if feats is None:
            feats = [c for c in t.columns if c not in DROP]
        Xs.append(t[feats].to_numpy(np.float32)); ys.append(t.y.to_numpy())
        del t; gc.collect()
    X, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs, ys; gc.collect()
    m = lgb.train(PAR, lgb.Dataset(X, y, feature_name=feats, categorical_feature=["pid"], free_raw_data=True), 1000)
    m.save_model(str(CACHE / "t2p_lgb_vt.txt"))


def cnn_data(p):
    """CNN tensors for the training windows of panel p, origin row from the masked layer."""
    from . import t2_cnn as C
    z = load(p, "train")
    Q, vcut = queue_truth(p, z=z)
    net = network(p)
    cap = net.capacity_vph.to_numpy()
    bm = np.zeros(len(net), np.float32)
    bm[bnecks()[p]] = 1
    dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
    w = pq.read_table(PARTS / f"{p}.parquet", columns=["d", "T"]).to_pandas().drop_duplicates()
    X, Y, E = [], [], []
    for d, T in zip(w.d.to_numpy(), w["T"].to_numpy()):
        X.append(C.window_tensor(visible_t(z["speed"][d], z["m_speed"][d], T), visible_t(z["flow"][d], z["m_flow"][d], T),
                                 vcut, cap, T, dow[d], bm, T2P.index(p)))
        Y.append(Q[d, T + 1:T + 7]); E.append(z["elig"][d, T + 1:T + 7] == 1)
    return np.stack(X), np.stack(Y).astype(np.float32), np.stack(E)


def train_crop(data, epochs, seed, crop=64, bs=48, lr=2e-3):
    """t2_cnn.train_model with a random spatial crop per batch (the net is fully convolutional, so it runs on the
    whole corridor at inference). Positional channels are computed on the full corridor before cropping."""
    import torch
    import torch.nn as nn
    from . import t2_cnn as C
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = C.Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    batches = [(p, i) for p, (X, Y, E) in data.items() for i in range(0, len(X), bs)]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(batches))
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    perm = {p: rng.permutation(len(v[0])) for p, v in data.items()}
    for ep in range(epochs):
        tot = 0.0
        for bi in rng.permutation(len(batches)):
            p, i = batches[bi]
            X, Y, E = data[p]
            idx = perm[p][i:i + bs]
            L = X.shape[2]
            a = int(rng.integers(0, L - crop + 1)) if L > crop else 0
            sl = slice(a, a + crop)
            x = torch.from_numpy(X[idx][:, :, sl]); y = torch.from_numpy(Y[idx][:, :, sl])
            e = torch.from_numpy(E[idx][:, :, sl].astype(np.float32))
            loss = (lossf(net(x), y) * (0.2 + e)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach())
        for p in perm:
            perm[p] = rng.permutation(len(data[p][0]))
        print(f"  epoch {ep} loss {tot / len(batches):.4f}", flush=True)
    return net


def train_cnn(seed, epochs):
    import torch
    data = {p: cnn_data(p) for p in T2P}
    print("cnn data", {p: len(v[0]) for p, v in data.items()}, flush=True)
    net = train_crop(data, epochs, seed)
    torch.save(net.state_dict(), CACHE / f"t2p_cnn_vt_s{seed}.pt")


def window_inputs(p, s):
    """(window_id, T0, d, hs, hf) for the ongoing windows: window history T-60..T-5 + masked-layer origin row."""
    net = network(p)
    li = {x: i for i, x in enumerate(net.link_id)}
    z = load(p, s)
    di = {d: i for i, d in enumerate(z["dates"].tolist())}
    w = pd.read_csv(REL / "task2" / p / s / "window_index.csv")
    h = pd.read_parquet(REL / "task2" / p / s / "window_history.parquet")
    out = []
    for r in w[w.condition == "queue_ongoing"].itertuples():
        T0 = pd.Timestamp(r.forecast_origin)
        T = T0.hour * 12 + T0.minute // 5
        d = di[T0.strftime("%Y-%m-%d")]
        hs, hf = og.history_arrays(h[h.window_id == r.window_id], T0, li, len(net))
        assert np.isnan(hs[-1]).all()            # the window history never carries the origin row
        hs[-1], hf[-1] = z["m_speed"][d, T], z["m_flow"][d, T]
        out.append((r.window_id, T0, d, hs, hf))
    return out


def ongoing_maps(seeds):
    import torch
    from . import t2_cnn as C
    m = lgb.Booster(model_file=str(CACHE / "t2p_lgb_vt.txt"))
    feats = m.feature_name()
    nets = []
    for sd in seeds:
        n = C.Net()
        n.load_state_dict(torch.load(CACHE / f"t2p_cnn_vt_s{sd}.pt"))
        nets.append(n)
    bn = bnecks()
    maps = {}
    for p in T2P:
        net = network(p)
        vcut = 0.6 * net.free_speed_kmh.to_numpy()
        cap = net.capacity_vph.to_numpy()
        bm = np.zeros(len(net), np.float32)
        bm[bn[p]] = 1
        for s in SPLITS:
            z = load(p, s)
            for wid, T0, d, hs, hf in window_inputs(p, s):
                T = T0.hour * 12 + T0.minute // 5
                early = og.early_features(z["m_speed"][d], T, vcut)
                pl = np.zeros((6, len(net)), np.float32)
                Xw = og.cell_features(hs, hf, vcut, cap, T, T0.dayofweek, margin=12, bneck=bn[p], early=early)
                if Xw is not None:
                    Xw["pid"] = T2P.index(p)
                    pl[Xw.k.to_numpy() - 1, Xw.link.to_numpy()] = m.predict(Xw[feats].to_numpy(np.float32))
                x = C.window_tensor(hs, hf, vcut, cap, T, T0.dayofweek, bm, T2P.index(p))
                pc = np.mean([C.predict(n, x[None])[0] for n in nets], axis=0)
                maps[wid] = (pl, pc)
    return maps


def onset_windows():
    """V10a onset (t2_v10 with w = 0)."""
    from .t2_onset_shift import fit_general
    from .t2_v10 import onset_windows as ow
    df = pd.read_parquet(CACHE / "t2_onset.parquet")
    df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
    df = o2.add_cluster(df)
    cands = {p: sorted(df[df.panel == p].link.unique().tolist()) for p in T2P}
    models = [fit_general(df), fit_general(df, drop=("tod", "dow"))]
    return {s: ow(s, models, cands, 0.0) for s in SPLITS}


def assemble(name, seeds=(0, 1, 2, 3)):
    import pickle
    seeds = [s for s in seeds if (CACHE / f"t2p_cnn_vt_s{s}.pt").exists()]
    print("cnn seeds", seeds, flush=True)
    maps = ongoing_maps(seeds)
    pickle.dump(maps, open(CACHE / f"t2p_maps_{name}.pkl", "wb"))
    ong = {wid: np.where(pl > 0, 0.5 * pl + 0.5 * pc, pc) > 0.5 for wid, (pl, pc) in maps.items()}
    ons = onset_windows()
    q = pd.concat([predict(s, onset_windows=ons[s], ongoing_windows=ong) for s in SPLITS], ignore_index=True)
    q.to_csv(OUT / f"queue_{name}.csv", index=False)
    print("wrote", name, "queued cells", int(q.queue_pred.sum()), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "lgb":
        train_lgb()
    elif cmd == "cnn":
        train_cnn(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 25)
    elif cmd == "assemble":
        assemble(sys.argv[2])
