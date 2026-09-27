"""Ongoing 排队的一维全卷积网络：13 步历史 × 路段 → 6 步 × 路段 的排队概率。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_events import queue_truth
from .t2_ongoing import is_ongoing

torch.set_num_threads(4)
NCH = 4 * 13 + 2 + 2 + len(T2P)


def window_tensor(hs, hf, vcut, cap, T, dow, bneck_mask, pid):
    """hs/hf (13, L) visible history -> (NCH, L) float32."""
    L = hs.shape[1]
    r = hs / vcut[None, :]
    miss = ~np.isfinite(r)
    rf = pd.DataFrame(r).ffill().bfill().to_numpy()
    rf = np.nan_to_num(rf, nan=1.6)
    ff = np.nan_to_num(pd.DataFrame(hf / cap[None, :]).ffill().bfill().to_numpy(), nan=0.5)
    q = np.nan_to_num(hs <= vcut[None, :]).astype(np.float32)
    parts = [np.clip(rf, 0, 2), np.clip(ff, 0, 2), q, miss.astype(np.float32),
             bneck_mask[None, :], (np.arange(L) / max(L - 1, 1))[None, :],
             np.full((1, L), T / 288.0), np.full((1, L), dow / 6.0)]
    oh = np.zeros((len(T2P), L), np.float32)
    oh[pid] = 1
    return np.concatenate(parts + [oh], 0).astype(np.float32)


def panel_windows(panel: str, wins: pd.DataFrame):
    """Tensors for the given (d, T) windows of a train panel: X (n, NCH, L), Y (n, 6, L), E (n, 6, L)."""
    z = load(panel, "train")
    Q, vcut = queue_truth(panel, z=z)
    net = network(panel)
    cap = net.capacity_vph.to_numpy()
    bn = pd.read_parquet(CACHE / "t2_onset.parquet", columns=["panel", "link"])
    bmask = np.zeros(len(net), np.float32)
    bmask[bn[bn.panel == panel].link.unique()] = 1
    dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
    pid = T2P.index(panel)
    X, Y, E = [], [], []
    for d, T in zip(wins.d.to_numpy(), wins["T"].to_numpy()):
        X.append(window_tensor(z["speed"][d, T - 12:T + 1], z["flow"][d, T - 12:T + 1], vcut, cap, T, dow[d], bmask, pid))
        Y.append(Q[d, T + 1:T + 7])
        E.append(z["elig"][d, T + 1:T + 7] == 1)
    return np.stack(X), np.stack(Y).astype(np.float32), np.stack(E)


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.c2 = nn.Conv1d(c, c, 1)
        self.n = nn.GroupNorm(8, c)
        self.a = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.a(self.n(self.c1(x))))


class Net(nn.Module):
    def __init__(self, c=96):
        super().__init__()
        self.inp = nn.Conv1d(NCH, c, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 1, 2, 4, 1)])
        self.out = nn.Conv1d(c, 6, 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


def train_model(data: dict, epochs: int = 12, seed: int = 0, lr: float = 2e-3):
    """data: {panel: (X, Y, E)}. Batches are drawn per panel (different L)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    batches = []
    for p, (X, Y, E) in data.items():
        n = len(X)
        for i in range(0, n, 48):
            batches.append((p, i))
    total = epochs * len(batches)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        order = rng.permutation(len(batches))
        tot = 0.0
        for bi in order:
            p, i = batches[bi]
            X, Y, E = data[p]
            idx = rng.permutation(len(X))[i:i + 48] if False else slice(i, i + 48)
            x = torch.from_numpy(X[idx]); y = torch.from_numpy(Y[idx]); e = torch.from_numpy(E[idx].astype(np.float32))
            logit = net(x)
            loss = (lossf(logit, y) * (0.2 + e)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        print(f"  epoch {ep} loss {tot / len(batches):.4f}", flush=True)
    return net


def predict(net, X):
    net.eval()
    with torch.no_grad():
        out = [torch.sigmoid(net(torch.from_numpy(X[i:i + 128]))).numpy() for i in range(0, len(X), 128)]
    net.train()
    return np.concatenate(out)
