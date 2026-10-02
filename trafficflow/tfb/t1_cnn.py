"""Round 8b: spatiotemporal CNN imputer for Task 1 (traffic-imputation family: GRIN / SAITS / Graph WaveNet).

Input window: W slots x all links of one panel-split masked layer, channels = normalised speed (v / split plateau - 1),
flow / capacity, occupancy, observed mask, pct/100, corridor common mode (per slot), tod sin/cos, link position, lanes.
Factorised dilated residual blocks (time 3x1, link 1x3). Output: normalised speed and flow at every cell. Loss on cells
that are masked in the input and have truth (train), so it learns exactly the Task 1 reconstruction.
Restart-safe training (checkpoint per epoch). Usage:
  TFB_CF=1 python -m tfb.t1_cnn train [epochs]     -> cache/t1cnn.pt
  TFB_CF=1 python -m tfb.t1_cnn eval              -> out-of-scenario hidden cells, LightGBM (t1a) vs CNN vs blends
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from . import features as FE
from .data import CACHE, load, network, panels

torch.set_num_threads(4)
W, C_IN, CH = 36, 11, 48
CKPT = CACHE / "t1cnn.ckpt"


def tensors(p, z, plat):
    """(C_IN, D*288 slots stacked per day, L) float32 inputs; returns per-day arrays (D, C, 288, L)."""
    net = network(p)
    cap = net.capacity_vph.to_numpy(np.float32)
    lanes = net.lanes.to_numpy(np.float32)
    v, q, o, pct = z["m_speed"], z["m_flow"], z["m_occ"], z["pct"]
    obs = np.isfinite(v) & np.isfinite(q)
    s = np.where(obs, v / plat - 1.0, 0.0)
    f = np.where(obs, q / cap, 0.0)
    oc = np.where(np.isfinite(o), o, 0.0)
    ffm = np.isfinite(v) & (v > 0.85 * plat)
    cf = np.nan_to_num(np.nanmean(np.where(ffm, v / plat - 1, np.nan), 2))[..., None] * np.ones_like(s)
    D, T, L = v.shape
    t = np.arange(T, dtype=np.float32)[None, :, None] * np.ones((D, 1, L), np.float32)
    pos = (np.arange(L, dtype=np.float32) / max(L - 1, 1))[None, None, :] * np.ones((D, T, 1), np.float32)
    X = np.stack([s, f, oc, obs.astype(np.float32), np.nan_to_num(pct) / 100.0, cf,
                  np.sin(2 * np.pi * t / 288), np.cos(2 * np.pi * t / 288), pos,
                  lanes[None, None, :] / 5.0 * np.ones((D, T, 1), np.float32),
                  np.isfinite(o).astype(np.float32)], 1).astype(np.float32)
    return X, cap


class Block(nn.Module):
    def __init__(self, c, dt, dl):
        super().__init__()
        self.t = nn.Conv2d(c, c, (3, 1), padding=(dt, 0), dilation=(dt, 1))
        self.l = nn.Conv2d(c, c, (1, 3), padding=(0, dl), dilation=(1, dl))
        self.n = nn.GroupNorm(8, c)
        self.a = nn.GELU()

    def forward(self, x):
        return x + self.l(self.a(self.t(self.a(self.n(x)))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Conv2d(C_IN, CH, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(CH, dt, dl) for dt, dl in ((1, 1), (2, 2), (4, 4), (1, 8), (2, 16), (4, 1), (1, 2), (2, 4))])
        self.out = nn.Conv2d(CH, 2, 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


def panel_data(p):
    z = load(p, "train")
    plat = FE.plateau(p, "train")
    X, cap = tensors(p, z, plat)
    tgt = np.isnan(z["m_speed"]) & np.isfinite(z["speed"]) & np.isfinite(z["flow"]) & ~np.isnan(z["m_speed"]).all(2)[..., None]
    Y = np.stack([np.where(tgt, z["speed"] / plat - 1.0, 0.0), np.where(tgt, z["flow"] / cap, 0.0)], 1).astype(np.float32)
    return X, Y, tgt.astype(np.float32), plat, cap


def train(epochs=12, per_epoch=6000, bs=16, lr=2e-3):
    data = {p: panel_data(p) for p in panels()}
    print("data ready", flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (per_epoch // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    start = 0
    if CKPT.exists():
        st = torch.load(CKPT, weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; start = st["epoch"] + 1
        print("resumed at", start, flush=True)
    pl = list(data)
    wts = np.array([data[p][0].shape[0] for p in pl], float); wts /= wts.sum()
    for ep in range(start, epochs):
        tot = 0.0
        for it in range(per_epoch // bs):
            p = pl[rng.choice(len(pl), p=wts)]
            X, Y, M = data[p][:3]
            d = rng.integers(0, X.shape[0], bs); t0 = rng.integers(0, 288 - W, bs)
            xb = torch.from_numpy(np.stack([X[i, :, a:a + W] for i, a in zip(d, t0)]))
            yb = torch.from_numpy(np.stack([Y[i, :, a:a + W] for i, a in zip(d, t0)]))
            mb = torch.from_numpy(np.stack([M[i, a:a + W] for i, a in zip(d, t0)]))[:, None]
            pr = net(xb)
            err = (pr - yb) ** 2 * mb
            loss = (err[:, 0].sum() * 25.0 + err[:, 1].sum() * 4.0) / mb.sum().clamp(min=1)   # ~equal speed/flow weight
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach())
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "epoch": ep}, CKPT)
        print(f"epoch {ep} loss {tot / (per_epoch // bs):.4f}", flush=True)
    torch.save(net.state_dict(), CACHE / "t1cnn.pt")


def predict_day(net, X):
    """X (C, 288, L) -> (2, 288, L) by overlapping windows (stride W/2), averaged."""
    out = np.zeros((2,) + X.shape[1:], np.float32); cnt = np.zeros(X.shape[1:], np.float32)
    starts = list(range(0, 288 - W + 1, W // 2))
    if starts[-1] != 288 - W:
        starts.append(288 - W)
    with torch.no_grad():
        xb = torch.from_numpy(np.stack([X[:, a:a + W] for a in starts]))
        pr = net(xb).numpy()
    for k, a in enumerate(starts):
        out[:, a:a + W] += pr[k]; cnt[a:a + W] += 1
    return out / cnt


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 12)


def evaluate():
    """Out-of-scenario hidden cells (t1_shift_eval protocol): LightGBM ensemble (0.75 t1a + 0.25 t1c) vs CNN vs blends."""
    import lightgbm as lgb
    import pandas as pd
    from .t1_shift_eval import SPLITS, hidden
    out = CACHE / "t1_cnn_res.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    net = Net(); net.load_state_dict(torch.load(CACHE / "t1cnn.pt")); net.eval()
    M = {pre: (lgb.Booster(model_file=str(CACHE / f"{pre}_speed.txt")), lgb.Booster(model_file=str(CACHE / f"{pre}_flow.txt")))
         for pre in ("t1a", "t1c")}
    FE.SCEN = "split"
    for p in panels():
        for s in SPLITS:
            key = f"{p}|{s}"
            if key in res:
                continue
            z, zz, idx = hidden(p, s, False)
            plat = FE.plateau_of(zz["m_speed"]); zz["_plat"] = plat
            X = FE.build(p, zz, idx, prof=FE.profile(p, "train"))
            ys, yf, ln = z["m_speed"][idx], z["m_flow"][idx], X.lanes.to_numpy()
            ens = [0.0, 0.0]
            for pre, w in (("t1a", 0.75), ("t1c", 0.25)):
                ms, mf = M[pre]
                ens[0] = ens[0] + w * (X.speed_lin.to_numpy() + ms.predict(X[ms.feature_name()]))
                ens[1] = ens[1] + w * (X.flow_lin.to_numpy() + mf.predict(X[mf.feature_name()]) * ln)
            T, cap = tensors(p, zz, plat)
            d, t, l = idx
            cs, cfl = np.empty(len(d), np.float32), np.empty(len(d), np.float32)
            for dd in np.unique(d):
                pr = predict_day(net, T[dd])
                m = d == dd
                cs[m] = (pr[0, t[m], l[m]] + 1.0) * plat[l[m]]
                cfl[m] = pr[1, t[m], l[m]] * cap[l[m]]
            r = {}
            for w in (0.0, 0.2, 0.35, 0.5, 1.0):
                sp = (1 - w) * ens[0] + w * cs
                fl = (1 - w) * ens[1] + w * np.clip(cfl, 0, None)
                r[str(w)] = [float(np.mean((sp - ys) ** 2)), float(np.mean(((fl - yf) / ln) ** 2))]
            res[key] = r
            out.write_text(json.dumps(res))
            print(key, {k: (round(v[0] ** .5, 3), round(v[1] ** .5, 2)) for k, v in r.items()}, flush=True)
    dfr = pd.DataFrame([dict(panel=k.split("|")[0], split=k.split("|")[1], w=float(n), speed=v[0] ** .5, flow=v[1] ** .5)
                        for k, r in res.items() for n, v in r.items()])
    print(dfr.groupby(["w", "split"])[["speed", "flow"]].mean().unstack("split").round(4))
    for w in (0.2, 0.35, 0.5):
        a = dfr[dfr.w == 0.0].set_index(["panel", "split"]); b = dfr[dfr.w == w].set_index(["panel", "split"])
        print(f"w={w}: speed better on {int((b.speed < a.speed).sum())}/{len(a)}, flow better on {int((b.flow < a.flow).sum())}/{len(a)}")


if __name__ == "__main__" and sys.argv[1] == "eval":
    evaluate()
