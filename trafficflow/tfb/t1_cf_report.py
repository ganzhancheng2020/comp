"""Report for the T1 common-mode A/B (needs t1ab_base.npy / t1ab_cf.npy from t1_cf_ab).
S_LWR proxy: target cells are mostly isolated (neighbours observed), so sum|dN residual| ~ 2 sum|N error| at target
cells, N = q/v * length. The relative change of sum|N error| scales the (1 - S_LWR) term."""
import numpy as np
import pandas as pd

from .data import CACHE, network

va = pd.read_parquet(CACHE / "t1_va.parquet", columns=["panel", "y_speed", "y_flow", "lanes", "len", "plat", "regime"])
P = {n: np.load(CACHE / f"t1ab_{n}.npy") for n in ("base", "cf")}
ys, yf = va.y_speed.to_numpy(), va.y_flow.to_numpy()
cong = ys < 0.8 * va.plat.to_numpy()
print("congested share", cong.mean().round(4))
for n, (s, f) in P.items():
    eN = np.abs(va.len.to_numpy() * (f / np.maximum(s, 1) - yf / np.maximum(ys, 1)))
    rs = lambda m: np.sqrt(np.mean((s[m] - ys[m]) ** 2))
    print(f"{n:5s} speed RMSE all {rs(np.ones(len(s), bool)):.4f} free {rs(~cong):.4f} congested {rs(cong):.4f} | "
          f"flow/lane RMSE {np.sqrt(np.mean(((f - yf) / va.lanes.to_numpy()) ** 2)):.3f} | sum|N err| {eN.sum():.1f}")
eb = np.abs(va.len.to_numpy() * (P["base"][1] / np.maximum(P["base"][0], 1) - yf / np.maximum(ys, 1)))
ec = np.abs(va.len.to_numpy() * (P["cf"][1] / np.maximum(P["cf"][0], 1) - yf / np.maximum(ys, 1)))
r = pd.DataFrame({"panel": va.panel, "b": eb, "c": ec}).groupby("panel").sum()
r["ratio"] = r.c / r.b
print(r.ratio.round(4).to_dict(), "mean ratio", r.ratio.mean().round(4))
