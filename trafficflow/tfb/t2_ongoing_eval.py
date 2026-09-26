import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CACHE
from .t2_ongoing import T2P, decode

df = pd.read_parquet(CACHE / "t2_ongoing.parquet")
df["pid"] = df.panel.map({p: i for i, p in enumerate(T2P)})
# cap windows per panel for speed
rng = np.random.default_rng(0)
keep = []
for p, g in df.groupby("panel"):
    w = g[["d", "T"]].drop_duplicates()
    w = w.sample(min(len(w), 3000), random_state=0)
    keep.append(g.merge(w, on=["d", "T"]))
df = pd.concat(keep, ignore_index=True)
feats = [c for c in df.columns if c not in ("y", "d", "T", "panel", "n_fut_total", "n_fut_out")]
tr, te = df[df.d % 2 == 0], df[df.d % 2 == 1]
print(len(tr), len(te), flush=True)
m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2),
              lgb.Dataset(tr[feats], tr.y, categorical_feature=["pid"]), 600)
m.save_model(str(CACHE / "t2_ongoing_eval.txt"))
te = te.assign(p=m.predict(te[feats]))
res = []
for (p, d, T), g in te.groupby(["panel", "d", "T"]):
    y = g.y.to_numpy().astype(bool)
    out = g.n_fut_out.iloc[0]
    pers = g.lastq.to_numpy() > 0.5
    sel = decode(g.p.to_numpy())
    sel_half = g.p.to_numpy() > 0.5

    def sc(s):
        i = (s & y).sum()
        u = (s | y).sum() + out
        return i / u if u else 1.0
    res.append((p, sc(pers), sc(sel), sc(sel_half)))
r = pd.DataFrame(res, columns=["panel", "pers", "model", "p05"])
r["selectable"] = r.pers <= 0.9
print(r.groupby("panel")[["pers", "model", "p05"]].mean().round(3))
print("all", r[["pers", "model", "p05"]].mean().round(4).to_dict())
print("pers<=0.9", r[r.selectable][["pers", "model", "p05"]].mean().round(4).to_dict())
