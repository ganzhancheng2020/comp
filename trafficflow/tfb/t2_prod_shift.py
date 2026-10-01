"""Out-of-scenario check of the production ongoing models (vt inputs): LightGBM alone, CNN mean alone, and blends,
on ongoing windows mined from the validation/private masked layer (evaluation only), selector-style filter.
Usage: python -m tfb.t2_prod_shift"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from . import t2_cnn as C
from . import t2_ongoing as og
from .data import CACHE, load, network
from .t2 import T2_PANELS as T2P
from .t2_ongoing_shift import mined_windows, iou
from .t2_prod import bnecks


def main():
    m = lgb.Booster(model_file=str(CACHE / "t2p_lgb_vt.txt"))
    feats = m.feature_name()
    nets = []
    for sd in range(4):
        f = CACHE / f"t2p_cnn_vt_s{sd}.pt"
        if f.exists():
            n = C.Net(); n.load_state_dict(torch.load(f)); nets.append(n)
    print("cnn seeds", len(nets), flush=True)
    bn = bnecks()
    rng = np.random.default_rng(0)
    rows = []
    for p in T2P:
        net = network(p); vc = 0.6 * net.free_speed_kmh.to_numpy(); cap = net.capacity_vph.to_numpy()
        bm = np.zeros(len(net), np.float32); bm[bn[p]] = 1
        for s in ("validation", "private"):
            z = load(p, s); dow = pd.to_datetime(z["dates"]).dayofweek.to_numpy()
            for d, T, tru, e, pers in mined_windows(p, s, z, rng):
                hs, hf = z["m_speed"][d, T - 12:T + 1].astype(float), z["m_flow"][d, T - 12:T + 1].astype(float)
                ef = og.early_features(z["m_speed"][d], T, vc)
                pl = np.zeros((6, len(net)), np.float32)
                Xw = og.cell_features(hs, hf, vc, cap, T, dow[d], margin=12, bneck=bn[p], early=ef)
                if Xw is not None:
                    Xw["pid"] = T2P.index(p)
                    pl[Xw.k.to_numpy() - 1, Xw.link.to_numpy()] = m.predict(Xw[feats].to_numpy(np.float32))
                x = C.window_tensor(hs, hf, vc, cap, T, dow[d], bm, T2P.index(p))
                pcs = [C.predict(n, x[None])[0] for n in nets]
                pc = np.mean(pcs, axis=0)
                bl = lambda w, thr=0.5: np.where(pl > 0, w * pl + (1 - w) * pc, pc) > thr
                r = dict(panel=p, split=s, pers=iou(pers, tru, e), lgb=iou(pl > 0.5, tru, e), cnn=iou(pc > 0.5, tru, e),
                         blend=iou(bl(0.5), tru, e), b_t45=iou(bl(0.5, 0.45), tru, e), b_t55=iou(bl(0.5, 0.55), tru, e),
                         b_w3=iou(bl(0.3), tru, e), b_w7=iou(bl(0.7), tru, e))
                for k in range(len(pcs)):
                    r[f"cnn{k}"] = iou(pcs[k] > 0.5, tru, e)
                rows.append(r)
        print(p, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / "t2_prod_shift.csv", index=False)
    cols = [c for c in r.columns if c not in ("panel", "split")]
    print(r.groupby(["split", "panel"])[cols].mean().groupby("split").mean().T.round(4))
    for a, b in (("blend", "lgb"), ("blend", "cnn"), ("b_t45", "blend"), ("b_t55", "blend"), ("b_w3", "blend"), ("b_w7", "blend")):
        dd = r[a] - r[b]
        print(f"{a} - {b}: {dd.mean():+.4f} ± {dd.std() / np.sqrt(len(dd)):.4f}  "
              + " ".join(f"{s} {g.mean():+.4f}" for s, g in dd.groupby(r.split)))


if __name__ == "__main__":
    main()
