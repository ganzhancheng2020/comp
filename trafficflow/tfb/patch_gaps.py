"""Replace the gap-slot rows of a state file with the gap-specialist model."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CACHE, panels
from .submit import OUT
from .t1_model import gap_rows, predict_rows

src, dst = sys.argv[1], sys.argv[2]
st = pd.read_csv(OUT / src)
import os
gm = {k: lgb.Booster(model_file=str(CACHE / f"{os.environ.get('GAP_MODEL', 't1gap')}_{k}.txt")) for k in ("speed", "flow")}
off = 0
for p in panels():
    for s in ("validation", "private"):
        n = int(((st.panel == p).to_numpy() & st.timestamp.str.startswith("2031-03" if s == "validation" else "2031-04").to_numpy()).sum())
        rows = gap_rows(p, s)
        if len(rows):
            r, sp, fl = predict_rows(gm, p, s, rows)
            idx = off + r
            assert (st.panel.to_numpy()[idx] == p).all()
            print(p, s, len(rows), "old", st.loc[idx, "flow_vph"].mean().round(1), "new", fl.mean().round(1), flush=True)
            st.loc[idx, "speed_kmh"] = np.round(sp, 4)
            st.loc[idx, "flow_vph"] = np.round(fl, 3)
        off += n
assert off == len(st)
st.to_csv(OUT / dst, index=False)
