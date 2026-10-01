"""Merge per-task files into the Kaggle upload and zip it.
Usage: python -m tfb.make_sub <state.csv> <queue.csv> <odme.csv> <name>   (files in out/tfb) -> out/tfb/sub_<name>.zip"""
import sys
import zipfile

import pandas as pd

from .submit import OUT, merge

if __name__ == "__main__":
    state, queue, odme, name = sys.argv[1:5]
    path = merge(str(OUT / state), str(OUT / queue), str(OUT / odme), name)
    s = pd.read_csv(path)
    assert s.notna().all().all(), "blank values"
    print("rows", len(s), s.task.value_counts().to_dict(), flush=True)
    with zipfile.ZipFile(OUT / f"sub_{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname=f"submission_{name}.csv")
    print("wrote", OUT / f"sub_{name}.zip")
