"""Learn the MDL feature prior from oracle rules on a set of tasks.

cost(f) = -log2 p(f) / log2(10), i.e. measured in "table entry" units (one
entry stores one of ten colors). p(f) is the Laplace-smoothed frequency with
which f appears in the minimal-DL oracle rules of the given tasks.

Usage: python learn_prior.py out/train_v2_oracle.json [task_id_filter_file] > arcsolver/prior.json
"""
import json
import math
import re
import sys
from collections import Counter

from arcsolver.features import cell_features
import numpy as np


def learn(oracle, ids=None):
    cnt = Counter()
    for tid, r in oracle.items():
        if ids is not None and tid not in ids:
            continue
        if r["expressible"] and "rule" in (r["how"] or ""):
            for f in eval(re.search(r"on (\[.*?\])", r["how"]).group(1)):
                cnt[f] += 1
    names = sorted(cell_features(np.zeros((3, 3), dtype=np.int8)))
    names = sorted(set(names) | set(cnt))
    total = sum(cnt.values()) + len(names)
    unit = math.log2(10)
    prior = {n: round(-math.log2((cnt[n] + 1) / total) / unit, 3) for n in names}
    prior["__default__"] = round(-math.log2(1 / total) / unit, 3)
    return prior


if __name__ == "__main__":
    oracle = json.load(open(sys.argv[1]))
    ids = set(open(sys.argv[2]).read().split()) if len(sys.argv) > 2 else None
    print(json.dumps(learn(oracle, ids), indent=1, sort_keys=True))
