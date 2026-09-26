"""Expressibility vs. learnability of compact local rules on ARC tasks.

For every task, and for every shape-preserving transform T, we ask:
  * expressible: is there a compact rule (<= MAX entries) consistent with
    the train pairs AND the (held-out) test pairs? (oracle, uses test outputs)
  * learned: does the solver, using train pairs only, output the right answer?
The difference is the induction gap: the rule class contains the answer but
the train examples under-determine it.

Usage: python analysis_oracle.py data_arc2/data/evaluation out/eval_v2.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from arcsolver import dsl
from arcsolver.grid import to_np
from arcsolver.rules import MAX_COMPRESSIVE_ENTRIES, learn_rules
from arcsolver.solver import _apply_all


def oracle(path):
    task = json.load(open(path))
    tid = os.path.basename(path)[:-5]
    ins = [to_np(p["input"]) for p in task["train"] + task["test"]]
    outs = [to_np(p["output"]) for p in task["train"] + task["test"]]
    ntr = len(task["train"])
    res = {"shape": False, "expressible": False, "how": None}
    for name, fn, cost in dsl.TRANSFORMS:
        xs = _apply_all(fn, ins)
        if xs is None or any(x.shape != y.shape for x, y in zip(xs, outs)):
            continue
        res["shape"] = True
        if all((x == y).all() for x, y in zip(xs, outs)):
            res.update(expressible=True, how=f"exact:{name}")
            break
        try:
            # Fit on ALL pairs (train+test); a dummy test input keeps the API.
            rules = learn_rules(list(zip(xs, outs)), [xs[0]], max_rules=1)
        except Exception:
            continue
        if rules and len(rules[0][2].table) <= MAX_COMPRESSIVE_ENTRIES:
            res.update(expressible=True, how=f"{name}+{rules[0][2].describe()}")
            break
    return tid, res


def main():
    d, solved_path = sys.argv[1], sys.argv[2]
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    with Pool(4) as p:
        res = dict(p.map(oracle, files, chunksize=1))
    solved = json.load(open(solved_path))
    n = len(res)
    shape = sum(r["shape"] for r in res.values())
    expr = sum(r["expressible"] for r in res.values())
    learned = sum(1 for t, r in res.items() if solved[t]["score"] > 0)
    learned_expr = sum(1 for t, r in res.items() if r["expressible"] and solved[t]["score"] > 0)
    print(f"tasks={n} shape_reachable={shape} expressible={expr} solved={learned} solved&expressible={learned_expr}")
    for t, r in sorted(res.items()):
        if r["expressible"] and solved[t]["score"] == 0:
            print("GAP", t, r["how"][:120])
    out = solved_path.replace(".json", "_oracle.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
