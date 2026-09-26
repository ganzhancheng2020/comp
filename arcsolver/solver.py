"""Top-level solver: transform search + MDL local-rule induction + model averaging."""
import os
import time
from collections import defaultdict

import numpy as np

from . import dsl
from .config import on
from .grid import to_list, to_np
from .rules import learn_rules, load_prior

_PRIOR = os.environ.get("ARC_PRIOR_PATH") or os.path.join(os.path.dirname(__file__), "prior.json")
if os.path.exists(_PRIOR) and os.environ.get("ARC_PRIOR", "1") != "0":
    load_prior(_PRIOR)

GEOM = [t for t in dsl.TRANSFORMS if t[0] in (
    "rot90", "rot180", "rot270", "flipud", "fliplr", "transpose", "antitranspose", "crop_nonbg",
    "dedupe_rc", "drop_bg_lines", "remove_seps")]


def _key(grids):
    return tuple((g.shape, g.tobytes()) for g in grids)


def _apply_all(fn, grids):
    out = []
    for g in grids:
        r = dsl.apply(fn, g)
        if r is None:
            return None
        out.append(r)
    return out


def solve_task(task, time_limit=30.0, max_rule_transforms=6, verbose=False):
    t0 = time.time()
    tr_in = [to_np(p["input"]) for p in task["train"]]
    tr_out = [to_np(p["output"]) for p in task["train"]]
    te_in = [to_np(p["input"]) for p in task["test"]]

    votes = defaultdict(lambda: -np.inf)
    preds = {}
    explain = {}

    def add(grids, logw, why):
        k = _key(grids)
        # Bayesian model averaging pools evidence; the ablation keeps only the best rule.
        votes[k] = np.logaddexp(votes[k], logw) if on("bma") else max(votes[k], logw)
        preds[k] = grids
        if k not in explain or logw > explain[k][0]:
            explain[k] = (logw, why)

    # 1. Single transforms (exact, or as a basis for rule learning).
    shape_ok = []
    seen_inter = set()
    for name, fn, cost in dsl.TRANSFORMS:
        xs = _apply_all(fn, tr_in)
        if xs is None or any(x.shape != y.shape for x, y in zip(xs, tr_out)):
            continue
        ts = _apply_all(fn, te_in)
        if ts is None:
            continue
        if all((x == y).all() for x, y in zip(xs, tr_out)):
            add(ts, -cost, f"exact:{name}")
            continue
        k = _key(xs) + _key(ts)
        if k in seen_inter:
            continue
        seen_inter.add(k)
        acc = np.mean([(x == y).mean() for x, y in zip(xs, tr_out)])
        shape_ok.append((-(acc), cost, name, xs, ts))

    # 2. Depth-2 exact compositions (cheap check only).
    if not votes:
        for n1, f1, c1 in dsl.TRANSFORMS:
            if time.time() - t0 > time_limit * 0.3:
                break
            xs1 = _apply_all(f1, tr_in)
            if xs1 is None or n1 == "id":
                continue
            for n2, f2, c2 in GEOM:
                xs = _apply_all(f2, xs1)
                if xs is None or any(x.shape != y.shape for x, y in zip(xs, tr_out)):
                    continue
                if all((x == y).all() for x, y in zip(xs, tr_out)):
                    ts1 = _apply_all(f1, te_in)
                    ts = ts1 and _apply_all(f2, ts1)
                    if ts:
                        add(ts, -(c1 + c2), f"exact:{n1}>{n2}")

    # 3. MDL rule induction on top of shape-preserving transforms.
    shape_ok.sort(key=lambda t: (t[0], t[1]))
    for negacc, cost, name, xs, ts in shape_ok[:max_rule_transforms]:
        if time.time() - t0 > time_limit:
            break
        try:
            rules = learn_rules(list(zip(xs, tr_out)), ts)
        except Exception as e:  # keep going on odd inputs
            if verbose:
                print("rule error", name, e)
            continue
        for grids, w, model in rules:
            add(grids, w - cost, f"{name}+{model.describe()}")

    ranked = sorted(votes, key=lambda k: -votes[k])
    attempts = [preds[k] for k in ranked[:2]]
    info = [explain[k][1] for k in ranked[:2]]
    while len(attempts) < 2:
        attempts.append(te_in)
        info.append("fallback:identity")
    out = []
    for i in range(len(te_in)):
        out.append({"attempt_1": to_list(attempts[0][i]), "attempt_2": to_list(attempts[1][i])})
    return out, info
