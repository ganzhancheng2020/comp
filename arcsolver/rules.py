"""MDL local-rule induction.

Given aligned (input-like grid, output grid) pairs of equal shape, search for a
small set of per-cell features S and a target encoding E such that the lookup
table key(S) -> label(E) is consistent with every training cell. Each
consistent rule gets a description length; predictions from all consistent
rules are pooled with weights exp(-DL) (a crude Bayesian model average).
"""
import bisect
from itertools import combinations

import numpy as np

from .config import on
from .features import COLOR_FEATURES, NONE, cell_features

KEEP, COPY = 11, 12
FEATURE_COST_DEFAULT = 1.5
FEATURE_COST = {}  # learned prior: name -> cost in table-entry units (see learn_prior.py)


def load_prior(path):
    """Load a learned feature prior (JSON mapping feature name -> cost)."""
    import json
    FEATURE_COST.clear()
    FEATURE_COST.update(json.load(open(path)))


def feature_cost(name):
    return FEATURE_COST.get(name, FEATURE_COST.get("__default__", FEATURE_COST_DEFAULT))
MAX_COMPRESSIVE_ENTRIES = 40
MEMORIZATION_PENALTY = 1000.0
MAX_INTERP_KEYS = 64
BASE = 64


class RuleModel:
    def __init__(self, names, subset, enc, table, cost, copy_feat=None):
        self.names, self.subset, self.enc = names, subset, enc
        self.table, self.cost, self.copy_feat = table, cost, copy_feat

    def describe(self):
        feats = [self.names[i] for i in self.subset]
        return f"rule[{self.enc}{':' + self.copy_feat if self.copy_feat else ''}] on {feats} ({len(self.table)} entries, DL={self.cost:.1f})"


def _pack(F, subset):
    key = np.zeros(F.shape[1], dtype=np.int64)
    for i in subset:
        key = key * BASE + F[i]
    return key


def _impurity(key, y):
    ky = key * 16 + y
    u, cnt = np.unique(ky, return_counts=True)
    k = u // 16
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return len(key) - np.maximum.reduceat(cnt, starts).sum(), len(starts)


def _feature_matrix(feat_dicts, names):
    return np.stack([np.concatenate([fd[n].ravel() for fd in feat_dicts]) for n in names]).astype(np.int64)


def learn_rules(pairs, test_inputs, max_rules=40, beam=24, time_budget_feats=3, max_build=120):
    """pairs: list of (x, y) numpy grids with x.shape == y.shape.

    Returns list of (prediction grids for test_inputs, log-weight, RuleModel).
    """
    tr_feats = [cell_features(x) for x, _ in pairs]
    te_feats = [cell_features(x) for x in test_inputs]
    names = sorted(set.intersection(*[set(f) for f in tr_feats + te_feats]))
    # Drop features that are constant across train and test (useless).
    Ftr = _feature_matrix(tr_feats, names)
    Fte = _feature_matrix(te_feats, names)
    keep = [i for i in range(len(names)) if len(np.unique(np.r_[Ftr[i], Fte[i]])) > 1 or names[i] == "c"]
    names = [names[i] for i in keep]
    Ftr, Fte = Ftr[keep], Fte[keep]
    ci = names.index("c")
    out = np.concatenate([y.ravel() for _, y in pairs]).astype(np.int64)
    own = Ftr[ci]
    changed = out != own

    encodings = [("abs", out, None, 0.0)]
    if changed.any() and (~changed).any():
        if on("keep"):
            encodings.append(("keep", np.where(changed, out, KEEP), None, 0.0))
        for j, n in enumerate(names):
            if on("copy") and n in COLOR_FEATURES and n != "c" and (Ftr[j][changed] == out[changed]).all():
                if len(np.unique(out[changed])) > 1:
                    y = np.where(changed, COPY, KEEP)
                    encodings.append(("copy", y, n, 1.0))
    elif not changed.any():
        return []

    results = []
    nf = len(names)
    for enc, y, cf, enc_cost in encodings:
        # Level 1 and 2: exhaustive; level 3: beam from best impure pairs.
        scored = []
        for i in range(nf):
            scored.append(((i,), *_impurity(Ftr[i], y)))
        pairs2 = []
        for i, j in combinations(range(nf), 2):
            imp, nk = _impurity(Ftr[i] * BASE + Ftr[j], y)
            pairs2.append(((i, j), imp, nk))
        scored += pairs2
        if time_budget_feats >= 3:
            best_pairs = sorted([p for p in pairs2 if p[1] > 0], key=lambda t: (t[1], t[2]))[:beam]
            seen = set()
            for (i, j), _, _ in best_pairs:
                base = Ftr[i] * BASE + Ftr[j]
                for k in range(nf):
                    if k in (i, j):
                        continue
                    t = tuple(sorted((i, j, k)))
                    if t in seen:
                        continue
                    seen.add(t)
                    imp, nk = _impurity(base * BASE + Ftr[k], y)
                    if imp == 0:
                        scored.append((t, imp, nk))
        consistent = sorted((t for t in scored if t[1] == 0), key=lambda t: (t[2], len(t[0])))
        for subset, imp, nk in consistent[:max_build]:
            m = _build(names, Ftr, Fte, y, subset, enc, cf, enc_cost, te_feats, test_inputs)
            if m is not None:
                results.append(m)
    results.sort(key=lambda r: r[2].cost)
    return results[:max_rules]


ORDINAL = {"m_size", "o_size", "m_rank_small", "m_rank_large", "o_rank_small", "o_rank_large", "cnt4",
           "cnt8", "same4", "col_rank", "m_ncol", "r", "c_idx", "rb", "cb"}


def _interpolate(names, Ftr, Fte, subset, table, unseen):
    """Resolve unseen test keys by monotone interval interpolation.

    An unseen tuple may take a label if it matches seen tuples on every
    position except one ordinal feature, and the nearest seen values below
    and above it (on that feature) agree on the label. One-sided
    extrapolation is allowed at a higher description-length cost.
    """
    ords = [p for p, i in enumerate(subset) if names[i] in ORDINAL]
    if not ords:
        return {}, 0.0
    seen = {}
    for k, lab in table.items():
        tup = []
        for _ in subset:
            tup.append(k % BASE)
            k //= BASE
        seen[tuple(reversed(tup))] = lab
    # Index seen tuples by (ordinal position, all other positions) -> sorted (value, label).
    index = {}
    for t, lab in seen.items():
        for p in ords:
            index.setdefault((p, t[:p] + t[p + 1:]), []).append((t[p], lab))
    for v in index.values():
        v.sort()
    te_tuples = [tuple(int(Fte[i][j]) for i in subset) for j in range(Fte.shape[1])]
    resolved, cache, cost = {}, {}, 0.0
    for j in np.flatnonzero(unseen):
        u = te_tuples[j]
        if u not in cache:
            if len(cache) >= MAX_INTERP_KEYS:
                cache[u] = (None, 0.0)
                continue
            labs = set()
            extra = 0.0
            for p in ords:
                pts = index.get((p, u[:p] + u[p + 1:]))
                if not pts:
                    continue
                k = bisect.bisect_left([v for v, _ in pts], u[p])
                lo = pts[k - 1][1] if k > 0 else None
                hi = pts[k][1] if k < len(pts) else None
                if lo is not None and hi is not None:
                    if lo == hi:
                        labs.add(lo)
                        extra = max(extra, 1.0)
                    else:
                        labs.add(None)
                else:
                    labs.add(lo if lo is not None else hi)
                    extra = max(extra, 2.0)
            lab = labs.pop() if len(labs) == 1 else None
            cache[u] = (lab, extra)
            if lab is not None:
                cost += extra
        lab = cache[u][0]
        if lab is not None:
            resolved[j] = lab
    return resolved, cost


def _build(names, Ftr, Fte, y, subset, enc, cf, enc_cost, te_feats, test_inputs):
    ktr = _pack(Ftr, subset)
    kte = _pack(Fte, subset)
    table = dict(zip(ktr.tolist(), y.tolist()))
    labels = np.array([table.get(k, -1) for k in kte.tolist()], dtype=np.int64)
    unseen = labels < 0
    interp_cost = 0.0
    if unseen.any() and on("interp"):
        resolved, interp_cost = _interpolate(names, Ftr, Fte, subset, table, unseen)
        for i, lab in resolved.items():
            labels[i] = lab
        unseen = labels < 0
    n_unseen_keys = len(set(kte[unseen].tolist()))
    if unseen.any() and enc == "abs":
        return None  # no sensible default for absolute encodings
    labels[unseen] = KEEP
    # Description length: table entries + feature choice + generalisation penalties.
    cost = len(table) * 1.0 + sum(feature_cost(names[i]) for i in subset) + enc_cost + 4.0 * n_unseen_keys + interp_cost
    if len(table) > MAX_COMPRESSIVE_ENTRIES and on("gate"):
        cost += MEMORIZATION_PENALTY  # fails the compression test: keep only as last resort
    # Reward rules whose training support is large relative to table size.
    own = Fte[names.index("c")]
    pred = np.where(labels == KEEP, own, labels)
    if enc == "copy":
        src = Fte[names.index(cf)]
        pred = np.where(labels == COPY, src, pred)
    if (pred >= 10).any():
        return None
    grids, off = [], 0
    for x in test_inputs:
        n = x.size
        grids.append(pred[off:off + n].reshape(x.shape).astype(np.int8))
        off += n
    return grids, -cost / 2.0, RuleModel(names, subset, enc, table, cost, cf)
