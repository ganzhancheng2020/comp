"""MDL local-rule induction.

Given aligned (input-like grid, output grid) pairs of equal shape, search for a
small set of per-cell features S and a target encoding E such that the lookup
table key(S) -> label(E) is consistent with every training cell. Each
consistent rule gets a description length; predictions from all consistent
rules are pooled with weights exp(-DL) (a crude Bayesian model average).
"""
from itertools import combinations

import numpy as np

from .features import COLOR_FEATURES, NONE, cell_features

KEEP, COPY = 11, 12
MAX_COMPRESSIVE_ENTRIES = 40
MEMORIZATION_PENALTY = 1000.0
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


def learn_rules(pairs, test_inputs, max_rules=40, beam=24, time_budget_feats=3):
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
        encodings.append(("keep", np.where(changed, out, KEEP), None, 0.0))
        for j, n in enumerate(names):
            if n in COLOR_FEATURES and n != "c" and (Ftr[j][changed] == out[changed]).all():
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
        for subset, imp, nk in scored:
            if imp:
                continue
            m = _build(names, Ftr, Fte, y, subset, enc, cf, enc_cost, te_feats, test_inputs)
            if m is not None:
                results.append(m)
    results.sort(key=lambda r: r[2].cost)
    return results[:max_rules]


def _build(names, Ftr, Fte, y, subset, enc, cf, enc_cost, te_feats, test_inputs):
    ktr = _pack(Ftr, subset)
    kte = _pack(Fte, subset)
    table = dict(zip(ktr.tolist(), y.tolist()))
    labels = np.array([table.get(k, -1) for k in kte.tolist()], dtype=np.int64)
    unseen = labels < 0
    n_unseen_keys = len(set(kte[unseen].tolist()))
    if unseen.any() and enc == "abs":
        return None  # no sensible default for absolute encodings
    labels[unseen] = KEEP
    # Description length: table entries + feature choice + unseen penalty.
    cost = len(table) * 1.0 + 1.5 * len(subset) + enc_cost + 4.0 * n_unseen_keys
    if len(table) > MAX_COMPRESSIVE_ENTRIES:
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
