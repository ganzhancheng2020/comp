"""Per-cell feature extraction.

Every feature is an integer array with the same shape as the grid. Values are
small non-negative integers (colors are 0..9; "none/outside" is encoded as 10)
so feature tuples can be packed into a single int64 key.
"""
import numpy as np
from scipy import ndimage

from .grid import background, enclosed, label_map

NONE = 10  # "outside grid" / "no such cell"
DIRS4 = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
DIRS8 = dict(DIRS4, ul=(-1, -1), ur=(-1, 1), dl=(1, -1), dr=(1, 1))

# Features whose value is a color (can be used as copy source for targets).
COLOR_FEATURES = set()


def _shift(a, dr, dc, fill=NONE):
    h, w = a.shape
    out = np.full((h, w), fill, dtype=np.int16)
    rs = slice(max(0, -dr), min(h, h - dr))
    rd = slice(max(0, dr), min(h, h + dr))
    cs = slice(max(0, -dc), min(w, w - dc))
    cd = slice(max(0, dc), min(w, w + dc))
    out[rs, cs] = a[rd, cd]
    return out


def _ray(a, bg, dr, dc):
    """First non-background color seen walking from each cell in (dr, dc)."""
    h, w = a.shape
    out = np.full((h, w), NONE, dtype=np.int16)
    rows = range(h) if dr <= 0 else range(h - 1, -1, -1)
    cols = range(w) if dc <= 0 else range(w - 1, -1, -1)
    # Dynamic programming over the walk direction.
    for r in rows:
        for c in cols:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[r, c] = a[rr, cc] if a[rr, cc] != bg else out[rr, cc]
    return out


def _line_color(a, bg, axis):
    """Most common non-bg color along the row (axis=1) or column (axis=0)."""
    h, w = a.shape
    out = np.full((h, w), NONE, dtype=np.int16)
    n = h if axis == 1 else w
    for i in range(n):
        line = a[i] if axis == 1 else a[:, i]
        vals = line[line != bg]
        if len(vals):
            c = np.bincount(vals, minlength=10).argmax()
            if axis == 1:
                out[i, :] = c
            else:
                out[:, i] = c
    return out


def cell_features(a: np.ndarray, bg: int = None) -> dict:
    a = a.astype(np.int16)
    if bg is None:
        bg = background(a)
    h, w = a.shape
    F = {}

    def color_feat(name, arr):
        F[name] = arr
        COLOR_FEATURES.add(name)

    color_feat("c", a)
    for k, (dr, dc) in DIRS8.items():
        color_feat("n_" + k, _shift(a, dr, dc))
    for k, (dr, dc) in DIRS8.items():
        color_feat("ray_" + k, _ray(a, bg, dr, dc))
    color_feat("mir_h", a[:, ::-1].copy())
    color_feat("mir_v", a[::-1, :].copy())
    color_feat("mir_hv", a[::-1, ::-1].copy())
    if h == w:
        color_feat("mir_t", a.T.copy())
        color_feat("mir_at", a[::-1, ::-1].T.copy())
    color_feat("row_col", _line_color(a, bg, 1))
    color_feat("col_col", _line_color(a, bg, 0))

    rr, cc = np.indices((h, w))
    F["is_bg"] = (a == bg).astype(np.int16)
    for m in (2, 3):
        F[f"r%{m}"] = (rr % m).astype(np.int16)
        F[f"c%{m}"] = (cc % m).astype(np.int16)
    F["r"] = np.minimum(rr, 31).astype(np.int16)
    F["c_idx"] = np.minimum(cc, 31).astype(np.int16)
    F["rb"] = np.minimum(h - 1 - rr, 31).astype(np.int16)
    F["cb"] = np.minimum(w - 1 - cc, 31).astype(np.int16)
    F["border"] = ((rr == 0) | (cc == 0) | (rr == h - 1) | (cc == w - 1)).astype(np.int16)
    F["diag"] = ((rr == cc) | (rr + cc == w - 1)).astype(np.int16) if h == w else (rr == cc).astype(np.int16)

    nonbg = (a != bg).astype(np.int16)
    k8 = np.ones((3, 3), dtype=np.int16)
    k8[1, 1] = 0
    F["cnt8"] = ndimage.convolve(nonbg, k8, mode="constant").astype(np.int16)
    k4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int16)
    F["cnt4"] = ndimage.convolve(nonbg, k4, mode="constant").astype(np.int16)
    same4 = np.zeros((h, w), dtype=np.int16)
    for dr, dc in DIRS4.values():
        same4 += (_shift(a, dr, dc) == a).astype(np.int16)
    F["same4"] = same4

    F["enclosed"] = enclosed(a, bg).astype(np.int16)
    # Between two cells of the same color horizontally / vertically.
    rl, rr_ = F["ray_l"], F["ray_r"]
    F["between_h"] = np.where((rl == rr_) & (rl != NONE), rl, NONE).astype(np.int16)
    ru, rd = F["ray_u"], F["ray_d"]
    F["between_v"] = np.where((ru == rd) & (ru != NONE), ru, NONE).astype(np.int16)
    COLOR_FEATURES.update(["between_h", "between_v"])

    # Object-level features (single-color 4-connected components).
    for tag, diag, byc in (("o", False, True), ("m", True, False)):
        lab, sizes = label_map(a, bg, diag=diag, by_color=byc)
        sizes = np.array(sizes)
        size_map = sizes[lab]
        F[tag + "_size"] = np.minimum(size_map, 31).astype(np.int16)
        uniq = sorted(set(sizes[1:].tolist()))
        if uniq:
            rank_small = {s: i for i, s in enumerate(uniq)}
            rank_large = {s: i for i, s in enumerate(reversed(uniq))}
            F[tag + "_rank_small"] = np.vectorize(lambda s: min(rank_small.get(s, 30), 30) if s else NONE + 20)(size_map).astype(np.int16)
            F[tag + "_rank_large"] = np.vectorize(lambda s: min(rank_large.get(s, 30), 30) if s else NONE + 20)(size_map).astype(np.int16)
        else:
            F[tag + "_rank_small"] = np.full((h, w), NONE + 20, dtype=np.int16)
            F[tag + "_rank_large"] = np.full((h, w), NONE + 20, dtype=np.int16)
        n = lab.max()
        if n:
            idx = np.arange(1, n + 1)
            touch = np.zeros(n + 1, dtype=np.int16)
            border_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            touch[border_labels] = 1
            touch[0] = 0
            F[tag + "_touch"] = touch[lab]
            if not byc:
                ncol = np.zeros(n + 1, dtype=np.int16)
                mcol = np.full(n + 1, NONE, dtype=np.int16)
                for i in idx:
                    vals = a[lab == i]
                    bc = np.bincount(vals, minlength=10)
                    ncol[i] = (bc > 0).sum()
                    # least-frequent color within the object (the "marker")
                    nz = np.nonzero(bc)[0]
                    mcol[i] = nz[np.argmin(bc[nz])]
                F[tag + "_ncol"] = ncol[lab]
                F[tag + "_minor"] = mcol[lab]
                COLOR_FEATURES.add(tag + "_minor")
        else:
            F[tag + "_touch"] = np.zeros((h, w), dtype=np.int16)
            if not byc:
                F[tag + "_ncol"] = np.zeros((h, w), dtype=np.int16)
                F[tag + "_minor"] = np.full((h, w), NONE, dtype=np.int16)
                COLOR_FEATURES.add(tag + "_minor")

    # Color frequency rank of own color (0 = most common).
    counts = np.bincount(a.ravel(), minlength=11)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(11, dtype=np.int16)
    rank[order] = np.arange(11)
    F["col_rank"] = rank[a]
    return F
