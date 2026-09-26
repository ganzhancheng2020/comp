"""Grid utilities: background detection, connected components, geometry."""
from collections import Counter

import numpy as np
from scipy import ndimage

Grid = np.ndarray

_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
_STRUCT8 = np.ones((3, 3), dtype=int)


def to_np(g):
    return np.array(g, dtype=np.int8)


def to_list(a):
    return [[int(v) for v in row] for row in a]


def background(a: Grid) -> int:
    """Most frequent color; ties broken toward 0."""
    vals, counts = np.unique(a, return_counts=True)
    best = counts.max()
    cands = vals[counts == best]
    return 0 if 0 in cands else int(cands[0])


def components(a: Grid, bg: int, diag: bool = False, by_color: bool = True):
    """Connected components of non-background cells.

    Returns list of dicts with mask, color(s), bbox, size.
    """
    objs = []
    struct = _STRUCT8 if diag else _STRUCT4
    if by_color:
        for c in np.unique(a):
            if c == bg:
                continue
            lab, n = ndimage.label(a == c, structure=struct)
            for i in range(1, n + 1):
                objs.append(_obj(a, lab == i))
    else:
        lab, n = ndimage.label(a != bg, structure=struct)
        for i in range(1, n + 1):
            objs.append(_obj(a, lab == i))
    return objs


def _obj(a, mask):
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    colors = Counter(a[mask].tolist())
    return {
        "mask": mask,
        "bbox": (int(r0), int(r1), int(c0), int(c1)),
        "size": int(mask.sum()),
        "color": colors.most_common(1)[0][0],
        "ncolors": len(colors),
    }


def crop(a: Grid, bbox):
    r0, r1, c0, c1 = bbox
    return a[r0:r1, c0:c1]


def nonbg_bbox(a: Grid, bg: int):
    rows, cols = np.nonzero(a != bg)
    if len(rows) == 0:
        return None
    return (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)


def label_map(a: Grid, bg: int, diag: bool = False, by_color: bool = True):
    """Integer label per cell (0 = background) plus list of component sizes."""
    struct = _STRUCT8 if diag else _STRUCT4
    out = np.zeros(a.shape, dtype=np.int32)
    sizes = [0]
    nxt = 1
    if by_color:
        for c in np.unique(a):
            if c == bg:
                continue
            lab, n = ndimage.label(a == c, structure=struct)
            for i in range(1, n + 1):
                m = lab == i
                out[m] = nxt
                sizes.append(int(m.sum()))
                nxt += 1
    else:
        lab, n = ndimage.label(a != bg, structure=struct)
        out = lab.astype(np.int32)
        sizes = [0] + [int((lab == i).sum()) for i in range(1, n + 1)]
    return out, sizes


def enclosed(a: Grid, bg: int):
    """Mask of background cells not 4-connected to the border."""
    lab, n = ndimage.label(a == bg, structure=_STRUCT4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    m = (lab > 0) & ~np.isin(lab, list(border))
    return m


def separators(a: Grid):
    """Detect full rows/cols of a single color forming grid lines.

    Returns (color, row_idx list, col_idx list) or None.
    """
    h, w = a.shape
    best = None
    for c in np.unique(a):
        rows = [r for r in range(h) if (a[r] == c).all()]
        cols = [k for k in range(w) if (a[:, k] == c).all()]
        if not rows and not cols:
            continue
        if len(rows) == h or len(cols) == w:
            continue
        score = len(rows) + len(cols)
        if best is None or score > best[0]:
            best = (score, int(c), rows, cols)
    if best is None:
        return None
    return best[1], best[2], best[3]


def split_cells(a: Grid):
    """Split grid into sub-grids separated by separator lines."""
    sep = separators(a)
    if sep is None:
        return None
    c, rows, cols = sep
    h, w = a.shape

    def spans(idx, n):
        out, start = [], 0
        for i in idx + [n]:
            if i > start:
                out.append((start, i))
            start = i + 1
        return out

    rs, cs = spans(rows, h), spans(cols, w)
    cells = [[a[r0:r1, c0:c1] for (c0, c1) in cs] for (r0, r1) in rs]
    return c, cells


def mode_color(a: Grid, exclude=()):
    cnt = Counter(a.ravel().tolist())
    for c, _ in cnt.most_common():
        if c not in exclude:
            return c
    return None
