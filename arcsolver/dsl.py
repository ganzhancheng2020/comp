"""Grid-level transforms. Each returns a grid or None when not applicable."""
from collections import Counter

import numpy as np

from .grid import background, components, crop, nonbg_bbox, split_cells

TRANSFORMS = []  # (name, fn, cost)


def T(name, cost=1.0):
    def deco(fn):
        TRANSFORMS.append((name, fn, cost))
        return fn
    return deco


T("id", 0.0)(lambda a: a)
T("rot90")(lambda a: np.rot90(a, 1))
T("rot180")(lambda a: np.rot90(a, 2))
T("rot270")(lambda a: np.rot90(a, 3))
T("flipud")(lambda a: a[::-1])
T("fliplr")(lambda a: a[:, ::-1])
T("transpose")(lambda a: a.T)
T("antitranspose")(lambda a: np.rot90(a, 2).T)


@T("crop_nonbg")
def _crop_nonbg(a):
    bb = nonbg_bbox(a, background(a))
    return None if bb is None else crop(a, bb)


def _crop_obj(select, diag=True, by_color=False):
    def fn(a):
        bg = background(a)
        objs = components(a, bg, diag=diag, by_color=by_color)
        if not objs:
            return None
        o = select(objs, a)
        return None if o is None else crop(a, o["bbox"])
    return fn


def _unique_by(key):
    def sel(objs, a):
        vals = Counter(key(o, a) for o in objs)
        cands = [o for o in objs if vals[key(o, a)] == 1]
        return cands[0] if len(cands) == 1 else None
    return sel


def _argbest(key, largest=True):
    def sel(objs, a):
        ks = [key(o, a) for o in objs]
        b = max(ks) if largest else min(ks)
        c = [o for o, k in zip(objs, ks) if k == b]
        return c[0] if len(c) == 1 else None
    return sel


_bbox_area = lambda o, a: (o["bbox"][1] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][2])
for diag, by_color, tag in ((True, False, "m"), (False, True, "o")):
    T(f"crop_{tag}_largest", 1.5)(_crop_obj(_argbest(lambda o, a: o["size"]), diag, by_color))
    T(f"crop_{tag}_smallest", 1.5)(_crop_obj(_argbest(lambda o, a: o["size"], False), diag, by_color))
    T(f"crop_{tag}_bbox_largest", 1.5)(_crop_obj(_argbest(_bbox_area), diag, by_color))
    T(f"crop_{tag}_unique_color", 1.5)(_crop_obj(_unique_by(lambda o, a: o["color"]), diag, by_color))
    T(f"crop_{tag}_unique_size", 1.5)(_crop_obj(_unique_by(lambda o, a: o["size"]), diag, by_color))
    T(f"crop_{tag}_unique_shape", 2.0)(_crop_obj(
        _unique_by(lambda o, a: crop(o["mask"], o["bbox"]).tobytes() + bytes(crop(o["mask"], o["bbox"]).shape)), diag, by_color))
T("crop_m_most_colors", 1.5)(_crop_obj(_argbest(lambda o, a: len(np.unique(a[o["mask"]]))), True, False))
T("crop_m_densest", 2.0)(_crop_obj(_argbest(lambda o, a: o["size"] / max(1, _bbox_area(o, a))), True, False))


def _crop_color(rank):
    def fn(a):
        bg = background(a)
        cnt = Counter(a[a != bg].tolist())
        if len(cnt) <= rank:
            return None
        order = [c for c, _ in sorted(cnt.items(), key=lambda t: (t[1], t[0]))]
        c = order[rank]
        rows, cols = np.nonzero(a == c)
        return a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return fn


T("crop_rarest_color", 1.5)(_crop_color(0))
T("crop_2nd_rarest_color", 2.0)(_crop_color(1))


def _crop_inside_rarest(a):
    g = _crop_color(0)(a)
    if g is None or g.shape[0] < 3 or g.shape[1] < 3:
        return None
    return g[1:-1, 1:-1]


T("crop_inside_rarest", 2.0)(_crop_inside_rarest)

for k in (2, 3, 4, 5):
    T(f"upscale{k}", 1.0)(lambda a, k=k: np.kron(a, np.ones((k, k), dtype=a.dtype)))


def _downscale(k, mode):
    def fn(a):
        h, w = a.shape
        if h % k or w % k:
            return None
        b = a.reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(h // k, w // k, k * k)
        bg = background(a)
        if mode == "first":
            return b[:, :, 0]
        out = np.empty((h // k, w // k), dtype=a.dtype)
        for i in range(h // k):
            for j in range(w // k):
                v = b[i, j]
                nz = v[v != bg]
                out[i, j] = Counter(nz.tolist()).most_common(1)[0][0] if (mode == "any" and len(nz)) else (
                    Counter(v.tolist()).most_common(1)[0][0] if mode == "maj" else bg)
        return out
    return fn


for k in (2, 3, 4, 5):
    T(f"down{k}_maj", 1.5)(_downscale(k, "maj"))
    T(f"down{k}_any", 1.5)(_downscale(k, "any"))

for n in (1, 2, 3, 4):
    for m in (1, 2, 3, 4):
        if n * m > 1:
            T(f"tile{n}x{m}", 1.0)(lambda a, n=n, m=m: np.tile(a, (n, m)))

T("mirror_h", 1.0)(lambda a: np.hstack([a, a[:, ::-1]]))
T("mirror_h2", 1.0)(lambda a: np.hstack([a[:, ::-1], a]))
T("mirror_v", 1.0)(lambda a: np.vstack([a, a[::-1]]))
T("mirror_v2", 1.0)(lambda a: np.vstack([a[::-1], a]))
T("mirror_4", 1.0)(lambda a: np.vstack([np.hstack([a, a[:, ::-1]]), np.hstack([a[::-1], a[::-1, ::-1]])]))
T("mirror_4b", 1.0)(lambda a: np.vstack([np.hstack([a[::-1, ::-1], a[::-1]]), np.hstack([a[:, ::-1], a])]))
T("rot_4", 1.5)(lambda a: np.vstack([np.hstack([a, np.rot90(a, 3)]), np.hstack([np.rot90(a, 1), np.rot90(a, 2)])])
                if a.shape[0] == a.shape[1] else None)
def _self_kron(a):
    if a.size > 36:
        return None
    bg = background(a)
    h, w = a.shape
    return np.where(np.kron(a != bg, np.ones((h, w), dtype=bool)), np.tile(a, (h, w)), bg).astype(a.dtype)


T("self_kron", 2.0)(_self_kron)


def _parts(a):
    """Sub-grids from separator lines or, failing that, equal halves."""
    s = split_cells(a)
    out = []
    if s is not None:
        _, cells = s
        flat = [c for row in cells for c in row]
        if len(flat) >= 2 and all(c.shape == flat[0].shape for c in flat):
            out.append(("sep", flat))
    h, w = a.shape
    if w % 2 == 0:
        out.append(("lr", [a[:, : w // 2], a[:, w // 2:]]))
    if h % 2 == 0:
        out.append(("tb", [a[: h // 2], a[h // 2:]]))
    return out


def _overlay(kind, which):
    def fn(a):
        bg = background(a)
        for tag, parts in _parts(a):
            if tag != which:
                continue
            st = np.stack(parts)
            nb = st != bg
            if kind == "count":
                return nb.sum(0).astype(a.dtype)
            if kind == "mask_and":
                return nb.all(0).astype(a.dtype)
            if kind == "mask_or":
                return nb.any(0).astype(a.dtype)
            if kind == "mask_xor":
                return (nb.sum(0) == 1).astype(a.dtype)
            if kind == "bits":
                # Each part contributes a bit; rule learner maps codes to colors.
                return sum((nb[i].astype(int) << i) for i in range(min(len(parts), 3))).astype(a.dtype)
            if kind in ("first", "last"):
                order = range(len(parts)) if kind == "last" else range(len(parts) - 1, -1, -1)
                out = np.full(parts[0].shape, bg, dtype=a.dtype)
                for i in order:
                    out = np.where(nb[i], st[i], out)
                return out
            if kind in ("most", "least", "unique"):
                dens = nb.reshape(len(parts), -1).sum(1)
                if kind == "unique":
                    keys = [p.tobytes() for p in parts]
                    cnt = Counter(keys)
                    u = [i for i, k in enumerate(keys) if cnt[k] == 1]
                    return parts[u[0]] if len(u) == 1 else None
                b = dens.max() if kind == "most" else dens.min()
                idx = np.flatnonzero(dens == b)
                return parts[idx[0]] if len(idx) == 1 else None
        return None
    return fn


for which in ("sep", "lr", "tb"):
    for kind in ("count", "bits", "mask_and", "mask_or", "mask_xor", "first", "last", "most", "least", "unique"):
        T(f"parts_{which}_{kind}", 1.5)(_overlay(kind, which))


def _cell_summary(a):
    s = split_cells(a)
    if s is None:
        return None
    sep, cells = s
    out = np.empty((len(cells), len(cells[0])), dtype=a.dtype)
    for i, row in enumerate(cells):
        if len(row) != out.shape[1]:
            return None
        for j, c in enumerate(row):
            v = c[(c != sep)]
            bgc = background(a)
            nz = v[v != bgc]
            out[i, j] = Counter(nz.tolist()).most_common(1)[0][0] if len(nz) else bgc
    return out


T("cell_summary", 1.5)(_cell_summary)


def _remove_separators(a):
    s = split_cells(a)
    if s is None:
        return None
    _, cells = s
    try:
        return np.vstack([np.hstack(row) for row in cells])
    except ValueError:
        return None


T("remove_seps", 1.5)(_remove_separators)


def _dedupe(a):
    rows = [0] + [i for i in range(1, a.shape[0]) if not (a[i] == a[i - 1]).all()]
    b = a[rows]
    cols = [0] + [j for j in range(1, b.shape[1]) if not (b[:, j] == b[:, j - 1]).all()]
    return b[:, cols]


T("dedupe_rc", 1.5)(_dedupe)


def _drop_bg_lines(a):
    bg = background(a)
    rows = [i for i in range(a.shape[0]) if (a[i] != bg).any()]
    cols = [j for j in range(a.shape[1]) if (a[:, j] != bg).any()]
    if not rows or not cols:
        return None
    return a[rows][:, cols]


T("drop_bg_lines", 1.5)(_drop_bg_lines)


def apply(fn, a):
    try:
        r = fn(a)
    except Exception:
        return None
    if r is None or r.ndim != 2 or r.size == 0 or r.shape[0] > 30 or r.shape[1] > 30:
        return None
    return np.ascontiguousarray(r)


# --- Regularity repair: make the input exactly symmetric / periodic. ---

def _best_axis(mask, axis):
    """Mirror axis (as r0+r1 sum, i.e. doubled coordinate) maximising self-overlap."""
    idx = np.nonzero(mask)[axis]
    lo, hi = idx.min(), idx.max()
    best, best_s = None, -1
    for s in range(2 * lo, 2 * hi + 1):
        m = _mirror(mask, axis, s)
        ov = (m & mask).sum()
        if ov > best_s:
            best, best_s = s, ov
    return best


def _mirror(mask, axis, s):
    out = np.zeros_like(mask)
    rr, cc = np.nonzero(mask)
    if axis == 0:
        rr = s - rr
    else:
        cc = s - cc
    ok = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    out[rr[ok], cc[ok]] = True
    return out


def _sym_repair(axis, mode):
    def fn(a):
        bg = background(a)
        out = np.full_like(a, bg)
        changed = False
        for c in np.unique(a):
            if c == bg:
                continue
            mask = a == c
            s = _best_axis(mask, axis)
            m = _mirror(mask, axis, s)
            new = (mask & m) if mode == "and" else (mask | m)
            if mode == "or":
                new &= (a == bg) | mask
            changed |= bool((new != mask).any())
            out[new] = c
        return out if changed else None
    return fn


for ax, tag in ((1, "v"), (0, "h")):
    for mode in ("and", "or"):
        T(f"sym_repair_{tag}_{mode}", 1.5)(_sym_repair(ax, mode))


def _period_fix_line(line, max_err=0.25):
    """Return line with its dominant periodic pattern enforced (or None)."""
    n = len(line)
    best = None
    for trim in range(0, 4):
        seg = line[trim:n - trim] if trim else line
        m = len(seg)
        for p in range(1, m // 2 + 1):
            if m < 3 * p:
                break
            fixed = seg.copy()
            err = 0
            for r in range(p):
                vals = seg[r::p]
                bc = np.bincount(vals, minlength=10)
                maj = bc.argmax()
                if bc[maj] * 2 <= len(vals) and len(vals) > 2:
                    err = m
                    break
                err += len(vals) - bc[maj]
                fixed[r::p] = maj
            if err > max_err * m:
                continue
            key = (round(err / m, 3), -m, p)  # lowest error rate, then widest coverage, then shortest period
            if best is None or key < best[0]:
                full = line.copy()
                full[trim:n - trim if trim else n] = fixed
                best = (key, full)
            break  # shortest valid period for this trim
    return None if best is None else best[1]


def _period_repair(axis):
    def fn(a):
        b = a if axis == 1 else a.T
        out = b.copy()
        for i in range(b.shape[0]):
            f = _period_fix_line(b[i])
            if f is not None:
                out[i] = f
        out = out if axis == 1 else out.T
        return out if (out != a).any() else None
    return fn


T("period_repair_rows", 1.5)(_period_repair(1))
T("period_repair_cols", 1.5)(_period_repair(0))
