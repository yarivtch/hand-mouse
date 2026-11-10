"""
Smart shapes: turn a rough freehand stroke into a clean shape.

recognize(points)            -> shape dict or None (keep freehand)
merge_arrow_head(prev, pts)  -> arrow shape when `pts` is a small "V" drawn at
                                the end of a previous line, else None

A shape dict is {"kind": str, "lines": [polyline, ...], "closed": [bool, ...]}
where each polyline is a list of (x, y) tuples in the same units as the input.
Pure numpy + OpenCV, no GUI, so it can be tested on its own.
"""

import math

import cv2
import numpy as np

MIN_SIZE = 40            # strokes smaller than this (px) stay freehand (writing, dots)
FIT_LIMIT = 0.05         # max mean distance to the fitted shape, as a fraction of size
LINE_DEVIATION = 0.07    # max sideways wobble of a straight line, fraction of length
SNAP_ANGLE = 8           # degrees: lines / rectangles this close to level get leveled


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def _resample(pts, n=64):
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 2:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] == 0:
        return p[:1]
    t = np.linspace(0, cum[-1], n)
    return np.column_stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])])


def _path_len(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _dist_to_polyline(p, poly, closed):
    """Mean distance from points p (N,2) to polyline poly (M,2)."""
    q = np.asarray(poly, dtype=np.float64)
    if closed:
        q = np.vstack([q, q[:1]])
    a, b = q[:-1], q[1:]
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(1), 1e-9)
    ap = p[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(2) / ab2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    d = np.hypot(*(p[:, None, :] - proj).transpose(2, 0, 1))
    return d.min(1)


def _ellipse_points(cx, cy, ax, ay, ang_deg, n=96):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    x, y = ax * np.cos(t), ay * np.sin(t)
    return np.column_stack([cx + x * c - y * s, cy + x * s + y * c])


def _snap_angle(a):
    """Angle (radians) snapped to the nearest 0/45/90... when within SNAP_ANGLE."""
    step = math.pi / 4
    near = round(a / step) * step
    return near if abs(a - near) < math.radians(SNAP_ANGLE) else a


def _tolist(arr):
    return [(float(x), float(y)) for x, y in arr]


def _arrow_head(tip, direction, size):
    """Two wing points for an arrowhead at `tip` pointing along `direction`."""
    ang = math.atan2(direction[1], direction[0])
    wings = []
    for da in (math.radians(150), -math.radians(150)):
        wings.append((tip[0] + size * math.cos(ang + da), tip[1] + size * math.sin(ang + da)))
    return [wings[0], tuple(tip), wings[1]]


# --------------------------------------------------------------------------- #
# individual recognizers
# --------------------------------------------------------------------------- #

def _as_line(p, size):
    a, b = p[0], p[-1]
    chord = math.hypot(*(b - a))
    if chord < MIN_SIZE:
        return None
    dev = _dist_to_polyline(p, [a, b], False).max() / chord
    if dev > LINE_DEVIATION or _path_len(p) / chord > 1.15:
        return None
    ang = _snap_angle(math.atan2(b[1] - a[1], b[0] - a[0]))
    mid = (a + b) / 2
    h = chord / 2
    d = np.array([math.cos(ang), math.sin(ang)]) * h
    return {"kind": "line", "lines": [_tolist([mid - d, mid + d])], "closed": [False]}


def _as_arrow(p, size):
    """Single-stroke arrow: a shaft, then a head drawn at the far end."""
    start = p[0]
    d0 = np.hypot(*(p - start).T)
    tip_i = int(d0.argmax())
    if tip_i < len(p) * 0.4 or tip_i > len(p) - 6:
        return None
    shaft, tail = p[: tip_i + 1], p[tip_i:]
    tip = p[tip_i]
    shaft_len = math.hypot(*(tip - start))
    if shaft_len < MIN_SIZE * 1.5:
        return None
    tail_len = _path_len(tail)
    tail_reach = np.hypot(*(tail - tip).T).max()
    if not (0.15 * shaft_len < tail_len < 1.2 * shaft_len) or tail_reach > 0.45 * shaft_len:
        return None
    if tail_reach < 0.06 * shaft_len:
        return None
    # the head must fold back: tail points sit behind the tip along the shaft
    u = (tip - start) / shaft_len
    back = ((tail - tip) @ u)
    if back.min() > -0.04 * shaft_len:
        return None
    head = 0.18 * shaft_len
    straight = _dist_to_polyline(shaft, [start, tip], False).max() / shaft_len < LINE_DEVIATION
    if straight:
        ang = _snap_angle(math.atan2(u[1], u[0]))
        u = np.array([math.cos(ang), math.sin(ang)])
        tip = start + u * shaft_len
        shaft_line = [start, tip]
        direction = u
    else:  # curved arrow: keep the shaft, smooth it a little
        sm = _resample(shaft, 40)
        k = np.ones(5) / 5
        sm = np.column_stack([np.convolve(np.pad(sm[:, i], 2, mode="edge"), k, "valid") for i in (0, 1)])
        sm[0], sm[-1] = start, tip
        shaft_line = sm
        direction = sm[-1] - sm[-6]
    return {"kind": "arrow",
            "lines": [_tolist(shaft_line), _arrow_head(tuple(tip), direction, head)],
            "closed": [False, False]}


def _closed_candidates(p, size):
    pts = p.astype(np.float32).reshape(-1, 1, 2)
    out = []

    # ellipse / circle
    try:
        (cx, cy), (w, h), ang = cv2.fitEllipse(pts)
        ax, ay = w / 2, h / 2
        if min(ax, ay) > 0 and max(ax, ay) / max(min(ax, ay), 1e-6) < 6:
            if min(ax, ay) / max(ax, ay) > 0.82:      # round enough: perfect circle
                r = (ax + ay) / 2
                ax = ay = r
                ang = 0
            poly = _ellipse_points(cx, cy, ax, ay, ang)
            err = _dist_to_polyline(p, poly, True).mean() / size
            kind = "circle" if ax == ay else "ellipse"
            out.append((err, {"kind": kind, "lines": [_tolist(poly)], "closed": [True]}))
    except cv2.error:
        pass

    # rectangle
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if min(w, h) > MIN_SIZE * 0.5:
        a = math.radians(ang)
        a_s = _snap_angle(a)
        if a_s != a:  # level it
            box = cv2.boxPoints(((cx, cy), (w, h), math.degrees(a_s)))
        else:
            box = cv2.boxPoints(((cx, cy), (w, h), ang))
        err = _dist_to_polyline(p, box, True).mean() / size
        out.append((err * 1.05, {"kind": "rectangle", "lines": [_tolist(box)], "closed": [True]}))

    # triangle
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    for eps in (0.04, 0.06, 0.08, 0.10, 0.12):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) <= 3:
            break
    if len(approx) == 3:
        tri = approx.reshape(-1, 2).astype(np.float64)
        err = _dist_to_polyline(p, tri, True).mean() / size
        out.append((err, {"kind": "triangle", "lines": [_tolist(tri)], "closed": [True]}))
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def recognize(points):
    if len(points) < 8:
        return None
    raw = np.asarray(points, dtype=np.float64)
    x0, y0 = raw.min(0)
    x1, y1 = raw.max(0)
    size = math.hypot(x1 - x0, y1 - y0)
    if size < MIN_SIZE:
        return None
    p = _resample(raw, 64)

    line = _as_line(p, size)
    if line:
        return line

    gap = math.hypot(*(p[-1] - p[0]))
    closed = gap < 0.25 * size
    if closed:
        cands = _closed_candidates(p, size)
        if cands:
            err, shape = min(cands, key=lambda c: c[0])
            if err < FIT_LIMIT:
                return shape
        return None

    arrow = _as_arrow(p, size)
    if arrow:
        return arrow
    return None


def merge_arrow_head(prev_shape, pts):
    """If `pts` is a small V drawn at an end of a straight line, return an arrow."""
    if not prev_shape or prev_shape.get("kind") != "line" or len(pts) < 5:
        return None
    a, b = (np.array(v) for v in prev_shape["lines"][0])
    length = math.hypot(*(b - a))
    p = _resample(pts, 32)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    vsize = math.hypot(x1 - x0, y1 - y0)
    if vsize > 0.6 * length or vsize < 8:
        return None
    # apex = the point farthest from the line joining the V's two ends
    ends = [p[0], p[-1]]
    dev = _dist_to_polyline(p, ends, False)
    apex = p[int(dev.argmax())]
    if dev.max() < 0.2 * vsize:
        return None
    for tip, tail in ((b, a), (a, b)):
        if math.hypot(*(apex - tip)) < 0.3 * length:
            direction = tip - tail
            return {"kind": "arrow",
                    "lines": [_tolist([tail, tip]),
                              _arrow_head(tuple(tip), direction, 0.18 * length)],
                    "closed": [False, False]}
    return None
