"""4D geometry and robust Euclidean distance queries, independent of CAD."""

from itertools import combinations
import numpy as np

EPS = 1e-10
AXES = "xyzw"


def hypercube(dimension=4):
    """Vertex i has coordinate +1 where bit i is set; edges have stable IDs."""
    vertices = np.array([[1.0 if i & (1 << axis) else -1.0 for axis in range(dimension)] for i in range(2**dimension)])
    edges = np.array([(i, i ^ (1 << axis)) for i in range(2**dimension) for axis in range(dimension) if not i & (1 << axis)], dtype=int)
    return vertices, edges


def rotate(vertices, angle_deg, plane="xw"):
    i, j = (AXES.index(a) for a in plane)
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    result = np.array(vertices, dtype=float, copy=True)
    result[..., i] = c * vertices[..., i] - s * vertices[..., j]
    result[..., j] = s * vertices[..., i] + c * vertices[..., j]
    return result


def project(vertices, method="perspective", distance=4.0):
    """Invalid points are NaN internally; they become null in JSON/CSV outputs."""
    if method == "orthographic":
        return vertices[..., :3].copy()
    if method != "perspective":
        raise ValueError("unknown projection")
    denominator = distance - vertices[..., 3]
    # Points on or beyond the eye plane have no usable physical projection.
    valid = denominator > EPS
    factor = np.full(denominator.shape, np.nan)
    np.divide(distance, denominator, out=factor, where=valid)
    return vertices[..., :3] * factor[..., None]


def samples(start, end, step):
    """Include endpoints, without accumulating floating point increments."""
    count = int(np.floor((end - start) / step + 1e-9))
    values = start + np.arange(count + 1) * step
    if end - values[-1] > 1e-8:
        values = np.append(values, end)
    else:
        values[-1] = end
    return values


def point_segment_distance(point, start, end):
    point, start, end = np.broadcast_arrays(np.asarray(point, float), np.asarray(start, float), np.asarray(end, float))
    direction = end - start
    norm2 = np.sum(direction * direction, axis=-1)
    t = np.zeros(norm2.shape)
    np.divide(np.sum((point - start) * direction, axis=-1), norm2, out=t, where=norm2 > EPS**2)
    nearest = start + np.clip(t, 0, 1)[..., None] * direction
    return np.linalg.norm(point - nearest, axis=-1)


def segment_distance(a, b, c, d):
    """Exact closest distance of finite segments, vectorized over leading axes.

    The interior/interior solution plus four endpoint/segment candidates cover
    skew, parallel, collinear and degenerate segments without unstable division.
    """
    a, b, c, d = np.broadcast_arrays(*(np.asarray(p, float) for p in (a, b, c, d)))
    u, v, w = b - a, d - c, a - c
    normal = np.cross(u, v)
    denominator = np.sum(normal * normal, axis=-1)
    usable = denominator > 0
    s, t = np.zeros_like(denominator), np.zeros_like(denominator)
    np.divide(np.sum(np.cross(-w, v) * normal, axis=-1), denominator, out=s, where=usable)
    np.divide(np.sum(np.cross(-w, u) * normal, axis=-1), denominator, out=t, where=usable)
    interior = usable & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    middle = np.where(interior, np.linalg.norm(w + s[..., None]*u - t[..., None]*v, axis=-1), np.inf)
    return np.minimum.reduce([middle, point_segment_distance(a, c, d), point_segment_distance(b, c, d), point_segment_distance(c, a, b), point_segment_distance(d, a, b)])


def corner_pairs(edges, n_vertices=16):
    """(vertex, first edge ID, second edge ID, first neighbor, second neighbor)."""
    result = []
    for vertex in range(n_vertices):
        incident = np.flatnonzero(np.any(edges == vertex, axis=1))
        for i, j in combinations(incident, 2):
            ni = int(edges[i, 0] if edges[i, 1] == vertex else edges[i, 1])
            nj = int(edges[j, 0] if edges[j, 1] == vertex else edges[j, 1])
            result.append((vertex, int(i), int(j), ni, nj))
    return np.array(result)


def corner_angles(points, corners):
    a = points[corners[:, 3]] - points[corners[:, 0]]
    b = points[corners[:, 4]] - points[corners[:, 0]]
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.full(norms.shape, np.nan)
    np.divide(np.sum(a*b, axis=1), norms, out=cosine, where=norms > EPS**2)
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
