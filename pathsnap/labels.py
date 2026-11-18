"""Per-point labels from the reference route (monotonic alignment).

The .route file is the ordered list of arcs the vehicle drove, but it does not say which GPS
point belongs to which arc. We assign every point i a route position j(i) such that

    j(0) <= j(1) <= ... <= j(n-1)          (the vehicle never moves backwards along its route)

and the total distance  sum_i  d(point_i, route_arc_j(i))  is minimal. Dynamic programming:

    D[i, j] = c[i, j] + min_{k <= j} D[i-1, k]          c = capped point-to-arc distance

A running prefix minimum makes each row O(m), so the whole alignment is O(n * m). Working
with route *positions* (not arc ids) handles routes that visit an arc twice.

The label of point i is route[j(i)]. If the point is further than ``unreliable_m`` from that
arc (off-network dwell, route mismatch) it is flagged ``unreliable`` and is excluded from
training labels and from metrics. The reference route is not perfect ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.candidates import TripGeometry
from pathsnap.geo import point_segment_distance

COST_CAP_M = 200.0   # caps the influence of far-off points (parked clusters) on the alignment


def align_to_route(geom: TripGeometry, route: np.ndarray, unreliable_m: float = 50.0) -> pd.DataFrame:
    """Return one row per GPS point: obs_idx, route_pos, label_arc, label_dist_m, unreliable."""
    n, m = len(geom.px), len(route)
    ra = geom.arcs[route]
    ax, ay = geom.nx[ra[:, 0]], geom.ny[ra[:, 0]]
    bx, by = geom.nx[ra[:, 1]], geom.ny[ra[:, 1]]

    back = np.empty((n, m), dtype=np.int32)       # argmin of the prefix minimum of the previous row
    prev = None
    for i in range(n):
        c, _ = point_segment_distance(geom.px[i], geom.py[i], ax, ay, bx, by)
        c = np.minimum(c, COST_CAP_M)
        if prev is None:
            cur = c
            back[i] = np.arange(m)
        else:
            # prefix minimum of the previous row (value acc[j], position pm_idx[j]) — vectorised
            acc = np.minimum.accumulate(prev)
            changed = np.r_[True, acc[1:] < acc[:-1]]
            pm_idx = np.maximum.accumulate(np.where(changed, np.arange(m), 0)).astype(np.int32)
            cur = c + acc
            back[i] = pm_idx
        prev = cur

    # backtrack
    pos = np.empty(n, dtype=np.int32)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    lab = route[pos]
    d, _ = geom.point_arc_distance(np.arange(n), lab)
    return pd.DataFrame({"obs_idx": np.arange(n, dtype=np.int32), "route_pos": pos, "label_arc": lab.astype(np.int32),
                         "label_dist_m": d.astype(np.float32), "unreliable": d > unreliable_m})
