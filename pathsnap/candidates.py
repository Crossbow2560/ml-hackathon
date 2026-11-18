"""Per-trip geometry and candidate generation.

``TripGeometry`` projects a trip into metres once (per-trip UTM) and builds a shapely
STRtree over all arcs. ``candidates()`` then lists, for every GPS point, the directed arcs
within ``radius_m`` (keeping the ``max_k`` nearest), with the perpendicular distance and
the projection position along the arc. Both directions of a two-way road are separate
candidates with the same distance — the classifier separates them with heading features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely

from pathsnap.geo import LocalProjector, point_segment_distance


@dataclass
class TripGeometry:
    px: np.ndarray        # GPS x (m)
    py: np.ndarray        # GPS y (m)
    t: np.ndarray         # timestamps (s)
    nx: np.ndarray        # node x (m)
    ny: np.ndarray        # node y (m)
    arcs: np.ndarray      # (M, 2) from, to
    lines: np.ndarray     # shapely LineString per arc
    tree: shapely.STRtree
    proj: LocalProjector

    @classmethod
    def from_trip(cls, trip) -> "TripGeometry":
        tr = trip.track
        proj = LocalProjector.for_track(tr.lon.to_numpy(), tr.lat.to_numpy())
        px, py = proj.forward(tr.lon.to_numpy(), tr.lat.to_numpy())
        nx, ny = proj.forward(trip.nodes[:, 0], trip.nodes[:, 1])
        a = trip.arcs
        seg = np.stack([np.c_[nx[a[:, 0]], ny[a[:, 0]]], np.c_[nx[a[:, 1]], ny[a[:, 1]]]], axis=1)
        lines = shapely.linestrings(seg)
        return cls(px, py, tr.t.to_numpy(), nx, ny, a, lines, shapely.STRtree(lines), proj)

    def point_arc_distance(self, obs_idx, arc_ids):
        """Distance (m) and projection fraction t for paired arrays of points and arcs."""
        a = self.arcs[arc_ids]
        return point_segment_distance(self.px[obs_idx], self.py[obs_idx],
                                      self.nx[a[:, 0]], self.ny[a[:, 0]], self.nx[a[:, 1]], self.ny[a[:, 1]])


def candidates(geom: TripGeometry, radius_m: float = 50.0, max_k: int | None = 20) -> pd.DataFrame:
    """All (obs_idx, cand_arc) pairs within radius_m, at most max_k nearest per point.

    Columns: obs_idx, cand_arc, dist_m, proj_t. Points with no arc in range simply have no rows.
    """
    pts = shapely.points(geom.px, geom.py)
    obs, arc = geom.tree.query(pts, predicate="dwithin", distance=radius_m)
    dist, t = geom.point_arc_distance(obs, arc)
    df = pd.DataFrame({"obs_idx": obs.astype(np.int32), "cand_arc": arc.astype(np.int32),
                       "dist_m": dist.astype(np.float32), "proj_t": t.astype(np.float32)})
    df = df.sort_values(["obs_idx", "dist_m", "cand_arc"], kind="stable")
    if max_k is not None:
        df = df[df.groupby("obs_idx").cumcount() < max_k]
    return df.reset_index(drop=True)
