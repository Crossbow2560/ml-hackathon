"""Geometry helpers.

Longitude/latitude are angles, so Euclidean distance in degrees is wrong (one degree of
longitude shrinks with cos(latitude)). We use two tools:

* ``haversine_m`` – great-circle distance in metres, for quick per-pair distances.
* ``LocalProjector`` – projects a trip into its own UTM zone (metres, conformal, < 0.1 %
  scale error inside a 6° zone). Every trip covers at most a few tens of km, so a single
  zone per trip is accurate; planar geometry (point-to-segment distance, bearings) is
  then valid in metres.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1, lat1, lon2, lat2):
    """Vectorised great-circle distance in metres."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


class LocalProjector:
    """lon/lat <-> metres in the UTM zone chosen from a reference point."""

    def __init__(self, ref_lon: float, ref_lat: float):
        self.epsg = utm_epsg(ref_lon, ref_lat)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    @classmethod
    def for_track(cls, lon, lat) -> "LocalProjector":
        """Use the median track position so the zone fits the whole trip."""
        return cls(float(np.nanmedian(lon)), float(np.nanmedian(lat)))

    def forward(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def inverse(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (broadcasting, planar metres).

    Returns (distance, t) where t in [0, 1] is the position of the closest point along AB
    (0 = at A, 1 = at B). Zero-length segments fall back to distance to A.
    """
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = np.where(seg_len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(ax, ay, bx, by):
    """Planar bearing of A->B in degrees, 0 = north (+y), clockwise, range [0, 360)."""
    return (np.degrees(np.arctan2(bx - ax, by - ay)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)
