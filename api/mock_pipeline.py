"""Mock pipeline with the same two functions as pathsnap.pipeline (PATHSNAP_USE_MOCK=1).

match_trip snaps every GPS point to its nearest arc (no model, no HMM, confidence 1.0);
score_trip reads the reference route and reports the share of points on a route arc.
Used to test the API and UI end to end without trained model files.
"""
from __future__ import annotations

import time

import numpy as np
import shapely

from pathsnap.candidates import TripGeometry
from pathsnap.io import load_trip_dir


def match_trip(trip_dir):
    t0 = time.perf_counter()
    trip = load_trip_dir(trip_dir, load_route=False, allow_evaluator=True)
    g = TripGeometry.from_trip(trip)
    pts = shapely.points(g.px, g.py)
    pi, li = g.tree.query_nearest(pts, all_matches=False)
    arc = np.full(len(pts), -1)
    arc[pi] = li
    lat, lon = trip.track.lat.to_numpy(), trip.track.lon.to_numpy()
    total = (time.perf_counter() - t0) * 1000
    return {
        "trip_id": trip.trip_id,
        "gps": np.c_[lat, lon].tolist(),
        "matched_points": [{"idx": i, "arc_id": int(arc[i]) if arc[i] >= 0 else None, "lat": float(lat[i]),
                            "lon": float(lon[i]), "confidence": 1.0} for i in range(len(arc))],
        "route_geometry": np.c_[lat, lon].tolist(),
        "timing_ms": {"load": total, "candidates": 0.0, "classify": 0.0, "decode": 0.0, "total": total},
        "model": {"name": "mock nearest-arc", "version": "mock"},
    }


def score_trip(trip_dir, prediction):
    trip = load_trip_dir(trip_dir, load_route=True, allow_evaluator=True)
    arcs = np.array([-1 if p["arc_id"] is None else p["arc_id"] for p in prediction["matched_points"]])
    on_route = np.isin(arcs, trip.route) if trip.route is not None else np.zeros(len(arcs), bool)
    return {"top1": float(on_route.mean()), "macro_f1": 0.0, "route_continuity": 0.0, "route_precision": 0.0,
            "route_recall": 0.0, "route_f1": 0.0, "candidate_miss_rate": 0.0, "n_points": int(trip.n_points)}
