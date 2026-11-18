"""Candidate features (one row per GPS observation x candidate arc).

``trip_features`` uses only the GPS track and the road network — never the reference
route — so the exact same code runs at inference time in the API. Labels are attached
separately by ``add_labels`` (training / evaluation only).

All features are geometry-relative (metres, degrees, counts). Absolute coordinates and
node/arc ids are deliberately excluded: trips are worldwide and ids are local to a trip.
See artifacts/feature_list.json for the column contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.candidates import TripGeometry, candidates
from pathsnap.geo import angle_diff_deg, bearing_deg, point_segment_distance
from pathsnap.route_metrics import arc_lengths

HEADING_HALF_WINDOW = 2      # heading/speed from point i-2 to i+2
HEADING_MIN_MOVE_M = 2.0     # below this displacement the GPS heading is noise
WINDOW_HALF = 3              # window_*_dist over points i-3 .. i+3
SOFTMIN_SIGMA_M = 10.0

FEATURES = ["dist_m", "dist_rank", "dist_minus_min_m", "softmin_share", "n_candidates", "proj_t",
            "arc_length_m", "arc_two_way", "from_in_degree", "to_out_degree", "heading_diff_deg",
            "heading_valid", "speed_mps", "dt_prev_s", "window_mean_dist_m", "window_min_dist_m"]


def point_motion(geom: TripGeometry):
    """Per-point smoothed heading (deg), validity flag, speed (m/s) and Δt to the previous point."""
    n = len(geom.px)
    i = np.arange(n)
    lo = np.clip(i - HEADING_HALF_WINDOW, 0, n - 1)
    hi = np.clip(i + HEADING_HALF_WINDOW, 0, n - 1)
    dx, dy = geom.px[hi] - geom.px[lo], geom.py[hi] - geom.py[lo]
    move = np.hypot(dx, dy)
    dt_win = geom.t[hi] - geom.t[lo]
    heading = bearing_deg(0, 0, dx, dy)
    valid = move > HEADING_MIN_MOVE_M
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt_win > 0, move / dt_win, 0.0)
    dt_prev = np.r_[1.0, np.diff(geom.t)]
    return heading, valid, speed, dt_prev


def trip_features(trip, geom: TripGeometry | None = None, radius_m: float = 50.0, max_k: int = 30) -> pd.DataFrame:
    """Candidate rows with id columns (trip_id, obs_idx, cand_arc) + FEATURES. No labels."""
    geom = geom or TripGeometry.from_trip(trip)
    c = candidates(geom, radius_m=radius_m, max_k=max_k)          # sorted by obs, then distance
    obs, arc = c.obs_idx.to_numpy(), c.cand_arc.to_numpy()
    arcs = geom.arcs
    n_nodes = len(geom.nx)

    # --- distance relative to the other candidates of the same observation
    g = c.groupby("obs_idx", sort=False)
    c["dist_rank"] = g.cumcount().astype(np.float32)
    dmin = g.dist_m.transform("min")
    c["dist_minus_min_m"] = c.dist_m - dmin
    w = np.exp(-(c.dist_m - dmin) / SOFTMIN_SIGMA_M)     # shift by min for numerical stability
    c["softmin_share"] = w / w.groupby(c.obs_idx).transform("sum")
    c["n_candidates"] = g.dist_m.transform("size").astype(np.float32)

    # --- arc attributes
    length = arc_lengths(trip.nodes, arcs)
    key, rev = arcs[:, 0] * n_nodes + arcs[:, 1], arcs[:, 1] * n_nodes + arcs[:, 0]
    two_way = np.isin(rev, key)
    indeg = np.bincount(arcs[:, 1], minlength=n_nodes)
    outdeg = np.bincount(arcs[:, 0], minlength=n_nodes)
    c["arc_length_m"] = length[arc]
    c["arc_two_way"] = two_way[arc].astype(np.float32)
    c["from_in_degree"] = indeg[arcs[arc, 0]].astype(np.float32)
    c["to_out_degree"] = outdeg[arcs[arc, 1]].astype(np.float32)

    # --- motion: GPS heading vs arc bearing (separates the two directions of a road)
    heading, valid, speed, dt_prev = point_motion(geom)
    a0, a1 = arcs[arc, 0], arcs[arc, 1]
    arc_bearing = bearing_deg(geom.nx[a0], geom.ny[a0], geom.nx[a1], geom.ny[a1])
    hd = angle_diff_deg(heading[obs], arc_bearing)
    c["heading_valid"] = valid[obs].astype(np.float32)
    c["heading_diff_deg"] = np.where(valid[obs], hd, 90.0)      # 90° = uninformative when not moving
    c["speed_mps"] = speed[obs]
    c["dt_prev_s"] = dt_prev[obs]

    # --- temporal consistency: how close do the neighbouring points stay to this arc?
    n = len(geom.px)
    offs = range(-WINDOW_HALF, WINDOW_HALF + 1)
    dists = np.empty((len(offs), len(c)), dtype=np.float32)
    ax, ay, bx, by = geom.nx[a0], geom.ny[a0], geom.nx[a1], geom.ny[a1]
    for r, o in enumerate(offs):
        j = np.clip(obs + o, 0, n - 1)
        dists[r], _ = point_segment_distance(geom.px[j], geom.py[j], ax, ay, bx, by)
    c["window_mean_dist_m"] = dists.mean(axis=0)
    c["window_min_dist_m"] = dists.min(axis=0)

    c.insert(0, "trip_id", trip.trip_id)
    out = c[["trip_id", "obs_idx", "cand_arc"] + FEATURES].copy()
    out[FEATURES] = out[FEATURES].astype(np.float32)
    return out


def add_labels(feats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach label (cand_arc == aligned route arc) and unreliable flag from pathsnap.labels."""
    f = feats.merge(labels[["obs_idx", "label_arc", "unreliable"]], on="obs_idx", how="left")
    f["label"] = (f.cand_arc == f.label_arc).astype(np.int8)
    return f.drop(columns="label_arc")
