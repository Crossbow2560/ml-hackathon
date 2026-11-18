"""Phase 0 / step 04 — how far GPS points are from the road network and the route.

Each trip is projected to its own UTM zone (metres). Every arc becomes a 2-point line and is
put in a shapely STRtree spatial index. For every GPS point we compute:

* distance to the nearest reference-route arc   -> GPS noise relative to the "truth"
* distance to the nearest arc of any kind
* whether that nearest arc is (either direction of) a route arc -> naive nearest-arc baseline
* number of arcs within 25 / 50 / 75 / 100 m    -> candidate-set size per radius
* whether a route arc lies within each radius   -> how often a radius would MISS the truth

These numbers justify the candidate radius R and cap K used later.

Outputs (results/analysis/<partition>/):
    gps_route_distance.csv         one row per trip (percentiles, shares)
    gps_route_distance_sample.csv  every 5th GPS point (for CDF / box plots)
    candidate_density.csv          one row per (trip, radius)
Run: python -m scripts.analysis.04_gps_vs_route --partition train
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from pathsnap.geo import LocalProjector
from scripts.analysis._common import iter_trips, out_dir, parse_args, save

RADII_M = (25, 50, 75, 100)
SAMPLE_EVERY = 5


def analyse(trip):
    tr = trip.track
    ok = np.isfinite(tr[["lon", "lat"]].to_numpy()).all(axis=1)
    lon, lat = tr.lon.to_numpy()[ok], tr.lat.to_numpy()[ok]
    idx = np.flatnonzero(ok)
    proj = LocalProjector.for_track(lon, lat)
    px, py = proj.forward(lon, lat)
    nx, ny = proj.forward(trip.nodes[:, 0], trip.nodes[:, 1])

    n_nodes = len(trip.nodes)
    valid = ((trip.arcs >= 0) & (trip.arcs < n_nodes)).all(axis=1)
    arc_ids = np.flatnonzero(valid)
    a = trip.arcs[valid]
    coords = np.stack([np.c_[nx[a[:, 0]], ny[a[:, 0]]], np.c_[nx[a[:, 1]], ny[a[:, 1]]]], axis=1)
    lines = shapely.linestrings(coords)
    pts = shapely.points(px, py)
    n = len(pts)

    # nearest arc of any kind
    tree = shapely.STRtree(lines)
    (pi, li), dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    d_any = np.full(n, np.nan); d_any[pi] = dist
    nearest = np.full(n, -1); nearest[pi] = arc_ids[li]

    # nearest reference-route arc
    route = trip.route[(trip.route >= 0) & (trip.route < len(trip.arcs))]
    route = route[valid[route]]
    pos = np.searchsorted(arc_ids, route)
    rtree = shapely.STRtree(lines[pos])
    (pi, _), dist = rtree.query_nearest(pts, return_distance=True, all_matches=False)
    d_route = np.full(n, np.nan); d_route[pi] = dist

    # is the nearest arc on the route (ignoring direction)?
    und = lambda e: np.minimum(e[:, 0], e[:, 1]) * n_nodes + np.maximum(e[:, 0], e[:, 1])
    route_keys = np.unique(und(trip.arcs[route]))
    nearest_on_route = np.isin(und(trip.arcs[nearest]), route_keys)

    # where/when are the far-off (> 50 m) points? A compact cluster that lasts a long time is
    # off-network dwell (parking, private ground); far points spread along the trip suggest a
    # route that does not match the drive.
    far = d_route > 50
    t = tr.t.to_numpy()[ok]
    dt = np.diff(t, append=t[-1])
    far_spread = float(max(np.ptp(px[far]), np.ptp(py[far]))) if far.any() else 0.0
    far_time_share = float(dt[far].sum() / max(t[-1] - t[0], 1e-9))

    q = lambda p: float(np.nanpercentile(d_route, p))
    summary = {
        "trip_id": trip.trip_id, "n_points": n,
        "d_route_p50_m": q(50), "d_route_p90_m": q(90), "d_route_p95_m": q(95),
        "d_route_p99_m": q(99), "d_route_max_m": float(np.nanmax(d_route)),
        "d_route_mean_m": float(np.nanmean(d_route)),
        **{f"share_d_route_gt_{r}m": float(np.mean(d_route > r)) for r in (25, 50, 100)},
        "d_any_p50_m": float(np.nanmedian(d_any)), "d_any_p95_m": float(np.nanpercentile(d_any, 95)),
        "nearest_arc_on_route_share": float(nearest_on_route.mean()),
        "far_spread_m": far_spread,
        "far_time_share": far_time_share,
    }

    dens = []
    for r in RADII_M:
        hit = tree.query(pts, predicate="dwithin", distance=r)
        cnt = np.bincount(hit[0], minlength=n)
        dens.append({
            "trip_id": trip.trip_id, "radius_m": r,
            "cand_mean": float(cnt.mean()), "cand_p50": float(np.median(cnt)),
            "cand_p95": float(np.percentile(cnt, 95)), "cand_max": int(cnt.max()),
            "share_zero_candidates": float(np.mean(cnt == 0)),
            "route_within_radius_share": float(np.mean(d_route <= r)),
        })

    s = slice(None, None, SAMPLE_EVERY)
    sample = pd.DataFrame({"trip_id": trip.trip_id, "idx": idx[s],
                           "d_route_m": d_route[s], "d_any_m": d_any[s]})
    return summary, pd.DataFrame(dens), sample


def run(partition: str, root: str | None = None, out: str | None = None):
    od = out_dir(partition, out)
    rows, dens, samples = [], [], []
    for tid, trip, err in iter_trips(partition, root, "gps-vs-route"):
        if trip is None or trip.route is None:
            rows.append({"trip_id": tid, "load_error": err or "no route"})
            continue
        s, d, smp = analyse(trip)
        rows.append(s); dens.append(d); samples.append(smp)
    df = pd.DataFrame(rows)
    de = pd.concat(dens, ignore_index=True)
    sm = pd.concat(samples, ignore_index=True)
    save(df, od / "gps_route_distance.csv")
    save(sm, od / "gps_route_distance_sample.csv")
    save(de, od / "candidate_density.csv")

    print(f"  GPS->route distance (sampled points) p50/p90/p99/max: "
          f"{sm.d_route_m.quantile(.5):.1f}/{sm.d_route_m.quantile(.9):.1f}/"
          f"{sm.d_route_m.quantile(.99):.1f}/{sm.d_route_m.max():.1f} m")
    print(f"  naive nearest-arc-on-route share (mean over trips): {df.nearest_arc_on_route_share.mean():.3f}")
    agg = de.groupby("radius_m").agg(cand_mean=("cand_mean", "mean"),
                                     route_cov_mean=("route_within_radius_share", "mean"),
                                     route_cov_min=("route_within_radius_share", "min"))
    print(agg.round(3).to_string())
    return df, de, sm


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.partition, a.data_dir, a.out_dir)
