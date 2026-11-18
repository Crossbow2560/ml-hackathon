"""Phase 0 / step 01 — GPS track description and validation.

Per trip: missing / invalid values, coordinate plausibility (incl. a lon/lat swap check
against the road-network bounding box), duplicate and non-monotonic timestamps, sampling
interval (Δt) distribution and gaps, duration, path length, speeds, stationary share and
geographical extent.

Outputs (results/analysis/<partition>/):
    tracks_summary.csv   one row per trip
    track_gaps.csv       one row per time gap > 10 s
    track_issues.csv     one row per trip with data-quality counts
    tracks_describe.csv  pandas describe() of lon/lat/t/dt/speed over the whole partition
Run: python -m scripts.analysis.01_describe_tracks --partition train
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.geo import LocalProjector, haversine_m
from scripts.analysis._common import iter_trips, out_dir, parse_args, save

GAP_REPORT_S = 10          # every gap above this goes to track_gaps.csv
GAP_LEVELS_S = (10, 60, 300)
IMPLAUSIBLE_SPEED_MPS = 70  # 252 km/h
STATIONARY_SPEED_MPS = 0.5


def _share_inside(lon, lat, bbox):
    lo0, la0, lo1, la1 = bbox
    return float(np.mean((lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)))


def describe_track(trip):
    tr = trip.track
    lon, lat, t = (tr[c].to_numpy() for c in ("lon", "lat", "t"))
    n = len(tr)
    vals = tr.to_numpy()
    n_nan = int(np.isnan(vals).any(axis=1).sum())
    n_inf = int(np.isinf(vals).any(axis=1).sum())
    finite = np.isfinite(vals).all(axis=1)
    n_oor = int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum())

    # lon/lat order check: points should fall inside the network bbox; if swapping
    # the columns makes far more points fall inside, the order is probably wrong.
    nb = (trip.nodes[:, 0].min(), trip.nodes[:, 1].min(), trip.nodes[:, 0].max(), trip.nodes[:, 1].max())
    inside = _share_inside(lon[finite], lat[finite], nb)
    inside_swapped = _share_inside(lat[finite], lon[finite], nb)

    dup_rows = int(tr.duplicated().sum())
    dup_t = int(tr["t"].duplicated().sum())
    dt = np.diff(t)
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)
    sp = speed[np.isfinite(speed)]

    proj = LocalProjector.for_track(lon[finite], lat[finite])
    x, y = proj.forward(lon[finite], lat[finite])

    summary = {
        "trip_id": trip.trip_id,
        "n_points": n,
        "lon_dtype": str(tr["lon"].dtype), "lat_dtype": str(tr["lat"].dtype), "t_dtype": str(tr["t"].dtype),
        "t_start": float(t[0]), "t_end": float(t[-1]), "duration_s": float(t[-1] - t[0]),
        "dt_share_1s": float(np.isclose(dt, 1.0, atol=1e-6).mean()) if len(dt) else np.nan,
        "dt_min": float(dt.min()) if len(dt) else np.nan,
        "dt_p50": float(np.median(dt)) if len(dt) else np.nan,
        "dt_p95": float(np.percentile(dt, 95)) if len(dt) else np.nan,
        "dt_max": float(dt.max()) if len(dt) else np.nan,
        **{f"n_gaps_gt_{g}s": int((dt > g).sum()) for g in GAP_LEVELS_S},
        "path_length_m": float(np.nansum(step)),
        "speed_p50_mps": float(np.median(sp)) if len(sp) else np.nan,
        "speed_p95_mps": float(np.percentile(sp, 95)) if len(sp) else np.nan,
        "speed_max_mps": float(sp.max()) if len(sp) else np.nan,
        "stationary_share": float(np.mean(sp < STATIONARY_SPEED_MPS)) if len(sp) else np.nan,
        "lon_min": float(np.nanmin(lon)), "lon_max": float(np.nanmax(lon)),
        "lat_min": float(np.nanmin(lat)), "lat_max": float(np.nanmax(lat)),
        "centroid_lon": float(np.nanmean(lon)), "centroid_lat": float(np.nanmean(lat)),
        "extent_x_km": float((x.max() - x.min()) / 1000), "extent_y_km": float((y.max() - y.min()) / 1000),
        "utm_epsg": proj.epsg,
    }
    issues = {
        "trip_id": trip.trip_id,
        "n_points": n,
        "n_nan_rows": n_nan,
        "n_inf_rows": n_inf,
        "n_out_of_range": n_oor,
        "share_in_network_bbox": inside,
        "share_in_bbox_if_swapped": inside_swapped,
        "lonlat_swap_suspected": bool(inside_swapped > inside + 0.5),
        "n_duplicate_rows": dup_rows,
        "n_duplicate_t": dup_t,
        "n_non_monotonic_t": int((dt < 0).sum()),
        "n_zero_dt": int((dt == 0).sum()),
        "n_non_integer_dt": int((~np.isclose(dt, np.round(dt), atol=1e-6)).sum()),
        f"n_speed_gt_{IMPLAUSIBLE_SPEED_MPS}mps": int((sp > IMPLAUSIBLE_SPEED_MPS).sum()),
        "load_issues": "; ".join(trip.issues),
    }
    gi = np.where(dt > GAP_REPORT_S)[0]
    gaps = pd.DataFrame({
        "trip_id": trip.trip_id,
        "idx_before": gi,
        "t_before": t[gi], "t_after": t[gi + 1],
        "dt_s": dt[gi],
        "jump_m": step[gi],
        "implied_speed_mps": step[gi] / dt[gi],
    })
    frame = pd.DataFrame({"lon": lon, "lat": lat, "t": t,
                          "dt": np.r_[np.nan, dt], "speed_mps": np.r_[np.nan, speed]})
    return summary, issues, gaps, frame


def run(partition: str, root: str | None = None, out: str | None = None):
    od = out_dir(partition, out)
    summaries, issues, gaps, frames, failures = [], [], [], [], []
    for tid, trip, err in iter_trips(partition, root, "tracks"):
        if trip is None:
            failures.append({"trip_id": tid, "load_issues": err})
            continue
        s, i, g, f = describe_track(trip)
        summaries.append(s); issues.append(i); gaps.append(g); frames.append(f)

    summ = pd.DataFrame(summaries)
    iss = pd.concat([pd.DataFrame(issues), pd.DataFrame(failures)], ignore_index=True)
    gap = pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame()
    desc = pd.concat(frames, ignore_index=True).describe().T.reset_index().rename(columns={"index": "column"})

    save(summ, od / "tracks_summary.csv")
    save(gap, od / "track_gaps.csv")
    save(iss, od / "track_issues.csv")
    save(desc, od / "tracks_describe.csv")

    print(f"  points total: {summ['n_points'].sum():,}  per trip min/median/max: "
          f"{summ['n_points'].min()}/{int(summ['n_points'].median())}/{summ['n_points'].max()}")
    print(f"  share of Δt exactly 1 s (point-weighted): "
          f"{np.average(summ['dt_share_1s'], weights=summ['n_points'] - 1):.3f}")
    print(f"  trips with a gap > 60 s: {(summ['n_gaps_gt_60s'] > 0).sum()}, > 300 s: {(summ['n_gaps_gt_300s'] > 0).sum()}")
    print(f"  rows with NaN: {iss['n_nan_rows'].sum()}, out-of-range: {iss['n_out_of_range'].sum()}, "
          f"swap suspected: {iss['lonlat_swap_suspected'].sum()}, non-monotonic t: {iss['n_non_monotonic_t'].sum()}")
    return summ, gap, iss


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.partition, a.data_dir, a.out_dir)
