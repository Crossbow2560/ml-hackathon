"""Phase 0 / step 03 — reference-route description and validation.

The .route file is an ordered list of arc ids. It does NOT say which GPS point belongs to
which arc, so we check how trustworthy and how connected it is before deriving labels.

Per trip: route length (arcs, metres), invalid arc ids, repeated arcs, immediate U-turns,
continuity (arc i ends where arc i+1 starts) and every break with its size, the share of
route arcs inside the network's largest strongly connected component, and how far the
route's start/end are from the first/last GPS point.

Outputs (results/analysis/<partition>/):
    routes_summary.csv  one row per trip
    route_breaks.csv    one row per discontinuity
Run: python -m scripts.analysis.03_describe_routes --partition train
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from pathsnap.geo import LocalProjector, haversine_m, point_segment_distance
from scripts.analysis._common import iter_trips, out_dir, parse_args, save

_net = importlib.import_module("scripts.analysis.02_describe_network")


def describe_route(trip):
    nodes, arcs, route = trip.nodes, trip.arcs, trip.route
    tid = trip.trip_id
    if route is None:
        return {"trip_id": tid, "route_present": False}, pd.DataFrame()

    m = len(arcs)
    in_range = (route >= 0) & (route < m)
    r = route[in_range]
    a = arcs[r]
    frm, to = a[:, 0], a[:, 1]
    seg_len = haversine_m(nodes[frm, 0], nodes[frm, 1], nodes[to, 0], nodes[to, 1])

    connected = to[:-1] == frm[1:]
    u_turn = (frm[1:] == to[:-1]) & (to[1:] == frm[:-1])
    bi = np.where(~connected)[0]
    gap_m = haversine_m(nodes[to[bi], 0], nodes[to[bi], 1], nodes[frm[bi + 1], 0], nodes[frm[bi + 1], 1])
    breaks = pd.DataFrame({"trip_id": tid, "route_pos": bi, "arc_before": r[bi], "arc_after": r[bi + 1],
                           "gap_m": gap_m})

    # route arcs inside the largest strongly connected component?
    _, lab = _net.components(arcs, len(nodes), "strong")
    sizes = np.bincount(lab)
    big = np.argmax(sizes)
    in_scc = (lab[frm] == big) & (lab[to] == big)

    # distance from the first / last GPS point to the first / last route arc (metres)
    tr = trip.track
    proj = LocalProjector.for_track(tr.lon, tr.lat)
    px, py = proj.forward(tr.lon.to_numpy()[[0, -1]], tr.lat.to_numpy()[[0, -1]])
    nx, ny = proj.forward(nodes[:, 0], nodes[:, 1])
    d_start, _ = point_segment_distance(px[0], py[0], nx[frm[0]], ny[frm[0]], nx[to[0]], ny[to[0]])
    d_end, _ = point_segment_distance(px[1], py[1], nx[frm[-1]], ny[frm[-1]], nx[to[-1]], ny[to[-1]])

    summary = {
        "trip_id": tid,
        "route_present": True,
        "route_n_arcs": int(len(route)),
        "n_invalid_arc_ids": int((~in_range).sum()),
        "n_unique_arcs": int(len(np.unique(r))),
        "n_repeated_arcs": int(len(r) - len(np.unique(r))),
        "n_consecutive_duplicates": int((np.diff(r) == 0).sum()),
        "n_u_turns": int(u_turn.sum()),
        "continuity": float(connected.mean()) if len(connected) else np.nan,
        "n_breaks": int(len(bi)),
        "max_break_gap_m": float(gap_m.max()) if len(bi) else 0.0,
        "route_length_m": float(seg_len.sum()),
        "route_arc_len_p50_m": float(np.median(seg_len)),
        "share_in_largest_scc": float(in_scc.mean()),
        "start_dist_m": float(d_start),
        "end_dist_m": float(d_end),
    }
    return summary, breaks


def run(partition: str, root: str | None = None, out: str | None = None):
    od = out_dir(partition, out)
    rows, brks = [], []
    for tid, trip, err in iter_trips(partition, root, "routes"):
        if trip is None:
            rows.append({"trip_id": tid, "load_error": err})
            continue
        s, b = describe_route(trip)
        rows.append(s); brks.append(b)
    df = pd.DataFrame(rows)
    br = pd.concat(brks, ignore_index=True) if brks else pd.DataFrame()
    save(df, od / "routes_summary.csv")
    save(br, od / "route_breaks.csv")
    print(f"  routes present: {df.route_present.sum()}/{len(df)}; arcs per route min/median/max: "
          f"{df.route_n_arcs.min()}/{int(df.route_n_arcs.median())}/{df.route_n_arcs.max()}")
    print(f"  trips with breaks: {(df.n_breaks > 0).sum()} (total breaks {df.n_breaks.sum()}), "
          f"min continuity {df.continuity.min():.3f}")
    print(f"  invalid ids: {df.n_invalid_arc_ids.sum()}, repeated arcs: {df.n_repeated_arcs.sum()}, "
          f"U-turns: {df.n_u_turns.sum()}")
    print(f"  start/end distance to route p95: {df.start_dist_m.quantile(.95):.1f} m / {df.end_dist_m.quantile(.95):.1f} m")
    return df, br


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.partition, a.data_dir, a.out_dir)
