"""Phase 0 / step 06 — visualise a trip on a map (raw data only, no model).

Interactive folium map (HTML) per trip with toggleable layers:
    * Road network within NETWORK_BUFFER_M of the track (grey = two-way, orange = one-way)
    * Reference route (green; split into pieces wherever the route is discontinuous)
    * GPS track (thin blue line) + GPS point markers (subsampled, tooltip: idx, t, distance to route)
    * GPS points > FAR_M from the route (red) — these become `unreliable` labels later
    * Time gaps > GAP_S (purple dashed jump)
    * Start / end markers, legend, layer control, two base maps

Optionally (--static) a matplotlib PNG of the same layers in projected metres.
Base map: OpenStreetMap tiles (free, no API key).

Outputs: results/analysis/<partition>/maps/<trip_id>.html  (and .png with --static)
Run:     python -m scripts.analysis.06_visualise_trip --partition train --trip 00000001 00000054
         python -m scripts.analysis.06_visualise_trip --partition student_test --all --static
"""
from __future__ import annotations

import argparse

import folium
import numpy as np
import shapely

from pathsnap.geo import LocalProjector
from pathsnap.io import load_trip
from scripts.analysis._common import ANALYSIS_PARTITIONS, out_dir
from pathsnap.io import list_trips

NETWORK_BUFFER_M = 200
FAR_M = 50
GAP_S = 60
MAX_MARKERS = 3000

COLORS = {"two_way": "#8c8c8c", "one_way": "#f28e2b", "route": "#2ca02c", "gps": "#1f77b4",
          "far": "#d62728", "gap": "#9467bd"}


def prepare(trip):
    """Project the trip and compute everything both renderers need."""
    tr = trip.track
    lon, lat, t = tr.lon.to_numpy(), tr.lat.to_numpy(), tr.t.to_numpy()
    proj = LocalProjector.for_track(lon, lat)
    px, py = proj.forward(lon, lat)
    nodes, arcs = trip.nodes, trip.arcs
    nx, ny = proj.forward(nodes[:, 0], nodes[:, 1])
    seg = np.stack([np.c_[nx[arcs[:, 0]], ny[arcs[:, 0]]], np.c_[nx[arcs[:, 1]], ny[arcs[:, 1]]]], axis=1)
    lines = shapely.linestrings(seg)

    # arcs near the track (keeps the map small: big networks have 300k arcs)
    # spatial-index query: arcs within the buffer of any GPS point
    hits = shapely.STRtree(lines).query(shapely.points(px, py), predicate="dwithin", distance=NETWORK_BUFFER_M)
    near = np.unique(hits[1])

    n = len(nodes)
    key, rev = arcs[:, 0] * n + arcs[:, 1], arcs[:, 1] * n + arcs[:, 0]
    two_way = np.isin(rev, key)

    route = trip.route
    d_route = np.full(len(px), np.nan)
    pieces = []
    if route is not None and len(route):
        rl = lines[route]
        (pi, _), dist = shapely.STRtree(rl).query_nearest(shapely.points(px, py), return_distance=True,
                                                          all_matches=False)
        d_route[pi] = dist
        # split the route into connected pieces (break where arc i does not end where arc i+1 starts)
        ra = arcs[route]
        cuts = np.flatnonzero(ra[:-1, 1] != ra[1:, 0]) + 1
        for chunk in np.split(np.arange(len(route)), cuts):
            node_seq = np.r_[ra[chunk[0], 0], ra[chunk, 1]]
            pieces.append(node_seq)

    gaps = np.flatnonzero(np.diff(t) > GAP_S)
    return dict(lon=lon, lat=lat, t=t, px=px, py=py, nodes=nodes, nx=nx, ny=ny, arcs=arcs,
                near=near, two_way=two_way, d_route=d_route, route_pieces=pieces, gaps=gaps)


def _legend_html(trip_id, partition, n_points, n_near, n_far):
    items = [("Road network, two-way", COLORS["two_way"], "line"),
             ("Road network, one-way", COLORS["one_way"], "line"),
             ("Reference route", COLORS["route"], "line"),
             ("GPS track / points", COLORS["gps"], "dot"),
             (f"GPS > {FAR_M} m from route", COLORS["far"], "dot"),
             (f"Time gap > {GAP_S} s", COLORS["gap"], "line")]
    rows = "".join(
        f'<div><span style="display:inline-block;width:18px;height:{4 if k == "line" else 10}px;'
        f'background:{c};border-radius:{0 if k == "line" else 5}px;margin-right:6px;vertical-align:middle"></span>{l}</div>'
        for l, c, k in items)
    return (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;'
            f'border:1px solid #999;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
            f'<b>Trip {trip_id}</b> ({partition})<br>{n_points:,} GPS points · {n_near:,} arcs shown · '
            f'{n_far:,} far points<hr style="margin:4px 0">{rows}</div>')


def folium_map(trip, d) -> folium.Map:
    lat, lon, nodes, arcs = d["lat"], d["lon"], d["nodes"], d["arcs"]
    m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], tiles=None, control_scale=True)
    # OpenStreetMap tiles: free, no API key, so nothing secret ends up in the HTML.
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    def arc_coords(ids):  # folium wants [lat, lon]
        return [[[nodes[a, 1], nodes[a, 0]], [nodes[b, 1], nodes[b, 0]]] for a, b in arcs[ids]]

    net = folium.FeatureGroup(name=f"Road network (≤ {NETWORK_BUFFER_M} m)")
    tw = d["near"][d["two_way"][d["near"]]]
    ow = d["near"][~d["two_way"][d["near"]]]
    if len(tw):
        folium.PolyLine(arc_coords(tw), color=COLORS["two_way"], weight=2, opacity=0.7).add_to(net)
    if len(ow):
        folium.PolyLine(arc_coords(ow), color=COLORS["one_way"], weight=2, opacity=0.8).add_to(net)
    net.add_to(m)

    if d["route_pieces"]:
        rg = folium.FeatureGroup(name=f"Reference route ({len(d['route_pieces'])} piece(s))")
        for i, seq in enumerate(d["route_pieces"]):
            folium.PolyLine([[nodes[k, 1], nodes[k, 0]] for k in seq], color=COLORS["route"], weight=5,
                            opacity=0.8, tooltip=f"route piece {i + 1}").add_to(rg)
        rg.add_to(m)

    tg = folium.FeatureGroup(name="GPS track (line)")
    folium.PolyLine(np.c_[lat, lon].tolist(), color=COLORS["gps"], weight=1.5, opacity=0.6).add_to(tg)
    tg.add_to(m)

    step = max(1, int(np.ceil(len(lat) / MAX_MARKERS)))
    pg = folium.FeatureGroup(name=f"GPS points (every {step})")
    for i in range(0, len(lat), step):
        dr = d["d_route"][i]
        folium.CircleMarker([lat[i], lon[i]], radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"idx {i} · t={d['t'][i]:.0f}s · {dr:.1f} m to route").add_to(pg)
    pg.add_to(m)

    far = np.flatnonzero(d["d_route"] > FAR_M)
    fg = folium.FeatureGroup(name=f"GPS > {FAR_M} m from route ({len(far)})")
    for i in far[:: max(1, int(np.ceil(len(far) / MAX_MARKERS)))]:
        folium.CircleMarker([lat[i], lon[i]], radius=3, color=COLORS["far"], fill=True, fill_opacity=0.9,
                            weight=0, tooltip=f"idx {i} · {d['d_route'][i]:.0f} m from route").add_to(fg)
    fg.add_to(m)

    gg = folium.FeatureGroup(name=f"Time gaps > {GAP_S} s ({len(d['gaps'])})")
    for i in d["gaps"]:
        dt = d["t"][i + 1] - d["t"][i]
        folium.PolyLine([[lat[i], lon[i]], [lat[i + 1], lon[i + 1]]], color=COLORS["gap"], weight=3,
                        dash_array="6 6", tooltip=f"gap {dt:.0f} s after idx {i}").add_to(gg)
    gg.add_to(m)

    se = folium.FeatureGroup(name="Start / end")
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(se)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(se)
    se.add_to(m)

    m.get_root().html.add_child(folium.Element(
        _legend_html(trip.trip_id, trip.partition, len(lat), len(d["near"]), len(far))))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    return m


def static_png(trip, d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    nx, ny, arcs = d["nx"], d["ny"], d["arcs"]
    x0, y0 = d["px"].min(), d["py"].min()  # plot in km relative to the track's corner

    def segs(ids):
        a = arcs[ids]
        return np.stack([np.c_[nx[a[:, 0]] - x0, ny[a[:, 0]] - y0], np.c_[nx[a[:, 1]] - x0, ny[a[:, 1]] - y0]], 1) / 1000

    fig, ax = plt.subplots(figsize=(10, 10))
    near = d["near"]
    ax.add_collection(LineCollection(segs(near[d["two_way"][near]]), colors=COLORS["two_way"], lw=0.6, label="two-way road"))
    ax.add_collection(LineCollection(segs(near[~d["two_way"][near]]), colors=COLORS["one_way"], lw=0.6, label="one-way road"))
    for i, seq in enumerate(d["route_pieces"]):
        ax.plot((nx[seq] - x0) / 1000, (ny[seq] - y0) / 1000, color=COLORS["route"], lw=2.5, alpha=0.8,
                label="reference route" if i == 0 else None)
    sc = ax.scatter((d["px"] - x0) / 1000, (d["py"] - y0) / 1000, c=np.clip(d["d_route"], 0, 100), cmap="viridis_r",
                    s=3, zorder=3, label="GPS (colour = m to route)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="distance to route (m, clipped at 100)")
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_xlabel("x (km, UTM)"); ax.set_ylabel("y (km, UTM)")
    ax.set_title(f"Trip {trip.trip_id} ({trip.partition}) — {len(d['px']):,} GPS points")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partition", default="train", choices=ANALYSIS_PARTITIONS)
    p.add_argument("--trip", nargs="*", default=[], help="trip id(s), e.g. 00000001 (or just 1)")
    p.add_argument("--all", action="store_true", help="every trip in the partition")
    p.add_argument("--static", action="store_true", help="also write a matplotlib PNG")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    ids = list_trips(a.partition, a.data_dir) if a.all else [t.zfill(8) for t in a.trip]
    if not ids:
        p.error("give --trip <id> ... or --all")
    od = out_dir(a.partition, a.out_dir) / "maps"
    od.mkdir(exist_ok=True)
    for tid in ids:
        trip = load_trip(tid, a.partition, a.data_dir)
        d = prepare(trip)
        html = od / f"{tid}.html"
        folium_map(trip, d).save(str(html))
        msg = f"{tid}: {html} ({html.stat().st_size / 1e6:.1f} MB, {len(d['near'])} arcs)"
        if a.static:
            png = od / f"{tid}.png"
            static_png(trip, d, png)
            msg += f" + {png.name}"
        print(msg)


if __name__ == "__main__":
    main()
