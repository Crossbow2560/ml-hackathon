"""Map and plot builders shared by the analysis scripts, make_maps.py, the report and the API.

Main entry point:

    trip_map(trip, pred_arcs=None, show_route=True, confidence=None) -> folium.Map

Layers (each can be switched on/off in the map's layer control):
    * Road network within NETWORK_BUFFER_M of the track (grey = two-way, orange = one-way)
    * Reference route (green, split into connected pieces)          — if show_route and route known
    * Predicted route (red, built from one predicted arc per GPS point) — if pred_arcs given
    * GPS track line + GPS point markers (blue, or red→green by confidence if given)
    * GPS points > FAR_M from the reference route (red)              — if route known
    * Time gaps > GAP_S (purple dashed)
    * Start / end markers, legend, layer control

Everything is computed in a per-trip UTM projection (metres); folium itself needs [lat, lon].
Base map: OpenStreetMap tiles (free, no API key).
"""
from __future__ import annotations

import branca.colormap as cm
import folium
import numpy as np
import shapely

from pathsnap.geo import LocalProjector

NETWORK_BUFFER_M = 200   # only draw arcs this close to the track (big networks have 300k arcs)
FAR_M = 50               # GPS points further than this from the route are highlighted
GAP_S = 60               # time gaps longer than this are drawn as dashed jumps
MAX_MARKERS = 3000       # subsample point markers above this (keeps the HTML light)

COLORS = {"two_way": "#8c8c8c", "one_way": "#f28e2b", "route": "#2ca02c", "gps": "#1f77b4",
          "far": "#d62728", "gap": "#9467bd", "pred": "#e31a1c"}


# --------------------------------------------------------------------------- data prep
def prepare(trip) -> dict:
    """Project the trip to metres and compute everything the renderers need."""
    tr = trip.track
    lon, lat, t = tr.lon.to_numpy(), tr.lat.to_numpy(), tr.t.to_numpy()
    proj = LocalProjector.for_track(lon, lat)
    px, py = proj.forward(lon, lat)
    nodes, arcs = trip.nodes, trip.arcs
    nx, ny = proj.forward(nodes[:, 0], nodes[:, 1])
    seg = np.stack([np.c_[nx[arcs[:, 0]], ny[arcs[:, 0]]], np.c_[nx[arcs[:, 1]], ny[arcs[:, 1]]]], axis=1)
    lines = shapely.linestrings(seg)
    pts = shapely.points(px, py)

    # spatial-index query: arcs within the buffer of any GPS point
    hits = shapely.STRtree(lines).query(pts, predicate="dwithin", distance=NETWORK_BUFFER_M)
    near = np.unique(hits[1])

    n = len(nodes)
    key, rev = arcs[:, 0] * n + arcs[:, 1], arcs[:, 1] * n + arcs[:, 0]
    two_way = np.isin(rev, key)

    route = trip.route
    d_route = np.full(len(px), np.nan)
    pieces = []
    if route is not None and len(route):
        (pi, _), dist = shapely.STRtree(lines[route]).query_nearest(pts, return_distance=True, all_matches=False)
        d_route[pi] = dist
        pieces = arc_pieces(route, arcs)

    gaps = np.flatnonzero(np.diff(t) > GAP_S)
    return dict(lon=lon, lat=lat, t=t, px=px, py=py, nodes=nodes, nx=nx, ny=ny, arcs=arcs, lines=lines,
                near=near, two_way=two_way, d_route=d_route, route_pieces=pieces, gaps=gaps)


def arc_pieces(arc_seq, arcs) -> list[np.ndarray]:
    """Turn an arc sequence into node sequences, one per connected piece.

    Consecutive duplicates are collapsed first. A new piece starts wherever arc i does not
    end at the node where arc i+1 starts (a discontinuity).
    """
    seq = np.asarray([a for a in arc_seq if a is not None and a >= 0], dtype=np.int64)
    if len(seq) == 0:
        return []
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    ra = arcs[seq]
    cuts = np.flatnonzero(ra[:-1, 1] != ra[1:, 0]) + 1
    return [np.r_[ra[c[0], 0], ra[c, 1]] for c in np.split(np.arange(len(seq)), cuts)]


def nearest_arcs(d: dict) -> np.ndarray:
    """Nearest arc per GPS point — a NAIVE BASELINE used only to demo the predicted-route layer."""
    (pi, li) = shapely.STRtree(d["lines"]).query_nearest(shapely.points(d["px"], d["py"]), all_matches=False)
    out = np.full(len(d["px"]), -1)
    out[pi] = li
    return out


# --------------------------------------------------------------------------- folium
def _legend_html(title, subtitle, items, extra=""):
    rows = "".join(
        f'<div><span style="display:inline-block;width:18px;height:{4 if k == "line" else 10}px;'
        f'background:{c};border-radius:{0 if k == "line" else 5}px;margin-right:6px;vertical-align:middle"></span>{l}</div>'
        for l, c, k in items)
    return (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;'
            f'border:1px solid #999;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
            f'<b>{title}</b><br>{subtitle}<hr style="margin:4px 0">{rows}{extra}</div>')


def trip_map(trip, pred_arcs=None, show_route: bool = True, confidence=None, prepared: dict | None = None,
             pred_label: str = "Predicted route") -> folium.Map:
    """Interactive folium map of one trip (see module docstring for the layers).

    pred_arcs:  sequence with one predicted arc id per GPS point (None / -1 = no prediction)
    confidence: sequence with one value in [0, 1] per GPS point (colours the GPS markers)
    prepared:   output of prepare(trip), to avoid recomputing it
    """
    d = prepared if prepared is not None else prepare(trip)
    lat, lon, nodes, arcs = d["lat"], d["lon"], d["nodes"], d["arcs"]
    has_route = show_route and len(d["route_pieces"]) > 0

    m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    def arc_coords(ids):  # folium wants [lat, lon]
        return [[[nodes[a, 1], nodes[a, 0]], [nodes[b, 1], nodes[b, 0]]] for a, b in arcs[ids]]

    def node_line(seq):
        return [[nodes[k, 1], nodes[k, 0]] for k in seq]

    # road network
    net = folium.FeatureGroup(name=f"Road network (≤ {NETWORK_BUFFER_M} m)")
    tw = d["near"][d["two_way"][d["near"]]]
    ow = d["near"][~d["two_way"][d["near"]]]
    if len(tw):
        folium.PolyLine(arc_coords(tw), color=COLORS["two_way"], weight=2, opacity=0.7).add_to(net)
    if len(ow):
        folium.PolyLine(arc_coords(ow), color=COLORS["one_way"], weight=2, opacity=0.8).add_to(net)
    net.add_to(m)

    # reference route
    if has_route:
        rg = folium.FeatureGroup(name=f"Reference route ({len(d['route_pieces'])} piece(s))")
        for i, seq in enumerate(d["route_pieces"]):
            folium.PolyLine(node_line(seq), color=COLORS["route"], weight=6, opacity=0.7,
                            tooltip=f"reference route piece {i + 1}").add_to(rg)
        rg.add_to(m)

    # predicted route
    pred_pieces = arc_pieces(pred_arcs, arcs) if pred_arcs is not None else []
    if pred_pieces:
        pg_ = folium.FeatureGroup(name=f"{pred_label} ({len(pred_pieces)} piece(s))")
        for i, seq in enumerate(pred_pieces):
            folium.PolyLine(node_line(seq), color=COLORS["pred"], weight=3, opacity=0.9,
                            tooltip=f"{pred_label.lower()} piece {i + 1}").add_to(pg_)
        pg_.add_to(m)

    # GPS track + points
    tg = folium.FeatureGroup(name="GPS track (line)")
    folium.PolyLine(np.c_[lat, lon].tolist(), color=COLORS["gps"], weight=1.5, opacity=0.6).add_to(tg)
    tg.add_to(m)

    conf = None if confidence is None else np.clip(np.asarray(confidence, dtype=float), 0, 1)
    cmap = cm.LinearColormap(["#d62728", "#ffbf00", "#2ca02c"], vmin=0, vmax=1, caption="Point confidence")
    step = max(1, int(np.ceil(len(lat) / MAX_MARKERS)))
    pg = folium.FeatureGroup(name=f"GPS points (every {step})" + (" — coloured by confidence" if conf is not None else ""))
    for i in range(0, len(lat), step):
        tip = f"idx {i} · t={d['t'][i]:.0f}s"
        if has_route:
            tip += f" · {d['d_route'][i]:.1f} m to route"
        if conf is not None:
            tip += f" · confidence {conf[i]:.2f}"
        if pred_arcs is not None:
            tip += f" · arc {pred_arcs[i]}"
        color = cmap(conf[i]) if conf is not None else COLORS["gps"]
        folium.CircleMarker([lat[i], lon[i]], radius=2.5, color=color, fill=True, fill_color=color,
                            fill_opacity=0.9, weight=0, tooltip=tip).add_to(pg)
    pg.add_to(m)
    if conf is not None:
        cmap.add_to(m)

    # far-off points (only meaningful with a route)
    far = np.flatnonzero(d["d_route"] > FAR_M) if has_route else np.array([], dtype=int)
    if has_route:
        fg = folium.FeatureGroup(name=f"GPS > {FAR_M} m from route ({len(far)})")
        for i in far[:: max(1, int(np.ceil(len(far) / MAX_MARKERS)))]:
            folium.CircleMarker([lat[i], lon[i]], radius=3, color=COLORS["far"], fill=True, fill_opacity=0.9,
                                weight=0, tooltip=f"idx {i} · {d['d_route'][i]:.0f} m from route").add_to(fg)
        fg.add_to(m)

    # time gaps
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

    items = [("Road network, two-way", COLORS["two_way"], "line"),
             ("Road network, one-way", COLORS["one_way"], "line")]
    if has_route:
        items.append(("Reference route", COLORS["route"], "line"))
    if pred_pieces:
        items.append((pred_label, COLORS["pred"], "line"))
    items.append(("GPS track / points", COLORS["gps"], "dot"))
    if has_route:
        items.append((f"GPS > {FAR_M} m from route", COLORS["far"], "dot"))
    items.append((f"Time gap > {GAP_S} s", COLORS["gap"], "line"))
    subtitle = f"{len(lat):,} GPS points · {len(d['near']):,} arcs shown" + (f" · {len(far):,} far points" if has_route else "")
    m.get_root().html.add_child(folium.Element(
        _legend_html(f"Trip {trip.trip_id} ({trip.partition})", subtitle, items)))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    return m


# --------------------------------------------------------------------------- matplotlib
def static_png(trip, d: dict, path, pred_arcs=None) -> None:
    """Static PNG of the same layers in projected metres (for the report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    nx, ny, arcs = d["nx"], d["ny"], d["arcs"]
    x0, y0 = d["px"].min(), d["py"].min()  # km relative to the track's corner

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
    for i, seq in enumerate(arc_pieces(pred_arcs, arcs) if pred_arcs is not None else []):
        ax.plot((nx[seq] - x0) / 1000, (ny[seq] - y0) / 1000, color=COLORS["pred"], lw=1.2,
                label="predicted route" if i == 0 else None)
    sc = ax.scatter((d["px"] - x0) / 1000, (d["py"] - y0) / 1000, c=np.clip(d["d_route"], 0, 100), cmap="viridis_r",
                    s=3, zorder=3, label="GPS (colour = m to route)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="distance to route (m, clipped at 100)")
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_xlabel("x (km, UTM)"); ax.set_ylabel("y (km, UTM)")
    ax.set_title(f"Trip {trip.trip_id} ({trip.partition}) — {len(d['px']):,} GPS points")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
