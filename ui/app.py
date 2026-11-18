"""PathSnap evaluator UI (Laptop 2) — Streamlit front end for the PathSnap API.

The UI only talks to the API over HTTP (no model, no dataset, no reference routes here):
    1. GET  /health  — connection check
    2. GET  /trips   — the evaluation trips to choose from
    3. POST /match   — run map-matching for the chosen trip; show route, scores and timings

Set the API address in the sidebar or with PATHSNAP_API_URL (e.g. http://192.168.1.20:8000).
Run: streamlit run ui/app.py
"""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

LOW_CONF = 0.6
MAX_MARKERS = 3000

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap — GPS map-matching")

with st.sidebar:
    st.header("API connection")
    api = st.text_input("API URL", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")).rstrip("/")
    if st.button("Check connection"):
        try:
            h = requests.get(f"{api}/health", timeout=5).json()
            st.success(f"Connected — partition `{h['partition']}`")
            st.json(h["model"])
        except Exception as exc:
            st.error(f"Cannot reach {api}: {exc}")


@st.cache_data(ttl=60, show_spinner=False)
def get_trips(url: str):
    r = requests.get(f"{url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()


try:
    trips = get_trips(api)
except Exception as exc:
    st.error(f"Could not load trips from {api}/trips — is the API running? ({exc})")
    st.stop()

st.caption(f"Partition served by the API: **{trips['partition']}** — {len(trips['trips'])} trips")
labels = {f"{t['trip_id']}  ({t['n_points']:,} GPS points)": t["trip_id"] for t in trips["trips"]}
choice = st.selectbox("Choose a trip", list(labels))
if st.button("Run map-matching", type="primary"):
    with st.spinner("Running the pipeline on the API laptop..."):
        try:
            r = requests.post(f"{api}/match", json={"trip_id": labels[choice]}, timeout=600)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except Exception as exc:
            st.session_state.pop("result", None)
            detail = getattr(getattr(exc, "response", None), "text", "")
            st.error(f"Map-matching failed: {exc} {detail}")

res = st.session_state.get("result")
if not res:
    st.info("Pick a trip and press **Run map-matching**.")
    st.stop()


def fmt(v, pct=True):
    return "n/a" if v is None else (f"{v:.1%}" if pct else f"{v:,}")


m = res["metrics"]
st.subheader(f"Trip {res['trip_id']} — {res['model']['name']} (v{res['model']['version']})")
c = st.columns(4)        # two rows so the values are not truncated on narrower screens
c[0].metric("Top-1 accuracy", fmt(m["top1"]))
c[1].metric("Macro-F1", fmt(m["macro_f1"]))
c[2].metric("Route continuity", fmt(m["route_continuity"]))
c[3].metric("Candidate miss rate", fmt(m["candidate_miss_rate"]))
c = st.columns(4)
c[0].metric("Route precision", fmt(m["route_precision"]))
c[1].metric("Route recall", fmt(m["route_recall"]))
c[2].metric("Route F1", fmt(m["route_f1"]))
c[3].metric("GPS points", fmt(m["n_points"], pct=False))

# ---- map
gps = res["gps"]
pts = res["matched_points"]
lats = [p[0] for p in gps]
lons = [p[1] for p in gps]
fmap = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], tiles="OpenStreetMap", control_scale=True)
g1 = folium.FeatureGroup(name="Raw GPS track")
folium.PolyLine(gps, color="#1f77b4", weight=2, opacity=0.6).add_to(g1)
g1.add_to(fmap)
if res["route_geometry"]:
    g2 = folium.FeatureGroup(name="Predicted route")
    folium.PolyLine(res["route_geometry"], color="#e31a1c", weight=4, opacity=0.9).add_to(g2)
    g2.add_to(fmap)
cmap = LinearColormap(["#d62728", "#ffbf00", "#2ca02c"], vmin=0, vmax=1, caption="Matched point confidence")
g3 = folium.FeatureGroup(name="Matched points (colour = confidence)")
step = max(1, len(pts) // MAX_MARKERS)
for p in pts[::step]:
    if p["arc_id"] is None:
        continue
    col = cmap(p["confidence"])
    folium.CircleMarker([p["lat"], p["lon"]], radius=3, color=col, fill=True, fill_color=col, fill_opacity=0.9, weight=0,
                        tooltip=f"point {p['idx']} · arc {p['arc_id']} · confidence {p['confidence']:.2f}").add_to(g3)
g3.add_to(fmap)
cmap.add_to(fmap)
folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
legend = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:6px 10px;'
          'border:1px solid #999;border-radius:6px;font:12px sans-serif;color:#222">'
          '<span style="color:#1f77b4">━━</span> raw GPS track<br>'
          '<span style="color:#e31a1c">━━</span> predicted route<br>'
          '<span style="color:#2ca02c">●</span>/<span style="color:#d62728">●</span> matched point, high/low confidence</div>')
fmap.get_root().html.add_child(folium.Element(legend))
folium.LayerControl(collapsed=False).add_to(fmap)
fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
st_folium(fmap, height=620, use_container_width=True, returned_objects=[])

# ---- timings and low-confidence stretches
left, right = st.columns(2)
with left:
    st.markdown("**Server timings (ms)**")
    st.table({k: [round(v, 1)] for k, v in res["timing_ms"].items()})
    st.caption(f"{m['n_points']:,} GPS points")
with right:
    st.markdown(f"**Low-confidence stretches (confidence < {LOW_CONF})**")
    runs, start = [], None
    for p in pts:
        low = p["arc_id"] is not None and p["confidence"] < LOW_CONF
        if low and start is None:
            start = p["idx"]
        if not low and start is not None:
            runs.append((start, p["idx"] - 1)); start = None
    if start is not None:
        runs.append((start, pts[-1]["idx"]))
    if runs:
        runs.sort(key=lambda r: r[1] - r[0], reverse=True)
        st.table([{"from point": a, "to point": b, "length (points)": b - a + 1} for a, b in runs[:15]])
        st.caption(f"{len(runs)} stretches in total (longest 15 shown)")
    else:
        st.write("None — every matched point is above the threshold.")
