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
The drawing code lives in pathsnap/viz.py (shared with scripts/make_maps.py and the API).

Outputs: results/analysis/<partition>/maps/<trip_id>.html  (and .png with --static)
Run:     python -m scripts.analysis.06_visualise_trip --partition train --trip 00000001 00000054
         python -m scripts.analysis.06_visualise_trip --partition student_test --all --static
"""
from __future__ import annotations

import argparse

from pathsnap.io import list_trips, load_trip
from pathsnap.viz import prepare, static_png, trip_map
from scripts.analysis._common import ANALYSIS_PARTITIONS, out_dir


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
        trip_map(trip, prepared=d).save(str(html))
        msg = f"{tid}: {html} ({html.stat().st_size / 1e6:.1f} MB, {len(d['near'])} arcs)"
        if a.static:
            png = od / f"{tid}.png"
            static_png(trip, d, png)
            msg += f" + {png.name}"
        print(msg)


if __name__ == "__main__":
    main()
