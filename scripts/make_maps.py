"""Write folium trip maps to results/maps/<trip_id>.html (layers: road network, reference
route, predicted route, GPS points, far-off points, gaps — see pathsnap/viz.py).

Predicted route source (pick at most one):
    --pred FILE.csv        columns idx, arc_id[, confidence] — one row per GPS point (model output)
    --nearest-baseline     nearest arc per GPS point; a NAIVE BASELINE, labelled as such on the map

Trip selection:
    --trip 1 33 ...        explicit ids
    --pick4                4 representative train trips chosen from the Phase 0 CSVs:
                           shortest, longest clean trip, most gaps > 60 s, densest network

Examples:
    python -m scripts.make_maps --partition train --pick4 --nearest-baseline
    python -m scripts.make_maps --partition student_test --trip 52 --pred results/hmm/pred_00000052.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.io import REPO_ROOT, load_trip
from pathsnap.viz import nearest_arcs, prepare, trip_map

PARTITIONS = ("train", "student_test")   # evaluator_test is never mapped outside the API


def pick_representative(partition: str) -> dict[str, str]:
    """Choose 4 distinct, representative trips from results/analysis/<partition>/*.csv."""
    a = REPO_ROOT / "results" / "analysis" / partition
    trk = pd.read_csv(a / "tracks_summary.csv", dtype={"trip_id": str})
    net = pd.read_csv(a / "network_summary.csv", dtype={"trip_id": str})
    gvr = pd.read_csv(a / "gps_route_distance.csv", dtype={"trip_id": str})
    df = trk.merge(net, on="trip_id").merge(gvr[["trip_id", "share_d_route_gt_50m"]], on="trip_id")
    df["trip_id"] = df.trip_id.str.zfill(8)
    clean = df[df.share_d_route_gt_50m <= 0.05]          # skip off-network-dwell / mismatch trips

    chosen: dict[str, str] = {}

    def take(label, frame, col, largest):
        frame = frame[~frame.trip_id.isin(chosen.values())]
        row = frame.nlargest(1, col) if largest else frame.nsmallest(1, col)
        chosen[label] = row.trip_id.iloc[0]

    take("short", clean, "path_length_m", largest=False)
    take("long", clean, "path_length_m", largest=True)
    take("gaps", df, "n_gaps_gt_60s", largest=True)
    take("dense_network", clean, "n_arcs", largest=True)
    return chosen


def read_pred(path: str, n_points: int):
    p = pd.read_csv(path)
    arcs = np.full(n_points, -1)
    arcs[p["idx"].to_numpy()] = p["arc_id"].fillna(-1).astype(int).to_numpy()
    conf = None
    if "confidence" in p:
        conf = np.zeros(n_points)
        conf[p["idx"].to_numpy()] = p["confidence"].to_numpy()
    return arcs, conf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", default="train", choices=PARTITIONS)
    ap.add_argument("--trip", nargs="*", default=[])
    ap.add_argument("--pick4", action="store_true")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--pred", help="CSV with idx, arc_id[, confidence] (needs exactly one --trip)")
    src.add_argument("--nearest-baseline", action="store_true")
    ap.add_argument("--no-route", action="store_true", help="hide the reference route layer")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "maps"))
    a = ap.parse_args()

    labels = {}
    if a.pick4:
        labels = {tid: lab for lab, tid in pick_representative(a.partition).items()}
    ids = list(labels) + [t.zfill(8) for t in a.trip if t.zfill(8) not in labels]
    if not ids:
        ap.error("give --trip <id> ... or --pick4")
    if a.pred and len(ids) != 1:
        ap.error("--pred needs exactly one trip")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for tid in ids:
        trip = load_trip(tid, a.partition, a.data_dir)
        d = prepare(trip)
        pred, conf, label = None, None, "Predicted route"
        if a.pred:
            pred, conf = read_pred(a.pred, trip.n_points)
        elif a.nearest_baseline:
            pred, label = nearest_arcs(d), "Nearest-arc baseline (not the model)"
        m = trip_map(trip, pred_arcs=pred, confidence=conf, show_route=not a.no_route, prepared=d, pred_label=label)
        path = out / f"{tid}.html"
        m.save(str(path))
        tag = f" [{labels[tid]}]" if tid in labels else ""
        print(f"{tid}{tag}: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path} "
              f"({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
