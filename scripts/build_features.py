"""P3 — build the candidate feature table for one partition.

For every trip: geometry -> labels (monotonic alignment to the route) -> candidates within
R = radius_m (K nearest) -> 16 features -> label / unreliable columns. Writes

    data_cache/features_<partition>.parquet     (gitignored; deterministic, rebuild any time)
    results/p3/features_<partition>_summary.json

and checks the contract: no NaNs, <= K candidates and <= 1 positive per observation.

Run: python -m scripts.build_features --partition train
     python -m scripts.build_features --partition student_test
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from pathsnap.candidates import TripGeometry
from pathsnap.config import load_config
from pathsnap.features import FEATURES, add_labels, trip_features
from pathsnap.io import REPO_ROOT, list_trips, load_trip
from pathsnap.labels import align_to_route


def build(partition: str) -> pd.DataFrame:
    cfg = load_config()
    R, K = cfg["candidates"]["radius_m"], cfg["candidates"]["max_k"]
    unrel_m = cfg["labels"]["unreliable_m"]
    parts, no_cand, t0 = [], 0, time.time()
    ids = list_trips(partition)
    for i, tid in enumerate(ids, 1):
        trip = load_trip(tid, partition, require_route=True)
        geom = TripGeometry.from_trip(trip)
        lab = align_to_route(geom, trip.route, unreliable_m=unrel_m)
        f = add_labels(trip_features(trip, geom, R, K), lab)
        no_cand += trip.n_points - f.obs_idx.nunique()
        parts.append(f)
        print(f"\r{partition} {i}/{len(ids)} {tid} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    df = pd.concat(parts, ignore_index=True)
    df["trip_id"] = df["trip_id"].astype("string")
    df["obs_idx"] = df["obs_idx"].astype(np.int32)
    df["cand_arc"] = df["cand_arc"].astype(np.int32)
    df["unreliable"] = df["unreliable"].astype(bool)
    df.attrs["points_without_candidates"] = int(no_cand)
    return df


def check(df: pd.DataFrame, K: int) -> dict:
    g = df.groupby(["trip_id", "obs_idx"], sort=False)
    pos = g.label.sum()
    rel = ~g.unreliable.first()
    summ = {
        "rows": int(len(df)),
        "observations": int(g.ngroups),
        "points_without_candidates": df.attrs.get("points_without_candidates", 0),
        "nan_values": int(df[FEATURES].isna().sum().sum()),
        "max_candidates_per_obs": int(g.size().max()),
        "max_positives_per_obs": int(pos.max()),
        "positive_rate_rows": float(df.label.mean()),
        "unreliable_obs_share": float(1 - rel.mean()),
        "reliable_obs_with_positive": float((pos[rel] == 1).mean()),
        "nearest_is_label_reliable": float(df[(df.dist_rank == 0) & ~df.unreliable].label.mean()),
    }
    assert summ["nan_values"] == 0, "NaN in features"
    assert summ["max_candidates_per_obs"] <= K, "more than K candidates"
    assert summ["max_positives_per_obs"] <= 1, "more than one positive per observation"
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", required=True, choices=("train", "student_test"))
    a = ap.parse_args()
    K = load_config()["candidates"]["max_k"]
    t0 = time.time()
    df = build(a.partition)
    summ = check(df, K)
    summ["build_seconds"] = round(time.time() - t0, 1)
    cache = REPO_ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"features_{a.partition}.parquet"
    df.to_parquet(path, index=False)
    out = REPO_ROOT / "results" / "p3"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"features_{a.partition}_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
