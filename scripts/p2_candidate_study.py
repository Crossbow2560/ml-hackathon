"""P2 — label quality and candidate miss rate vs radius R and cap K.

For every train and student_test trip:
  1. align GPS points to the reference route (pathsnap.labels) -> per-point labels
  2. generate candidates at the largest radius once, then evaluate every (R, K):
       miss rate  = share of RELIABLE points whose label arc is not among the candidates
       cands/pt   = mean number of candidates kept per point
The (R, K) choice trades recall of the true arc (miss rate) against classifier/HMM cost.

Outputs (results/p2/):
    label_summary.csv     one row per trip: unreliable share, label distance percentiles, route coverage
    candidate_study.csv   one row per (partition, trip, R, K)
    candidate_study_summary.csv / candidate_miss_rate.png
Run: python -m scripts.p2_candidate_study
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.candidates import TripGeometry, candidates
from pathsnap.io import REPO_ROOT, list_trips, load_trip
from pathsnap.labels import align_to_route
from pathsnap.route_metrics import continuity

RADII = (25, 50, 75, 100)
KS = (10, 20, 30, 45, 60, None)
OUT = REPO_ROOT / "results" / "p2"


def study_trip(trip):
    g = TripGeometry.from_trip(trip)
    lab = align_to_route(g, trip.route)
    rel = lab[~lab.unreliable]
    lsum = {"trip_id": trip.trip_id, "n_points": len(lab), "unreliable_share": float(lab.unreliable.mean()),
            "label_dist_p50_m": float(rel.label_dist_m.median()) if len(rel) else np.nan,
            "label_dist_p95_m": float(rel.label_dist_m.quantile(.95)) if len(rel) else np.nan,
            "route_positions_used": len(np.unique(lab.route_pos)) / len(trip.route),
            "label_seq_continuity": continuity(lab.label_arc, trip.arcs)}

    c = candidates(g, radius_m=max(RADII), max_k=None)          # sorted by distance within each point
    c["rank"] = c.groupby("obs_idx").cumcount()
    c = c.merge(lab[["obs_idx", "label_arc"]], on="obs_idx")
    c["is_label"] = c.cand_arc == c.label_arc
    rel_ids = rel.obs_idx.to_numpy()
    rows = []
    for r in RADII:
        cr = c[c.dist_m <= r]
        for k in KS:
            ck = cr if k is None else cr[cr["rank"] < k]
            hit = ck.loc[ck.is_label, "obs_idx"].unique()
            per_pt = ck.groupby("obs_idx").size().reindex(lab.obs_idx, fill_value=0)
            rows.append({"trip_id": trip.trip_id, "radius_m": r, "k": "all" if k is None else k,
                         "miss_rate": float(1 - np.isin(rel_ids, hit).mean()) if len(rel_ids) else np.nan,
                         "cands_per_point": float(per_pt.mean()),
                         "share_no_candidates": float((per_pt == 0).mean())})
    return lsum, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labs, rows = [], []
    t0 = time.time()
    for part in ("train", "student_test"):
        ids = list_trips(part)
        for i, tid in enumerate(ids, 1):
            ls, rs = study_trip(load_trip(tid, part))
            labs.append({"partition": part, **ls})
            rows += [{"partition": part, **r} for r in rs]
            print(f"\r{part} {i}/{len(ids)} ({time.time() - t0:.0f}s)", end="", flush=True)
        print()
    lab = pd.DataFrame(labs)
    st = pd.DataFrame(rows)
    lab.to_csv(OUT / "label_summary.csv", index=False)
    st.to_csv(OUT / "candidate_study.csv", index=False)

    # point-weighted summary (each trip weighted by its number of points)
    st = st.merge(lab[["partition", "trip_id", "n_points"]], on=["partition", "trip_id"])
    summ = (st.groupby(["partition", "radius_m", "k"], sort=False)
              .apply(lambda g: pd.Series({
                  "miss_rate_weighted": np.average(g.miss_rate.fillna(0), weights=g.n_points),
                  "miss_rate_trip_p90": g.miss_rate.quantile(.9),
                  "cands_per_point": np.average(g.cands_per_point, weights=g.n_points)}), include_groups=False)
              .reset_index())
    summ.to_csv(OUT / "candidate_study_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, part in zip(axes, ("train", "student_test")):
        s = summ[summ.partition == part]
        for r in RADII:
            sr = s[s.radius_m == r]
            ax.plot(sr.cands_per_point, sr.miss_rate_weighted * 100, "o-", label=f"R = {r} m")
            for _, row in sr.iterrows():
                ax.annotate(str(row.k), (row.cands_per_point, row.miss_rate_weighted * 100), fontsize=6,
                            xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("mean candidates per point"); ax.set_title(f"Candidate miss rate vs cost — {part} (labels = K)")
        ax.legend(); ax.grid(alpha=.3)
    axes[0].set_ylabel("miss rate on reliable points (%)")
    fig.savefig(OUT / "candidate_miss_rate.png", dpi=150, bbox_inches="tight")

    print(f"done in {time.time() - t0:.0f}s")
    print(lab.groupby("partition")[["unreliable_share", "label_dist_p50_m", "label_dist_p95_m",
                                    "route_positions_used", "label_seq_continuity"]].mean().round(3))
    print(summ[summ.radius_m.isin([50, 75])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
