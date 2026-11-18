"""P6 — train the learned HMM transition model (our novel component).

Training pairs come from the 80 train trips: for a sample of consecutive observation pairs
(t-1, t) — both reliable, both containing their label arc, Δt <= gap_s — every reachable
candidate pair (a, b) becomes a row of geometric pair features (pathsnap.transitions) with
target 1 if (a, b) is the true (label_{t-1}, label_t) pair, else 0. No classifier scores are
used, so this model cannot leak labels into the emissions.

Evaluated on student_test pairs (same sampling): pair ROC-AUC and "true pair ranked first
among all pairs of the step", compared with the exponential baseline score -|d_route - d_gc|.

Outputs (results/hmm/): transition_learned.joblib [gitignored], transition_model.json
Run: python -m scripts.train_transitions [--step-every 5]
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pathsnap.candidates import TripGeometry
from pathsnap.config import load_config
from pathsnap.graph import RoadGraph
from pathsnap.hmm import arc_bearings, build_steps
from pathsnap.io import REPO_ROOT, load_trip
from pathsnap.transitions import TRANS_FEATURES, LearnedTransition, feature_matrix, pair_features

OUT = REPO_ROOT / "results" / "hmm"


def collect_pairs(partition: str, every: int, gap_s: float, v_max: float, radius: float):
    feats = pd.read_parquet(REPO_ROOT / "data_cache" / f"features_{partition}.parquet",
                            columns=["trip_id", "obs_idx", "cand_arc", "proj_t", "label", "unreliable"])
    feats["trip_id"] = feats["trip_id"].astype(str)
    Xs, ys, step_ids = [], [], []
    sid = 0
    t0 = time.time()
    trips = sorted(feats.trip_id.unique())
    for k, tid in enumerate(trips, 1):
        f = feats[feats.trip_id == tid]
        trip = load_trip(tid, partition)
        geom, graph = TripGeometry.from_trip(trip), RoadGraph(trip.nodes, trip.arcs)
        bear = arc_bearings(geom)
        st = build_steps(f, "label")                       # ps = 0/1 labels per candidate
        unrel = f.groupby("obs_idx").unreliable.first()
        for t in range(1, len(st.obs), every):
            o0, o1 = st.obs[t - 1], st.obs[t]
            dt = geom.t[o1] - geom.t[o0]
            if dt > gap_s or unrel[o0] or unrel[o1] or st.ps[t - 1].max() < 1 or st.ps[t].max() < 1:
                continue
            d_gc = float(np.hypot(geom.px[o1] - geom.px[o0], geom.py[o1] - geom.py[o0]))
            pf = pair_features(graph, bear, st.arcs[t - 1], st.ts[t - 1], st.arcs[t], st.ts[t], d_gc, dt,
                               v_max * max(dt, 1.0) + 2 * radius)
            reach = pf["reachable"].ravel()
            y = np.outer(st.ps[t - 1], st.ps[t]).ravel()
            if not (reach & (y == 1)).any():
                continue
            Xs.append(feature_matrix(pf)[reach]); ys.append(y[reach]); step_ids.append(np.full(reach.sum(), sid))
            sid += 1
        print(f"\r{partition} {k}/{len(trips)} pairs so far {sum(len(y) for y in ys):,} ({time.time() - t0:.0f}s)",
              end="", flush=True)
    print()
    return np.concatenate(Xs), np.concatenate(ys).astype(int), np.concatenate(step_ids)


def ranked_first(score: np.ndarray, y: np.ndarray, steps: np.ndarray) -> float:
    df = pd.DataFrame({"s": score, "y": y, "g": steps})
    return float(df.loc[df.groupby("g").s.idxmax(), "y"].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step-every", type=int, default=5, help="use every N-th step of each trip")
    a = ap.parse_args()
    cfg = load_config()
    gap_s, v_max, R = cfg["hmm"]["gap_s"], cfg["hmm"]["v_max_mps"], cfg["candidates"]["radius_m"]
    OUT.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, gtr = collect_pairs("train", a.step_every, gap_s, v_max, R)
    Xst, yst, gst = collect_pairs("student_test", a.step_every, gap_s, v_max, R)
    t0 = time.time()
    model = LearnedTransition(C=1.0).fit(Xtr, ytr)
    fit_s = time.time() - t0

    i_abs = TRANS_FEATURES.index("abs_diff_m")
    res = {"features": TRANS_FEATURES, "coefficients_standardised": model.coefficients(),
           "train_pairs": int(len(ytr)), "train_steps": int(gtr.max() + 1), "train_positive_rate": float(ytr.mean()),
           "student_test_pairs": int(len(yst)), "step_every": a.step_every, "fit_seconds": round(fit_s, 1)}
    for part, X, y, g in (("train", Xtr, ytr, gtr), ("student_test", Xst, yst, gst)):
        p = model.model.predict_proba(X)[:, 1]
        res[f"{part}_auc_learned"] = float(roc_auc_score(y, p))
        res[f"{part}_auc_exponential"] = float(roc_auc_score(y, -X[:, i_abs]))
        res[f"{part}_true_pair_first_learned"] = ranked_first(p, y, g)
        res[f"{part}_true_pair_first_exponential"] = ranked_first(-X[:, i_abs], y, g)
    joblib.dump(model, OUT / "transition_learned.joblib")
    (OUT / "transition_model.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
