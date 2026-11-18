"""P6 — classifier-only vs classifier + HMM (exponential / learned transitions) on student_test.

Uses the chosen classifier (results/models/<model>.joblib, refit on all train trips) to score
every student_test candidate, then decodes with the tuned decoder weights
(results/hmm/decoder_params.json). Reports, point-weighted over the 10 trips:

    top1, candidate macro-F1 / log loss (HMM: forward-backward posteriors), point continuity,
    stitched-route continuity, length-weighted route precision / recall / F1,
    and the reference labels' own continuity as the ceiling.

Outputs (results/hmm/): hmm_comparison.csv, hmm_per_trip.csv, pred/<trip>_<method>.csv,
                        plots/*.png, maps/<trip>.html (best and worst HMM gain)
Run: python -m scripts.evaluate_hmm --model lgbm_rank_tuned
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.evaluation import build_context, evaluate
from pathsnap.hmm import HMMParams
from pathsnap.io import REPO_ROOT, load_trip
from pathsnap.transitions import ExpTransition
from pathsnap.viz import prepare, trip_map

OUT = REPO_ROOT / "results" / "hmm"
METHOD_ORDER = ["classifier_only", "hmm_exponential", "hmm_learned", "reference_labels"]


def weighted(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["top1", "macro_f1", "log_loss", "precision", "recall", "point_continuity", "route_continuity",
            "route_precision", "route_recall", "route_f1"]
    g = df.groupby("method")
    out = g.apply(lambda x: pd.Series({c: np.average(x[c], weights=x.n_reliable) if x[c].notna().all() else np.nan
                                       for c in cols}), include_groups=False)
    out["decode_seconds"] = g.decode_seconds.sum() if "decode_seconds" in df else np.nan
    return out.reindex([m for m in METHOD_ORDER if m in out.index]).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lgbm_rank_tuned")
    a = ap.parse_args()
    dp = json.loads((OUT / "decoder_params.json").read_text())
    params = HMMParams(gap_s=dp["gap_s"], v_max_mps=dp["v_max_mps"], radius_m=dp["radius_m"])
    models = {"exponential": ExpTransition(), "learned": joblib.load(OUT / "transition_learned.joblib")}
    settings = {v: (dp[v]["alpha"], dp[v]["weight"]) for v in models}

    clf = joblib.load(REPO_ROOT / "results" / "models" / f"{a.model}.joblib")
    feats = pd.read_parquet(REPO_ROOT / "data_cache" / "features_student_test.parquet")
    feats["trip_id"] = feats["trip_id"].astype(str)
    feats = feats.sort_values(["trip_id", "obs_idx"], kind="stable").reset_index(drop=True)
    feats["p"] = clf.predict_proba(feats)

    (OUT / "pred").mkdir(parents=True, exist_ok=True)
    rows = []
    for tid, f in feats.groupby("trip_id"):
        trip = load_trip(tid, "student_test")
        t0 = time.time()
        ctx = build_context(trip, f[["obs_idx", "cand_arc", "proj_t", "label", "p"]], models, params)
        prep = time.time() - t0
        r, preds = evaluate(ctx, settings)
        for x in r:
            x["decode_seconds"] = prep if x["method"].startswith("hmm") else 0.0
        rows += r
        for m, p in preds.items():
            p.to_csv(OUT / "pred" / f"{tid}_{m}.csv", index=False)
        print(tid, " | ".join(f"{x['method']} top1={x['top1']:.3f} cont={x.get('route_continuity', np.nan):.3f}"
                              for x in r))
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "hmm_per_trip.csv", index=False)
    comp = weighted(per)
    comp.insert(1, "model", a.model)
    comp.to_csv(OUT / "hmm_comparison.csv", index=False)
    print(comp.round(4).to_string(index=False))

    # plots
    (OUT / "plots").mkdir(exist_ok=True)
    m = comp[comp.method != "reference_labels"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes, ["top1", "route_continuity", "route_f1"],
                              ["Top-1 accuracy", "Stitched-route continuity", "Route F1 (length-weighted)"]):
        ax.bar(m.method, m[col], color=["#9ecae9", "#f58518", "#54a24b"][:len(m)])
        ref = comp.loc[comp.method == "reference_labels", col]
        if col != "top1" and len(ref):
            ax.axhline(float(ref.iloc[0]), color="k", ls=":", label="reference labels")
            ax.legend()
        ax.set_ylim(max(0, m[col].min() - 0.1), 1.0); ax.set_title(f"{title} — student_test"); ax.tick_params(axis="x", rotation=15)
    fig.savefig(OUT / "plots" / "hmm_vs_classifier.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    piv = per.pivot(index="trip_id", columns="method", values="top1")
    fig, ax = plt.subplots(figsize=(8, 4))
    gain = (piv["hmm_learned"] - piv["classifier_only"]).sort_values()
    ax.bar(gain.index.str.lstrip("0"), gain * 100, color=np.where(gain >= 0, "#54a24b", "#e45756"))
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Top-1 change (percentage points)")
    ax.set_title("HMM (learned) minus classifier-only Top-1 per trip — student_test")
    fig.savefig(OUT / "plots" / "hmm_gain_per_trip.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # maps for the best and worst HMM gain
    (OUT / "maps").mkdir(exist_ok=True)
    for tid in {gain.index[-1], gain.index[0]}:
        trip = load_trip(tid, "student_test")
        pts = pd.read_csv(OUT / "pred" / f"{tid}_hmm_learned.csv")
        trip_map(trip, pred_arcs=pts.arc.to_numpy(), confidence=pts.confidence.to_numpy(), prepared=prepare(trip),
                 pred_label="Predicted route (classifier + HMM)").save(str(OUT / "maps" / f"{tid}.html"))
        print("map:", OUT / "maps" / f"{tid}.html")


if __name__ == "__main__":
    main()
