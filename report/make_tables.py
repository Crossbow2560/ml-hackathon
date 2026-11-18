"""Report tables and numbers, generated ONLY from committed result files.

report/build_report.py replaces
    {{table:<name>}}                 with the markdown table TABLES[<name>]()
    {{v:<model>:<column>[:fmt]}}     with a value from results/models/model_comparison.csv
    {{j:<file>:<key.path>[:fmt]}}    with a value from a JSON result file (path relative to repo)
so every number in the PDF can be traced back to (and regenerated from) the repository.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from pathsnap.io import REPO_ROOT

R = REPO_ROOT / "results"
PARTS = ("train", "student_test")


def md(df: pd.DataFrame, digits: int = 3) -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "N/A" if np.isnan(v) else f"{v:,.{digits}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False))


@lru_cache(None)
def comparison() -> pd.DataFrame:
    return pd.read_csv(R / "models" / "model_comparison.csv").set_index("model")


def an(part, name):
    return pd.read_csv(R / "analysis" / part / name)


# ----------------------------------------------------------------------------- data
def dataset_overview():
    rows = []
    for p in PARTS:
        inv, trk, net, rte = an(p, "inventory.csv"), an(p, "tracks_summary.csv"), an(p, "network_summary.csv"), an(p, "routes_summary.csv")
        rows.append({"Partition": p, "Trips": len(inv), "GPS points": int(trk.n_points.sum()),
                     "Points/trip (min–median–max)": f"{trk.n_points.min():,}–{int(trk.n_points.median()):,}–{trk.n_points.max():,}",
                     "Nodes/trip (median)": int(net.n_nodes.median()), "Arcs/trip (median)": int(net.n_arcs.median()),
                     "Route arcs/trip (median)": int(rte.route_n_arcs.median()),
                     "Share Δt = 1 s": float(np.average(trk.dt_share_1s, weights=trk.n_points - 1)),
                     "Trips with gap > 60 s": int((trk.n_gaps_gt_60s > 0).sum()),
                     "Two-way arc share (median)": float(net.two_way_share.median()),
                     "Route continuity (min)": float(rte.continuity.min())})
    return md(pd.DataFrame(rows).set_index("Partition").T.reset_index().rename(columns={"index": "Quantity"}))


def missing_values():
    cols = {"n_nan_rows": "Rows with NaN", "n_inf_rows": "Rows with inf", "n_out_of_range": "Out-of-range coordinates",
            "lonlat_swap_suspected": "Trips with lon/lat swap suspected", "n_duplicate_rows": "Duplicate rows",
            "n_duplicate_t": "Duplicate timestamps", "n_non_monotonic_t": "Non-monotonic time steps",
            "n_non_integer_dt": "Non-integer Δt steps", "n_speed_gt_70mps": "Steps implying > 70 m/s"}
    out = {}
    for p in PARTS:
        i = an(p, "track_issues.csv")
        i["lonlat_swap_suspected"] = i["lonlat_swap_suspected"].astype(str).eq("True")
        out[p] = [int(i[c].sum()) for c in cols]
    inv_issues = {p: int((an(p, "inventory.csv")["issues"].fillna("") != "").sum()) for p in PARTS}
    df = pd.DataFrame(out, index=list(cols.values()))
    df.loc["Trips with missing files / parse errors"] = [inv_issues[p] for p in PARTS]
    return md(df.reset_index().rename(columns={"index": "Check"}))


def gps_noise():
    rows = []
    for p in PARTS:
        s, g = an(p, "gps_route_distance_sample.csv"), an(p, "gps_route_distance.csv")
        q = s.d_route_m.quantile
        rows.append({"Partition": p, "p50 (m)": q(.5), "p90 (m)": q(.9), "p99 (m)": q(.99), "max (m)": s.d_route_m.max(),
                     "Share > 50 m": (s.d_route_m > 50).mean(),
                     "Nearest arc on route (any direction)": g.nearest_arc_on_route_share.mean()})
    return md(pd.DataFrame(rows), 3)


def far_trips():
    g = an("train", "gps_route_distance.csv")
    f = g[g.share_d_route_gt_50m > 0.05].copy()
    f["Cause"] = np.where((f.far_spread_m <= 1500) & (f.far_time_share >= 0.3), "off-network dwell", "route mismatch / detour")
    f = f[["trip_id", "n_points", "share_d_route_gt_50m", "far_spread_m", "far_time_share", "Cause"]]
    f.columns = ["Trip (train)", "GPS points", "Share > 50 m from route", "Spread of far points (m)", "Share of trip time", "Cause"]
    return md(f, 2)


def candidate_study():
    s = pd.read_csv(R / "p2" / "candidate_study_summary.csv")
    s = s[(s.partition == "train") & s.radius_m.isin([25, 50, 75]) & s.k.astype(str).isin(["10", "20", "30", "all"])]
    s = s[["radius_m", "k", "miss_rate_weighted", "cands_per_point"]]
    s.columns = ["Radius R (m)", "Cap K", "Candidate miss rate (train, reliable points)", "Candidates per point"]
    return md(s, 4)


def label_summary():
    l = pd.read_csv(R / "p2" / "label_summary.csv")
    g = l.groupby("partition")[["unreliable_share", "label_dist_p50_m", "label_dist_p95_m", "route_positions_used",
                                "label_seq_continuity"]].mean().reset_index()
    g.columns = ["Partition", "Unreliable points", "Point→label arc p50 (m)", "p95 (m)", "Route arcs that receive a point",
                 "Continuity of per-point labels"]
    return md(g, 3)


def feature_table():
    f = json.loads((REPO_ROOT / "artifacts" / "feature_list.json").read_text())
    return md(pd.DataFrame([{"Feature": x["name"], "Unit": x["unit"], "Meaning": x["description"]} for x in f["features"]]))


def rule_baselines():
    b = pd.read_csv(R / "p3" / "rule_baselines.csv").pivot(index="rule", columns="partition", values="top1")
    b = b.reindex(["nearest", "nearest_heading", "window_heading"])[["train", "student_test"]].reset_index()
    b.columns = ["Hand rule (no learning)", "Top-1 train", "Top-1 student_test"]
    return md(b, 3)


# ----------------------------------------------------------------------------- models
ORDER = ["logreg", "gnb", "knn", "rf", "lgbm", "mlp", "lgbm_rank", "lgbm_rank_tuned"]


def _cfg(s: str) -> str:
    import re
    s = s.replace("penalty=deprecated", "penalty=l2 (default)")
    return re.sub(r"\d+\.\d{5,}", lambda m: f"{float(m.group()):.4g}", s)


def model_comparison():
    c = comparison().reindex(ORDER)
    df = pd.DataFrame({
        "Model": c.index, "Algorithm family": c.family, "Key configuration": [_cfg(x) for x in c.key_config],
        "CV Macro F1 (mean ± sd)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(c.cv_macro_f1_mean, c.cv_macro_f1_std)],
        "Top-1 Accuracy (CV / student_test)": [f"{a:.4f} / {b:.4f}" for a, b in zip(c.cv_top1, c.st_top1)],
        "Route continuity (student_test, classifier only)": c.st_route_continuity,
        "Student-Test Macro F1": c.st_macro_f1, "Log Loss (student_test)": c.st_log_loss,
        "Inference time (ms / 1,000 points)": c.infer_ms_per_1k_obs})
    return md(df, 4)


def model_fit():
    c = comparison().reindex(ORDER)
    df = pd.DataFrame({"Model": c.index, "Train Top-1": c.train_top1, "CV Top-1": c.cv_top1, "Student-test Top-1": c.st_top1,
                       "Train − CV gap": c.train_top1 - c.cv_top1, "Train log loss": c.train_log_loss,
                       "CV log loss": c.cv_log_loss, "ST precision": c.st_precision, "ST recall": c.st_recall,
                       "Fit time (s)": c.train_time_s})
    return md(df, 4)


def hpo_space():
    s = json.loads((R / "optuna" / "search_space.json").read_text())
    rows = [{"Hyperparameter": k, "Type": v[0], "Range": f"[{v[1]}, {v[2]}]", "Scale": "log" if len(v) > 3 and v[3] else "linear"}
            for k, v in s["space"].items()]
    return md(pd.DataFrame(rows))


def hpo_best():
    b = json.loads((R / "optuna" / "best_params.json").read_text())
    untuned = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "subsample": 1.0,
               "colsample_bytree": 1.0, "reg_lambda": 0.0, "subsample_freq": 0}
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    return md(pd.DataFrame([{"Hyperparameter": k, "Untuned (initial)": fmt(untuned.get(k)),
                             "Best (trial %d)" % b["best_trial"]: fmt(v)} for k, v in b["params"].items()]))


def tuned_vs_untuned():
    c = comparison().loc[["lgbm_rank", "lgbm_rank_tuned"]]
    rows = [("CV macro-F1 (GroupKFold 5, train)", "cv_macro_f1_mean"), ("CV Top-1", "cv_top1"),
            ("Student-test Top-1", "st_top1"), ("Student-test macro-F1", "st_macro_f1"),
            ("Student-test log loss", "st_log_loss"), ("Train Top-1 (overfitting check)", "train_top1"),
            ("Inference time (ms / 1,000 points)", "infer_ms_per_1k_obs"), ("Fit time (s)", "train_time_s")]
    return md(pd.DataFrame([{"Metric": n, "Untuned lgbm_rank": c.loc["lgbm_rank", k], "Tuned lgbm_rank": c.loc["lgbm_rank_tuned", k]}
                            for n, k in rows]), 4)


# ----------------------------------------------------------------------------- HMM
def transition_eval():
    t = json.loads((R / "hmm" / "transition_model.json").read_text())
    rows = []
    for p in PARTS:
        for v in ("exponential", "learned"):
            rows.append({"Pairs from": p, "Transition model": v, "Pair ROC-AUC": t[f"{p}_auc_{v}"],
                         "True pair ranked first in its step": t[f"{p}_true_pair_first_{v}"]})
    return md(pd.DataFrame(rows), 4)


def transition_coef():
    t = json.loads((R / "hmm" / "transition_model.json").read_text())
    return md(pd.DataFrame([{"Pair feature (standardised)": k, "Coefficient": v}
                            for k, v in t["coefficients_standardised"].items()]), 3)


def decoder_params():
    d = json.loads((R / "hmm" / "decoder_params.json").read_text())
    rows = [{"Transition model": "exponential", "α (emission weight)": d["exponential"]["alpha"],
             "Transition weight": f"1/β, β = {d['exponential']['beta']:.1f} m", "Top-1 (30 train trips, OOF)": d["exponential"]["best_top1"]},
            {"Transition model": "learned", "α (emission weight)": d["learned"]["alpha"],
             "Transition weight": f"γ = {d['learned']['gamma']:.2f}", "Top-1 (30 train trips, OOF)": d["learned"]["best_top1"]},
            {"Transition model": "none (classifier only)", "α (emission weight)": float("nan"), "Transition weight": "—",
             "Top-1 (30 train trips, OOF)": d["classifier_only_top1"]}]
    return md(pd.DataFrame(rows), 4)


def hmm_comparison():
    h = pd.read_csv(R / "hmm" / "hmm_comparison.csv")
    h = h[["method", "top1", "macro_f1", "log_loss", "point_continuity", "route_continuity", "route_precision", "route_recall", "route_f1"]]
    h.columns = ["Method (student_test)", "Top-1", "Macro-F1", "Log loss", "Point continuity", "Stitched-route continuity",
                 "Route precision", "Route recall", "Route F1"]
    return md(h, 4)


def hmm_per_trip():
    h = pd.read_csv(R / "hmm" / "hmm_per_trip.csv", dtype={"trip_id": str})
    p = h.pivot(index="trip_id", columns="method", values="top1")[["classifier_only", "hmm_exponential", "hmm_learned"]]
    n = h.drop_duplicates("trip_id").set_index("trip_id").n_points
    p.insert(0, "GPS points", n)
    p["Gain (learned − classifier)"] = p.hmm_learned - p.classifier_only
    p = p.reset_index().rename(columns={"trip_id": "Trip", "classifier_only": "Classifier only", "hmm_exponential": "HMM exponential",
                                        "hmm_learned": "HMM learned"})
    return md(p, 3)


TABLES = {f.__name__: f for f in [dataset_overview, missing_values, gps_noise, far_trips, candidate_study, label_summary,
                                  feature_table, rule_baselines, model_comparison, model_fit, hpo_space, hpo_best,
                                  tuned_vs_untuned, transition_eval, transition_coef, decoder_params, hmm_comparison,
                                  hmm_per_trip]}


def value(model: str, col: str, fmt: str = ".4f") -> str:
    return format(comparison().loc[model, col], fmt)


def json_value(path: str, key: str, fmt: str = ".4f") -> str:
    v = json.loads((REPO_ROOT / path).read_text())
    for k in key.split("."):
        v = v[k]
    return format(v, fmt) if fmt else str(v)
