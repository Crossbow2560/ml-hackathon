"""P4 — train and compare candidate-scoring models.

For each model:
  1. GroupKFold(5) on the 80 train trips (a trip never spans folds): fit on 4 folds, predict
     the 5th -> per-fold macro-F1 / Top-1 / log loss, plus out-of-fold (OOF) predictions.
  2. Refit on all train trips (timed); score on train (over/underfitting evidence) and on the
     10 student_test trips: candidate metrics, Top-1, log loss, confusion matrix, inference
     time and classifier-only route continuity (argmax arc per point, see route_metrics).
Unreliable observations are removed from fitting and scoring.

Outputs (results/models/, or --out):
    model_comparison.csv   one row per model (re-running a model replaces its row)
    cv_folds.csv           one row per (model, fold)
    mlp_history.csv        epoch-wise train/val log loss of the MLP refit
    <model>.joblib, <model>_st_pred.parquet, <model>_oof.parquet (--save-oof)   [gitignored]
Run:
    python -m scripts.train_models --models all
    python -m scripts.train_models --models logreg lgbm --max-trips 5 --out results/models_dryrun
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.io import REPO_ROOT, load_trip
from pathsnap.metrics import all_metrics, argmax_rows
from pathsnap.models import FAMILIES, SEED, MLPModel, get_model
from pathsnap.route_metrics import continuity

N_FOLDS = 5
OBS = ["trip_id", "obs_idx"]


def load_features(partition: str, max_trips: int | None) -> pd.DataFrame:
    df = pd.read_parquet(REPO_ROOT / "data_cache" / f"features_{partition}.parquet")
    df = df[~df["unreliable"]].copy()
    df["trip_id"] = df["trip_id"].astype(str)
    if max_trips:
        df = df[df.trip_id.isin(sorted(df.trip_id.unique())[:max_trips])]
    return df.sort_values(OBS, kind="stable").reset_index(drop=True)


def route_continuity(pred_df: pd.DataFrame, partition: str) -> float:
    """Mean over trips of the continuity of the per-point argmax arc sequence."""
    best = argmax_rows(pred_df, "p").sort_values(OBS)
    vals = [continuity(g.cand_arc.to_numpy(), load_trip(tid, partition).arcs) for tid, g in best.groupby("trip_id")]
    return float(np.nanmean(vals))


def upsert(path: Path, rows: pd.DataFrame, key: str = "model"):
    if path.exists():
        old = pd.read_csv(path)
        rows = pd.concat([old[~old[key].isin(rows[key])], rows], ignore_index=True)
    rows.to_csv(path, index=False)


def run_model(name: str, train: pd.DataFrame, test: pd.DataFrame, out: Path, save_oof: bool, params: dict | None = None,
              label: str | None = None) -> dict:
    label = label or name
    params = params or {}
    n_folds = min(N_FOLDS, train.trip_id.nunique())
    folds, oof = [], np.zeros(len(train))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(train, groups=train.trip_id)):
        m = get_model(name, **params).fit(train.iloc[tr])
        oof[va] = m.predict_proba(train.iloc[va])
        fm = all_metrics(train.iloc[va].assign(p=oof[va]))
        folds.append({"model": label, "fold": k, **{x: fm[x] for x in ("macro_f1", "top1", "log_loss", "n_obs")}})
        print(f"  [{label}] fold {k}: macro_f1={fm['macro_f1']:.4f} top1={fm['top1']:.4f} log_loss={fm['log_loss']:.4f}")
    fdf = pd.DataFrame(folds)

    t0 = time.time()
    model = get_model(name, **params).fit(train)
    train_time = time.time() - t0
    trm = all_metrics(train.assign(p=model.predict_proba(train)))
    t0 = time.time()
    p_st = model.predict_proba(test)
    infer_s = time.time() - t0
    st = test.assign(p=p_st)
    stm = all_metrics(st)

    row = {
        "model": label, "family": FAMILIES[name], "key_config": model.config(),
        "cv_macro_f1_mean": fdf.macro_f1.mean(), "cv_macro_f1_std": fdf.macro_f1.std(),
        "cv_top1": fdf.top1.mean(), "cv_top1_std": fdf.top1.std(), "cv_log_loss": fdf.log_loss.mean(),
        "train_macro_f1": trm["macro_f1"], "train_top1": trm["top1"], "train_log_loss": trm["log_loss"],
        "st_top1": stm["top1"], "st_macro_f1": stm["macro_f1"], "st_precision": stm["precision"],
        "st_recall": stm["recall"], "st_accuracy": stm["accuracy"], "st_log_loss": stm["log_loss"],
        "st_route_continuity": route_continuity(st, "student_test"),
        "st_tn": stm["tn"], "st_fp": stm["fp"], "st_fn": stm["fn"], "st_tp": stm["tp"],
        "train_time_s": train_time, "infer_ms_per_1k_obs": 1e6 * infer_s / stm["n_obs"],   # seconds -> ms, per 1,000 observations
        "n_train_trips": train.trip_id.nunique(), "seed": SEED,
    }
    upsert(out / "model_comparison.csv", pd.DataFrame([row]))
    upsert(out / "cv_folds.csv", fdf)
    if isinstance(model, MLPModel):
        pd.DataFrame(model.history).assign(model=label).to_csv(out / f"{label}_history.csv", index=False)
    joblib.dump(model, out / f"{label}.joblib")
    st[OBS + ["cand_arc", "label", "p"]].to_parquet(out / f"{label}_st_pred.parquet", index=False)
    if save_oof:
        train[OBS + ["cand_arc", "label"]].assign(p=oof).to_parquet(out / f"{label}_oof.parquet", index=False)
    print(f"  [{label}] CV macro_f1 {row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f} | "
          f"student_test top1 {row['st_top1']:.4f} macro_f1 {row['st_macro_f1']:.4f} log_loss {row['st_log_loss']:.4f} "
          f"continuity {row['st_route_continuity']:.3f} | fit {train_time:.1f}s")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--max-trips", type=int, default=None, help="dry run: first N train / N student_test trips")
    ap.add_argument("--save-oof", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "models"))
    a = ap.parse_args()
    names = list(FAMILIES) if a.models == ["all"] else a.models
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = load_features("train", a.max_trips)
    test = load_features("student_test", a.max_trips)
    print(f"train: {train.trip_id.nunique()} trips, {len(train):,} rows | student_test: {test.trip_id.nunique()} trips, "
          f"{len(test):,} rows")
    t0 = time.time()
    for n in names:
        print(f"== {n}")
        s = time.time()
        run_model(n, train, test, out, a.save_oof)
        print(f"   {n} total {time.time() - s:.0f}s")
    print(f"== all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
