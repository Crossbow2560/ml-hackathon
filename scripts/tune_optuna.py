"""P5 — Bayesian hyperparameter optimisation (Optuna TPE) for the chosen model.

Objective: mean macro-F1 over GroupKFold(5) on the 80 train trips (a trip never spans folds).
Top-1 accuracy is recorded for every trial as well. The evaluator partition is never used,
and student_test is NOT used during the search (only for the final tuned-vs-untuned report).

Cost control (stated in the report):
  * temporal thinning — trials use every ``--thin``-th GPS observation per trip; points 1 s
    apart are near-duplicates, so this keeps the information while cutting rows ~``thin``x
  * median pruning — a trial whose running fold score is below the median of earlier trials
    after the first 2 folds is stopped early

After the search the best parameters are refit with exactly the P4 procedure
(scripts.train_models.run_model: full data, same folds, same metrics, student_test) and added
to results/models/model_comparison.csv as ``<model>_tuned``.

Outputs (results/optuna/): trials.csv, best_params.json, search_space.json,
                           optimisation_history.png, param_importance.png
Run: python -m scripts.tune_optuna --model lgbm_rank --trials 60
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.io import REPO_ROOT
from pathsnap.metrics import all_metrics
from pathsnap.models import SEED, get_model
from scripts.train_models import load_features, run_model

OUT = REPO_ROOT / "results" / "optuna"

# search spaces: (type, low, high[, log]) — justified in the report
SPACES = {
    "lgbm_rank": {"n_estimators": ("int", 100, 800), "learning_rate": ("float", 0.01, 0.2, True),
                  "num_leaves": ("int", 15, 255, True), "min_child_samples": ("int", 10, 300, True),
                  "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.5, 1.0),
                  "reg_lambda": ("float", 1e-3, 10.0, True)},
    "lgbm": {"n_estimators": ("int", 100, 800), "learning_rate": ("float", 0.01, 0.2, True),
             "num_leaves": ("int", 15, 255, True), "min_child_samples": ("int", 10, 300, True),
             "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.5, 1.0),
             "reg_lambda": ("float", 1e-3, 10.0, True)},
    "rf": {"n_estimators": ("int", 100, 500), "max_depth": ("int", 6, 40, True),
           "min_samples_leaf": ("int", 1, 50, True), "max_features": ("float", 0.2, 1.0)},
    "logreg": {"C": ("float", 1e-3, 100.0, True)},
}
FIXED = {"lgbm_rank": {"subsample_freq": 1}, "lgbm": {"subsample_freq": 1}}


def suggest(trial: optuna.Trial, space: dict) -> dict:
    p = {}
    for k, spec in space.items():
        kind, lo, hi, *log = spec
        log = bool(log and log[0])
        p[k] = trial.suggest_int(k, lo, hi, log=log) if kind == "int" else trial.suggest_float(k, lo, hi, log=log)
    return p


def thin(df: pd.DataFrame, every: int) -> pd.DataFrame:
    return df[df.obs_idx % every == 0] if every > 1 else df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(SPACES))
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--thin", type=int, default=4)
    ap.add_argument("--max-trips", type=int, default=None, help="dry run only")
    ap.add_argument("--no-refit", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train = load_features("train", a.max_trips)
    sub = thin(train, a.thin).reset_index(drop=True)
    folds = list(GroupKFold(n_splits=min(5, sub.trip_id.nunique())).split(sub, groups=sub.trip_id))
    print(f"search data: {sub.trip_id.nunique()} trips, {len(sub):,} rows (every {a.thin}th observation)")

    def objective(trial):
        params = {**suggest(trial, SPACES[a.model]), **FIXED.get(a.model, {})}
        f1s, tops = [], []
        for k, (tr, va) in enumerate(folds):
            m = get_model(a.model, **params).fit(sub.iloc[tr])
            met = all_metrics(sub.iloc[va].assign(p=m.predict_proba(sub.iloc[va])))
            f1s.append(met["macro_f1"]); tops.append(met["top1"])
            trial.report(float(np.mean(f1s)), k)
            if trial.should_prune():
                trial.set_user_attr("top1", float(np.mean(tops)))
                raise optuna.TrialPruned()
        trial.set_user_attr("top1", float(np.mean(tops)))
        trial.set_user_attr("macro_f1_std", float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0)
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize", study_name=f"{a.model}_macro_f1",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials,
                   callbacks=[lambda s, t: print(f"trial {t.number:3d} {t.state.name:8s} "
                                                 f"value={t.value if t.value is not None else float('nan'):.4f} "
                                                 f"best={s.best_value:.4f} ({time.time() - t0:.0f}s)", flush=True)])
    search_s = time.time() - t0

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials["duration_s"] = trials.pop("duration").dt.total_seconds()
    trials.to_csv(OUT / "trials.csv", index=False)
    best = {**study.best_params, **FIXED.get(a.model, {})}
    (OUT / "best_params.json").write_text(json.dumps({"model": a.model, "best_cv_macro_f1_thinned": study.best_value,
                                                      "best_trial": study.best_trial.number, "params": best}, indent=2))
    (OUT / "search_space.json").write_text(json.dumps({
        "model": a.model, "space": SPACES[a.model], "fixed": FIXED.get(a.model, {}),
        "objective": "mean macro-F1, GroupKFold(5) by trip, train partition",
        "sampler": "TPESampler(seed=42)", "pruner": "MedianPruner(n_startup_trials=8, n_warmup_steps=1)",
        "n_trials": a.trials, "thin_every": a.thin, "search_rows": len(sub),
        "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
        "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
        "search_seconds": round(search_s, 1)}, indent=2))

    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    ax = plot_optimization_history(study); ax.set_title(f"Optuna optimisation history — {a.model} (train CV macro-F1)")
    ax.figure.savefig(OUT / "optimisation_history.png", dpi=150, bbox_inches="tight"); plt.close("all")
    try:
        ax = plot_param_importances(study); ax.set_title(f"Hyperparameter importance (fANOVA) — {a.model}")
        ax.figure.savefig(OUT / "param_importance.png", dpi=150, bbox_inches="tight"); plt.close("all")
    except Exception as exc:  # needs >= 2 completed trials with varied params
        print("param importance skipped:", exc)
    print(f"search done in {search_s:.0f}s; best thinned CV macro-F1 {study.best_value:.4f} with {best}")

    if not a.no_refit:
        test = load_features("student_test", a.max_trips)
        print("refitting tuned model with the P4 procedure ...")
        run_model(a.model, train, test, REPO_ROOT / "results" / "models", save_oof=True, params=best,
                  label=f"{a.model}_tuned")


if __name__ == "__main__":
    main()
