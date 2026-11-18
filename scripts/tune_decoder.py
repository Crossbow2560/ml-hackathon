"""P6 — tune the HMM decoder weights with Optuna (train partition only, out-of-fold scores).

Emissions are the chosen classifier's OUT-OF-FOLD probabilities on train trips
(results/models/<model>_oof.parquet), so the decoder is tuned on scores the classifier did not
see during fitting. For each transition variant, Optuna (TPE, seed 42) searches

    alpha  (emission weight)      log-uniform [0.2, 5]
    weight (transition weight)    exponential: 1/beta with beta log-uniform [1, 100] m
                                  learned:     gamma       log-uniform [0.2, 5]

maximising point-weighted Top-1 over a fixed random subset of train trips. Shortest-path /
transition matrices are computed once per trip and reused by every trial. Each trial decodes
the trips in parallel (worker processes forked after the matrices are built, so they share
them) and scores the Viterbi path only (no forward-backward needed for Top-1). gap_s and v_max are
fixed from the data analysis (configs/default.yaml).

Outputs (results/hmm/): decoder_trials_<variant>.csv, decoder_params.json
Run: python -m scripts.tune_decoder --model lgbm_rank_tuned --trips 30 --trials 40
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd

from pathsnap.config import load_config
from pathsnap.evaluation import build_context, classifier_only, top1, viterbi_top1_correct
from pathsnap.hmm import HMMParams
from pathsnap.io import REPO_ROOT, load_trip
from pathsnap.transitions import ExpTransition

OUT = REPO_ROOT / "results" / "hmm"
CTXS: list = []          # filled before the worker pool is forked; workers read it, never copy it


def _correct(args):
    i, variant, alpha, w = args
    return viterbi_top1_correct(CTXS[i], variant, alpha, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lgbm_rank_tuned")
    ap.add_argument("--trips", type=int, default=30)
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    cfg = load_config()
    params = HMMParams(gap_s=cfg["hmm"]["gap_s"], v_max_mps=cfg["hmm"]["v_max_mps"], radius_m=cfg["candidates"]["radius_m"])
    models = {"exponential": ExpTransition(), "learned": joblib.load(OUT / "transition_learned.joblib")}

    oof = pd.read_parquet(REPO_ROOT / "results" / "models" / f"{a.model}_oof.parquet")
    feats = pd.read_parquet(REPO_ROOT / "data_cache" / "features_train.parquet",
                            columns=["trip_id", "obs_idx", "cand_arc", "proj_t"])
    oof["trip_id"] = oof["trip_id"].astype(str); feats["trip_id"] = feats["trip_id"].astype(str)
    oof = oof.merge(feats, on=["trip_id", "obs_idx", "cand_arc"])
    rng = np.random.default_rng(42)
    trips = sorted(rng.choice(sorted(oof.trip_id.unique()), size=min(a.trips, oof.trip_id.nunique()), replace=False))

    t0 = time.time()
    ctxs = CTXS
    for k, tid in enumerate(trips, 1):
        ctxs.append(build_context(load_trip(tid, "train"), oof[oof.trip_id == tid], models, params))
        print(f"\rcontext {k}/{len(trips)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    prep_s = time.time() - t0
    n_rel = sum(int(c.reliable.sum()) for c in ctxs)
    base_top1 = sum(top1(c, classifier_only(c)) * c.reliable.sum() for c in ctxs) / n_rel
    print(f"classifier-only Top-1 on these {len(ctxs)} trips (OOF): {base_top1:.4f}")

    order = sorted(range(len(ctxs)), key=lambda i: -len(ctxs[i].steps.obs))   # biggest trips first
    pool = mp.get_context("fork").Pool(processes=min(len(ctxs), os.cpu_count() or 1))

    def objective_for(variant):
        def objective(trial):
            alpha = trial.suggest_float("alpha", 0.2, 5.0, log=True)
            if variant == "exponential":
                beta = trial.suggest_float("beta", 1.0, 100.0, log=True); w = 1.0 / beta
            else:
                w = trial.suggest_float("gamma", 0.2, 5.0, log=True)
            correct = sum(pool.imap_unordered(_correct, [(i, variant, alpha, w) for i in order], chunksize=1))
            return correct / n_rel
        return objective

    out = {"model": a.model, "trips": trips, "n_reliable_points": n_rel, "classifier_only_top1": base_top1,
           "gap_s": params.gap_s, "v_max_mps": params.v_max_mps, "radius_m": params.radius_m,
           "precompute_seconds": round(prep_s, 1)}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for variant in models:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        s = time.time()
        study.optimize(objective_for(variant), n_trials=a.trials,
                       callbacks=[lambda st, tr: print(f"  {variant} trial {tr.number:2d} top1={tr.value:.4f} "
                                                       f"best={st.best_value:.4f}", flush=True)])
        df = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
        df["duration_s"] = df.pop("duration").dt.total_seconds()
        df.to_csv(OUT / f"decoder_trials_{variant}.csv", index=False)
        bp = study.best_params
        weight = 1.0 / bp["beta"] if variant == "exponential" else bp["gamma"]
        out[variant] = {**bp, "weight": weight, "best_top1": study.best_value, "search_seconds": round(time.time() - s, 1)}
    pool.close()
    (OUT / "decoder_params.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "trips"}, indent=2))


if __name__ == "__main__":
    main()
