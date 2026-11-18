"""Candidate- and observation-level metrics.

Rows are (GPS observation, candidate arc) pairs with a binary ``label``.

* candidate_metrics  — binary classification quality over candidate rows: accuracy,
                       precision, recall, macro-F1 (threshold 0.5), log loss, confusion matrix.
* top1_accuracy      — per GPS observation, is the highest-scoring candidate the label arc?
                       (the main map-matching metric; observations whose label arc is not
                       among the candidates count as wrong)
* candidate_miss_rate — share of observations whose label arc is not among the candidates.

Unreliable observations (label > 50 m away, see pathsnap/labels.py) are removed first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score)

OBS_KEYS = ["trip_id", "obs_idx"]
EPS = 1e-7


def reliable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows of unreliable observations (if the column exists)."""
    return df[~df["unreliable"]] if "unreliable" in df else df


def candidate_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def argmax_rows(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """The highest-scoring candidate row of every observation."""
    idx = df.groupby(OBS_KEYS, sort=False)[score_col].idxmax()
    return df.loc[idx.to_numpy()]


def top1_accuracy(df: pd.DataFrame, score_col: str) -> float:
    return float(argmax_rows(df, score_col)["label"].mean())


def candidate_miss_rate(df: pd.DataFrame) -> float:
    return float(1 - df.groupby(OBS_KEYS, sort=False)["label"].max().mean())


def all_metrics(df: pd.DataFrame, score_col: str = "p") -> dict:
    """Candidate metrics + Top-1 + miss rate on reliable rows."""
    d = reliable(df)
    m = candidate_metrics(d["label"], d[score_col])
    m["top1"] = top1_accuracy(d, score_col)
    m["candidate_miss_rate"] = candidate_miss_rate(d)
    m["n_obs"] = int(d.groupby(OBS_KEYS, sort=False).ngroups)
    return m
