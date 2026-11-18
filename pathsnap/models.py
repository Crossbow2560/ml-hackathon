"""Model zoo for candidate scoring. Every model has the same interface:

    model = get_model(name, **params)
    model.fit(train_df)                 # rows sorted by (trip_id, obs_idx); uses FEATURES + label
    p = model.predict_proba(df)         # probability that each candidate row is the true arc

Families: linear (logreg), probabilistic (gnb), instance-based (knn), bagging ensemble (rf),
boosting (lgbm), neural network (mlp), learning-to-rank (lgbm_rank). All use the same 16
features from pathsnap.features and seed 42. No class weighting: the HMM needs honest
probabilities, and weighting would distort them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.features import FEATURES

SEED = 42
EPS = 1e-7

FAMILIES = {"logreg": "Linear", "gnb": "Probabilistic", "knn": "Instance-based", "rf": "Ensemble (bagging)",
            "lgbm": "Boosting", "mlp": "Neural network", "lgbm_rank": "Learning-to-rank (boosting)"}


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    """Rows per observation, in order of appearance (df must be sorted by trip_id, obs_idx)."""
    key = df["trip_id"].astype(str) + ":" + df["obs_idx"].astype(str)
    change = np.r_[True, key.to_numpy()[1:] != key.to_numpy()[:-1]]
    starts = np.flatnonzero(change)
    return np.diff(np.r_[starts, len(df)])


def group_softmax(scores: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Softmax of scores within each observation (turns ranking scores into probabilities)."""
    gid = np.repeat(np.arange(len(sizes)), sizes)
    mx = np.maximum.reduceat(scores, np.r_[0, np.cumsum(sizes)[:-1]])
    e = np.exp(scores - mx[gid])
    s = np.add.reduceat(e, np.r_[0, np.cumsum(sizes)[:-1]])
    return e / s[gid]


class SklearnModel:
    """Wraps any sklearn/LightGBM binary classifier."""

    def __init__(self, name: str, estimator, max_train_rows: int | None = None):
        self.name, self.estimator, self.max_train_rows = name, estimator, max_train_rows

    def fit(self, df: pd.DataFrame):
        d = df
        if self.max_train_rows and len(df) > self.max_train_rows:
            # stratified subsample (keeps the positive rate) — used for KNN only
            frac = self.max_train_rows / len(df)
            d = df.groupby("label", group_keys=False).sample(frac=frac, random_state=SEED)
        self.estimator.fit(d[FEATURES].to_numpy(), d["label"].to_numpy())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(df[FEATURES].to_numpy())[:, 1]

    def config(self) -> str:
        est = self.estimator.steps[-1][1] if hasattr(self.estimator, "steps") else self.estimator
        keys = {"logreg": ["C", "penalty", "max_iter"], "gnb": ["var_smoothing"],
                "knn": ["n_neighbors", "weights"], "rf": ["n_estimators", "min_samples_leaf", "max_features"],
                "lgbm": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"]}.get(self.name, [])
        cfg = ", ".join(f"{k}={est.get_params()[k]}" for k in keys)
        return cfg + (f", train subsample={self.max_train_rows:,}" if self.max_train_rows else "")


class RankerModel:
    """LightGBM LambdaRank: one query group per GPS observation; optimises ranking (≈ Top-1)."""

    def __init__(self, **params):
        p = dict(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=63,
                 min_child_samples=50, random_state=SEED, n_jobs=-1, verbose=-1)
        p.update(params)
        self.estimator = LGBMRanker(**p)

    def fit(self, df: pd.DataFrame):
        self.estimator.fit(df[FEATURES].to_numpy(), df["label"].to_numpy(), group=group_sizes(df))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return group_softmax(self.estimator.predict(df[FEATURES].to_numpy()), group_sizes(df))

    def config(self) -> str:
        g = self.estimator.get_params()
        return (f"lambdarank, n_estimators={g['n_estimators']}, learning_rate={g['learning_rate']}, "
                f"num_leaves={g['num_leaves']}, per-observation softmax")


class MLPModel:
    """MLP trained epoch by epoch so real train/validation log-loss curves can be recorded."""

    def __init__(self, hidden=(64, 32), max_epochs=50, patience=5, batch_size=2048, learning_rate_init=1e-3,
                 alpha=1e-4, val_frac=0.1):
        self.hidden, self.max_epochs, self.patience = hidden, max_epochs, patience
        self.batch_size, self.lr, self.alpha, self.val_frac = batch_size, learning_rate_init, alpha, val_frac
        self.history: list[dict] = []

    def fit(self, df: pd.DataFrame):
        rng = np.random.default_rng(SEED)
        trips = df["trip_id"].unique()
        val_trips = set(rng.choice(trips, size=max(1, int(round(len(trips) * self.val_frac))), replace=False))
        is_val = df["trip_id"].isin(val_trips).to_numpy()
        Xtr, ytr = df.loc[~is_val, FEATURES].to_numpy(), df.loc[~is_val, "label"].to_numpy()
        Xva, yva = df.loc[is_val, FEATURES].to_numpy(), df.loc[is_val, "label"].to_numpy()
        self.scaler = StandardScaler().fit(Xtr)
        Xtr, Xva = self.scaler.transform(Xtr), self.scaler.transform(Xva)
        self.net = MLPClassifier(hidden_layer_sizes=self.hidden, batch_size=self.batch_size,
                                 learning_rate_init=self.lr, alpha=self.alpha, random_state=SEED)
        sub = rng.choice(len(Xtr), size=min(len(Xtr), 200_000), replace=False)   # train-loss estimate
        best, best_state, bad = np.inf, None, 0
        self.history = []
        for epoch in range(1, self.max_epochs + 1):
            perm = rng.permutation(len(Xtr))
            for s in range(0, len(perm), self.batch_size * 50):          # partial_fit over the epoch
                b = perm[s:s + self.batch_size * 50]
                self.net.partial_fit(Xtr[b], ytr[b], classes=[0, 1])
            tr = log_loss(ytr[sub], np.clip(self.net.predict_proba(Xtr[sub])[:, 1], EPS, 1 - EPS), labels=[0, 1])
            va = log_loss(yva, np.clip(self.net.predict_proba(Xva)[:, 1], EPS, 1 - EPS), labels=[0, 1])
            self.history.append({"epoch": epoch, "train_log_loss": tr, "val_log_loss": va})
            if va < best - 1e-5:
                best, bad = va, 0
                best_state = ([w.copy() for w in self.net.coefs_], [b.copy() for b in self.net.intercepts_])
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.coefs_, self.net.intercepts_ = best_state         # restore the best epoch
        self.best_epoch = int(np.argmin([h["val_log_loss"] for h in self.history]) + 1)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.net.predict_proba(self.scaler.transform(df[FEATURES].to_numpy()))[:, 1]

    def config(self) -> str:
        return (f"hidden={self.hidden}, adam lr={self.lr}, alpha={self.alpha}, batch={self.batch_size}, "
                f"early stop patience={self.patience} on {int(self.val_frac * 100)}% held-out trips")


def get_model(name: str, **params):
    if name == "logreg":
        p = dict(C=1.0, max_iter=1000); p.update(params)
        return SklearnModel(name, make_pipeline(StandardScaler(), LogisticRegression(**p)))
    if name == "gnb":
        return SklearnModel(name, GaussianNB(**params))
    if name == "knn":
        p = dict(n_neighbors=15, weights="distance", n_jobs=-1); p.update(params)
        return SklearnModel(name, make_pipeline(StandardScaler(), KNeighborsClassifier(**p)), max_train_rows=200_000)
    if name == "rf":
        p = dict(n_estimators=300, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED)
        p.update(params)
        return SklearnModel(name, RandomForestClassifier(**p))
    if name == "lgbm":
        p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                 random_state=SEED, n_jobs=-1, verbose=-1)
        p.update(params)
        return SklearnModel(name, LGBMClassifier(**p))
    if name == "mlp":
        return MLPModel(**params)
    if name == "lgbm_rank":
        return RankerModel(**params)
    raise ValueError(f"unknown model {name!r}; choose from {list(FAMILIES)}")
