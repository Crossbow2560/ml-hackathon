"""HMM transition models between candidate arcs of consecutive GPS observations.

For a step (observation t-1 -> t) with candidates A (arcs, projection t_a) and B (arcs, t_b),
``pair_features`` builds an |A| x |B| table of geometric features:

    d_route     shortest driving distance from (a, t_a) to (b, t_b)          [m, inf = unreachable]
    d_gc        straight-line distance between the two GPS fixes              [m]
    abs_diff_m  |d_route - d_gc|   (Newson & Krumm: true pairs have d_route ≈ d_gc)
    log_ratio   log((d_route + 1) / (d_gc + 1))
    route_speed_mps  d_route / Δt  (implausibly fast = wrong pair)
    dt_s, same_arc, u_turn (b is the reverse of a), turn_angle_deg (between arc bearings)

Two models turn these into log transition scores (unreachable pairs always get -inf):

* ExpTransition     — textbook baseline:  log T = -abs_diff / beta
* LearnedTransition — our model: logistic regression trained on train trips where the positive
                      class is the true (label_{t-1}, label_t) pair; log T = log P(true pair).
                      It uses geometry only (no classifier scores), so it cannot leak labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.geo import angle_diff_deg

TRANS_FEATURES = ["abs_diff_m", "log_ratio", "route_speed_mps", "dt_s", "same_arc", "u_turn", "turn_angle_deg",
                  "log_d_route"]


def pair_features(graph, bearings, arcs_a, t_a, arcs_b, t_b, d_gc: float, dt: float, limit: float) -> dict:
    """|A| x |B| feature matrices for one step (see module docstring)."""
    arcs_a, arcs_b = np.asarray(arcs_a), np.asarray(arcs_b)
    d_route = graph.arc_distance_matrix(arcs_a, np.asarray(t_a, float), arcs_b, np.asarray(t_b, float), limit)
    reach = np.isfinite(d_route)
    dr = np.where(reach, d_route, limit)
    A, B = graph.arcs[arcs_a], graph.arcs[arcs_b]
    dt = max(float(dt), 1e-3)
    shape = d_route.shape
    return {
        "reachable": reach,
        "d_route": d_route,
        "abs_diff_m": np.abs(dr - d_gc),
        "log_ratio": np.log((dr + 1.0) / (d_gc + 1.0)),
        "route_speed_mps": dr / dt,
        "dt_s": np.full(shape, dt),
        "same_arc": (arcs_a[:, None] == arcs_b[None, :]).astype(float),
        "u_turn": ((A[:, 0][:, None] == B[:, 1][None, :]) & (A[:, 1][:, None] == B[:, 0][None, :])).astype(float),
        "turn_angle_deg": angle_diff_deg(bearings[arcs_a][:, None], bearings[arcs_b][None, :]),
        "log_d_route": np.log1p(dr),
    }


def feature_matrix(pf: dict) -> np.ndarray:
    """Flatten the pair-feature dict to (|A|*|B|, n_features) in TRANS_FEATURES order."""
    return np.stack([pf[k].ravel() for k in TRANS_FEATURES], axis=1)


class ExpTransition:
    name = "exponential"

    def __init__(self, beta: float = 5.0):
        self.beta = beta

    def log_prob(self, pf: dict) -> np.ndarray:
        return np.where(pf["reachable"], -pf["abs_diff_m"] / self.beta, -np.inf)


class LearnedTransition:
    name = "learned"

    def __init__(self, C: float = 1.0):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def log_prob(self, pf: dict) -> np.ndarray:
        p = self.model.predict_proba(feature_matrix(pf))[:, 1].reshape(pf["reachable"].shape)
        return np.where(pf["reachable"], np.log(np.clip(p, 1e-12, 1.0)), -np.inf)

    def coefficients(self) -> dict:
        lr = self.model.steps[-1][1]
        return dict(zip(TRANS_FEATURES, lr.coef_[0].round(4).tolist())) | {"intercept": float(lr.intercept_[0])}
