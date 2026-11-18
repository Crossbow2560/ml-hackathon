"""PathSnap inference pipeline — the functions the API calls.

    pred    = match_trip(trip_dir)            # prediction only, never reads .route
    metrics = score_trip(trip_dir, pred)      # evaluation only, the ONLY place .route is read
    response = {**pred, "metrics": metrics}

Splitting prediction from scoring enforces the leakage rule structurally: match_trip loads
the trip with ``load_route=False``, so reference labels are never in memory while
predicting. Ground truth stays server-side; the response never contains the reference route.

match_trip: GPS track + road network -> candidates (R, K) -> 16 features -> classifier
probabilities -> HMM (Viterbi path + forward-backward confidence) -> stitched route.
Artifacts (artifacts/classifier.joblib, transition.joblib, pipeline_config.yaml) are loaded
once. Candidate posteriors are kept in a small server-side cache (never in the response) so
score_trip can compute candidate-level metrics without re-running the model.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import yaml

from pathsnap.candidates import TripGeometry
from pathsnap.features import trip_features
from pathsnap.graph import RoadGraph
from pathsnap.hmm import HMMParams, decode_trip, stitch_route
from pathsnap.io import REPO_ROOT, load_trip_dir
from pathsnap.labels import align_to_route
from pathsnap.metrics import candidate_metrics
from pathsnap.route_metrics import continuity, route_prf
from pathsnap.transitions import ExpTransition

ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", REPO_ROOT / "artifacts"))
PIPELINE_VERSION = "1.0"
_INTERNAL: dict = {}      # trip_dir -> candidate posteriors (server-side only, last few trips)


# ----------------------------------------------------------------------------- schema
class MatchedPoint(TypedDict):
    idx: int                 # row index of the GPS point in the .track file
    arc_id: int | None       # predicted arc id (local to the trip); None = no candidate
    lat: float               # GPS point snapped onto the predicted arc (raw GPS if arc_id is None)
    lon: float
    confidence: float        # posterior probability of the chosen arc, in [0, 1]


class Timing(TypedDict):
    load: float              # milliseconds
    candidates: float
    classify: float
    decode: float
    total: float


class ModelInfo(TypedDict):
    name: str
    version: str


class Prediction(TypedDict):
    trip_id: str
    gps: list[list[float]]              # [[lat, lon], ...] raw GPS, same order as .track
    matched_points: list[MatchedPoint]  # one per GPS point
    route_geometry: list[list[float]]   # [[lat, lon], ...] polyline of the stitched predicted route
    timing_ms: Timing
    model: ModelInfo


class Metrics(TypedDict):
    top1: float                  # observation-level Top-1 accuracy (reliable labels only)
    macro_f1: float              # candidate-level binary macro F1
    route_continuity: float      # share of consecutive predicted arcs that are connected
    route_precision: float       # length-weighted, predicted route vs reference route
    route_recall: float
    route_f1: float
    candidate_miss_rate: float   # share of points whose true arc is not among candidates
    n_points: int


PREDICTION_KEYS = tuple(Prediction.__annotations__)
POINT_KEYS = tuple(MatchedPoint.__annotations__)
TIMING_KEYS = tuple(Timing.__annotations__)
METRIC_KEYS = tuple(Metrics.__annotations__)


# ----------------------------------------------------------------------------- API
@lru_cache(maxsize=1)
def load_artifacts():
    """(classifier, transition model, config) — loaded once per process."""
    cfg = yaml.safe_load((ARTIFACTS / "pipeline_config.yaml").read_text())
    clf = joblib.load(ARTIFACTS / cfg["classifier"]["file"])
    h = cfg["hmm"]
    trans = joblib.load(ARTIFACTS / h["file"]) if h["transition"] == "learned" else ExpTransition(beta=1.0)
    return clf, trans, cfg


def _hmm_params(cfg) -> HMMParams:
    h = cfg["hmm"]
    return HMMParams(alpha=h["alpha"], weight=h["weight"], gap_s=h["gap_s"], v_max_mps=h["v_max_mps"],
                     radius_m=cfg["candidates"]["radius_m"])


def match_trip(trip_dir: str | os.PathLike) -> Prediction:
    """Run the full prediction pipeline on one trip folder (the .route file is never opened)."""
    clf, trans, cfg = load_artifacts()
    t0 = time.perf_counter()
    trip = load_trip_dir(trip_dir, load_route=False, allow_evaluator=True)
    geom = TripGeometry.from_trip(trip)
    graph = RoadGraph(trip.nodes, trip.arcs)
    t1 = time.perf_counter()
    feats = trip_features(trip, geom, cfg["candidates"]["radius_m"], cfg["candidates"]["max_k"])
    t2 = time.perf_counter()
    feats["p"] = clf.predict_proba(feats)
    t3 = time.perf_counter()
    pts, post = decode_trip(geom, graph, feats[["obs_idx", "cand_arc", "proj_t", "p"]], trans, _hmm_params(cfg))
    route = stitch_route(pts.arc.to_numpy(), graph)
    t4 = time.perf_counter()

    # snap each GPS point onto its predicted arc (projection fraction from the candidate table)
    tlook = feats.set_index(["obs_idx", "cand_arc"]).proj_t
    lat, lon = trip.track.lat.to_numpy(), trip.track.lon.to_numpy()
    arc = pts.arc.to_numpy()
    has = arc >= 0
    tt = tlook.reindex(list(zip(np.flatnonzero(has), arc[has]))).to_numpy()
    a = trip.arcs[arc[has]]
    sx = geom.nx[a[:, 0]] + tt * (geom.nx[a[:, 1]] - geom.nx[a[:, 0]])
    sy = geom.ny[a[:, 0]] + tt * (geom.ny[a[:, 1]] - geom.ny[a[:, 0]])
    slon, slat = geom.proj.inverse(sx, sy)
    plat, plon = lat.copy(), lon.copy()
    plat[has], plon[has] = slat, slon
    conf = pts.confidence.to_numpy()
    nodes = trip.nodes
    geometry = []
    if route:
        seq = [int(trip.arcs[route[0], 0])] + [int(trip.arcs[r, 1]) for r in route]
        geometry = [[float(nodes[k, 1]), float(nodes[k, 0])] for k in seq]

    _INTERNAL[str(Path(trip_dir).resolve())] = {"post": feats[["obs_idx", "cand_arc"]].merge(
        post, on=["obs_idx", "cand_arc"], how="left").fillna({"posterior": 0.0}), "route": route}
    while len(_INTERNAL) > 20:
        _INTERNAL.pop(next(iter(_INTERNAL)))

    return {
        "trip_id": trip.trip_id,
        "gps": np.c_[lat, lon].tolist(),
        "matched_points": [{"idx": int(i), "arc_id": int(arc[i]) if arc[i] >= 0 else None, "lat": float(plat[i]),
                            "lon": float(plon[i]), "confidence": float(conf[i])} for i in range(len(arc))],
        "route_geometry": geometry,
        "timing_ms": {"load": (t1 - t0) * 1e3, "candidates": (t2 - t1) * 1e3, "classify": (t3 - t2) * 1e3,
                      "decode": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3},
        "model": {"name": f"{cfg['classifier']['name']} + HMM ({cfg['hmm']['transition']} transitions)",
                  "version": str(cfg["version"])},
    }


def score_trip(trip_dir: str | os.PathLike, prediction: Prediction) -> Metrics:
    """Score a prediction against the server-side reference route (the only reader of .route)."""
    _, _, cfg = load_artifacts()
    trip = load_trip_dir(trip_dir, load_route=True, allow_evaluator=True)
    if trip.route is None:
        raise ValueError(f"no reference route for trip {trip.trip_id}")
    geom = TripGeometry.from_trip(trip)
    graph = RoadGraph(trip.nodes, trip.arcs)
    lab = align_to_route(geom, trip.route, cfg["labels"]["unreliable_m"]).set_index("obs_idx")
    label_arc, reliable = lab.label_arc.to_numpy(), ~lab.unreliable.to_numpy(bool)
    arc = np.array([-1 if p["arc_id"] is None else p["arc_id"] for p in prediction["matched_points"]])

    internal = _INTERNAL.get(str(Path(trip_dir).resolve()))
    if internal is None:                 # scoring without a cached run: recompute predictions
        match_trip(trip_dir)
        internal = _INTERNAL[str(Path(trip_dir).resolve())]
    post = internal["post"]
    post_rel = post[post.obs_idx.isin(np.flatnonzero(reliable))]
    y = (post_rel.cand_arc.to_numpy() == label_arc[post_rel.obs_idx.to_numpy()]).astype(int)
    cm = candidate_metrics(y, post_rel.posterior.to_numpy())
    hit = set(post_rel.obs_idx[y == 1])
    route = internal["route"] or stitch_route(arc, graph)
    rel_idx = np.flatnonzero(reliable)
    return {
        "top1": float(np.mean(arc[reliable] == label_arc[reliable])) if reliable.any() else float("nan"),
        "macro_f1": cm["macro_f1"],
        "route_continuity": continuity(route, trip.arcs),
        **route_prf(route, trip.route, graph.length),
        "candidate_miss_rate": float(1 - np.isin(rel_idx, list(hit)).mean()) if len(rel_idx) else float("nan"),
        "n_points": int(trip.n_points),
    }


# ----------------------------------------------------------------------------- checks
def validate_prediction(pred: dict) -> None:
    """Raise ValueError if ``pred`` does not follow the Prediction contract."""
    missing = [k for k in PREDICTION_KEYS if k not in pred]
    if missing:
        raise ValueError(f"prediction missing keys: {missing}")
    if "metrics" in pred or "route" in pred or "labels" in pred:
        raise ValueError("prediction must not contain metrics / reference route / labels")
    n = len(pred["gps"])
    if len(pred["matched_points"]) != n:
        raise ValueError(f"matched_points has {len(pred['matched_points'])} entries, gps has {n}")
    for p in pred["matched_points"][:5] + pred["matched_points"][-5:]:
        if set(p) != set(POINT_KEYS):
            raise ValueError(f"matched point keys {sorted(p)} != {sorted(POINT_KEYS)}")
        if not 0.0 <= p["confidence"] <= 1.0:
            raise ValueError(f"confidence out of [0, 1]: {p['confidence']}")
    if set(pred["timing_ms"]) != set(TIMING_KEYS):
        raise ValueError(f"timing_ms keys {sorted(pred['timing_ms'])} != {sorted(TIMING_KEYS)}")
    if set(pred["model"]) != {"name", "version"}:
        raise ValueError("model must have exactly 'name' and 'version'")


def validate_metrics(metrics: dict) -> None:
    """Raise ValueError if ``metrics`` does not follow the Metrics contract."""
    if set(metrics) != set(METRIC_KEYS):
        raise ValueError(f"metrics keys {sorted(metrics)} != {sorted(METRIC_KEYS)}")
