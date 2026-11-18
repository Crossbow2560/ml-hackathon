"""PathSnap inference pipeline — public contract (STUB, filled in during P2–P6).

The API (api/main.py) calls exactly two functions, in this order:

    pred    = match_trip(trip_dir)            # prediction only, never reads .route
    metrics = score_trip(trip_dir, pred)      # evaluation only, the ONLY place .route is read
    response = {**pred, "metrics": metrics}

Splitting prediction from scoring enforces the leakage rule structurally: match_trip loads
the trip with ``load_route=False``, so reference labels are never in memory while
predicting. Ground truth stays server-side; the response never contains the reference route.

Until the real pipeline exists both functions raise NotImplementedError; the API uses
api/mock_pipeline.py, which must return the same shapes (check with ``validate_prediction``
and ``validate_metrics``).
"""
from __future__ import annotations

import os
from typing import TypedDict

PIPELINE_VERSION = "0.0-stub"


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
def match_trip(trip_dir: str | os.PathLike) -> Prediction:
    """Run the full prediction pipeline on one trip folder.

    Steps (implemented in later phases):
      1. load_trip_dir(trip_dir, load_route=False, allow_evaluator=True)   (P0, done)
      2. project to metres, generate candidate arcs per GPS point        (P2)
      3. compute features, score candidates with the trained classifier  (P3/P4)
      4. HMM: Viterbi path + forward-backward confidence                 (P6)
      5. stitch the route geometry, fill the response
    """
    raise NotImplementedError("pipeline not implemented yet — use api/mock_pipeline.py")


def score_trip(trip_dir: str | os.PathLike, prediction: Prediction) -> Metrics:
    """Score a prediction against the server-side reference route (reads .route)."""
    raise NotImplementedError("scoring not implemented yet — use api/mock_pipeline.py")


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
