"""Per-trip evaluation of classifier-only vs classifier + HMM predictions.

``TripContext`` holds everything computed once per trip (geometry, graph, labels, candidate
scores, HMM steps and base transition matrices). ``evaluate`` then scores any decoder setting
cheaply — used by scripts/tune_decoder.py and scripts/evaluate_hmm.py.
Labels come from pathsnap.labels (monotonic alignment); unreliable points are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathsnap.candidates import TripGeometry
from pathsnap.graph import RoadGraph
from pathsnap.hmm import HMMParams, Steps, build_steps, compute_transitions, run_hmm, stitch_route, to_frames
from pathsnap.labels import align_to_route
from pathsnap.metrics import candidate_metrics
from pathsnap.route_metrics import continuity, route_prf


@dataclass
class TripContext:
    trip: object
    geom: TripGeometry
    graph: RoadGraph
    cand: pd.DataFrame          # obs_idx, cand_arc, proj_t, label, p  (sorted by obs_idx, cand_arc)
    label_arc: np.ndarray       # per point (nan if none)
    reliable: np.ndarray        # per point bool
    steps: Steps
    gap_break: np.ndarray
    base: dict                  # transition model name -> list of base matrices


def build_context(trip, cand: pd.DataFrame, models: dict, params: HMMParams, unreliable_m: float = 50.0) -> TripContext:
    geom = TripGeometry.from_trip(trip)
    graph = RoadGraph(trip.nodes, trip.arcs)
    lab = align_to_route(geom, trip.route, unreliable_m).set_index("obs_idx")
    cand = cand.sort_values(["obs_idx", "cand_arc"], kind="stable").reset_index(drop=True)
    steps = build_steps(cand, "p")
    gap_break, base = compute_transitions(geom, graph, steps, models, params)
    return TripContext(trip, geom, graph, cand, lab.label_arc.to_numpy(float), ~lab.unreliable.to_numpy(bool),
                       steps, gap_break, base)


def classifier_only(ctx: TripContext) -> np.ndarray:
    c = ctx.cand
    best = c.loc[c.groupby("obs_idx").p.idxmax(), ["obs_idx", "cand_arc"]]
    arc = np.full(ctx.trip.n_points, -1)
    arc[best.obs_idx.to_numpy()] = best.cand_arc.to_numpy()
    return arc


def decode(ctx: TripContext, model: str, alpha: float, weight: float):
    path, post, seg = run_hmm(ctx.steps, ctx.gap_break, ctx.base[model], alpha, weight)
    return to_frames(ctx.steps, ctx.trip.n_points, path, post, seg)


def viterbi_top1_correct(ctx: TripContext, model: str, alpha: float, weight: float) -> int:
    """Number of reliable points whose Viterbi arc equals the label (fast path for tuning)."""
    path, _, _ = run_hmm(ctx.steps, ctx.gap_break, ctx.base[model], alpha, weight, posteriors=False)
    arc = np.full(ctx.trip.n_points, -1)
    arc[ctx.steps.obs] = [ctx.steps.arcs[t][path[t]] for t in range(len(path))]
    r = ctx.reliable
    return int(np.sum(arc[r] == ctx.label_arc[r]))


def top1(ctx: TripContext, arc: np.ndarray) -> float:
    r = ctx.reliable
    return float(np.mean(arc[r] == ctx.label_arc[r])) if r.any() else float("nan")


def score(ctx: TripContext, method: str, arc: np.ndarray, cand_prob: np.ndarray, with_route: bool = True) -> dict:
    """All metrics for one method on one trip. cand_prob aligns with ctx.cand rows."""
    rel_rows = ctx.cand.obs_idx.isin(np.flatnonzero(ctx.reliable)).to_numpy()
    cm = candidate_metrics(ctx.cand.label.to_numpy()[rel_rows], cand_prob[rel_rows])
    row = {"trip_id": ctx.trip.trip_id, "method": method, "n_points": ctx.trip.n_points,
           "n_reliable": int(ctx.reliable.sum()), "top1": top1(ctx, arc), "macro_f1": cm["macro_f1"],
           "log_loss": cm["log_loss"], "precision": cm["precision"], "recall": cm["recall"],
           "point_continuity": continuity(arc, ctx.trip.arcs)}
    if with_route:
        st = stitch_route(arc, ctx.graph)
        row["route_continuity"] = continuity(st, ctx.trip.arcs)
        row.update(route_prf(st, ctx.trip.route, ctx.graph.length))
    return row


def evaluate(ctx: TripContext, settings: dict) -> tuple[list[dict], dict]:
    """settings: {model_name: (alpha, weight)}. Returns (metric rows, predictions per method)."""
    rows, preds = [], {}
    arc_c = classifier_only(ctx)
    rows.append(score(ctx, "classifier_only", arc_c, ctx.cand.p.to_numpy()))
    preds["classifier_only"] = pd.DataFrame({"obs_idx": np.arange(ctx.trip.n_points), "arc": arc_c})
    for name, (alpha, weight) in settings.items():
        pts, post = decode(ctx, name, alpha, weight)
        p = ctx.cand[["obs_idx", "cand_arc"]].merge(post, on=["obs_idx", "cand_arc"], how="left").posterior
        rows.append(score(ctx, f"hmm_{name}", pts.arc.to_numpy(), p.fillna(0).to_numpy()))
        preds[f"hmm_{name}"] = pts
    ref = np.where(np.isnan(ctx.label_arc), -1, ctx.label_arc).astype(int)
    ref_r = ref[ctx.reliable]
    st = stitch_route(ref_r, ctx.graph)
    rows.append({"trip_id": ctx.trip.trip_id, "method": "reference_labels", "n_points": ctx.trip.n_points,
                 "n_reliable": int(ctx.reliable.sum()), "top1": 1.0, "point_continuity": continuity(ref_r, ctx.trip.arcs),
                 "route_continuity": continuity(st, ctx.trip.arcs), **route_prf(st, ctx.trip.route, ctx.graph.length)})
    return rows, preds
