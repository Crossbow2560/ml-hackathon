"""Route-level metrics (continuity, length-weighted precision / recall / F1).

These judge the *route* a method produces, complementing the per-point metrics in
pathsnap/metrics.py:

* continuity  — after collapsing repeats, the share of consecutive predicted arcs where
                arc i ends at the node where arc i+1 starts. Independent classifier
                predictions often jump between unconnected arcs; an HMM should fix that.
* route P/R/F1 — length-weighted overlap between the predicted and the reference arc sets.
                 Direction matters: arc u->v is not arc v->u.
"""
from __future__ import annotations

import numpy as np

from pathsnap.geo import haversine_m


def arc_lengths(nodes: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    """Length of every arc in metres (great-circle between its two nodes)."""
    a, b = arcs[:, 0], arcs[:, 1]
    return haversine_m(nodes[a, 0], nodes[a, 1], nodes[b, 0], nodes[b, 1])


def collapse(arc_seq) -> np.ndarray:
    """Drop missing predictions (None / negative) and consecutive duplicates."""
    seq = np.asarray([a for a in arc_seq if a is not None and a >= 0], dtype=np.int64)
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def continuity(arc_seq, arcs: np.ndarray) -> float:
    """Share of consecutive (distinct) arcs that are connected head-to-tail. NaN if < 2 arcs."""
    seq = collapse(arc_seq)
    if len(seq) < 2:
        return float("nan")
    a = arcs[seq]
    return float(np.mean(a[:-1, 1] == a[1:, 0]))


def route_prf(pred_seq, ref_route, lengths: np.ndarray) -> dict:
    """Length-weighted precision / recall / F1 of the predicted arc set vs the reference set."""
    pred = np.unique(collapse(pred_seq))
    ref = np.unique(np.asarray(ref_route, dtype=np.int64))
    inter = np.intersect1d(pred, ref)
    lp, lr, li = lengths[pred].sum(), lengths[ref].sum(), lengths[inter].sum()
    p = li / lp if lp > 0 else 0.0
    r = li / lr if lr > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"route_precision": float(p), "route_recall": float(r), "route_f1": float(f)}
