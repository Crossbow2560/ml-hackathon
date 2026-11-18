"""HMM route reconstruction: Viterbi decoding + forward-backward confidence.

Hidden states  : at each GPS observation, its candidate arcs (<= K = 30).
Emissions      : log E_t(j) = alpha * log p_t(j), p = classifier probability renormalised
                 over the observation's candidates.
Transitions    : log T(i, j) = w * base(i, j)
                   exponential baseline: base = -|d_route - d_gc|   (w = 1 / beta)
                   learned model       : base = log P(true pair)   (w = gamma)
                 unreachable pairs are -inf.
Viterbi        : delta_t(j) = log E_t(j) + max_i [delta_{t-1}(i) + log T(i, j)]
Gaps / breaks  : if Δt > gap_s, or no candidate at t is reachable from any state at t-1,
                 the chain is split and decoding restarts (delta_t = log E_t).
Confidence     : forward-backward posterior P(state_t = j | the segment's observations);
                 a point's confidence is the posterior of its Viterbi arc.
Observations without candidates are returned with arc = -1, confidence = 0.

The expensive part (shortest paths for every candidate pair) does not depend on alpha / w,
so ``compute_transitions`` runs once per trip and ``run_hmm`` can be re-run cheaply when tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pathsnap.transitions import ExpTransition, pair_features


@dataclass
class HMMParams:
    alpha: float = 1.0        # emission weight
    weight: float = 1.0       # transition weight (1/beta for the exponential model, gamma for learned)
    gap_s: float = 60.0       # split chain if consecutive observations are further apart in time
    v_max_mps: float = 50.0   # bounds the shortest-path search: limit = v_max * Δt + 2R
    radius_m: float = 50.0


def arc_bearings(geom) -> np.ndarray:
    a = geom.arcs
    return (np.degrees(np.arctan2(geom.nx[a[:, 1]] - geom.nx[a[:, 0]], geom.ny[a[:, 1]] - geom.ny[a[:, 0]])) + 360) % 360


@dataclass
class Steps:
    obs: np.ndarray          # observation indices that have candidates, in time order
    arcs: list               # candidate arcs per step
    ts: list                 # projection fraction per candidate
    ps: list                 # classifier score per candidate


def build_steps(cand: pd.DataFrame, score_col: str = "p") -> Steps:
    cand = cand.sort_values(["obs_idx", "cand_arc"], kind="stable")
    obs, arcs, ts, ps = [], [], [], []
    for o, g in cand.groupby("obs_idx", sort=True):
        obs.append(int(o)); arcs.append(g.cand_arc.to_numpy()); ts.append(g.proj_t.to_numpy(float))
        ps.append(g[score_col].to_numpy(float) if score_col in g else np.ones(len(g)))
    return Steps(np.array(obs, dtype=int), arcs, ts, ps)


def compute_transitions(geom, graph, steps: Steps, models: dict, params: HMMParams):
    """Base log-transition matrices for every step and every model in ``models``.

    Returns (gap_break, base) where gap_break[t] is True if step t starts a new segment
    because of a time gap, and base[name][t] is the |A|x|B| base matrix (None at breaks).
    """
    T = len(steps.obs)
    bear = arc_bearings(geom)
    gap_break = np.zeros(T, dtype=bool)
    gap_break[0] = True
    base = {name: [None] * T for name in models}
    for t in range(1, T):
        o0, o1 = steps.obs[t - 1], steps.obs[t]
        dt = geom.t[o1] - geom.t[o0]
        if dt > params.gap_s:
            gap_break[t] = True
            continue
        d_gc = float(np.hypot(geom.px[o1] - geom.px[o0], geom.py[o1] - geom.py[o0]))
        limit = params.v_max_mps * max(dt, 1.0) + 2 * params.radius_m
        pf = pair_features(graph, bear, steps.arcs[t - 1], steps.ts[t - 1], steps.arcs[t], steps.ts[t], d_gc, dt, limit)
        for name, m in models.items():
            base[name][t] = ExpTransition(beta=1.0).log_prob(pf) if isinstance(m, ExpTransition) else m.log_prob(pf)
    return gap_break, base


def run_hmm(steps: Steps, gap_break: np.ndarray, base: list, alpha: float, weight: float, posteriors: bool = True):
    """Viterbi path + forward-backward posteriors. Returns (path, posteriors, segment_id).

    posteriors=False skips forward-backward (posteriors returned as None) — used when tuning,
    where only the Viterbi path is scored.
    """
    T = len(steps.obs)
    logE = []
    for p in steps.ps:
        q = np.clip(p, 1e-12, None)
        logE.append(alpha * np.log(q / q.sum()))
    brk = gap_break.copy()
    logT = [None] * T
    for t in range(1, T):
        if brk[t] or base[t] is None:
            brk[t] = True
            continue
        lt = weight * base[t]
        if not np.isfinite(lt).any():
            brk[t] = True
        else:
            logT[t] = lt

    delta, back = [None] * T, [None] * T
    for t in range(T):
        if brk[t]:
            delta[t] = logE[t].copy()
            continue
        s = delta[t - 1][:, None] + logT[t]
        back[t] = np.argmax(s, axis=0)
        delta[t] = s[back[t], np.arange(s.shape[1])] + logE[t]
        if not np.isfinite(delta[t]).any():          # no surviving path -> restart here
            brk[t] = True
            delta[t] = logE[t].copy()
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(np.argmax(delta[t - 1])) if brk[t] else int(back[t][path[t]])
    if not posteriors:
        return path, None, np.cumsum(brk) - 1

    fwd = [None] * T
    for t in range(T):
        f = logE[t].copy() if brk[t] else logE[t] + logsumexp(fwd[t - 1][:, None] + logT[t], axis=0)
        fwd[t] = f - logsumexp(f)
    bwd = [None] * T
    for t in range(T - 1, -1, -1):
        if t == T - 1 or brk[t + 1]:
            bwd[t] = np.zeros(len(steps.arcs[t]))
            continue
        b = logsumexp(logT[t + 1] + (logE[t + 1] + bwd[t + 1])[None, :], axis=1)
        bwd[t] = b - logsumexp(b) if np.isfinite(b).any() else np.zeros(len(steps.arcs[t]))
    post = []
    for t in range(T):
        lp = fwd[t] + bwd[t]
        post.append(np.exp(lp - logsumexp(lp)) if np.isfinite(lp).any() else np.full(len(lp), 1 / len(lp)))
    return path, post, np.cumsum(brk) - 1


def to_frames(steps: Steps, n_points: int, path, post, segment):
    """(points, posteriors) DataFrames from run_hmm output."""
    T = len(steps.obs)
    pts = pd.DataFrame({"obs_idx": np.arange(n_points), "arc": -1, "confidence": 0.0, "segment": -1})
    if T:
        pts.loc[steps.obs, "arc"] = [int(steps.arcs[t][path[t]]) for t in range(T)]
        pts.loc[steps.obs, "confidence"] = [float(post[t][path[t]]) for t in range(T)]
        pts.loc[steps.obs, "segment"] = segment
    posteriors = pd.DataFrame({"obs_idx": np.repeat(steps.obs, [len(a) for a in steps.arcs]),
                               "cand_arc": np.concatenate(steps.arcs) if T else [],
                               "posterior": np.concatenate(post) if T else []})
    return pts, posteriors


def decode_trip(geom, graph, cand: pd.DataFrame, model, params: HMMParams, score_col: str = "p"):
    """One-call decode (used by the API pipeline). Returns (points, posteriors)."""
    steps = build_steps(cand, score_col)
    gap_break, base = compute_transitions(geom, graph, steps, {"m": model}, params)
    path, post, seg = run_hmm(steps, gap_break, base["m"], params.alpha, params.weight)
    return to_frames(steps, len(geom.px), path, post, seg)


def stitch_route(arc_seq, graph, limit: float = 5000.0) -> list[int]:
    """Collapse repeats and join non-adjacent consecutive arcs with the shortest path."""
    out: list[int] = []
    for a in (int(x) for x in arc_seq if x is not None and x >= 0):
        if out and out[-1] == a:
            continue
        if out and graph.arcs[out[-1], 1] != graph.arcs[a, 0]:
            gap = graph.shortest_arc_path(out[-1], a, limit)
            if gap:
                out.extend(gap)
        out.append(a)
    return out
