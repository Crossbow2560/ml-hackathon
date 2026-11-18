"""Directed road graph with bounded shortest paths (used by the HMM and route stitching).

Nodes are the trip's .nodes rows, edges its .arcs rows, weighted by arc length in metres.
Shortest paths use scipy's Dijkstra with a distance cutoff (``limit``): between two GPS
fixes 1 s apart a car cannot drive kilometres, so searching the whole 300k-arc network is
wasted work. Results are cached per (source node, cutoff).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from pathsnap.route_metrics import arc_lengths


class RoadGraph:
    def __init__(self, nodes: np.ndarray, arcs: np.ndarray):
        self.nodes, self.arcs = nodes, arcs
        self.n_nodes = len(nodes)
        self.length = arc_lengths(nodes, arcs)
        # duplicate arcs: keep the shortest one per (u, v) for the matrix
        order = np.lexsort((self.length, arcs[:, 1], arcs[:, 0]))
        a = arcs[order]
        keep = np.r_[True, (a[1:] != a[:-1]).any(axis=1)]
        a, w, ids = a[keep], self.length[order][keep], order[keep]
        # zero weights would vanish in a sparse matrix; use a tiny epsilon instead
        self.matrix = csr_matrix((np.maximum(w, 1e-3), (a[:, 0], a[:, 1])), shape=(self.n_nodes,) * 2)
        self._edge_arc = {(int(u), int(v)): int(i) for (u, v), i in zip(a, ids)}
        self._reach_cache: dict = {}
        n = self.n_nodes
        key = arcs[:, 0] * n + arcs[:, 1]
        self.reverse_exists = np.isin(arcs[:, 1] * n + arcs[:, 0], key)

    # ------------------------------------------------------------------ fast many-to-many lookups
    def reach(self, src: int, limit: float):
        """Nodes reachable from src within ``limit`` metres: (sorted node ids, distances).

        Only reached nodes are stored, so thousands of cached sources stay small in memory.
        """
        key = (int(src), float(limit))
        hit = self._reach_cache.get(key)
        if hit is None:
            dist = dijkstra(self.matrix, directed=True, indices=int(src), limit=limit)
            idx = np.flatnonzero(np.isfinite(dist))
            hit = (idx, dist[idx])
            if len(self._reach_cache) > 50_000:
                self._reach_cache.clear()
            self._reach_cache[key] = hit
        return hit

    def node_dist_many(self, src: int, targets: np.ndarray, limit: float) -> np.ndarray:
        """Driving distance src -> each target node (inf if not reachable within limit)."""
        idx, d = self.reach(src, limit)
        pos = np.searchsorted(idx, targets)
        pos = np.clip(pos, 0, max(len(idx) - 1, 0))
        out = np.full(len(targets), np.inf)
        if len(idx):
            ok = idx[pos] == targets
            out[ok] = d[pos[ok]]
        return out

    def arc_distance_matrix(self, arcs_a, t_a, arcs_b, t_b, limit: float, back_tol_m: float = 20.0) -> np.ndarray:
        """Driving distances from every (arc_a, t_a) to every (arc_b, t_b): shape (len(a), len(b)).

        Same arc moving forward -> distance along the arc; otherwise rest of arc a + node path
        + first part of arc b. inf where unreachable within limit.

        GPS jitter: a slow or stopped vehicle's fixes often step slightly BACKWARDS. Without a
        tolerance that looks like a loop around the block (usually unreachable), so the true
        pair would get -inf. Backward moves up to ``back_tol_m`` along the same arc, or back
        across the from-node onto the preceding arc in the same direction, are allowed at a
        cost equal to the backward distance. Switching to the opposite-direction twin arc is
        NOT covered by this rule.
        """
        arcs_a, arcs_b = np.asarray(arcs_a), np.asarray(arcs_b)
        la, lb = self.length[arcs_a], self.length[arcs_b]
        to_a, from_b = self.arcs[arcs_a, 1], self.arcs[arcs_b, 0]
        out = np.empty((len(arcs_a), len(arcs_b)))
        for i, u in enumerate(to_a):
            out[i] = (1 - t_a[i]) * la[i] + self.node_dist_many(u, from_b, limit) + t_b * lb
        same = arcs_a[:, None] == arcs_b[None, :]
        fwd = same & (t_b[None, :] >= t_a[:, None])
        out[fwd] = ((t_b[None, :] - t_a[:, None]) * la[:, None])[fwd]
        if back_tol_m > 0:
            # backward along the same arc
            back_same = (t_a[:, None] - t_b[None, :]) * la[:, None]
            m = same & ~fwd & (back_same <= back_tol_m)
            out[m] = back_same[m]
            # backward across a's from-node onto b, where b -> a (b is the arc just before a)
            from_a, to_b = self.arcs[arcs_a, 0], self.arcs[arcs_b, 1]
            twin = (from_a[:, None] == to_b[None, :]) & (to_a[:, None] == from_b[None, :])
            pred = (to_b[None, :] == from_a[:, None]) & ~twin & ~same
            back_pred = t_a[:, None] * la[:, None] + (1 - t_b[None, :]) * lb[None, :]
            m = pred & (back_pred <= back_tol_m) & (back_pred < out)
            out[m] = back_pred[m]
        return out

    # ------------------------------------------------------------------ shortest paths
    @lru_cache(maxsize=2_000)
    def _sssp(self, src: int, limit: float):
        dist, pred = dijkstra(self.matrix, directed=True, indices=src, limit=limit, return_predecessors=True)
        return dist, pred

    def node_distance(self, u: int, v: int, limit: float = np.inf) -> float:
        """Shortest driving distance node u -> node v in metres (inf if unreachable within limit)."""
        if u == v:
            return 0.0
        return float(self._sssp(int(u), float(limit))[0][v])

    def network_distance(self, arc_a: int, t_a: float, arc_b: int, t_b: float, limit: float = np.inf) -> float:
        """Driving distance from fraction t_a along arc_a to fraction t_b along arc_b (metres).

        Same arc and moving forward: the distance along the arc. Otherwise: rest of arc_a +
        shortest path (to-node of a -> from-node of b) + the first part of arc_b.
        """
        la, lb = self.length[arc_a], self.length[arc_b]
        if arc_a == arc_b and t_b >= t_a:
            return float((t_b - t_a) * la)
        mid = self.node_distance(self.arcs[arc_a, 1], self.arcs[arc_b, 0], limit)
        return float((1 - t_a) * la + mid + t_b * lb)

    def shortest_arc_path(self, arc_a: int, arc_b: int, limit: float = np.inf) -> list[int] | None:
        """Arc ids strictly between arc_a and arc_b on the shortest path (None if unreachable)."""
        u, v = int(self.arcs[arc_a, 1]), int(self.arcs[arc_b, 0])
        if u == v:
            return []
        dist, pred = self._sssp(u, float(limit))
        if not np.isfinite(dist[v]):
            return None
        nodes = [v]
        while nodes[-1] != u:
            nodes.append(int(pred[nodes[-1]]))
        nodes.reverse()
        return [self._edge_arc[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]
