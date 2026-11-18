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

    # ------------------------------------------------------------------ shortest paths
    @lru_cache(maxsize=20_000)
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
