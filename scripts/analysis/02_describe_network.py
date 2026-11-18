"""Phase 0 / step 02 — road-network description and validation.

Per trip: node/arc counts, arcs pointing at non-existent nodes, self-loops, duplicate arcs,
share of two-way roads (arc u->v whose reverse v->u also exists), arc length statistics,
node degrees, isolated nodes and dead ends, weakly/strongly connected components, and how
well the network bounding box covers the GPS track.

Output: results/analysis/<partition>/network_summary.csv
Run:    python -m scripts.analysis.02_describe_network --partition train
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.geo import haversine_m
from scripts.analysis._common import iter_trips, out_dir, parse_args, save


def valid_arc_mask(arcs: np.ndarray, n_nodes: int) -> np.ndarray:
    return ((arcs >= 0) & (arcs < n_nodes)).all(axis=1)


def components(arcs: np.ndarray, n_nodes: int, connection: str):
    """Connected-component labels of the directed graph ('weak' or 'strong')."""
    a = arcs[valid_arc_mask(arcs, n_nodes)]
    g = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n_nodes, n_nodes)).tocsr()
    return connected_components(g, directed=True, connection=connection)


def describe_network(trip):
    nodes, arcs = trip.nodes, trip.arcs
    n, m = len(nodes), len(arcs)
    ok = valid_arc_mask(arcs, n)
    a = arcs[ok]
    frm, to = a[:, 0], a[:, 1]

    key = frm * n + to
    rev = to * n + frm
    two_way = np.isin(rev, key)
    length = haversine_m(nodes[frm, 0], nodes[frm, 1], nodes[to, 0], nodes[to, 1])

    indeg = np.bincount(to, minlength=n)
    outdeg = np.bincount(frm, minlength=n)
    isolated = (indeg + outdeg) == 0
    n_active = int((~isolated).sum())

    n_weak, lab_w = components(arcs, n, "weak")
    n_strong, lab_s = components(arcs, n, "strong")
    # isolated nodes each form their own component; report counts excluding them too
    w_sizes = np.bincount(lab_w[~isolated]) if n_active else np.array([0])
    s_sizes = np.bincount(lab_s[~isolated]) if n_active else np.array([0])
    w_sizes, s_sizes = w_sizes[w_sizes > 0], s_sizes[s_sizes > 0]

    tr = trip.track
    lo0, la0 = nodes.min(axis=0)
    lo1, la1 = nodes.max(axis=0)
    covered = ((tr.lon >= lo0) & (tr.lon <= lo1) & (tr.lat >= la0) & (tr.lat <= la1)).mean()

    return {
        "trip_id": trip.trip_id,
        "n_nodes": n,
        "n_arcs": m,
        "n_invalid_arcs": int((~ok).sum()),
        "n_self_loops": int((frm == to).sum()),
        "n_duplicate_arcs": int(len(key) - len(np.unique(key))),
        "two_way_share": float(two_way.mean()) if len(a) else np.nan,
        "arc_len_min_m": float(length.min()) if len(a) else np.nan,
        "arc_len_p50_m": float(np.median(length)) if len(a) else np.nan,
        "arc_len_mean_m": float(length.mean()) if len(a) else np.nan,
        "arc_len_p95_m": float(np.percentile(length, 95)) if len(a) else np.nan,
        "arc_len_max_m": float(length.max()) if len(a) else np.nan,
        "n_zero_length_arcs": int((length < 0.01).sum()),
        "network_length_km": float(length.sum() / 1000),
        "out_degree_mean": float(outdeg[~isolated].mean()) if n_active else np.nan,
        "out_degree_max": int(outdeg.max()) if n else 0,
        "in_degree_max": int(indeg.max()) if n else 0,
        "n_isolated_nodes": int(isolated.sum()),
        "n_dead_end_nodes": int(((outdeg == 0) & (indeg > 0)).sum()),     # can enter, cannot leave
        "n_source_only_nodes": int(((indeg == 0) & (outdeg > 0)).sum()),  # can leave, cannot enter
        "n_weak_components": int(len(w_sizes)),
        "n_strong_components": int(len(s_sizes)),
        "largest_weak_share": float(w_sizes.max() / n_active) if n_active else np.nan,
        "largest_strong_share": float(s_sizes.max() / n_active) if n_active else np.nan,
        "n_weak_components_incl_isolated": int(n_weak),
        "n_strong_components_incl_isolated": int(n_strong),
        "track_in_network_bbox_share": float(covered),
    }


def run(partition: str, root: str | None = None, out: str | None = None) -> pd.DataFrame:
    od = out_dir(partition, out)
    rows = []
    for tid, trip, err in iter_trips(partition, root, "network"):
        rows.append({"trip_id": tid, "load_error": err} if trip is None else describe_network(trip))
    df = pd.DataFrame(rows)
    save(df, od / "network_summary.csv")
    print(f"  nodes min/median/max: {df.n_nodes.min()}/{int(df.n_nodes.median())}/{df.n_nodes.max()}")
    print(f"  arcs  min/median/max: {df.n_arcs.min()}/{int(df.n_arcs.median())}/{df.n_arcs.max()}")
    print(f"  two-way share min/median/max: {df.two_way_share.min():.2f}/{df.two_way_share.median():.2f}/{df.two_way_share.max():.2f}")
    print(f"  invalid arcs: {df.n_invalid_arcs.sum()}, self-loops: {df.n_self_loops.sum()}, "
          f"duplicates: {df.n_duplicate_arcs.sum()}, isolated nodes: {df.n_isolated_nodes.sum()}")
    print(f"  trips with >1 weak component: {(df.n_weak_components > 1).sum()}, "
          f"largest strong-component share median: {df.largest_strong_share.median():.3f}")
    return df


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.partition, a.data_dir, a.out_dir)
