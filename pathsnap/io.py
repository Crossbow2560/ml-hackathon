"""Raw-data loading for PathSnap.

Each trip lives in ``<data_dir>/<partition>/<trip_id>/`` and has four whitespace-separated,
header-less files:

    <id>.track  lon lat t           GPS observations (~1 Hz)
    <id>.nodes  lon lat             road-network nodes, row index = node id
    <id>.arcs   from_node to_node   directed road segments, row index = arc id
    <id>.route  arc_id              ordered reference route (may be absent at inference)

Loading never silently "fixes" data: format problems are collected in ``Trip.issues`` so
that the Phase 0 analysis can report them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FILE_TYPES = ("track", "nodes", "arcs", "route")
EXPECTED_COLS = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}
EVALUATOR_PARTITION = "evaluator_test"


def data_dir(override: str | os.PathLike | None = None) -> Path:
    """Resolve the dataset root: explicit argument > $PATHSNAP_DATA_DIR > <repo>/dataset."""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("PATHSNAP_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "dataset"


def list_trips(partition: str, root: str | os.PathLike | None = None,
               allow_evaluator: bool = False) -> list[str]:
    """Sorted trip ids in a partition.

    The evaluator partition is refused unless ``allow_evaluator=True`` (only the API sets
    this), so that training/analysis code cannot accidentally read evaluator data.
    """
    if partition == EVALUATOR_PARTITION and not allow_evaluator:
        raise PermissionError("evaluator_test may only be read by the API (leakage rule)")
    pdir = data_dir(root) / partition
    if not pdir.is_dir():
        raise FileNotFoundError(f"partition folder not found: {pdir}")
    return sorted(p.name for p in pdir.iterdir() if p.is_dir())


def trip_dir(trip_id: str, partition: str, root: str | os.PathLike | None = None) -> Path:
    return data_dir(root) / partition / trip_id


def read_table(path: Path, kind: str) -> tuple[pd.DataFrame | None, list[str]]:
    """Read one raw file. Returns (frame or None, list of issue strings)."""
    issues: list[str] = []
    if not path.exists():
        return None, [f"{kind}: file missing"]
    if path.stat().st_size == 0:
        return None, [f"{kind}: file empty"]
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    except Exception as exc:  # malformed rows, ragged columns, ...
        return None, [f"{kind}: parse error ({exc.__class__.__name__}: {exc})"]
    if df.shape[1] != EXPECTED_COLS[kind]:
        issues.append(f"{kind}: expected {EXPECTED_COLS[kind]} columns, found {df.shape[1]}")
    return df, issues


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame            # columns lon, lat, t (float64)
    nodes: np.ndarray              # (N, 2) float64 lon, lat
    arcs: np.ndarray               # (M, 2) int64 from_node, to_node
    route: np.ndarray | None       # (R,) int64 arc ids, None if not supplied
    issues: list[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.track)


def load_trip(trip_id: str, partition: str, root: str | os.PathLike | None = None,
              require_route: bool = False, allow_evaluator: bool = False) -> Trip:
    """Load a trip. Coordinates are kept exactly as stored (lon, lat order)."""
    if partition == EVALUATOR_PARTITION and not allow_evaluator:
        raise PermissionError("evaluator_test may only be read by the API (leakage rule)")
    d = trip_dir(trip_id, partition, root)
    issues: list[str] = []
    frames = {}
    for kind in FILE_TYPES:
        df, iss = read_table(d / f"{trip_id}.{kind}", kind)
        frames[kind] = df
        issues += iss

    if frames["track"] is None or frames["nodes"] is None or frames["arcs"] is None:
        raise ValueError(f"trip {trip_id}: cannot load core files: {issues}")

    track = frames["track"].iloc[:, :3].astype(float)
    track.columns = ["lon", "lat", "t"]
    nodes = frames["nodes"].iloc[:, :2].to_numpy(dtype=float)
    arcs = frames["arcs"].iloc[:, :2].to_numpy(dtype=np.int64)
    route = None
    if frames["route"] is not None:
        route = frames["route"].iloc[:, 0].to_numpy(dtype=np.int64)
    elif require_route:
        raise ValueError(f"trip {trip_id}: route required but missing")

    return Trip(trip_id, partition, track.reset_index(drop=True), nodes, arcs, route, issues)
