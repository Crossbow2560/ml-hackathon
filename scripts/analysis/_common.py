"""Shared helpers for the Phase 0 analysis scripts."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pathsnap.io import REPO_ROOT, list_trips, load_trip

# evaluator_test is deliberately not a valid choice (leakage rule)
ANALYSIS_PARTITIONS = ("train", "student_test")


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--partition", default="train", choices=ANALYSIS_PARTITIONS)
    p.add_argument("--data-dir", default=None, help="dataset root (default: $PATHSNAP_DATA_DIR or ./dataset)")
    p.add_argument("--out-dir", default=None, help="output root (default: results/analysis)")
    return p.parse_args()


def out_dir(partition: str, override: str | None = None) -> Path:
    d = Path(override) if override else REPO_ROOT / "results" / "analysis"
    d = d / partition
    d.mkdir(parents=True, exist_ok=True)
    return d


def iter_trips(partition: str, root: str | None = None, label: str = ""):
    """Yield (trip_id, Trip or None, error message or None) with a progress line."""
    ids = list_trips(partition, root)
    t0 = time.time()
    for i, tid in enumerate(ids, 1):
        try:
            trip, err = load_trip(tid, partition, root), None
        except Exception as exc:  # report, do not crash the whole analysis
            trip, err = None, f"{exc.__class__.__name__}: {exc}"
        print(f"\r[{label}] {partition} {i}/{len(ids)} {tid} ({time.time() - t0:.0f}s)", end="", flush=True)
        yield tid, trip, err
    print()


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path} ({len(df)} rows)")
