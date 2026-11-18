"""P3 — non-learned reference baselines on the feature tables (no training).

Observation-level Top-1 accuracy (reliable points only) of simple hand rules. Any learned
model has to beat these to justify itself.

    nearest              closest candidate arc (ignores direction)
    nearest_heading      closest arc whose bearing is within 90° of the GPS heading
    window_heading       smallest mean distance over points i-3..i+3, same heading rule

Output: results/p3/rule_baselines.csv
Run:    python -m scripts.p3_rule_baselines
"""
from __future__ import annotations

import pandas as pd

from pathsnap.io import REPO_ROOT

RULES = {
    "nearest": lambda d: -d.dist_m,
    "nearest_heading": lambda d: -d.dist_m - 1000 * (d.heading_diff_deg > 90),
    "window_heading": lambda d: -d.window_mean_dist_m - 1000 * (d.heading_diff_deg > 90),
}


def top1(df: pd.DataFrame, score: pd.Series) -> float:
    best = df.assign(s=score).groupby(["trip_id", "obs_idx"]).s.idxmax()
    return float(df.loc[best, "label"].mean())


def main():
    rows = []
    for part in ("train", "student_test"):
        df = pd.read_parquet(REPO_ROOT / "data_cache" / f"features_{part}.parquet")
        df = df[~df.unreliable]
        for name, rule in RULES.items():
            rows.append({"partition": part, "rule": name, "top1": round(top1(df, rule(df)), 4)})
    out = pd.DataFrame(rows)
    path = REPO_ROOT / "results" / "p3" / "rule_baselines.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.pivot(index="rule", columns="partition", values="top1"))


if __name__ == "__main__":
    main()
