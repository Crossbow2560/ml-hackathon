"""Phase 0 / step 00 — dataset inventory.

For every trip folder: which of the four expected files exist, their sizes, row and column
counts, inferred dtypes, parse problems, and any unexpected extra files.

Output: results/analysis/<partition>/inventory.csv
Run:    python -m scripts.analysis.00_inventory --partition train
"""
from __future__ import annotations

import pandas as pd

from pathsnap.io import FILE_TYPES, data_dir, list_trips, read_table
from scripts.analysis._common import out_dir, parse_args, save


def run(partition: str, root: str | None = None, out: str | None = None) -> pd.DataFrame:
    od = out_dir(partition, out)
    pdir = data_dir(root) / partition
    rows = []
    for tid in list_trips(partition, root):
        d = pdir / tid
        expected = {f"{tid}.{k}" for k in FILE_TYPES}
        present = {p.name for p in d.iterdir()}
        row = {"trip_id": tid, "partition": partition,
               "unexpected_files": ";".join(sorted(present - expected))}
        issues = []
        for kind in FILE_TYPES:
            path = d / f"{tid}.{kind}"
            df, iss = read_table(path, kind)
            row[f"{kind}_present"] = path.exists()
            row[f"{kind}_bytes"] = path.stat().st_size if path.exists() else 0
            row[f"{kind}_rows"] = 0 if df is None else len(df)
            row[f"{kind}_cols"] = 0 if df is None else df.shape[1]
            row[f"{kind}_dtypes"] = "" if df is None else ",".join(map(str, df.dtypes))
            issues += iss
        row["issues"] = "; ".join(issues)
        rows.append(row)

    inv = pd.DataFrame(rows)
    save(inv, od / "inventory.csv")

    # console summary
    print(f"  trips: {len(inv)}")
    for kind in FILE_TYPES:
        print(f"  {kind:5s}: present in {inv[f'{kind}_present'].sum()}/{len(inv)} trips, "
              f"rows min/median/max = {inv[f'{kind}_rows'].min()}/"
              f"{int(inv[f'{kind}_rows'].median())}/{inv[f'{kind}_rows'].max()}")
    n_issue = (inv["issues"] != "").sum()
    print(f"  trips with format issues: {n_issue}")
    return inv


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.partition, a.data_dir, a.out_dir)
