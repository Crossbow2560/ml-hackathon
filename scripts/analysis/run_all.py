"""Run every Phase 0 analysis step (00 -> 05) for one partition.

Run: python -m scripts.analysis.run_all --partition train
     python -m scripts.analysis.run_all --partition student_test
"""
from __future__ import annotations

import importlib
import time

from scripts.analysis._common import parse_args

STEPS = ("00_inventory", "01_describe_tracks", "02_describe_network",
         "03_describe_routes", "04_gps_vs_route", "05_summary_report")


def main():
    a = parse_args(__doc__)
    t_all = time.time()
    for step in STEPS:
        print(f"== {step}")
        t0 = time.time()
        importlib.import_module(f"scripts.analysis.{step}").run(a.partition, a.data_dir, a.out_dir)
        print(f"   done in {time.time() - t0:.1f}s")
    print(f"== all steps finished in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
