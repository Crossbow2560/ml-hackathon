"""P7 — export the final inference artifacts used by the API (pathsnap.pipeline).

    artifacts/classifier.joblib     chosen candidate scorer (default: tuned LambdaRank)
    artifacts/transition.joblib     learned HMM transition model (only if the learned variant is used)
    artifacts/pipeline_config.yaml  everything else the pipeline needs (R, K, HMM weights, versions)
    artifacts/feature_list.json     already present (P3 contract)

The HMM variant (exponential vs learned) is chosen by the decoder-tuning score on TRAIN trips
(results/hmm/decoder_params.json) — student_test and evaluator data are not used for the choice.
Run: python -m scripts.export_artifacts [--model lgbm_rank_tuned] [--variant auto|learned|exponential]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil

import yaml

from pathsnap.config import load_config
from pathsnap.io import REPO_ROOT

MAX_MB = 50


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lgbm_rank_tuned")
    ap.add_argument("--variant", default="auto", choices=("auto", "learned", "exponential"))
    a = ap.parse_args()
    cfg = load_config()
    art = REPO_ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    dp = json.loads((REPO_ROOT / "results" / "hmm" / "decoder_params.json").read_text())
    variant = a.variant
    if variant == "auto":
        variant = max(("learned", "exponential"), key=lambda v: dp[v]["best_top1"])

    shutil.copy(REPO_ROOT / "results" / "models" / f"{a.model}.joblib", art / "classifier.joblib")
    if variant == "learned":
        shutil.copy(REPO_ROOT / "results" / "hmm" / "transition_learned.joblib", art / "transition.joblib")
    elif (art / "transition.joblib").exists():
        (art / "transition.joblib").unlink()

    best = json.loads((REPO_ROOT / "results" / "optuna" / "best_params.json").read_text())
    conf = {
        "version": f"1.0.{dt.date.today():%Y%m%d}",
        "classifier": {"name": a.model, "file": "classifier.joblib",
                       "params": best["params"] if a.model.endswith("_tuned") else "defaults (see pathsnap/models.py)"},
        "candidates": {"radius_m": cfg["candidates"]["radius_m"], "max_k": cfg["candidates"]["max_k"]},
        "labels": {"unreliable_m": cfg["labels"]["unreliable_m"]},
        "hmm": {"transition": variant, "file": "transition.joblib" if variant == "learned" else None,
                "alpha": float(dp[variant]["alpha"]), "weight": float(dp[variant]["weight"]),
                "gap_s": float(dp["gap_s"]), "v_max_mps": float(dp["v_max_mps"]),
                "chosen_by": "best decoder-tuning Top-1 on train OOF scores",
                "train_top1": {"learned": dp["learned"]["best_top1"], "exponential": dp["exponential"]["best_top1"]}},
    }
    (art / "pipeline_config.yaml").write_text(yaml.safe_dump(conf, sort_keys=False))
    for f in sorted(art.iterdir()):
        mb = f.stat().st_size / 1e6
        print(f"{f.name:24s} {mb:7.2f} MB")
        assert mb < MAX_MB, f"{f.name} exceeds {MAX_MB} MB"
    print(yaml.safe_dump(conf, sort_keys=False))


if __name__ == "__main__":
    main()
