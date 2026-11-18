"""P4/P5 — model comparison plots, generated ONLY from the result CSVs.

Reads results/models/{model_comparison.csv, cv_folds.csv, *_history.csv} and writes
results/models/plots/:
    01_cv_macro_f1.png          CV macro-F1 per model (mean ± std over the 5 folds, folds as dots)
    02_student_test_top1_f1.png student_test Top-1 and macro-F1 per model
    03_log_loss.png             CV and student_test log loss per model
    04_confusion_matrices.png   student_test confusion matrices (threshold 0.5)
    05_mlp_loss_curves.png      MLP train/validation log loss per epoch (only model with epochs)
    06_train_vs_test.png        train vs CV vs student_test Top-1 (over/underfitting evidence)
Run: python -m scripts.plot_models [--dir results/models]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.io import REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO_ROOT / "results" / "models"))
    d = Path(ap.parse_args().dir)
    out = d / "plots"
    out.mkdir(exist_ok=True)
    cmp_ = pd.read_csv(d / "model_comparison.csv").sort_values("cv_macro_f1_mean")
    folds = pd.read_csv(d / "cv_folds.csv")
    names = cmp_.model.tolist()
    x = np.arange(len(names))

    def save(fig, name):
        fig.savefig(out / name, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved", out / name)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, cmp_.cv_macro_f1_mean, yerr=cmp_.cv_macro_f1_std, capsize=4, color="#4c78a8")
    for i, n in enumerate(names):
        f = folds[folds.model == n].macro_f1
        ax.scatter(np.full(len(f), i), f, color="k", s=10, zorder=3)
    ax.set_xticks(x, names, rotation=20); ax.set_ylabel("macro-F1 (5-fold GroupKFold, train)")
    lo = max(0, (cmp_.cv_macro_f1_mean - cmp_.cv_macro_f1_std).min() - 0.05)
    ax.set_ylim(lo, 1); ax.set_title("Cross-validation macro-F1 by model (dots = folds) — train partition")
    save(fig, "01_cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(10, 4.5)); w = 0.38
    ax.bar(x - w / 2, cmp_.st_top1, w, label="Top-1 accuracy", color="#f58518")
    ax.bar(x + w / 2, cmp_.st_macro_f1, w, label="macro-F1", color="#54a24b")
    ax.set_xticks(x, names, rotation=20); ax.set_ylim(max(0, min(cmp_.st_top1.min(), cmp_.st_macro_f1.min()) - 0.05), 1)
    ax.set_title("student_test: Top-1 accuracy and macro-F1 by model"); ax.legend()
    save(fig, "02_student_test_top1_f1.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, cmp_.cv_log_loss, w, label="CV (train)", color="#4c78a8")
    ax.bar(x + w / 2, cmp_.st_log_loss, w, label="student_test", color="#e45756")
    ax.set_xticks(x, names, rotation=20); ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Log loss by model — train CV and student_test"); ax.legend()
    save(fig, "03_log_loss.png")

    n = len(names); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), cmp_.iterrows()):
        cm = np.array([[r.st_tn, r.st_fp], [r.st_fn, r.st_tp]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%}", ha="center", va="center",
                        color="white" if norm[i, j] > .5 else "black", fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(r.model, fontsize=9); ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("student_test confusion matrices (candidate rows, threshold 0.5)")
    fig.tight_layout(); save(fig, "04_confusion_matrices.png")

    hist = sorted(d.glob("*_history.csv"))
    if hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        for h in hist:
            df = pd.read_csv(h)
            nm = df.model.iloc[0]
            ax.plot(df.epoch, df.train_log_loss, "o-", label=f"{nm} train")
            ax.plot(df.epoch, df.val_log_loss, "s--", label=f"{nm} validation (held-out trips)")
        ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.legend()
        ax.set_title("MLP training vs validation loss — train partition")
        save(fig, "05_mlp_loss_curves.png")

    fig, ax = plt.subplots(figsize=(10, 4.5)); w = 0.27
    ax.bar(x - w, cmp_.train_top1, w, label="train (refit)", color="#9ecae9")
    ax.bar(x, cmp_.cv_top1, w, label="CV (held-out trips)", color="#4c78a8")
    ax.bar(x + w, cmp_.st_top1, w, label="student_test", color="#f58518")
    ax.set_xticks(x, names, rotation=20); ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(max(0, cmp_[["train_top1", "cv_top1", "st_top1"]].min().min() - 0.05), 1)
    ax.set_title("Train vs cross-validation vs student_test Top-1 (gap = overfitting)"); ax.legend()
    save(fig, "06_train_vs_test.png")


if __name__ == "__main__":
    main()
