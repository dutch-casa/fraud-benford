"""Evaluation metrics and plots for imbalanced binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.benford import BENFORD_PROBS


@dataclass(frozen=True, slots=True)
class EvalReport:
    auprc: float
    roc_auc: float
    precision_at_recall_80: float
    confusion_at_best_f1: np.ndarray
    best_f1_threshold: float
    best_f1: float


def evaluate(y_true: pd.Series, y_score: np.ndarray) -> EvalReport:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    auprc = float(average_precision_score(y_true_arr, y_score))
    roc = float(roc_auc_score(y_true_arr, y_score))

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score)

    recall_mask = recall[:-1] >= 0.80
    p_at_r80 = float(precision[:-1][recall_mask].max()) if recall_mask.any() else 0.0

    p = precision[:-1]
    r = recall[:-1]
    f1s = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    y_pred = (y_score >= best_thr).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred)

    return EvalReport(
        auprc=auprc,
        roc_auc=roc,
        precision_at_recall_80=p_at_r80,
        confusion_at_best_f1=cm,
        best_f1_threshold=best_thr,
        best_f1=best_f1,
    )


def plot_pr_curves(results: dict[str, tuple[pd.Series, np.ndarray]]) -> None:
    plt.figure(figsize=(9, 6))
    for name, (y_true, y_score) in results.items():
        precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_score))
        ap = average_precision_score(np.asarray(y_true), np.asarray(y_score))
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_roc_curves(results: dict[str, tuple[pd.Series, np.ndarray]]) -> None:
    plt.figure(figsize=(9, 6))
    for name, (y_true, y_score) in results.items():
        fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_score))
        auc = roc_auc_score(np.asarray(y_true), np.asarray(y_score))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_benford_histogram(
    digits: pd.Series, title: str = "Leading digit distribution", ax: plt.Axes | None = None
) -> None:
    valid = digits.dropna().astype(int)
    counts = valid.value_counts().sort_index()
    observed = [counts.get(d, 0) / len(valid) if len(valid) else 0.0 for d in range(1, 10)]
    expected = [BENFORD_PROBS[d] for d in range(1, 10)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    xs = np.arange(1, 10)
    ax.bar(xs, observed, alpha=0.7, label="Observed")
    ax.plot(xs, expected, "ro-", linewidth=2, markersize=8, label="Benford")
    ax.set_xticks(xs)
    ax.set_xlabel("Leading digit")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def model_comparison_table(reports: dict[str, EvalReport]) -> pd.DataFrame:
    rows = []
    for name, r in reports.items():
        rows.append(
            {
                "model": name,
                "AUPRC": r.auprc,
                "ROC-AUC": r.roc_auc,
                "P@R=0.80": r.precision_at_recall_80,
                "Best F1": r.best_f1,
                "Threshold": r.best_f1_threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("AUPRC", ascending=False).reset_index(drop=True)
