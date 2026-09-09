from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "labels": sorted(set(map(str, y_true)) | set(map(str, y_pred))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(set(map(str, y_true)) | set(map(str, y_pred)))).tolist(),
    }


def binary_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}
