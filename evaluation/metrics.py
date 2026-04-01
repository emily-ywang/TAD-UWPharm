"""
Evaluation metrics for ordinal 0/1/2 rubric scores.
Covers accuracy, macro F1, MAE, and quadratic weighted kappa (QWK).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

from config import DIMENSIONS


def compute_metrics(y_true: list | np.ndarray, y_pred: list | np.ndarray) -> dict:
    """Compute all evaluation metrics for a single scoring target.

    Returns:
        Dict with keys: accuracy, f1_macro, mae, qwk, n.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae":      float(mean_absolute_error(y_true, y_pred)),
        "qwk":      float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n":        int(len(y_true)),
    }


def compute_dimension_metrics(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    include_total: bool = True,
) -> pd.DataFrame:
    """Compute metrics for each rubric dimension (and optionally total_score).

    Args:
        df_true: DataFrame with ground-truth columns (what_score, why_score, how_score, total_score).
        df_pred: DataFrame with predicted columns using the same names.
        include_total: Whether to also evaluate total_score prediction.

    Returns:
        DataFrame indexed by target name with columns [accuracy, f1_macro, mae, qwk, n].
    """
    targets = [f"{d}_score" for d in DIMENSIONS]
    if include_total:
        targets.append("total_score")

    rows = []
    for target in targets:
        if target not in df_pred.columns:
            continue
        metrics = compute_metrics(df_true[target].values, df_pred[target].values)
        metrics["target"] = target
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("target")


def summarize_results(results_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-model metric DataFrames into a single wide comparison table.

    Args:
        results_by_model: mapping of model_name -> metrics DataFrame
                          (as returned by compute_dimension_metrics).

    Returns:
        Wide DataFrame with MultiIndex columns (model_name, metric).
    """
    frames = []
    for model_name, df_metrics in results_by_model.items():
        df = df_metrics.copy()
        df.columns = pd.MultiIndex.from_tuples([(model_name, col) for col in df.columns])
        frames.append(df)
    return pd.concat(frames, axis=1)


def compute_confusion_matrices(
    df_true: pd.DataFrame, df_pred: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Return a 3×3 confusion matrix per rubric dimension."""
    return {
        f"{d}_score": confusion_matrix(
            df_true[f"{d}_score"], df_pred[f"{d}_score"], labels=[0, 1, 2]
        )
        for d in DIMENSIONS
        if f"{d}_score" in df_pred.columns
    }
