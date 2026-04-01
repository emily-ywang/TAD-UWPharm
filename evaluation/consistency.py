"""
Consistency analysis for the LLM-based scorer.
Measures score stability across repeated runs of the same model,
and pairwise agreement across different LLM models.
"""
import time

import numpy as np
import pandas as pd

from config import (
    CONSISTENCY_N_RUNS,
    CONSISTENCY_VARIANCE_THRESHOLD,
    DIMENSIONS,
    LLM_MODELS,
)
from evaluation.metrics import compute_metrics
from scoring.llm_scorer import score_reflection


# ---------------------------------------------------------------------------
# Repeated-run analysis (single model)
# ---------------------------------------------------------------------------

def run_repeated_scoring(
    texts: list[str],
    model: str = "claude",
    n_runs: int = CONSISTENCY_N_RUNS,
    sleep_between: float = 1.0,
) -> list[list[dict]]:
    """Score each reflection n_runs times with the given model.

    Returns:
        Outer list: one entry per reflection.
        Inner list: n_runs result dicts, each tagged with 'run' index.
    """
    all_runs: list[list[dict]] = [[] for _ in texts]
    for run_idx in range(n_runs):
        for text_idx, text in enumerate(texts):
            result = score_reflection(text, model=model)
            result["run"] = run_idx
            all_runs[text_idx].append(result)
            time.sleep(sleep_between)
    return all_runs


def compute_score_variance(repeated_results: list[list[dict]]) -> pd.DataFrame:
    """Compute per-reflection score mean/std/range across repeated runs.

    Args:
        repeated_results: output of run_repeated_scoring.

    Returns:
        DataFrame with one row per reflection; columns: {dim}_mean, {dim}_std, {dim}_range.
    """
    rows = []
    for text_idx, runs in enumerate(repeated_results):
        row: dict = {"reflection_idx": text_idx}
        for dim in DIMENSIONS:
            scores = [r[f"{dim}_score"] for r in runs]
            row[f"{dim}_mean"]  = float(np.mean(scores))
            row[f"{dim}_std"]   = float(np.std(scores))
            row[f"{dim}_range"] = int(max(scores) - min(scores))
        rows.append(row)
    return pd.DataFrame(rows)


def flag_unstable_cases(
    variance_df: pd.DataFrame,
    threshold: float = CONSISTENCY_VARIANCE_THRESHOLD,
) -> pd.DataFrame:
    """Add an 'is_unstable' column; True when any dimension std dev exceeds threshold."""
    std_cols = [f"{d}_std" for d in DIMENSIONS]
    variance_df = variance_df.copy()
    variance_df["is_unstable"] = variance_df[std_cols].max(axis=1) > threshold
    return variance_df


# ---------------------------------------------------------------------------
# Cross-model agreement
# ---------------------------------------------------------------------------

def cross_model_agreement(
    texts: list[str],
    models: list[str] | None = None,
    sleep_between: float = 1.0,
) -> dict:
    """Score every reflection with each model and compute pairwise agreement metrics.

    Args:
        texts: list of reflection strings.
        models: subset of LLM_MODELS keys to compare. Defaults to all three.
        sleep_between: seconds between API calls.

    Returns:
        {
          "predictions":        {model: [result_dict, ...]},
          "agreement_metrics":  {"claude_vs_gpt4o": {"what": metrics_dict, ...}, ...},
        }
    """
    if models is None:
        models = list(LLM_MODELS.keys())

    predictions: dict[str, list[dict]] = {}
    for model in models:
        model_results = []
        for i, text in enumerate(texts):
            result = score_reflection(text, model=model)
            model_results.append(result)
            if i < len(texts) - 1:
                time.sleep(sleep_between)
        predictions[model] = model_results

    # Pairwise metrics per dimension
    agreement_metrics: dict[str, dict] = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            pair_key = f"{m1}_vs_{m2}"
            pair_metrics: dict[str, dict] = {}
            for dim in DIMENSIONS:
                y1 = [r[f"{dim}_score"] for r in predictions[m1]]
                y2 = [r[f"{dim}_score"] for r in predictions[m2]]
                pair_metrics[dim] = compute_metrics(y1, y2)
            agreement_metrics[pair_key] = pair_metrics

    return {"predictions": predictions, "agreement_metrics": agreement_metrics}
