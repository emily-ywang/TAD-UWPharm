"""
Cross-model agreement analysis for the LLM-based scorer.
Scores a sample of reflections with two models and reports:
  - pairwise exact-agreement rate and weighted κ per dimension (WHAT/WHY/HOW)
  - each model's QWK against human ground-truth scores (when provided)
"""
import time

from config import DIMENSIONS
from evaluation.metrics import compute_metrics
from scoring.llm_scorer import score_reflection


def cross_model_agreement(
    texts: list[str],
    models: list[str] | None = None,
    true_scores: dict[str, list] | None = None,
    sleep_between: float = 1.0,
) -> dict:
    """Score every reflection with each model and compute agreement metrics.

    Args:
        texts: list of reflection strings.
        models: model keys to compare. Defaults to ['claude', 'llama-4-scout'].
        true_scores: optional dict of human ground-truth scores keyed by dimension
                     e.g. {"what": [...], "why": [...], "how": [...]}.
                     When provided, each model's scores are also evaluated against
                     human labels and returned under "vs_human".
        sleep_between: seconds between API calls.

    Returns:
        {
          "predictions":       {model: [result_dict, ...]},
          "agreement_metrics": {"claude_vs_llama-4-scout": {"what": metrics_dict, ...}},
          "vs_human":          {"claude": {"what": metrics_dict, ...}, ...},  # only if true_scores given
        }
    """
    if models is None:
        models = ["claude", "llama-4-scout"]

    predictions: dict[str, list[dict]] = {}
    for model in models:
        model_results = []
        for i, text in enumerate(texts):
            result = score_reflection(text, model=model)
            model_results.append(result)
            if i < len(texts) - 1:
                time.sleep(sleep_between)
        predictions[model] = model_results

    # Pairwise inter-model agreement
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

    result = {"predictions": predictions, "agreement_metrics": agreement_metrics}

    # Each model vs human ground truth
    if true_scores is not None:
        vs_human: dict[str, dict] = {}
        for model in models:
            model_metrics: dict[str, dict] = {}
            for dim in DIMENSIONS:
                if dim not in true_scores:
                    continue
                y_pred = [r[f"{dim}_score"] for r in predictions[model]]
                model_metrics[dim] = compute_metrics(true_scores[dim], y_pred)
            vs_human[model] = model_metrics
        result["vs_human"] = vs_human

    return result
