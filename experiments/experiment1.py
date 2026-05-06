"""
Experiment 1: Rubric-decomposed scoring (predict WHAT + WHY + HOW separately, then sum)
vs. direct total-score prediction.

Usage:
    python3 experiments/experiment1.py
    python3 experiments/experiment1.py --data-path data/raw/SyntheticReflectionData_Experiment.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIMENSIONS, PROCESSED_DATA_DIR, RANDOM_SEED, RAW_CSV
from evaluation.metrics import compute_metrics
from preprocessing.loader import load_dataset
from scoring.embedding_classifier import EmbeddingClassifier
from scoring.tfidf_classifier import TFIDFClassifier


def run_decomposed_vs_direct(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    classifier_cls,
    **clf_kwargs,
) -> dict[str, dict]:
    """Train and evaluate both strategies with the given classifier class.

    Decomposed: predict each dimension separately, sum for total.
    Direct:     predict total_score in a single model.
    """
    true_total = df_test["total_score"].values
    dim_targets = [f"{d}_score" for d in DIMENSIONS]

    # --- Decomposed ---
    clf_dec = classifier_cls(**clf_kwargs)
    clf_dec.fit(df_train, targets=dim_targets)
    preds_dec = clf_dec.predict(df_test["reflection_text"].tolist())
    pred_total_dec = sum(preds_dec[t] for t in dim_targets)
    metrics_dec = compute_metrics(true_total, pred_total_dec)

    # --- Direct ---
    clf_dir = classifier_cls(**clf_kwargs)
    clf_dir.fit(df_train, targets=["total_score"])
    preds_dir = clf_dir.predict(df_test["reflection_text"].tolist())
    metrics_dir = compute_metrics(true_total, preds_dir["total_score"])

    return {"decomposed": metrics_dec, "direct": metrics_dir}


def main(args: argparse.Namespace) -> None:
    df = load_dataset(args.data_path)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Dataset loaded — train: {len(df_train)}, test: {len(df_test)}")

    all_results: dict[str, dict] = {}

    print("\nTF-IDF: decomposed vs direct...")
    all_results["tfidf"] = run_decomposed_vs_direct(
        df_train, df_test, TFIDFClassifier, model_type="logreg"
    )

    print("Embedding: decomposed vs direct...")
    all_results["embedding"] = run_decomposed_vs_direct(
        df_train, df_test, EmbeddingClassifier, model_type="logreg"
    )

    # Build comparison table
    rows = []
    for model_name, result in all_results.items():
        for approach, metrics in result.items():
            rows.append({"model": model_name, "approach": approach, **metrics})
    summary_df = pd.DataFrame(rows).set_index(["model", "approach"])

    print("\n=== Experiment 1 Results ===")
    print(summary_df.to_string())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment1_results.csv"
    summary_df.to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Decomposed vs direct total-score prediction"
    )
    parser.add_argument(
        "--data-path", default=str(RAW_CSV),
        help="Path to reflections CSV"
    )
    parser.add_argument(
        "--output-dir", default=str(PROCESSED_DATA_DIR / "experiments"),
        help="Directory to save results CSV"
    )
    main(parser.parse_args())
