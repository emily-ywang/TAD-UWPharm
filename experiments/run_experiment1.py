"""
Experiment 1: Compare TF-IDF, embedding, and LLM scoring approaches
across WHAT, WHY, and HOW dimensions.

Usage:
    python experiments/run_experiment1.py
    python experiments/run_experiment1.py --llm-models claude gpt4o gemini
    python experiments/run_experiment1.py --data-path data/raw/reflections.csv

Results are saved in three formats inside --output-dir:
    experiment1_results.csv   — wide multi-index table (overwritten each run)
    experiment1_results.json  — timestamped snapshot (one file per run)
    results_log.csv           — long-format log that appends across runs
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIMENSIONS, PROCESSED_DATA_DIR, RANDOM_SEED, RAW_CSV
from evaluation.metrics import compute_dimension_metrics, summarize_results
from preprocessing.loader import load_dataset
from scoring.embedding_classifier import EmbeddingClassifier
from scoring.llm_scorer import score_dataset
from scoring.tfidf_classifier import TFIDFClassifier

load_dotenv()


def run_tfidf(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    clf = TFIDFClassifier(model_type="logreg")
    clf.fit(df_train)
    raw_preds = clf.predict(df_test["reflection_text"].tolist())
    df_pred = pd.DataFrame(raw_preds)
    return compute_dimension_metrics(df_test, df_pred)


def run_embedding(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    clf = EmbeddingClassifier(model_type="logreg")
    clf.fit(df_train)
    raw_preds = clf.predict(df_test["reflection_text"].tolist())
    df_pred = pd.DataFrame(raw_preds)
    return compute_dimension_metrics(df_test, df_pred)


def run_llm(df_test: pd.DataFrame, model: str) -> pd.DataFrame:
    results = score_dataset(df_test["reflection_text"].tolist(), model=model)
    preds = {f"{dim}_score": [r[f"{dim}_score"] for r in results] for dim in DIMENSIONS}
    df_pred = pd.DataFrame(preds)
    return compute_dimension_metrics(df_test, df_pred)


def main(args: argparse.Namespace) -> None:
    df = load_dataset(args.data_path)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["what_score"]
    )
    print(f"Dataset loaded — train: {len(df_train)}, test: {len(df_test)}")

    all_results: dict[str, pd.DataFrame] = {}

    print("\n[1/3] TF-IDF classifier...")
    all_results["tfidf"] = run_tfidf(df_train, df_test)

    print("[2/3] Embedding classifier...")
    all_results["embedding"] = run_embedding(df_train, df_test)

    if args.llm_models:
        for model in args.llm_models:
            print(f"[LLM] Scoring with {model}...")
            all_results[f"llm_{model}"] = run_llm(df_test, model=model)

    summary = summarize_results(all_results)
    print("\n=== Experiment 1 Results ===")
    print(summary.to_string())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Wide CSV (overwritten each run — quick reference)
    # Flatten multi-index columns to "model_metric" so it opens cleanly in any spreadsheet.
    wide_path = out_dir / "experiment1_results.csv"
    flat = summary.copy()
    flat.columns = [f"{model}_{metric}" for model, metric in flat.columns]
    flat.to_csv(wide_path)

    # 2. Timestamped JSON snapshot (one per run — never overwritten)
    json_path = out_dir / f"experiment1_{run_ts}.json"
    snapshot = {
        "timestamp": run_ts,
        "data_path": args.data_path,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "results": {
            model_name: metrics_df.to_dict()
            for model_name, metrics_df in all_results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    # 3. Long-format log (appended across runs — useful for comparing over time)
    log_path = out_dir / "results_log.csv"
    log_rows = []
    for model_name, metrics_df in all_results.items():
        for target, row in metrics_df.iterrows():
            log_rows.append({
                "timestamp": run_ts,
                "data_path": args.data_path,
                "model": model_name,
                "target": target,
                **{k: v for k, v in row.items() if k != "n"},
                "n": int(row["n"]),
            })
    log_df = pd.DataFrame(log_rows)
    write_header = not log_path.exists()
    log_df.to_csv(log_path, mode="a", header=write_header, index=False)

    print(f"\nSaved:")
    print(f"  Wide CSV  → {wide_path}")
    print(f"  JSON snap → {json_path}")
    print(f"  Log (append) → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Dimension-level scoring across model families"
    )
    parser.add_argument(
        "--data-path", default=str(RAW_CSV),
        help="Path to reflections CSV"
    )
    parser.add_argument(
        "--llm-models", nargs="*", choices=["claude", "gpt4o", "llama"], default=[],
        help="LLM models to include (omit to skip LLM scoring)"
    )
    parser.add_argument(
        "--output-dir", default=str(PROCESSED_DATA_DIR / "experiments"),
        help="Directory to save results CSV"
    )
    main(parser.parse_args())
