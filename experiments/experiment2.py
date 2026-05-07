"""
Experiment 2: Compare TF-IDF, embedding, and LLM scoring approaches
across WHAT, WHY, and HOW dimensions.

Usage:
    python3 experiments/experiment2.py
    python3 experiments/experiment2.py --llm-models claude-haiku claude gpt4o llama-3.3-70b llama-4-scout gemini-flash
    python3 experiments/experiment2.py --data-path data/raw/reflections.csv

Results are saved in three formats inside --output-dir:
    experiment2_results.csv   — wide multi-index table (overwritten each run)
    experiment2_results.json  — timestamped snapshot (one file per run)
    results_log.csv           — long-format log that appends across runs
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIMENSIONS, PROCESSED_DATA_DIR, RANDOM_SEED, RAW_CSV
from evaluation.metrics import compute_confusion_matrices, compute_dimension_metrics, summarize_results
from preprocessing.loader import load_dataset
from scoring.embedding_classifier import EmbeddingClassifier
from scoring.llm_scorer import score_dataset, set_cache_path
from scoring.tfidf_classifier import TFIDFClassifier

set_cache_path(PROCESSED_DATA_DIR / "experiments" / "llm_score_cache.json")

load_dotenv()


def plot_results(
    all_results: dict[str, pd.DataFrame],
    all_preds: dict[str, pd.DataFrame],
    df_test: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save three plots to out_dir/plots/: QWK bar chart, metrics heatmap, confusion matrices."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    dim_targets = [f"{d}_score" for d in DIMENSIONS]
    model_names = list(all_results.keys())

    # 1. QWK grouped bar chart — one group per dimension, one bar per model
    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.5), 5))
    x = np.arange(len(DIMENSIONS))
    width = 0.8 / len(model_names)
    for i, model in enumerate(model_names):
        qwk_vals = [all_results[model].loc[t, "qwk"] if t in all_results[model].index else 0.0
                    for t in dim_targets]
        ax.bar(x + i * width - 0.4 + width / 2, qwk_vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in DIMENSIONS])
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("QWK")
    ax.set_title("Quadratic Weighted Kappa by Dimension and Model")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "qwk_by_dimension.png", dpi=150)
    plt.close(fig)

    # 2. Heatmap — models × dimensions, coloured by QWK
    qwk_df = pd.DataFrame(
        {model: [all_results[model].loc[t, "qwk"] if t in all_results[model].index else np.nan
                 for t in dim_targets]
         for model in model_names},
        index=[d.upper() for d in DIMENSIONS],
    )
    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.2), 3))
    sns.heatmap(
        qwk_df, annot=True, fmt=".2f", vmin=-0.1, vmax=1.0,
        cmap="RdYlGn", linewidths=0.5, ax=ax,
    )
    ax.set_title("QWK Heatmap — Models vs Dimensions")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "qwk_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Confusion matrices — one figure per model (3 subplots: WHAT / WHY / HOW)
    for model, df_pred in all_preds.items():
        cms = compute_confusion_matrices(df_test, df_pred)
        fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(4 * len(DIMENSIONS), 4))
        for ax, dim in zip(axes, DIMENSIONS):
            target = f"{dim}_score"
            cm = cms.get(target, np.zeros((3, 3), dtype=int))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[0, 1, 2], yticklabels=[0, 1, 2],
                ax=ax, cbar=False,
            )
            ax.set_title(dim.upper())
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        fig.suptitle(f"Confusion Matrices — {model}", fontsize=11)
        fig.tight_layout()
        safe_name = model.replace("/", "_").replace(":", "_")
        fig.savefig(plot_dir / f"confusion_{safe_name}.png", dpi=150)
        plt.close(fig)

    print(f"  Plots saved → {plot_dir}")


def run_tfidf(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clf = TFIDFClassifier(model_type="logreg")
    clf.fit(df_train)
    raw_preds = clf.predict(df_test["reflection_text"].tolist())
    df_pred = pd.DataFrame(raw_preds)
    return compute_dimension_metrics(df_test, df_pred), df_pred


def run_embedding(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clf = EmbeddingClassifier(model_type="logreg")
    clf.fit(df_train)
    raw_preds = clf.predict(df_test["reflection_text"].tolist())
    df_pred = pd.DataFrame(raw_preds)
    return compute_dimension_metrics(df_test, df_pred), df_pred


def run_llm(df_test: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = score_dataset(df_test["reflection_text"].tolist(), model=model)
    preds = {f"{dim}_score": [r[f"{dim}_score"] for r in results] for dim in DIMENSIONS}
    df_pred = pd.DataFrame(preds)
    return compute_dimension_metrics(df_test, df_pred), df_pred


def main(args: argparse.Namespace) -> None:
    df = load_dataset(args.data_path)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Dataset loaded — train: {len(df_train)}, test: {len(df_test)}")

    all_results: dict[str, pd.DataFrame] = {}
    all_preds: dict[str, pd.DataFrame] = {}

    print("\n[1/3] TF-IDF classifier...")
    all_results["tfidf"], all_preds["tfidf"] = run_tfidf(df_train, df_test)

    print("[2/3] Embedding classifier...")
    all_results["embedding"], all_preds["embedding"] = run_embedding(df_train, df_test)

    if args.llm_models:
        for model in args.llm_models:
            print(f"[LLM] Scoring with {model}...")
            key = f"llm_{model}"
            all_results[key], all_preds[key] = run_llm(df_test, model=model)

    summary = summarize_results(all_results)
    print("\n=== Experiment 2 Results ===")
    print(summary.to_string())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Wide CSV (overwritten each run — quick reference)
    # Flatten multi-index columns to "model_metric" so it opens cleanly in any spreadsheet.
    wide_path = out_dir / "experiment2_results.csv"
    flat = summary.copy()
    flat.columns = [f"{model}_{metric}" for model, metric in flat.columns]
    flat.to_csv(wide_path)

    # 2. Timestamped JSON snapshot (one per run — never overwritten)
    json_path = out_dir / f"experiment2_{run_ts}.json"
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

    print("\nGenerating plots...")
    plot_results(all_results, all_preds, df_test, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2: Dimension-level scoring across model families"
    )
    parser.add_argument(
        "--data-path", default="/Users/bellachang/Desktop/TAD-UWProject/TAD-UWPharm/data/raw/SyntheticReflectionData_Experiment.csv",
        help="Path to reflections CSV"
    )
    parser.add_argument(
        "--llm-models", nargs="*",
        choices=[
            # Proprietary — Anthropic (ANTHROPIC_API_KEY)
            "claude-haiku", "claude",
            # Proprietary — OpenAI (OPENAI_API_KEY)
            "gpt4o",
            # Groq open-source (GROQ_API_KEY)
            "llama-3.3-70b", "llama-4-scout",
        ],
        default=[
            # Proprietary
            "claude-haiku", "claude", "gpt4o",
            # Groq
            "llama-3.3-70b", "llama-4-scout",
        ],
        help=(
            "LLM models to run. Anthropic models need ANTHROPIC_API_KEY; "
            "gpt4o needs OPENAI_API_KEY; Groq models need GROQ_API_KEY."
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(PROCESSED_DATA_DIR / "experiments"),
        help="Directory to save results CSV"
    )
    main(parser.parse_args())
