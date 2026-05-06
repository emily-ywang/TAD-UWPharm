"""
Exploratory text analysis on the reflection corpus.
Call run_full_eda(df, output_dir=...) to generate all plots and save summaries.

Usage:
    python3 preprocessing/eda.py --data-path data/raw/SyntheticReflectionData_Experiment.csv
    python3 preprocessing/eda.py --data-path data/raw/mydata.csv --output-dir data/processed/my_run
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running directly from the preprocessing/ directory or the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DIMENSIONS, SCORE_LABELS, PROCESSED_DATA_DIR
from preprocessing.text_features import extract_features_batch


def run_full_eda(df: pd.DataFrame, save: bool = True, output_dir: Path | str | None = None) -> dict:
    """Run all EDA analyses and save outputs under the selected output directory."""
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    output_dir = Path(output_dir)
    plot_dir = output_dir / "eda_plots"
    if save:
        plot_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "score_distributions":  plot_score_distributions(df, save=save, output_dir=output_dir),
        "length_distributions": plot_length_distributions(df, save=save, output_dir=output_dir),
        "tfidf_keywords":       compute_tfidf_keywords_by_score(df, save=save, output_dir=output_dir),
        "marker_analysis":      plot_marker_analysis(df, save=save, output_dir=output_dir),
        "sentiment_analysis":   plot_sentiment_by_score(df, save=save, output_dir=output_dir),
    }

    if save:
        serialisable = {k: v.to_dict() if isinstance(v, pd.DataFrame) else v
                        for k, v in results.items()}
        with open(output_dir / "eda_summary.json", "w") as f:
            json.dump(serialisable, f, indent=2, default=str)

    return results


# ---------------------------------------------------------------------------
# Individual analyses
# ---------------------------------------------------------------------------

def plot_score_distributions(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Bar charts for WHAT, WHY, HOW score distributions."""
    plot_dir = Path(output_dir) / "eda_plots" if output_dir is not None else PROCESSED_DATA_DIR / "eda_plots"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    score_counts = {}
    for ax, dim in zip(axes, DIMENSIONS):
        col = f"{dim}_score"
        counts = df[col].value_counts().reindex(SCORE_LABELS, fill_value=0)
        score_counts[dim] = counts.to_dict()
        ax.bar(counts.index, counts.values, color="steelblue", edgecolor="black")
        ax.set_title(f"{dim.upper()} Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xticks(SCORE_LABELS)
    fig.suptitle("Score Distributions by Rubric Dimension", y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(plot_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(score_counts)


def plot_length_distributions(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Box plots of word / sentence count by total_score."""
    plot_dir = Path(output_dir) / "eda_plots" if output_dir is not None else PROCESSED_DATA_DIR / "eda_plots"
    features = extract_features_batch(df["reflection_text"].tolist())
    df = df.copy()
    df["word_count"]     = [f["word_count"]     for f in features]
    df["sentence_count"] = [f["sentence_count"] for f in features]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, label in zip(
        axes,
        ["word_count", "sentence_count"],
        ["Word Count", "Sentence Count"],
    ):
        groups = [df[df["total_score"] == s][col].tolist() for s in sorted(df["total_score"].unique())]
        ax.boxplot(groups, labels=sorted(df["total_score"].unique()))
        ax.set_title(f"{label} by Total Score")
        ax.set_xlabel("Total Score")
        ax.set_ylabel(label)
    fig.suptitle("Length Distributions by Total Score")
    plt.tight_layout()
    if save:
        fig.savefig(plot_dir / "length_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df[["word_count", "sentence_count"]].describe()


def compute_tfidf_keywords_by_score(
    df: pd.DataFrame,
    n_top: int = 10,
    save: bool = True,
    output_dir: Path | str | None = None,
) -> dict:
    """Top TF-IDF keywords per score level for each rubric dimension."""
    results: dict[str, dict] = {}
    for dim in DIMENSIONS:
        col = f"{dim}_score"
        dim_results: dict[int, list[str]] = {}
        for score in SCORE_LABELS:
            texts = df[df[col] == score]["reflection_text"].tolist()
            if not texts:
                dim_results[score] = []
                continue
            vec = TfidfVectorizer(
                max_features=500, stop_words="english", ngram_range=(1, 2)
            )
            matrix = vec.fit_transform(texts)
            mean_scores = matrix.mean(axis=0).A1
            top_idx = mean_scores.argsort()[::-1][:n_top]
            terms = vec.get_feature_names_out()
            dim_results[score] = [terms[i] for i in top_idx]
        results[dim] = dim_results

    if save:
        output_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
        with open(output_dir / "tfidf_keywords.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


def plot_marker_analysis(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Mean language-marker rates per dimension score — links rubric markers to scores."""
    plot_dir = Path(output_dir) / "eda_plots" if output_dir is not None else PROCESSED_DATA_DIR / "eda_plots"
    features = extract_features_batch(df["reflection_text"].tolist())
    df = df.copy()
    for col in ("future_marker_rate", "causal_marker_rate",
                "introspective_marker_rate", "specificity_marker_rate"):
        df[col] = [f[col] for f in features]

    # Primary marker expected to correlate with each dimension
    dim_marker_map = {
        "how":  "future_marker_rate",
        "why":  "causal_marker_rate",
        "what": "specificity_marker_rate",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    summary: dict[str, dict] = {}
    for ax, dim in zip(axes, DIMENSIONS):
        marker = dim_marker_map[dim]
        score_col = f"{dim}_score"
        means = df.groupby(score_col)[marker].mean().reindex(SCORE_LABELS, fill_value=0)
        summary[dim] = means.to_dict()
        ax.bar(means.index, means.values, color="coral", edgecolor="black")
        ax.set_title(f"{dim.upper()}: {marker}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Mean Marker Rate")
        ax.set_xticks(SCORE_LABELS)
    fig.suptitle("Language Marker Rates by Score")
    plt.tight_layout()
    if save:
        fig.savefig(plot_dir / "marker_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary)


def plot_sentiment_by_score(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Mean VADER compound sentiment per dimension score."""
    plot_dir = Path(output_dir) / "eda_plots" if output_dir is not None else PROCESSED_DATA_DIR / "eda_plots"
    features = extract_features_batch(df["reflection_text"].tolist())
    df = df.copy()
    df["sentiment_compound"] = [f["sentiment_compound"] for f in features]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    summary: dict[str, dict] = {}
    for ax, dim in zip(axes, DIMENSIONS):
        score_col = f"{dim}_score"
        means = df.groupby(score_col)["sentiment_compound"].mean().reindex(SCORE_LABELS, fill_value=0)
        summary[dim] = means.to_dict()
        ax.bar(means.index, means.values, color="mediumseagreen", edgecolor="black")
        ax.set_title(f"{dim.upper()} Sentiment by Score")
        ax.set_xlabel("Score")
        ax.set_ylabel("Mean VADER Compound")
        ax.set_xticks(SCORE_LABELS)
    fig.suptitle("VADER Sentiment by Score Level")
    plt.tight_layout()
    if save:
        fig.savefig(plot_dir / "sentiment_by_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary)


if __name__ == "__main__":
    from preprocessing.loader import load_dataset

    parser = argparse.ArgumentParser(description="Run EDA on a reflection dataset.")
    parser.add_argument("--data-path", required=True, help="Path to input CSV file")
    parser.add_argument("--output-dir", default=None, help="Directory to save plots and summary (default: data/processed)")
    args = parser.parse_args()

    df = load_dataset(args.data_path)
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_full_eda(df, save=True, output_dir=out_dir)
    print("EDA complete. Plots saved to", (out_dir or Path("data/processed")) / "eda_plots")
