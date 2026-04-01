"""
Exploratory text analysis on the reflection corpus.
Call run_full_eda(df) to generate all plots and save summaries to data/processed/eda_plots/.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DIMENSIONS, SCORE_LABELS, PROCESSED_DATA_DIR
from preprocessing.text_features import extract_features_batch

PLOT_DIR = PROCESSED_DATA_DIR / "eda_plots"


def run_full_eda(df: pd.DataFrame, save: bool = True) -> dict:
    """Run all EDA analyses. Returns a dict of result summaries."""
    if save:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "score_distributions":  plot_score_distributions(df, save=save),
        "length_distributions": plot_length_distributions(df, save=save),
        "tfidf_keywords":       compute_tfidf_keywords_by_score(df, save=save),
        "marker_analysis":      plot_marker_analysis(df, save=save),
        "sentiment_analysis":   plot_sentiment_by_score(df, save=save),
    }

    if save:
        serialisable = {k: v.to_dict() if isinstance(v, pd.DataFrame) else v
                        for k, v in results.items()}
        with open(PROCESSED_DATA_DIR / "eda_summary.json", "w") as f:
            json.dump(serialisable, f, indent=2, default=str)

    return results


# ---------------------------------------------------------------------------
# Individual analyses
# ---------------------------------------------------------------------------

def plot_score_distributions(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Bar charts for WHAT, WHY, HOW score distributions."""
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
        fig.savefig(PLOT_DIR / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(score_counts)


def plot_length_distributions(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Box plots of word / sentence count by total_score."""
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
        fig.savefig(PLOT_DIR / "length_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df[["word_count", "sentence_count"]].describe()


def compute_tfidf_keywords_by_score(
    df: pd.DataFrame, n_top: int = 10, save: bool = True
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
        with open(PROCESSED_DATA_DIR / "tfidf_keywords.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


def plot_marker_analysis(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Mean language-marker rates per dimension score — links rubric markers to scores."""
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
        fig.savefig(PLOT_DIR / "marker_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary)


def plot_sentiment_by_score(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Mean VADER compound sentiment per dimension score."""
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
        fig.savefig(PLOT_DIR / "sentiment_by_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary)
