"""
Lexical and structural NLP feature extraction.
Uses spaCy for tokenization / lemmatization and VADER for sentiment.
Run once to install the spaCy model:  python -m spacy download en_core_web_sm
"""
from typing import Any

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import FUTURE_MARKERS, CAUSAL_MARKERS, INTROSPECTIVE_MARKERS, SPECIFICITY_MARKERS

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Install it with: python -m spacy download en_core_web_sm"
    )

_vader = SentimentIntensityAnalyzer()


def extract_features(text: str) -> dict[str, Any]:
    """Return a flat dict of all lexical/structural features for a single reflection text."""
    doc = _nlp(text)
    sentences = list(doc.sents)
    words = [t for t in doc if t.is_alpha]

    features: dict[str, Any] = {}

    # --- Structural ---
    features["word_count"] = len(words)
    features["sentence_count"] = len(sentences)
    features["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0.0
    features["type_token_ratio"] = (
        len({t.lower_ for t in words}) / len(words) if words else 0.0
    )

    # --- Lemmas (useful for downstream TF-IDF on lemmatized text) ---
    features["lemmas"] = [t.lemma_.lower() for t in words]

    # --- Language marker counts (absolute and normalised by word count) ---
    text_lower = text.lower()
    wc = max(len(words), 1)
    for name, markers in [
        ("future_marker",       FUTURE_MARKERS),
        ("causal_marker",       CAUSAL_MARKERS),
        ("introspective_marker", INTROSPECTIVE_MARKERS),
        ("specificity_marker",  SPECIFICITY_MARKERS),
    ]:
        count = _count_markers(text_lower, markers)
        features[f"{name}_count"] = count
        features[f"{name}_rate"] = count / wc

    # --- VADER sentiment ---
    vader = _vader.polarity_scores(text)
    features["sentiment_compound"] = vader["compound"]
    features["sentiment_pos"] = vader["pos"]
    features["sentiment_neg"] = vader["neg"]
    features["sentiment_neu"] = vader["neu"]

    # --- Part-of-speech counts ---
    pos_counts: dict[str, int] = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    for pos in ("NOUN", "VERB", "ADJ", "ADV", "PRON"):
        features[f"pos_{pos.lower()}_count"] = pos_counts.get(pos, 0)

    return features


def extract_features_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Extract features for a list of texts."""
    return [extract_features(t) for t in texts]


def get_numeric_feature_names() -> list[str]:
    """Return names of all numeric (non-list) features produced by extract_features."""
    return [
        "word_count", "sentence_count", "avg_sentence_length", "type_token_ratio",
        "future_marker_count",       "future_marker_rate",
        "causal_marker_count",       "causal_marker_rate",
        "introspective_marker_count", "introspective_marker_rate",
        "specificity_marker_count",  "specificity_marker_rate",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "pos_noun_count", "pos_verb_count", "pos_adj_count",
        "pos_adv_count",  "pos_pron_count",
    ]


def _count_markers(text_lower: str, markers: list[str]) -> int:
    return sum(1 for m in markers if m in text_lower)
