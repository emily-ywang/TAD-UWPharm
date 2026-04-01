"""
Approach (a): TF-IDF + linear classifier baseline.
Trains one pipeline per rubric dimension (and optionally for total_score directly).
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import CV_FOLDS, DIMENSIONS, RANDOM_SEED


class TFIDFClassifier:
    """TF-IDF + linear classifier for rubric-decomposed or direct total-score prediction.

    Usage:
        clf = TFIDFClassifier(model_type="logreg")
        clf.fit(df_train)                          # fits WHAT, WHY, HOW, TOTAL
        preds = clf.predict(df_test["reflection_text"].tolist())
        cv_results = clf.cross_validate(df)
    """

    def __init__(self, model_type: str = "logreg", ngram_range: tuple = (1, 2)):
        """
        Args:
            model_type: 'logreg' (logistic regression) or 'svm' (linear SVM)
            ngram_range: TF-IDF unigram+bigram range by default
        """
        if model_type not in ("logreg", "svm"):
            raise ValueError(f"model_type must be 'logreg' or 'svm', got '{model_type}'")
        self.model_type = model_type
        self.ngram_range = ngram_range
        self.pipelines: dict[str, Pipeline] = {}

    def _build_pipeline(self) -> Pipeline:
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        if self.model_type == "logreg":
            clf = LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
            )
        else:
            clf = CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_SEED, class_weight="balanced"), cv=3
            )
        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    def fit(self, df: pd.DataFrame, targets: list[str] | None = None) -> None:
        """Fit one pipeline per target column.

        Args:
            df: must contain 'reflection_text' and all target columns.
            targets: defaults to [what_score, why_score, how_score, total_score].
        """
        if targets is None:
            targets = [f"{d}_score" for d in DIMENSIONS] + ["total_score"]
        texts = df["reflection_text"].tolist()
        for target in targets:
            pipe = self._build_pipeline()
            pipe.fit(texts, df[target].tolist())
            self.pipelines[target] = pipe

    def predict(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Return predictions for every fitted target."""
        return {target: pipe.predict(texts) for target, pipe in self.pipelines.items()}

    def cross_validate(
        self, df: pd.DataFrame, targets: list[str] | None = None
    ) -> dict[str, dict]:
        """Stratified k-fold CV; returns accuracy and macro-F1 per target."""
        if targets is None:
            targets = [f"{d}_score" for d in DIMENSIONS] + ["total_score"]
        texts = df["reflection_text"].tolist()
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        results = {}
        for target in targets:
            pipe = self._build_pipeline()
            scores = cross_validate(
                pipe, texts, df[target].tolist(),
                cv=cv, scoring=["accuracy", "f1_macro"],
            )
            results[target] = {
                "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "accuracy_std":  float(np.std(scores["test_accuracy"])),
                "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                "f1_macro_std":  float(np.std(scores["test_f1_macro"])),
            }
        return results
