"""
Approach (b): Sentence embedding + shallow classifier baseline.
Encodes reflections with SentenceTransformer, then fits a LogReg or SVM on top.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC

from config import CV_FOLDS, DIMENSIONS, EMBEDDING_MODEL, RANDOM_SEED


class EmbeddingClassifier:
    """Sentence embedding + shallow classifier for rubric-decomposed or direct scoring.

    Usage:
        clf = EmbeddingClassifier()
        clf.fit(df_train)
        preds = clf.predict(df_test["reflection_text"].tolist())
        cv_results = clf.cross_validate(df)
    """

    def __init__(self, model_type: str = "logreg", embedding_model: str = EMBEDDING_MODEL):
        if model_type not in ("logreg", "svm"):
            raise ValueError(f"model_type must be 'logreg' or 'svm', got '{model_type}'")
        self.model_type = model_type
        self.encoder = SentenceTransformer(embedding_model)
        self.classifiers: dict[str, LogisticRegression | CalibratedClassifierCV] = {}
        self._train_embeddings: np.ndarray | None = None

    def _build_classifier(self):
        if self.model_type == "logreg":
            return LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
            )
        return CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_SEED, class_weight="balanced"), cv=3
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts, show_progress_bar=True, batch_size=32)

    def fit(
        self,
        df: pd.DataFrame,
        targets: list[str] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Fit classifiers.  Pass pre-computed embeddings to avoid re-encoding."""
        if targets is None:
            targets = [f"{d}_score" for d in DIMENSIONS] + ["total_score"]
        if embeddings is None:
            embeddings = self.encode(df["reflection_text"].tolist())
            self._train_embeddings = embeddings
        for target in targets:
            clf = self._build_classifier()
            clf.fit(embeddings, df[target].tolist())
            self.classifiers[target] = clf

    def predict(
        self, texts: list[str], embeddings: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Return predictions for every fitted target."""
        if embeddings is None:
            embeddings = self.encode(texts)
        return {target: clf.predict(embeddings) for target, clf in self.classifiers.items()}

    def cross_validate(
        self,
        df: pd.DataFrame,
        targets: list[str] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> dict[str, dict]:
        """Stratified k-fold CV on embeddings; returns accuracy and macro-F1 per target."""
        if targets is None:
            targets = [f"{d}_score" for d in DIMENSIONS] + ["total_score"]
        if embeddings is None:
            embeddings = self.encode(df["reflection_text"].tolist())
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        results = {}
        for target in targets:
            clf = self._build_classifier()
            scores = cross_validate(
                clf, embeddings, df[target].tolist(),
                cv=cv, scoring=["accuracy", "f1_macro"],
            )
            results[target] = {
                "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "accuracy_std":  float(np.std(scores["test_accuracy"])),
                "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                "f1_macro_std":  float(np.std(scores["test_f1_macro"])),
            }
        return results
