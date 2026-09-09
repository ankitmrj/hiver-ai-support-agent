from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.intents.rules import heuristic_intent


class IntentClassifier:
    def __init__(self, vectorizer, model, labels):
        self.vectorizer = vectorizer
        self.model = model
        self.labels = labels

    @classmethod
    def fit(cls, X: pd.Series, y: pd.Series) -> "IntentClassifier":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=75_000, sublinear_tf=True)
        xx = vec.fit_transform(X.fillna(""))
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(xx, y)
        return cls(vec, model, list(model.classes_))

    def predict_one(self, text: str) -> tuple[str, float]:
        xx = self.vectorizer.transform([text])
        probs = self.model.predict_proba(xx)[0]
        i = int(probs.argmax())
        return str(self.model.classes_[i]), float(probs[i])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        return joblib.load(path)
