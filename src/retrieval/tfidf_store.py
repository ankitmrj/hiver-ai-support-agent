from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfStore:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, records: pd.DataFrame):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.records = records.reset_index(drop=True)

    @classmethod
    def fit(cls, records: pd.DataFrame) -> "TfidfStore":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
        mat = vec.fit_transform(records["customer_text"].fillna(""))
        return cls(vec, mat, records)

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(-sims)[:k]
        return [dict(self.records.iloc[i], similarity=float(sims[i])) for i in idx]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "records": self.records}, p)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfStore":
        x = joblib.load(path)
        return cls(x["vectorizer"], x["matrix"], x["records"])
