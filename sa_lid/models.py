"""Two predeclared, CPU-friendly pipelines, fitted on training rows only."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

CANDIDATES = {
    "word_tfidf_nb": {"analyzer": "word", "ngram_range": [1, 1], "max_features": 60000,
                      "min_df": 1, "sublinear_tf": False, "alpha": 1.0},
    "char_tfidf_svm": {"analyzer": "char_wb", "ngram_range": [3, 5], "max_features": 150000,
                       "min_df": 2, "sublinear_tf": True, "C": 1.0, "max_iter": 3000},
}


def build_model(name):
    config = CANDIDATES[name]
    vectorizer = TfidfVectorizer(
        analyzer=config["analyzer"], ngram_range=tuple(config["ngram_range"]),
        max_features=config["max_features"], min_df=config["min_df"],
        sublinear_tf=config["sublinear_tf"], lowercase=False, dtype=np.float32,
    )
    classifier = (MultinomialNB(alpha=config["alpha"]) if name == "word_tfidf_nb" else
                  LinearSVC(C=config["C"], dual=True, max_iter=config["max_iter"], random_state=42))
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
