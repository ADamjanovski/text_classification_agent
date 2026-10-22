import json
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import Any


def build_tfidf_logreg_pipeline(
        lowercase: bool = True,
        stop_words: list[str]  = None,
        ngram_range: tuple[float, float] = (1,1),
        min_df : int | float = 1,
        max_features : int = None,
        sublinear_tf : bool = False,
        max_iter : int = 100,
):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=lowercase,
                    stop_words=stop_words,
                    ngram_range= ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=sublinear_tf,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                ),
            ),
        ]
    )


MODELS_DIR = Path("../models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_sklearn_model_with_metrics(
    model,
    model_name,
    accuracy,
    macro_f1,
    fit_time,
):
    save_path = MODELS_DIR / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_path / "model.joblib")

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "fit_time_sec": float(fit_time),
    }

    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {save_path}")


def load_sklearn_model_with_metrics(model_path: str | Path) -> tuple[Any, dict[str, Any]]:
    load_path = Path(model_path)
    model = joblib.load(load_path / "model.joblib")

    metrics_path = load_path / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return model, metrics
