import time
import json
import joblib
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer

MODELS_DIR = Path("../models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_transformer_embedding_classifier(
    encoder,
    classifier,
    X_train,
    y_train,
    batch_size=64,
):
    start_embed = time.perf_counter()
    X_train_emb = encoder.encode(
        X_train,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embedding_time = time.perf_counter() - start_embed

    start_fit = time.perf_counter()
    classifier.fit(X_train_emb, y_train)
    classifier_fit_time = time.perf_counter() - start_fit

    total_train_time = embedding_time + classifier_fit_time

    return {
        "embedding_time_sec": embedding_time,
        "classifier_fit_time_sec": classifier_fit_time,
        "total_train_time_sec": total_train_time,
    }


def save_transformer_model_with_metrics(
    encoder,
    classifier,
    model_name,
    accuracy,
    macro_f1,
    fit_time,
):
    save_path = MODELS_DIR / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    encoder.save(str(save_path / "encoder"))
    joblib.dump(classifier, save_path / "classifier.joblib")

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "fit_time_sec": float(fit_time),
    }

    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {save_path}")


def load_transformer_model_with_metrics(
    model_path: str | Path,
) -> tuple[SentenceTransformer, Any, dict[str, Any]]:
    load_path = Path(model_path)

    encoder = SentenceTransformer(str(load_path / "encoder"))
    classifier = joblib.load(load_path / "classifier.joblib")

    metrics_path = load_path / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return encoder, classifier, metrics
