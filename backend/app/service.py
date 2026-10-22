import uuid
from typing import Optional
import numpy as np
from fastapi import HTTPException
from .schemas import ModelEntry, PredictionBase, PredictAllResponse, ModelRegistry
from .config import CLASS_MAPPING
from .database import insert_prediction

registry = ModelRegistry()

def run_inference(text: str, model: ModelEntry, threshold: float = 0.4, margin: float = 0.05):
    predictions = model.predict_proba(text)

    predicted = np.argmax(predictions)
    confidence = predictions[predicted]
    probabilities = {str(i + 1): float(score) for i, score in enumerate(predictions)}

    predicted_label = predicted + 1  
    predicted_name = CLASS_MAPPING[predicted_label]

    notes = []
    needs_review = False

    if len(text) < 50:
        needs_review = True
        notes.append("Text is very short (fewer than 50 characters) and may lack enough context for a reliable prediction.")

    if confidence < threshold:
        needs_review = True
        notes.append(f"Low confidence ({confidence * 100:.0f}% < {threshold * 100:.0f}%).")

    for i, score in enumerate(predictions):
        if i == predicted:
            continue
        if (confidence - score) < margin:
            needs_review = True
            notes.append(
                f"Small margin between '{predicted_name}' and "
                f"'{CLASS_MAPPING[i + 1]}' ({(confidence - score) * 100:.0f}% < {margin * 100:.0f}%)."
            )

    note = " ".join(notes)
    if needs_review:
        note = "This prediction needs to be manually reviewed. " + note

    return predicted_label, float(confidence), probabilities, needs_review, note


async def _predict_and_store(text: str, model_name: str, group_id: Optional[str] = None) -> PredictionBase:
    entry = registry.get(model_name)

    predicted_class, confidence, probabilities, needs_review, note = run_inference(text, entry.model)

    prediction = PredictionBase(
        prediction_id=str(uuid.uuid4()),
        prediction_group_id=group_id,
        text=text,
        model_name=model_name,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        needs_review=needs_review,
        note=note,
    )

    await insert_prediction(prediction)
    return prediction


async def predict(text: str, model_name: str) -> PredictionBase:
    return await _predict_and_store(text=text, model_name=model_name)


async def predict_all(text: str) -> PredictAllResponse:
    required_models = ["tfidf", "transformers"]
    missing = [name for name in required_models if name not in registry.available()]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Required models are not loaded: {', '.join(missing)}",
        )

    group_id = str(uuid.uuid4())
    predictions = []
    for model_name in required_models:
        pred = await _predict_and_store(text=text, model_name=model_name, group_id=group_id)
        predictions.append(pred)

    disagreement = len({p.predicted_class for p in predictions}) > 1
    return PredictAllResponse(disagreement=disagreement, predictions=predictions)

