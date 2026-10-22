from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional
from datetime import datetime, timezone
from fastapi import HTTPException

_no_ns = ConfigDict(protected_namespaces=())

class PredictionBase(BaseModel):
    model_config = _no_ns

    prediction_id:      str
    prediction_group_id: Optional[str] = None
    text:               str
    model_name:         Optional[str] = None
    predicted_class:    int
    confidence:         float  
    probabilities:      dict[str, float]
    needs_review:       bool = False
    note:        str   = ""
    created_at:         datetime = Field(
                        default_factory=lambda: datetime.now(timezone.utc)
                        )

class Prediction(PredictionBase):
    confirmed_class:    Optional[int] = None
    confirmed:          bool = False
    notes:              Optional[str] = None

class Model(BaseModel):
    name : str

class ModelInfo(BaseModel):
    name: str
    metrics: dict[str, Any]

class PredictRequest(BaseModel):
    model_config = _no_ns

    text:               str
    model_name:         str


class PredictAllRequest(BaseModel):
    model_config = _no_ns

    text: str

class PredictAllResponse(BaseModel):
    model_config = _no_ns

    disagreement: bool
    predictions: list[PredictionBase]

class ConfirmedPredictionRequest(BaseModel):
    model_config = _no_ns

    prediction_id :     str
    confirmed_class :   int
    notes :             Optional[str] = None

class ErrorResponse(BaseModel):
    status_code:    int
    message:        str

class PredictionListRequest(BaseModel):
    limit:          int  = Field(default=20, ge=1, le=100)
    offset:         int  = Field(default=0, ge=0)
    confirmed_only: bool = Field(default=False)
    by_model:       Optional[str] = Field(default=None)
    needs_review:   bool = Field(default=False)

class PredictionListResponse(BaseModel):
    total:   int
    limit:   int
    offset:  int
    results: list[Prediction]

class ConfirmPredictionRequest(BaseModel):
    confirmed_class: int      
    notes:           Optional[str] = None

@dataclass
class ModelEntry:
    model: Any
    metrics: dict[str, Any]

class ModelRegistry:
    def __init__(self):
        self._registry: dict[str, ModelEntry] = {}

    def register(self, model_name: str, model: Any, metrics: Optional[dict[str, Any]] = None):
        if hasattr(model, "eval"):
            model.eval()
        self._registry[model_name] = ModelEntry(model=model, metrics=metrics or {})

    def available(self) -> list[str]:
        """Only return models that are ready — hides ones retraining."""
        return [name for name, _ in self._registry.items()]

    def available_with_metrics(self) -> list[tuple[str, dict[str, Any]]]:
        return [(name, entry.metrics) for name, entry in self._registry.items()]

    def get(self, model_name: str) -> ModelEntry:
        entry = self._registry.get(model_name)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        return entry

class SklearnTextModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, text: str):
        return self.pipeline.predict_proba([text])[0]


class TransformerTextModel:
    def __init__(self, encoder, classifier):
        self.encoder = encoder
        self.classifier = classifier

    def predict_proba(self, text: str):
        embedding = self.encoder.encode([text], convert_to_numpy=True)
        return self.classifier.predict_proba(embedding)[0]
