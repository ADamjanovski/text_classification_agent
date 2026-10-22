from fastapi import APIRouter, Depends, HTTPException
from .schemas import PredictionListRequest, PredictionListResponse, PredictionBase, PredictRequest, PredictAllRequest, PredictAllResponse, Model, ModelInfo, Prediction, ConfirmPredictionRequest
from .database import list_predictions, get_prediction, confirm_prediction
from .service import predict, predict_all, registry
router = APIRouter()


@router.get("/predictions", response_model=PredictionListResponse)
async def get_predictions(params: PredictionListRequest= Depends()):
    records, total = await list_predictions(
        limit          = params.limit,
        offset         = params.offset,
        confirmed_only = params.confirmed_only,
        by_model       = params.by_model,
        needs_review   = params.needs_review
    )
    return PredictionListResponse(
        total   = total,
        limit   = params.limit,
        offset  = params.offset,
        results = records,
    )

@router.get("/predictions/{prediction_id}", response_model=Prediction)
async def get_predictiction_endpoint(prediction_id: str):
    prediction = await get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return Prediction(**prediction)

@router.get("/models", response_model=list[Model])
async def get_models():
    return [Model(name=name) for name in registry.available()]


@router.get("/models_info", response_model=list[ModelInfo])
async def get_models_info():
    return [ModelInfo(name=name, metrics=metrics) for name, metrics in registry.available_with_metrics()]


@router.post("/predict", response_model=PredictionBase)
async def predict_endpoint(params: PredictRequest):
    return await predict(params.text, params.model_name)


@router.post("/predict_all", response_model=PredictAllResponse)
async def predict_all_endpoint(params: PredictAllRequest):
    return await predict_all(params.text)


@router.post("/predictions/{prediction_id}/confirm", response_model=Prediction)
async def confirm_prediction_endpoint(
    prediction_id: str,
    body:          ConfirmPredictionRequest,
):
    updated = await confirm_prediction(
        prediction_id   = prediction_id,
        confirmed_class = body.confirmed_class,
        notes           = body.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction = await get_prediction(prediction_id)
    return Prediction(**prediction)


@router.get("/health")
async def health():
    return {"status": "ok"}
