from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from typing import Optional
from .schemas import Prediction
from .config import DB_NAME, MONGO_URI
# ── Connection ────────────────────────────────────────────────


client: AsyncIOMotorClient = None
db = None

async def connect_db():
    """Call on FastAPI startup."""
    global client, db
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    await db.predictions.create_index("prediction_id", unique=True)
    await db.predictions.create_index("prediction_group_id")
    await db.predictions.create_index("created_at")

async def close_db():
    """Call on FastAPI shutdown."""
    client.close()


async def insert_prediction(pred: Prediction) -> str:
    """Insert a new prediction document. Returns the prediction_id."""
    doc = pred.model_dump()
    await db.predictions.insert_one(doc)
    return pred.prediction_id


async def get_prediction(prediction_id: str) -> Optional[dict]:
    """Fetch one prediction by its ID."""
    return await db.predictions.find_one(
        {"prediction_id": prediction_id},
        {"_id": 0}          # exclude Mongo's internal _id from result
    )


async def confirm_prediction(
    prediction_id: str,
    confirmed_class: int,
    notes: str,
) -> bool:
    result = await db.predictions.update_one(
        {"prediction_id": prediction_id},
        {"$set": {
            "confirmed_class": confirmed_class,
            "confirmed": True,
            "notes": notes,
        }}
    )
    return result.modified_count == 1


async def list_predictions(
    limit:          int  = 20,
    offset:         int  = 0,
    confirmed_only: bool = False,
    by_model :      str  = None,
    needs_review :  bool = False,
) -> tuple[list[dict], int]:
    """
    Return a paginated list of predictions.
    Returns (records, total_count) so the caller can build pagination metadata.
    """
    query = {}

    if confirmed_only:
        query["confirmed"] = True

    if by_model:
        query["model_name"] = by_model

    if needs_review:
        query["needs_review"] = True

    total = await db.predictions.count_documents(query)

    cursor = (
        db.predictions
        .find(query, {"_id": 0})
        .sort("created_at", DESCENDING)
        .skip(offset)
        .limit(limit)
    )
    records = await cursor.to_list(length=limit)

    return records, total


async def delete_prediction(prediction_id: str) -> bool:
    result = await db.predictions.delete_one({"prediction_id": prediction_id})
    return result.deleted_count == 1

