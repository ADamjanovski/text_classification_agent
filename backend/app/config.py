import os
from pathlib import Path

MODELS_DIR = Path("/app/models")

MODEL_CONFIGS = {
    "tfidf": {
        "path": MODELS_DIR / "tfidf_model",
        "kind": "tfidf",
    },
    "transformers": {
        "path": MODELS_DIR / "transformers_model",
        "kind": "transformers",
    },
}

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME")

CLASS_MAPPING = {1 : "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
