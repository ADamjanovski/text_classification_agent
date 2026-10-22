import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ml_code.classical_model import load_sklearn_model_with_metrics
from ml_code.transformer_model import load_transformer_model_with_metrics
from .config import MODEL_CONFIGS
from .service import registry
from .router import router
from .database import connect_db, close_db
from .schemas import SklearnTextModel, TransformerTextModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()

    for name, conf in MODEL_CONFIGS.items():
        path = Path(conf["path"])
        kind = conf["kind"]

        if not path.exists() or not path.is_dir():
            logger.warning(f"Model directory not found: {path}")
            continue

        try:
            if kind == "tfidf":
                pipeline, metrics = load_sklearn_model_with_metrics(path)
                model = SklearnTextModel(pipeline)
            elif kind == "transformers":
                encoder, classifier, metrics = load_transformer_model_with_metrics(path)
                model = TransformerTextModel(encoder, classifier)
            else:
                logger.warning(f"Unsupported model kind '{kind}' for '{name}'")
                continue

            registry.register(name, model, metrics)
            logger.info(f"Loaded model: {name} ({kind})")
        except Exception as exc:
            logger.exception(f"Failed loading model '{name}' from '{path}': {exc}")

    yield

    await close_db()
    registry._registry.clear()
app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
