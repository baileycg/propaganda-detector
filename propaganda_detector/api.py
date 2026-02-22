"""
api.py – FastAPI backend that serves the propaganda / media-bias detector.

Run
---
    uvicorn api:app --reload --port 8000

Endpoints
---------
    GET  /health           – readiness check
    GET  /models           – list available model backends
    POST /predict          – classify a single text
    POST /predict/batch    – classify a list of texts
    POST /predict/url      – fetch & classify text from a URL
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the package root is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.predictor import PropagandaDetector, TransformerDetector, fetch_text_from_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singletons (loaded once at startup)
# ---------------------------------------------------------------------------
_detectors: Dict[str, PropagandaDetector | TransformerDetector] = {}


def _load_detector(model_type: str) -> PropagandaDetector | TransformerDetector:
    """Lazily load and cache a detector by type."""
    if model_type in _detectors:
        return _detectors[model_type]

    if model_type == "transformer":
        logger.info("Loading transformer (DistilBERT) model …")
        detector = TransformerDetector(model_name="distilbert_model")
    else:
        logger.info("Loading sklearn model …")
        detector = PropagandaDetector(model_name="best_model")

    _detectors[model_type] = detector
    return detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models on startup so the first request is fast."""
    models_dir = Path(__file__).parent / "models"

    # Load sklearn model if it exists
    if (models_dir / "best_model.joblib").exists():
        try:
            _load_detector("sklearn")
        except Exception as exc:
            logger.warning("Could not pre-load sklearn model: %s", exc)

    # Load transformer model if it exists
    if (models_dir / "distilbert_model").is_dir():
        try:
            _load_detector("transformer")
        except Exception as exc:
            logger.warning("Could not pre-load transformer model: %s", exc)

    if not _detectors:
        logger.warning(
            "No models found in %s. Train a model first or "
            "place model files in the models/ directory.",
            models_dir,
        )

    yield  # application runs
    _detectors.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Propaganda Detector API",
    description="Detect propaganda and media bias in text using ML models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify.")
    model_type: str = Field(
        "transformer",
        description="Model backend: 'sklearn' or 'transformer'.",
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for the 'Biased' label.",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "text": "The corrupt regime destroyed everything we worked for!!!",
                "model_type": "transformer",
                "threshold": 0.5,
            }
        ]
    }}


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to classify.")
    model_type: str = Field("transformer")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class URLPredictRequest(BaseModel):
    url: str = Field(..., description="URL to fetch and classify.")
    model_type: str = Field("transformer")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probability_biased: float
    probability_nonbiased: float
    signal_summary: Dict[str, float]
    triggered_words: List[str]
    emotions: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    biased_count: int
    nonbiased_count: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_detector(model_type: str, threshold: float):
    """Return the requested detector, raising 422 on bad model_type."""
    if model_type not in ("sklearn", "transformer"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid model_type '{model_type}'. Use 'sklearn' or 'transformer'.",
        )
    try:
        detector = _load_detector(model_type)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_type}' is not available. {exc}",
        )
    detector.threshold = threshold
    return detector


def _result_to_response(result) -> PredictionResponse:
    d = result.to_dict()
    return PredictionResponse(**d)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Check API health and which models are loaded."""
    return HealthResponse(
        status="ok",
        models_loaded=list(_detectors.keys()),
    )


@app.get("/models", tags=["meta"])
async def available_models():
    """List model backends that are currently loaded and ready."""
    return {
        "loaded": list(_detectors.keys()),
        "supported": ["sklearn", "transformer"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(req: PredictRequest):
    """Classify a single piece of text."""
    detector = _get_detector(req.model_type, req.threshold)
    result = detector.predict(req.text)
    return _result_to_response(result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(req: BatchPredictRequest):
    """Classify multiple texts in one request."""
    if len(req.texts) > 100:
        raise HTTPException(status_code=422, detail="Maximum 100 texts per batch request.")

    detector = _get_detector(req.model_type, req.threshold)
    results = detector.predict_batch(req.texts)
    responses = [_result_to_response(r) for r in results]

    biased = sum(1 for r in responses if r.label == "Propaganda / Biased")
    return BatchPredictionResponse(
        results=responses,
        total=len(responses),
        biased_count=biased,
        nonbiased_count=len(responses) - biased,
    )


@app.post("/predict/url", response_model=PredictionResponse, tags=["prediction"])
async def predict_url(req: URLPredictRequest):
    """Fetch text from a URL, then classify it."""
    try:
        text = fetch_text_from_url(req.url)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {exc}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the URL.")

    detector = _get_detector(req.model_type, req.threshold)
    result = detector.predict(text)
    return _result_to_response(result)


# ---------------------------------------------------------------------------
# Entry point (python api.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
