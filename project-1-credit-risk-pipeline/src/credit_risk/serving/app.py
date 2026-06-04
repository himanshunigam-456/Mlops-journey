"""FastAPI app — /predict, /healthz, /readyz endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Response

from credit_risk.serving.model_loader import ModelBundle, load_staging_model
from credit_risk.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger("credit_risk.serving")

# Module-level slot the lifespan handler fills in. Tests override via
# app.dependency_overrides on `get_bundle` so they never touch MLflow.
_bundle: ModelBundle | None = None


def get_bundle() -> ModelBundle | None:
    """Single source of truth for 'is a bundle available right now?'

    Returns the loaded bundle, or None if startup hasn't loaded one yet.
    Endpoints decide for themselves how to handle None: /healthz reports it,
    /readyz returns 503, /predict raises 503.

    Tests override this via ``app.dependency_overrides[get_bundle]``.
    """
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle
    try:
        _bundle = load_staging_model()
        logger.info(
            "Loaded model v%s with %d features", _bundle.version, len(_bundle.feature_names)
        )
    except Exception as exc:
        logger.exception("Failed to load model on startup: %s", exc)
        _bundle = None
    yield
    _bundle = None


app = FastAPI(
    title="Credit Risk Prediction API",
    version="0.2.0",
    description="Predicts default probability from already-encoded German Credit features.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def healthz(
    bundle: Annotated[ModelBundle | None, Depends(get_bundle)],
) -> HealthResponse:
    loaded = bundle is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.get("/readyz")
def readyz(
    response: Response,
    bundle: Annotated[ModelBundle | None, Depends(get_bundle)],
) -> dict:
    if bundle is None:
        response.status_code = 503
        return {"ready": False}
    return {"ready": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    req: PredictionRequest,
    bundle: Annotated[ModelBundle | None, Depends(get_bundle)],
) -> PredictionResponse:
    """Run inference. The request's `features` dict is reindexed to the
    model's training-time column order; missing columns default to 0."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if bundle.feature_names:
        row = pd.DataFrame([req.features]).reindex(columns=bundle.feature_names, fill_value=0.0)
    else:
        row = pd.DataFrame([req.features])

    X = row.to_numpy(dtype=float)
    pred = int(bundle.model.predict(X)[0])
    proba_default = float(bundle.model.predict_proba(X)[0, 1])

    return PredictionResponse(
        prediction=pred,
        probability_default=proba_default,
        model_version=str(bundle.version),
        request_id=req.request_id,
    )
