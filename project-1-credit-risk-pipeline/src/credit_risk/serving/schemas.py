"""Pydantic v2 schemas for the credit-risk prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """One prediction request — already-encoded features only.

    `features` is a dict of one-hot-encoded feature_name → float, matching
    the model's training-time column order. Use `examples/encode_and_call.py`
    to convert raw German Credit rows into this shape.
    """

    features: dict[str, float] = Field(min_length=1)
    request_id: str | None = None


class PredictionResponse(BaseModel):
    """One prediction result, with model lineage for observability."""

    prediction: Literal[0, 1] = Field(description="0=non-default, 1=default")
    probability_default: float = Field(ge=0.0, le=1.0)
    model_version: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: Literal["ok", "degraded"]
    model_loaded: bool

    @field_validator("status", mode="before")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v
