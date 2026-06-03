"""Tests for Pydantic v2 request/response schemas at the serving boundary."""

import pytest
from credit_risk.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from pydantic import ValidationError


def test_request_accepts_feature_dict():
    req = PredictionRequest(features={"duration": 12.0, "credit_amount": 1000.0})
    assert req.features["duration"] == 12.0


def test_request_rejects_empty_features():
    with pytest.raises(ValidationError):
        PredictionRequest(features={})


def test_request_accepts_optional_request_id():
    req = PredictionRequest(features={"x": 1.0}, request_id="abc-123")
    assert req.request_id == "abc-123"


def test_request_request_id_defaults_to_none():
    req = PredictionRequest(features={"x": 1.0})
    assert req.request_id is None


def test_response_default_class_is_int():
    resp = PredictionResponse(
        prediction=1, probability_default=0.83, model_version="6", request_id="abc"
    )
    assert resp.prediction == 1
    assert isinstance(resp.prediction, int)


def test_response_probability_must_be_unit_interval():
    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction=1, probability_default=1.5, model_version="6", request_id="abc"
        )


def test_response_prediction_must_be_zero_or_one():
    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction=2, probability_default=0.5, model_version="6", request_id="abc"
        )


def test_health_response_status_must_be_ok_or_degraded():
    HealthResponse(status="ok", model_loaded=True)
    HealthResponse(status="degraded", model_loaded=False)
    with pytest.raises(ValidationError):
        HealthResponse(status="weird", model_loaded=True)
