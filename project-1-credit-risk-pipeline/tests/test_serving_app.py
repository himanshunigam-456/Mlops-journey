"""Unit tests for the FastAPI app — mocks the model bundle to stay offline."""

from __future__ import annotations

import numpy as np
import pytest
from credit_risk.serving.app import app, get_bundle
from credit_risk.serving.model_loader import ModelBundle
from fastapi.testclient import TestClient


class _StubModel:
    """Mimics enough of sklearn's API for the predict endpoint to call."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        out = np.zeros((n, 2))
        out[:, 0] = 0.7
        out[:, 1] = 0.3
        return out


@pytest.fixture
def client():
    """Inject a stub model bundle so tests never touch MLflow."""
    stub = ModelBundle(
        model=_StubModel(),
        version=99,
        feature_names=["duration", "credit_amount"],
    )
    app.dependency_overrides[get_bundle] = lambda: stub
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_healthz_returns_ok_when_model_loaded(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_readyz_returns_200_when_model_loaded(client):
    r = client.get("/readyz")
    assert r.status_code == 200


def test_predict_returns_valid_response(client):
    r = client.post(
        "/predict",
        json={"features": {"duration": 12.0, "credit_amount": 1500.0}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["model_version"] == "99"


def test_predict_echoes_request_id(client):
    r = client.post(
        "/predict",
        json={"features": {"x": 1.0}, "request_id": "trace-abc"},
    )
    assert r.json()["request_id"] == "trace-abc"


def test_predict_rejects_empty_features(client):
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 422
