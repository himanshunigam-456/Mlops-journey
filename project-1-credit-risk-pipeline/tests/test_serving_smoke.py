"""Live smoke tests for the model loader. Skipped if MLflow is down."""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import pytest
from credit_risk.serving.model_loader import load_staging_model


@pytest.fixture(scope="module")
def mlflow_up() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        urllib.request.urlopen(uri + "/health", timeout=2)
    except Exception:
        pytest.skip(f"MLflow not reachable at {uri} — run `make up` first")
    return uri


def test_load_staging_returns_bundle(mlflow_up):
    bundle = load_staging_model()
    assert bundle.model is not None
    assert bundle.version >= 1
    assert len(bundle.feature_names) > 0


def test_loaded_model_can_predict(mlflow_up):
    bundle = load_staging_model()
    X = np.zeros((1, len(bundle.feature_names)))
    pred = bundle.model.predict(X)
    assert pred.shape == (1,)
    assert int(pred[0]) in (0, 1)
